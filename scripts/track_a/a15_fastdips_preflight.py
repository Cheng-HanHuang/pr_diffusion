#!/usr/bin/env python3
"""A1.5 *CPU-only* FAST-DIPS source/input/operator preflight.

Exposed A1.2 B22 frozen inputs only. No source downloads, no checkpoint/model
inference, no CUDA init, no new measurements, no original source mutations.
Writes a new receipt ONLY after every gate passes.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

CFG = Path(__file__).resolve().parents[2]/"configs/track_a/a15_fastdips_exposed_v1.json"

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""):h.update(b)
    return h.hexdigest()

def file_ok(path:Path,expected:str, label:str):
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"STOP {label}: file missing/symlink {path}")
    got=sha(path)
    if got!=expected:raise RuntimeError(f"STOP {label}: SHA mismatch {got}")
    return got

def load(path:Path):
    return json.loads(path.read_text(encoding="utf-8"))

def git(folder:Path,*args):
    r=subprocess.run(["git","-C",str(folder),*args],capture_output=True,
         text=True,check=False,timeout=40,env={**os.environ,"GIT_OPTIONAL_LOCKS":"0"})
    if r.returncode:raise RuntimeError(f"STOP git {folder.name} {args}: {r.stderr[:200]}")
    return r.stdout.strip()

def tensor_hash(t):
    import torch
    if not isinstance(t,torch.Tensor):raise RuntimeError("STOP non-tensor")
    t=t.detach().cpu().contiguous()
    h=hashlib.sha256(json.dumps({"dtype":str(t.dtype),"shape":list(t.shape)},
         sort_keys=True,separators=(",",":")).encode("utf-8")+b"\0")
    h.update(t.numpy().tobytes(order="C"));return h.hexdigest()

def load_tensor(path:Path, keys, expected_shape):
    import torch
    p=torch.load(str(path),map_location="cpu",weights_only=True)
    t=p if isinstance(p,torch.Tensor) else next(
       (p[k] for k in keys if isinstance(p,dict) and k in p
        and isinstance(p[k],torch.Tensor)),None)
    if t is None or t.device.type!="cpu" or t.dtype!=torch.float32 or \
       tuple(t.shape)!=tuple(expected_shape) or not bool(torch.isfinite(t).all()):
        raise RuntimeError(f"STOP wrong tensor schema {path}")
    return t

def check_source(cfg,head):
    me=Path(cfg["a15_checkout"]); fast=Path(cfg["fast_repo"])
    if git(me,"rev-parse","HEAD")!=head:
        raise RuntimeError("STOP A1.5 execution commit mismatch")
    if git(me,"branch","--show-current")!="codex/prdiff-track-a-a15-fastdips-exposed-20261020":
        raise RuntimeError("STOP A1.5 branch mismatch")
    if git(me,"status","--porcelain","--untracked-files=no"):
        raise RuntimeError("STOP A1.5 tracked source dirty")
    if git(fast,"rev-parse","HEAD")!=cfg["fast_head"]:
        raise RuntimeError("STOP FAST upstream source commit drift")
    if git(fast,"status","--porcelain","--untracked-files=no"):
        raise RuntimeError("STOP native FAST tracked code dirty")
    if (fast/"model.py").is_symlink() or (fast/"main.py").is_symlink():
        raise RuntimeError("STOP symlinked FAST source")
    # Detect source shadows: importer must be loaded from this exact tree.
    return {"a15_commit":head,"fast_head":cfg["fast_head"],
            "fast_operator_blob":git(fast,"rev-parse","HEAD:forward_operators.py"),
            "fast_main_blob":git(fast,"rev-parse","HEAD:main.py"),
            "fast_model_blob":git(fast,"rev-parse","HEAD:model.py")}

def cpu_gate(cfg):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP CUDA_VISIBLE_DEVICES must be empty for preflight")
    # Pixel-only native FAST imports: misc.py eagerly imports skimage,
    # torchmetrics, LPIPS and prettytable. Full upstream requirements also
    # install heavyweight LDM/StableDiffusion packages unnecessary here.
    wanted=["torch","numpy","PIL","scipy","packaging","yaml",
            "torchvision","imageio","tqdm","wandb","setproctitle",
            "torchmetrics","skimage","lpips","prettytable"]
    missing=[n for n in wanted if importlib.util.find_spec(n) is None]
    if missing:
        raise RuntimeError("STOP_MISSING_PYTHON_MODULES:"+",".join(missing))
    import torch
    # Ensure only the newly provisioned isolated interpreter is used.
    if Path(sys.executable).resolve()!=Path(cfg["python"]).resolve():
        raise RuntimeError("STOP wrong FAST-DIPS Python interpreter: "+sys.executable)
    if sys.version_info[:2]!=(3,10):
        raise RuntimeError("STOP FAST native environment requires Python 3.10")
    if torch.__version__.split("+")[0]!="2.3.0":
        raise RuntimeError("STOP FAST PyTorch pin differs from 2.3.0: "+torch.__version__)
    if torch.cuda.is_initialized():
        raise RuntimeError("STOP CUDA already initialized")
    if not hasattr(torch,"func"):
        raise RuntimeError("STOP FAST needs torch.func")
    fast=Path(cfg["fast_repo"])
    sys.path.insert(0,str(fast))
    if "forward_operators" in sys.modules:
        raise RuntimeError("STOP native module unexpectedly preimported")
    module=importlib.import_module("forward_operators")
    if Path(module.__file__).resolve()!=fast/"forward_operators.py":
        raise RuntimeError("STOP native operator module path")
    op=module.get_operator(name="phase_retrieval",oversample=2.0,
                           resolution=256,sigma=0.05)
    if op.pad!=64:raise RuntimeError("STOP native pad !=64")
    import torch.nn.functional as F
    g=torch.Generator(device="cpu").manual_seed(151517)
    x=torch.rand((1,3,256,256),generator=g)*2-1
    native=op(x)
    pad=F.pad(x*.5+.5,(64,64,64,64)).to(torch.complex64)
    indep=torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(
             pad,dim=(-2,-1)),dim=(-2,-1),norm="ortho"),
             dim=(-2,-1)).abs()
    error=float((native-indep).abs().max().item())
    rel=float(torch.linalg.vector_norm((native-indep).double())/
              torch.linalg.vector_norm(indep.double()).clamp_min(1e-12))
    if tuple(native.shape)!=(1,3,384,384) or error>2e-5 or rel>1e-6:
        raise RuntimeError(f"STOP native PR numerical mismatch abs={error} rel={rel}")
    return {"python":sys.version.split()[0],"torch":torch.__version__,
            "native_operator_absolute_max_error":error,"native_relative_l2":rel,
            "fft_centering":"ifftshift/ortho/fftshift","pad":64,
            "gpu_inference":0}

def verify_inputs(cfg):
    import torch
    frozen=Path(cfg["frozen_input_manifest"])
    file_ok(frozen,cfg["frozen_input_manifest_sha256"],"A1 frozen manifest")
    inp=load(frozen)
    if inp.get("status")!="PASS_EXPOSED_B22_INPUTS_FROZEN_NOT_HELDOUT" or \
       inp.get("protected_confirmation_access") is not False:
        raise RuntimeError("STOP original A1 input freeze status")
    rows=inp["rows"]
    if [str(v["image_id"]) for v in rows]!=cfg["ids"]:
        raise RuntimeError("STOP eight-image order")
    receipt=load(Path(cfg["a13_audit_receipt"]))
    if receipt.get("status")!="A13_SOURCE_AND_INPUT_AUDIT_PASS" or \
       receipt.get("audit_file_sha256")!=cfg["a13_audit_sha256"] or \
       receipt.get("candidate_count")!=128 or \
       receipt.get("operator_numeric_gate")!="PASS":
        raise RuntimeError("STOP A1.3 independent audit receipt")
    verified=[]
    for row in rows:
        p=Path(row["measurement_path"]); g=Path(row["ground_truth_tensor_path"])
        file_ok(p,row["measurement_file_sha256"],"frozen signed measurement")
        file_ok(g,row["ground_truth_tensor_file_sha256"],"frozen GT tensor")
        y=load_tensor(p,("measurement","y","observation"),(1,3,384,384))
        gt=load_tensor(g,("ground_truth","image","x"),(1,3,256,256))
        if tensor_hash(y)!=row["measurement_tensor_sha256"] or \
           tensor_hash(gt)!=row["ground_truth_tensor_sha256"]:
            raise RuntimeError("STOP original observation/GT tensor-content mismatch")
        verified.append({"image_id":str(row["image_id"]),
           "measurement_file_sha256":row["measurement_file_sha256"],
           "measurement_tensor_sha256":row["measurement_tensor_sha256"],
           "gt_file_sha256":row["ground_truth_tensor_file_sha256"],
           "signed_negative_fraction":float((y<0).float().mean().item())})
    return verified

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    cfg=load(CFG)
    stage=Path(cfg["output_root"])/cfg["preflight_stage"]
    if Path(args.out)!=stage or stage.exists() or stage.is_symlink():
        raise RuntimeError("STOP output stage not reserved/empty")
    source=check_source(cfg,args.expected_head)
    native=cpu_gate(cfg)
    rows=verify_inputs(cfg)
    ckpt=file_ok(Path(cfg["checkpoint"]),cfg["checkpoint_sha256"],"FFHQ model weights")
    data={"status":"A15_CPU_PREFLIGHT_PASS","source":source,
          "native_operator":native,"observations":rows,
          "checkpoint_sha256":ckpt,"cohort":cfg["cohort"],
          "fast_native_pixel":cfg["fast_native_pixel"],
          "gpu_runs":0,"new_measurements":0,
          "next_stage_automatically_authorized":False}
    # Write only fresh A15 output after source/numerical/input tests all pass.
    stage.mkdir(parents=True,exist_ok=False)
    with (stage/"A15_PREFLIGHT_PASS.json").open("x",encoding="utf8") as f:
        json.dump(data,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print(json.dumps({"status":"A15_CPU_PREFLIGHT_PASS",
         "images":len(rows),"native_operator_relerr":native["native_relative_l2"],
         "source_head":args.expected_head,"fast_head":cfg["fast_head"],
         "receipt_sha256":sha(stage/"A15_PREFLIGHT_PASS.json")},sort_keys=True))
    return 0

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as exc:
        print(f"A15_PREFLIGHT_STOP: {type(exc).__name__}: {exc}",file=sys.stderr)
        sys.exit(7)
