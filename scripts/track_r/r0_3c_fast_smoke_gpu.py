#!/usr/bin/env python3
"""R0.3c native FAST-DIPS smoke: one A1.5 replay + one frozen DEV80 root.

Run ONLY with the pinned Blackwell FAST interpreter, on one idle GPU.
Reads frozen signed y and DDPM only; NEVER opens ground truth or prior
candidate metrics. Reference comparisons and budget decision are CPU-only.
"""
from __future__ import annotations
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
FAST_PY=ROOT/"conda-envs/fast_dips_a15_blackwell/bin/python"
A15_CONFIG=WT/"configs/track_a/a15_fastdips_exposed_v1.json"
FROZEN_CONFIG=WT/"configs/track_r/r0_frozen_v1.json"
PATHS_CONFIG=WT/"configs/track_r/pac_paths_track_r_v1.json"
B26_INDEX=WT/"configs/b26/b26_dev80_manifest.json"
REFERENCE_ID="69073"
REFERENCE_REP=0
DEV80_ID="63070"
DEV80_REP=0
MANIFEST_SHA="8ab7dc375a39a94d1f5183dd58d82f6d5c1c20a3d7124c7249df6a685c5ef890"
R03B_HEAD="6767f1850235a3df0c94fea9b53135016dbe271f"

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()

def safe(path):
    p=Path(path)
    if (not p.is_file() or any(
        x in str(p).lower() or x in str(p.resolve(strict=False)).lower()
        for x in ("confirmation","confirm305"))):
        raise RuntimeError("STOP_R03C_PROTECTED_OR_MISSING_FILE "+str(p))
    return p

def read_json(path):
    return json.loads(safe(path).read_text(encoding="utf-8"))

def save_x(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")

def git_head(repo):
    return subprocess.check_output(
        ["git","-C",str(repo),"rev-parse","HEAD"],
        text=True,timeout=40,env={**os.environ,"GIT_OPTIONAL_LOCKS":"0"}).strip()

def assert_clean(repo):
    dirty=subprocess.check_output(
        ["git","-C",str(repo),"status","--porcelain","--untracked-files=no"],
        text=True,timeout=40,env={**os.environ,"GIT_OPTIONAL_LOCKS":"0"})
    if dirty:raise RuntimeError("STOP_R03C_TRACKED_REPO_DIRTY "+str(repo))

def gpu_preflight(physical):
    if physical not in range(4) or os.environ.get("CUDA_VISIBLE_DEVICES")!=str(physical):
        raise RuntimeError("STOP_R03C_GPU_BINDING")
    usage={}
    for line in subprocess.check_output(
        ["nvidia-smi","--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],text=True,timeout=30).splitlines():
        i,used=(v.strip() for v in line.split(","))
        usage[int(i)]=int(used)
    if physical not in usage or usage[physical]>1024:
        raise RuntimeError("STOP_R03C_GPU_NOT_IDLE "+str(usage))
    return usage[physical]

def seed_r0(source,image_id,rep):
    payload=f"R0FAST|{source}|{image_id}|{rep}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],"little")&((1<<63)-1)

def validate_native_config(cfg,frozen):
    f=frozen["fast_generation"]
    fields={"sampler":"sampler","T":"T","K_ADMM":"K_ADMM","S_x":"S_x",
            "rho":"rho","epsilon":"epsilon","backtrack":"backtrack",
            "alpha_max":"alpha_max"}
    for a,b in fields.items():
        if cfg[a]!=f[b]:
            raise RuntimeError("STOP_R03C_NATIVE_CONFIG_DRIFT "+a)

def gather_sources(paths):
    idx=read_json(B26_INDEX)
    s2=[r for r in idx["rows"] if str(r["image_id"]).zfill(5)==DEV80_ID]
    if len(s2)!=1 or s2[0]["method_role"]!="DEVELOPMENT":
        raise RuntimeError("STOP_R03C_S2_SOURCE_NOT_UNIQUE")
    s2=s2[0]
    freeze=paths["sources"]["S3_a12"]
    frozen_path=Path(freeze["inputs_root"])/"A1_2_INPUT_FREEZE_COMPLETE.json"
    if sha(safe(frozen_path))!=freeze["input_freeze_manifest_sha256"]:
        raise RuntimeError("STOP_R03C_A12_FREEZE_SHA")
    frozen=read_json(frozen_path)
    a15=[r for r in frozen["rows"] if str(r["image_id"])==REFERENCE_ID]
    if len(a15)!=1:
        raise RuntimeError("STOP_R03C_A15_SOURCE_NOT_UNIQUE")
    a15=a15[0]
    def as_case(source,iid,rep,measured,file_sha,tensor_sha,seed):
        meas=safe(measured)
        if sha(meas)!=file_sha:raise RuntimeError("STOP_R03C_MEAS_SHA "+source)
        return {"source":source,"image_id":iid,"rep":rep,
                "measurement_path":str(meas),"measurement_sha256":file_sha,
                "measurement_tensor_sha256":tensor_sha,"seed":seed}
    return [
        as_case("A15_REPLAY",REFERENCE_ID,REFERENCE_REP,a15["measurement_path"],
                a15["measurement_file_sha256"],a15["measurement_tensor_sha256"],
                15150000+int(REFERENCE_ID)+REFERENCE_REP),
        as_case("S2_b24_dev80",DEV80_ID,DEV80_REP,s2["measurement_path"],
                s2["measurement_file_sha256"],s2["measurement_tensor_sha256"],
                seed_r0("S2_b24_dev80",DEV80_ID,DEV80_REP))
    ]

def preflight(a):
    if Path(sys.executable).resolve()!=FAST_PY.resolve():
        raise RuntimeError("STOP_R03C_NOT_PINNED_BLACKWELL_INTERPRETER")
    if os.environ.get("PYTHONDONTWRITEBYTECODE")!="1":
        raise RuntimeError("STOP_R03C_BYTECODE_GUARD")
    if (not str(Path(a.out)).startswith(str(BASE)+"/") or
        Path(a.out).exists() or Path(a.out).is_symlink()):
        raise RuntimeError("STOP_R03C_OUTSIDE_OR_OCCUPIED_OUTPUT")
    if git_head(WT)!=a.expected_head:
        raise RuntimeError("STOP_R03C_EXECUTOR_HEAD")
    if subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],
                              text=True,timeout=40):
        raise RuntimeError("STOP_R03C_DIRTY_EXECUTOR")
    prior=read_json(a.g0_gpu_receipt)
    if (prior.get("status")!="G0_FULL_PASS_R0_3B_NOT_FULL_R0_3" or
        prior.get("head")!=R03B_HEAD or
        prior.get("comparison",{}).get("max_relative_difference",1)>=1e-5):
        raise RuntimeError("STOP_R03C_NO_G0_GPU_PASS")
    cfg=read_json(PATHS_CONFIG)
    native=read_json(A15_CONFIG)
    frozen=read_json(FROZEN_CONFIG)
    validate_native_config(native["fast_native_pixel"],frozen)
    external=cfg["external"]["fast_dips"]
    fast=Path(external["path"])
    if git_head(fast)!=external["commit"]:
        raise RuntimeError("STOP_R03C_FAST_SOURCE_HEAD")
    assert_clean(fast)
    if sha(safe(cfg["models"]["ffhq_ddpm"]["path"]))!=cfg["models"]["ffhq_ddpm"]["sha256"]:
        raise RuntimeError("STOP_R03C_CHECKPOINT_SHA")
    cases=gather_sources(cfg)
    usage=gpu_preflight(a.physical_gpu)
    return cfg,native["fast_native_pixel"],cases,usage

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--g0-gpu-receipt",required=True)
    ap.add_argument("--physical-gpu",type=int,required=True)
    a=ap.parse_args()
    cfg,conf,cases,used=preflight(a)
    import torch
    import numpy as np
    from PIL import Image
    from scripts.track_a.a15_fastdips_smoke import load_checkpoint_verified
    from scripts.track_a.a15_fastdips_preflight import load_tensor,tensor_hash
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("STOP_R03C_NOT_SINGLE_CUDA")
    prop=torch.cuda.get_device_properties(0)
    if (prop.major,prop.minor)!=(12,0):
        raise RuntimeError("STOP_R03C_NOT_BLACKWELL")
    fast=Path(cfg["external"]["fast_dips"]["path"])
    sys.path.insert(0,str(fast))
    import main as upstream
    import misc as settings
    from forward_operators import get_operator
    from model import get_model
    if Path(upstream.__file__).resolve()!=fast/"main.py":
        raise RuntimeError("STOP_R03C_FAST_IMPORT_ORIGIN")
    model_path=Path(cfg["models"]["ffhq_ddpm"]["path"])
    device=torch.device("cuda:0")
    # Model creation is outside timing. Per-root exact seeds are reset before
    # get_start(), matching A1.5's fresh-run sampling RNG state.
    ddpm=copy.deepcopy(settings.MODEL_PRESETS["ffhq256ddpm"])
    if ddpm["name"]!="ddpm":raise RuntimeError("STOP_R03C_NOT_PIXEL_DDPM")
    ddpm["model_config"]["model_path"]=str(model_path)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    model=get_model(**ddpm,device="cuda:0",requires_grad=False)
    ckpt_check=load_checkpoint_verified(
        model,model_path,cfg["models"]["ffhq_ddpm"]["sha256"])
    model.eval()
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    receipt={"schema":"prdiff.track_r.r0.3c.fast-smoke.v1",
             "status":"INCOMPLETE","head":a.expected_head,
             "physical_gpu":a.physical_gpu,"initial_used_mib":used,
             "upstream_head":cfg["external"]["fast_dips"]["commit"],
             "checkpoint_sha256":cfg["models"]["ffhq_ddpm"]["sha256"],
             "native_checkpoint_load":ckpt_check,
             "gt_read_by_worker":False,"cases":[],
             "start_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save_x(out/"RUNNING.json",receipt)
    try:
        for c in cases:
            sub=out/("a15_replay" if c["source"]=="A15_REPLAY" else "s2_new")
            sub.mkdir(parents=True,exist_ok=False)
            seed=c["seed"]
            random.seed(seed)
            np.random.seed(seed%(2**32))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            preset=copy.deepcopy(settings.SAMPLER_PRESETS[conf["sampler"]])
            if preset.pop("latent") is not False:
                raise RuntimeError("STOP_R03C_LATENT_PRESET")
            schedule=preset["diffusion_scheduler_config"]
            schedule["num_steps"]=conf["T"]
            sampler=upstream.FAST_DIPS(
                diffusion_scheduler_config=schedule,
                K=conf["K_ADMM"],S=conf["S_x"],rho=conf["rho"],
                epsilon=conf["epsilon"],backtrack=conf["backtrack"],
                alpha_max=conf["alpha_max"])
            operator=get_operator(
                name="phase_retrieval",oversample=2.0,resolution=256,sigma=0.05)
            y=load_tensor(Path(c["measurement_path"]),
                          ("measurement","y","observation"),(1,3,384,384))
            if tensor_hash(y)!=c["measurement_tensor_sha256"]:
                raise RuntimeError("STOP_R03C_OBSERVATION_CONTENT_SHA")
            x0=sampler.get_start(1,model)
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize()
            start=torch.cuda.Event(enable_timing=True)
            end=torch.cuda.Event(enable_timing=True)
            t0=time.perf_counter()
            start.record()
            # Native FAST-DIPS uses measurement gradients internally; do not
            # wrap sample() in torch.no_grad(), unlike the diffusion-prior scorer.
            output=sampler.sample(model,x0,operator,y.to(device),
                                  evaluator=None,record=False,verbose=False)
            end.record()
            torch.cuda.synchronize()
            seconds=time.perf_counter()-t0
            cuda_seconds=float(start.elapsed_time(end)/1000)
            native_seconds=float(sampler.last_sampling_time)
            if (tuple(output.shape)!=(1,3,256,256) or
                not bool(torch.isfinite(output).all())):
                raise RuntimeError("STOP_R03C_NATIVE_OUTPUT_SCHEMA")
            if not (0<cuda_seconds<=seconds*1.1 and
                    0<native_seconds<=seconds*1.1):
                raise RuntimeError("STOP_R03C_NATIVE_TIMINGS")
            cpu=output.detach().cpu().float()
            torch.save({"reconstruction":cpu},sub/"reconstruction.pt")
            image8=torch.round((cpu[0].clamp(-1,1)+1)*127.5).to(torch.uint8)
            Image.fromarray(image8.permute(1,2,0).numpy(),"RGB").save(
                sub/"reconstruction.png")
            case={
                "status":"R0_FAST_SMOKE_CASE_PASS",
                "source":c["source"],"image_id":c["image_id"],
                "rep":c["rep"],"seed":seed,
                "measurement_sha256":c["measurement_sha256"],
                "measurement_tensor_sha256":c["measurement_tensor_sha256"],
                "reconstruction_pt_sha256":sha(sub/"reconstruction.pt"),
                "reconstruction_png_sha256":sha(sub/"reconstruction.png"),
                "sample_only_wall_seconds":seconds,
                "sample_cuda_event_seconds":cuda_seconds,
                "native_step_sum_seconds":native_seconds,
                "gpu_peak_reserved_mib":torch.cuda.max_memory_reserved()/1048576,
                "output":str(sub),
                "new_measurement":False,"gt_access":False,
                "sampling_method":"UNMODIFIED_NATIVE_FAST_T150_ADMM2_S5"
            }
            save_x(sub/"R0_FAST_SMOKE_CASE_COMPLETE.json",case)
            receipt["cases"].append(case)
            print("R03C_FAST_CASE_PASS",c["source"],c["image_id"],
                  "seconds",round(seconds,3),"seed",seed,flush=True)
            del x0,output,cpu,image8,y,operator,sampler
            torch.cuda.empty_cache()
        receipt["status"]="R0_3C_FAST_GPU_SMOKE_COMPLETE_NEEDS_CPU_ADJUDICATION"
        save_x(out/"R0_3C_FAST_GPU_RECEIPT.json",receipt)
        print("R03C_GPU_TWO_CASES_COMPLETE",flush=True)
        return 0
    except Exception as e:
        receipt["status"]="STOP_R0_3C_FAST_GPU"
        receipt["error"]=repr(e)
        receipt["traceback"]=traceback.format_exc()[-2200:]
        save_x(out/"R0_3C_FAST_GPU_FAILURE.json",receipt)
        print("STOP_R03C_FAST",repr(e),flush=True)
        return 8

if __name__=="__main__":
    sys.exit(main())
