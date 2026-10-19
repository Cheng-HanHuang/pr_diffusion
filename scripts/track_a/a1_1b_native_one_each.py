#!/usr/bin/env python3
"""A1.1b: one DAPS/SITCOM/NP-LF/NP-S2 native integration smoke on A1 procedural data.

Uses frozen, read-only B22/B23/B24 helper *source*, but only new Track A outputs.
Never reads the protected B24 confirmation cohort or any FFHQ images. This
procedural test is a compatibility audit, not a reconstruction-quality study.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import signal
import subprocess
import sys
import time

ROOT=Path("/egr/research-pac/huang248")
REPO=ROOT/"pr_diffusion_track_a_a1_oracle_budget"
BASE=ROOT/"outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
PROBE=BASE/"a1_1a_synthetic_probe_v1"
OUT=BASE/"a1_1b_one_each_v1"
DAPS=ROOT/"pr_diffusion_b19_solver/external/daps"
SITCOM=ROOT/"external/SITCOM_ODE"
DIFFFPR=ROOT/"external/DiffFPR"
MODEL=ROOT/"models/ffhq_10m.pt"
DAPS_PY=ROOT/"conda-envs/daps/bin/python"
NP_PY=ROOT/"conda-envs/prdiff_ffhq/bin/python"
DAPS_HEAD="e7a77d094167084faed19b599b96673b7bb11447"
SITCOM_HEAD="275ab67efbd8146bffca20155171ba6be1169c09"
DIFFFPR_HEAD="a45ffe58f18fed8a63d3446600424e2b08733524"
DAPS_DIFF="fbb5b42369ecf0d3b9b67f8fc162053bc40ec32aed41dbd92a67e8d81dcfad69"
MODEL_SHA="81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa"
PROBE_GT_SHA="6eda5926b1cc56e23cdfed5fd70b743bdfe61067c955f778900683a42a6a1123"
PROBE_MEAS_SHA="992a33efa9bfb8e81d97ed61cf28369263ae1f11aeae19273c53973c2470acdb"
GPU_UUIDS={
  0:"GPU-8c9c6250-7b65-20d8-5c81-d6cb618810c3",
  1:"GPU-883c037a-34d2-48c4-467f-9a352fd8fdff",
  2:"GPU-c381c0f4-1dbc-004f-7d3a-1d7f7794dffe",
  3:"GPU-7d65c050-d7e8-5a6b-ee38-1d72d7a5696a",
}
METHODS=("DAPS","SITCOM","NP_LF","NP_S2")
SYNTHETIC_ID="11111"  # Never interpreted as a real FFHQ cohort entry
MAX_PER_METHOD_S=1500
MAX_AGGREGATE_S=3600
MAX_DEVICE_USED_MIB=52452
MIN_DEVICE_FREE_MIB=52096
MAX_INITIAL_DEVICE_USED_MIB=1024
RUN_WINDOW_OPEN_UTC=datetime.fromisoformat('2026-10-19T16:30:00+00:00')
RUN_LATEST_START_UTC=datetime.fromisoformat('2026-10-19T20:00:00+00:00')


def run_start_in_approved_window(now:datetime|None=None)->bool:
    """Conditional one-hour lease must start 12:30–16:00 Eastern Oct 19."""
    clock=(now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return RUN_WINDOW_OPEN_UTC<=clock<=RUN_LATEST_START_UTC


def sha_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def read_json(p:Path)->dict:
    v=json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(v,dict):
        raise ValueError(f"not a JSON dict: {p}")
    return v


def save_new(p:Path,v:dict)->None:
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8") as f:
        json.dump(v,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")


def git(repo:Path,*args:str)->str:
    c=subprocess.run(["git","-C",str(repo),*args],capture_output=True,text=True,
                     timeout=35,check=False,env={**os.environ,"GIT_OPTIONAL_LOCKS":"0"})
    if c.returncode:
        raise RuntimeError(f"Git identity error for {repo}: {args}")
    return c.stdout.strip()


def trusted_git()->dict:
    out={}
    for label,path,expected in (("DAPS",DAPS,DAPS_HEAD),("SITCOM",SITCOM,SITCOM_HEAD),
                                ("DiffFPR",DIFFFPR,DIFFFPR_HEAD)):
        head=git(path,"rev-parse","HEAD")
        if head!=expected:
            raise RuntimeError(f"{label} source HEAD drift: {head}")
        out[label]=head
    d=subprocess.run(["git","-C",str(DAPS),"diff","HEAD","--"],
                     capture_output=True,timeout=60,check=False)
    if d.returncode or hashlib.sha256(d.stdout).hexdigest()!=DAPS_DIFF:
        raise RuntimeError("native DAPS tracked patch differs from preflight")
    return out


def verified_probe()->dict:
    j=read_json(PROBE/"A1_1A_PROBE_MANIFEST.json")
    req={
        "schema":"prdiff.a1.procedural-input.v1",
        "synthetic_only":True,
        "ffhq_accessed":False,
        "gpu_used":False,
        "solver_executed":False,
        "native_daps_head":DAPS_HEAD,
        "ground_truth_tensor_sha256":PROBE_GT_SHA,
        "measurement_tensor_sha256":PROBE_MEAS_SHA,
        "shape":[1,3,384,384],
        "model_range_shape":[1,3,256,256],
        "dtype":"torch.float32",
        "sigma_y":0.05,
        "oversample":2.0,
    }
    for k,v in req.items():
        if j.get(k)!=v:
            raise RuntimeError(f"frozen procedural probe field drift: {k}")
    files={
        "image":("image_path","image_file_sha256","synthetic_probe.png"),
        "gt":("ground_truth_tensor_path","gt_file_sha256","ground_truth_tensor.pt"),
        "measurement":("measurement_path","measurement_file_sha256","measurement.pt"),
    }
    for label,(pk,dk,name) in files.items():
        p=Path(j[pk])
        if p!=PROBE/name or not p.is_file() or p.is_symlink() or sha_file(p)!=j[dk]:
            raise RuntimeError(f"frozen {label} file identity mismatch: {p}")
    return j


def gpu_state(gpu:int)->tuple[str,int,int]:
    c=subprocess.run(["nvidia-smi",f"--id={gpu}","--query-gpu=uuid,memory.free,memory.used",
                      "--format=csv,noheader,nounits"],capture_output=True,text=True,
                     check=False,timeout=15)
    if c.returncode:
        raise RuntimeError(f"nvidia-smi unavailable: {c.stderr[:200]}")
    fields=[x.strip() for x in c.stdout.strip().split(",")]
    if len(fields)!=3:
        raise RuntimeError(f"malformed GPU telemetry: {fields}")
    return fields[0],int(fields[1]),int(fields[2])


def check_gpu(gpu:int)->dict:
    uuid,free,used=gpu_state(gpu)
    if uuid!=GPU_UUIDS[gpu]:
        raise RuntimeError("physical GPU UUID drift")
    if free<MIN_DEVICE_FREE_MIB or used>MAX_INITIAL_DEVICE_USED_MIB:
        raise RuntimeError(f"GPU not safely free: free={free}, used={used} MiB")
    return {"physical_gpu":gpu,"uuid":uuid,"free_mib":free,"used_mib":used}


def import_b24_helper():
    # Reuses *functions* only, never B24 cohort/output/runner entrypoints.
    p=REPO/"scripts/b24/run_b24_1_method_smoke.py"
    spec=importlib.util.spec_from_file_location("a1_b24_frozen_helper",p)
    if spec is None or spec.loader is None:
        raise RuntimeError("frozen helper source not importable")
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def np_config(base_config:dict,family:str,seed:int)->dict:
    if family not in ("NP_LF","NP_S2") or not 0<=seed<2**32:
        raise ValueError("invalid native NP family/seed")
    cfg=json.loads(json.dumps(base_config))
    method=cfg["np1"]
    if family=="NP_LF":
        method.update(config_tag="lf",score_mode="lf",
                      score_reg_lambda=0.0,score_reg_lambda_schedule="constant")
    else:
        method.update(config_tag="s2_preproj_lam001",score_mode="prev_l2",
                      score_reg_lambda=0.01,score_reg_lambda_schedule="pre_projection_only")
    method["seed"]=seed
    method["log_every"]=0
    if method["num_steps"]!=1000 or method["soft_candidates"]!=5 or method["hard_candidates"]!=1:
        raise RuntimeError("B22 frozen NP work budget changed")
    return cfg


def build_input_item(j:dict)->dict:
    return {
      "image_id":SYNTHETIC_ID,
      "split":"A1_1A_PROCEDURAL_NON_FFHQ",
      "row_id":0,
      "measurement_path":j["measurement_path"],
      "measurement_file_sha256":j["measurement_file_sha256"],
      "measurement_tensor_sha256":j["measurement_tensor_sha256"],
      "measurement_shape":j["shape"],
      "measurement_dtype":j["dtype"],
      "ground_truth_tensor_path":j["ground_truth_tensor_path"],
      "ground_truth_tensor_sha256":j["ground_truth_tensor_sha256"],
      "ground_truth_source_path":j["image_path"],
      "ground_truth_source_sha256":j["image_file_sha256"],
    }


def make_native_parent(method:str, root:Path=OUT)->Path:
    """Create only this A1 run's empty method parent before B24 child_spec(rep0)."""
    if method not in ("DAPS","SITCOM") or root!=OUT and not root.is_dir():
        raise RuntimeError("invalid native parent method/root")
    if not root.is_dir():
        raise RuntimeError("A1 own output root missing")
    parent=root/method.lower()
    parent.mkdir(exist_ok=False)
    return parent


def kill_own_group(proc:subprocess.Popen)->None:
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=12)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait(timeout=10)


def run_bounded(spec:dict,gpu:int,run_started:float)->dict:
    if check_gpu(gpu)["free_mib"]<MIN_DEVICE_FREE_MIB:
        raise RuntimeError("GPU fit gate changed before native process")
    logfile=Path(spec["log"])
    if logfile.exists():
        raise RuntimeError("nonoverwrite failure: log already exists")
    t0=time.perf_counter()
    max_used=0
    with logfile.open("x",encoding="utf-8") as f:
        proc=subprocess.Popen(spec["command"],cwd=spec["run_dir"],env=spec["env"],
                              stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while proc.poll() is None:
                if time.perf_counter()-t0>MAX_PER_METHOD_S or \
                   time.perf_counter()-run_started>MAX_AGGREGATE_S:
                    raise RuntimeError("candidate/aggregate wall-time ceiling reached")
                uuid,free,used=gpu_state(gpu)
                if uuid!=GPU_UUIDS[gpu] or used>MAX_DEVICE_USED_MIB:
                    raise RuntimeError(f"GPU UUID/memory hard gate: uuid={uuid}, used={used}")
                max_used=max(max_used,used)
                time.sleep(2)
            if proc.returncode:
                raise RuntimeError(f"{spec['method']} native process returned rc={proc.returncode}; see {logfile}")
        finally:
            kill_own_group(proc)
    timing=read_json(spec["timing"])
    if timing.get("status")!=0:
        raise RuntimeError("missing valid CUDA event timer")
    term=Path(spec["terminal"])
    if not term.is_file() or term.stat().st_size==0:
        raise RuntimeError("native expected terminal file absent/empty")
    return {
      "method":spec["method"],
      "native_seed32":int(spec["native_seed"]),
      "terminal":str(term),
      "terminal_sha256":sha_file(term),
      "timing_json":str(spec["timing"]),
      "log_file":str(logfile),
      "wall_seconds_subprocess":time.perf_counter()-t0,
      "gpu_active_seconds_event":float(timing["gpu_active_seconds"]),
      "timing_runtime_counters":timing.get("runtime_counters"),
      "peak_torch_reserved_mib":float(timing.get("peak_reserved_bytes",0))/(1024**2),
      "max_device_used_mib_observed":max_used,
    }


def prepare_np_spec(family:str,item:dict,seed:int,gpu:int,base_config:dict)->dict:
    label=family.lower()
    run_root=OUT/label
    run_root.mkdir(exist_ok=False)
    input_dir=run_root/"input"
    input_dir.mkdir()
    helper=import_b24_helper()
    save_new(input_dir/"input_manifest.json",helper.compatible_input_manifest(item,REPO))
    cfg=np_config(base_config,family,seed)
    config_path=run_root/"frozen_runtime_config.json"
    save_new(config_path,cfg)
    timing=run_root/"cuda_timing.json"
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=":4096:8",
               PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(REPO)+os.pathsep+env.get("PYTHONPATH",""))
    for key in ("B21_CONT_ENABLE","B21_CONT_STATE_PATH","B21_CONT_NOISE_SEED",
                "B21_SAVE_STATE_STEPS","B23_TRACE_PATH"):
        env.pop(key,None)
    return {
      "method":family,"native_seed":seed,"run_dir":run_root,"env":env,
      "log":run_root/"run.log","timing":timing,
      "terminal":run_root/"np1/reconstruction.pt",
      "command":[str(NP_PY),str(REPO/"scripts/b23/run_python_cuda_timed.py"),
          "--script",str(REPO/"scripts/b22/run_b22_1_np_smoke.py"),
          "--cwd",str(REPO/"scripts/b22"),"--timing-json",str(timing),"--",
          "--config",str(config_path),"--repo_root",str(REPO),"--run_root",str(run_root)]
    }


def score_pngs(rows:list[dict],j:dict)->dict:
    """Offline common 8-bit saved PNG primary score; GT only loaded here."""
    import numpy as np
    import torch
    from PIL import Image
    gt_payload=torch.load(j["ground_truth_tensor_path"],map_location="cpu",weights_only=True)
    gt=gt_payload["ground_truth"]
    if tuple(gt.shape)!=(1,3,256,256):
        raise RuntimeError("GT shape changed before offline scoring")
    gt01=torch.round(((gt.float().clamp(-1,1)+1)/2)*255).clamp(0,255)/255
    def image01(p):
        im=np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)
        if im.shape!=(256,256,3):
            raise RuntimeError(f"saved PNG is not RGB256: {p}")
        return torch.from_numpy(im.copy()).permute(2,0,1).unsqueeze(0)/255.0
    for row in rows:
        if row["method"]=="DAPS":
            path=Path(row["terminal"])
        else:
            path=Path(row["terminal"]).with_name("reconstruction.png")
        if not path.is_file():
            raise RuntimeError(f"missing comparable saved PNG: {path}")
        pred=image01(path)
        mse=(pred-gt01).square().mean().clamp_min(1e-12)
        rotmse=(torch.rot90(pred,2,dims=(-2,-1))-gt01).square().mean().clamp_min(1e-12)
        row["comparable_saved_png_sha256"]=sha_file(path)
        row["offline_raw_png_psnr_db"]=float(-10*torch.log10(mse))
        row["offline_rot180_png_psnr_db"]=float(-10*torch.log10(rotmse))
        row["offline_ambiguity_aware_png_psnr_db"]=max(row["offline_raw_png_psnr_db"],
                                                     row["offline_rot180_png_psnr_db"])
        row["metrics_caution"]="PROCEDURAL_NOT_FFHQ_NO_PAPER_QUALITY_CLAIMS"
    return {"representation":"ALL_METHODS_SAVED_8BIT_RGB_PNG_QUANTIZED_GT",
            "raw_orientation_primary":True,"procedural_probe_only":True,
            "gt_runtime_selection":False}


def precheck(expected_head:str)->tuple[dict,dict]:
    if git(REPO,"rev-parse","HEAD")!=expected_head or \
       git(REPO,"branch","--show-current")!="codex/prdiff-track-a-a1-oracle-budget-20261016":
        raise RuntimeError("A1 runner checkout branch/HEAD not pinned")
    if git(REPO,"status","--porcelain","--untracked-files=no"):
        raise RuntimeError("dirty A1 tracked source")
    source=trusted_git()
    probe=verified_probe()
    if sha_file(MODEL)!=MODEL_SHA:
        raise RuntimeError("model checkpoint hash mismatch")
    for python in (DAPS_PY,ROOT/"conda-envs/sitcom_ode_bw/bin/python",NP_PY):
        if not python.is_file():
            raise RuntimeError(f"missing frozen solver interpreter {python}")
    # Confirm all source files are pinned under this A1 checkout.
    for p in ("scripts/b24/run_b24_1_method_smoke.py",
              "scripts/b22/run_b22_1_sitcom_smoke.py",
              "scripts/b22/run_b22_1_np_smoke.py",
              "scripts/b23/run_python_cuda_timed.py",
              "configs/b22/b22_1_smoke.json"):
        if not (REPO/p).is_file():
            raise RuntimeError(f"frozen B22/B24 execution source unavailable: {p}")
    return source,probe


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=("plan","execute"),required=True)
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--physical-gpu",type=int,choices=range(4))
    ap.add_argument("--reservation-ack",default="")
    args=ap.parse_args()
    source,j=precheck(args.expected_head)
    seeds={}
    from scripts.track_a.a1_protocol import seed32_numpy
    for m in METHODS:seeds[m]=seed32_numpy(m,SYNTHETIC_ID,0)
    if len(set(seeds.values()))!=4:
        raise RuntimeError("native 32bit seed collision")
    planned={
        "schema":"a1.1b.procedural-four-native-smoke.v1",
        "source_head":args.expected_head,
        "procedure":"ONE_FULL_NATIVE_CANDIDATE_PER_METHOD_NOT_K4",
        "probe_synthetic_non_ffhq":True,"protected_confirmation_access":False,
        "method_order":list(METHODS),"native_seeds32":seeds,
        "measurement_content_sha256":PROBE_MEAS_SHA,
        "source_heads":source,
        "checkpoint_sha256":MODEL_SHA,
        "max_each_wall_seconds":MAX_PER_METHOD_S,
        "max_aggregate_wall_seconds":MAX_AGGREGATE_S,
        "max_device_used_mib":MAX_DEVICE_USED_MIB,
        "min_free_mib":MIN_DEVICE_FREE_MIB,
        "max_initial_device_used_mib":MAX_INITIAL_DEVICE_USED_MIB,
        "conditional_run_window_utc":["2026-10-19T16:30:00Z","2026-10-19T20:00:00Z"],
        "offline_metric":"COMMON_SAVED_RGB_8BIT_RAW_AND_ROT180",
        "no_paper_quality_claims":True,
    }
    if args.mode=="plan":
        print(json.dumps({"status":"A1_1B_CPU_PREFLIGHT_PASS","gpu_run_authorized":False,**planned},
                         sort_keys=True))
        return 0
    if args.physical_gpu is None or args.reservation_ack!="TRACK_A_A1_1B_PROCEDURAL_GPU_SMOKE_APPROVED":
        raise RuntimeError("physical GPU and explicit issue #41 RUN RESERVATION acknowledgement required")
    if not run_start_in_approved_window():
        raise RuntimeError("GPU start outside the conditional Oct 19 12:30-16:00 EDT window; renew reservation")
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite A1.1b execution: {OUT}")
    check_gpu(args.physical_gpu)
    OUT.mkdir(parents=True,exist_ok=False)
    planned["gpu"]=check_gpu(args.physical_gpu)
    save_new(OUT/"RUN_IDENTITY.json",planned)
    item=build_input_item(j)
    helper=import_b24_helper()
    result=[]
    started=time.perf_counter()
    try:
        for method in METHODS:
            if time.perf_counter()-started>MAX_AGGREGATE_S:
                raise RuntimeError("aggregate GPU time ceiling")
            if method=="DAPS":
                # Historical B24 child_spec creates parent/rep0 without parents=True.
                native_parent=make_native_parent("DAPS")
                data_name="a1-1b-procedural-"+hashlib.sha256(str(OUT).encode()).hexdigest()[:10]
                data_dir=config_file=None
                try:
                    data_dir,config_file=helper.prepare_daps_dataset(item,data_name)
                    spec=helper.child_spec("DAPS",REPO,item,data_name,native_parent,0,seeds["DAPS"],args.physical_gpu)
                    # Redirect Hydra's own run directory into A1 outputs only.
                    spec["command"].append(f"hydra.run.dir={spec['run_dir'] / 'hydra'}")
                    spec["env"].pop("B23_TRACE_PATH",None)
                    row=run_bounded(spec,args.physical_gpu,started)
                finally:
                    if config_file is not None and config_file.is_file():
                        config_file.unlink()
                    if data_dir is not None and data_dir.is_dir():
                        sample=data_dir/f"{SYNTHETIC_ID}.png"
                        if sample.is_symlink(): sample.unlink()
                        data_dir.rmdir()  # refuses unexpected files; never recursive delete
            elif method=="SITCOM":
                native_parent=make_native_parent("SITCOM")
                spec=helper.child_spec("SITCOM",REPO,item,None,native_parent,0,seeds["SITCOM"],args.physical_gpu)
                spec["env"].pop("B23_TRACE_PATH",None)
                row=run_bounded(spec,args.physical_gpu,started)
            else:
                frozen=read_json(REPO/"configs/b22/b22_1_smoke.json")
                spec=prepare_np_spec(method,item,seeds[method],args.physical_gpu,frozen)
                row=run_bounded(spec,args.physical_gpu,started)
            row["measurement_content_sha256"]=PROBE_MEAS_SHA
            result.append(row)
            save_new(OUT/f"{method}_COMPLETE.json",row)
            print(json.dumps({"status":"A1_1B_METHOD_PASS","method":method,
                 "wall_seconds":row["wall_seconds_subprocess"],
                 "event_gpu_seconds":row["gpu_active_seconds_event"]},sort_keys=True),flush=True)
        metric=score_pngs(result,j)
        if sha_file(Path(j["measurement_path"]))!=j["measurement_file_sha256"]:
            raise RuntimeError("frozen observation changed during solver execution")
        summary={
            "status":"PASS_A1_1B_PROCEDURAL_FOUR_NATIVE_PIPELINE",
            "methods":result,"comparison":metric,
            "total_wall_seconds":time.perf_counter()-started,
            "sum_gpu_event_seconds":sum(v["gpu_active_seconds_event"] for v in result),
            "sum_subprocess_wall_seconds":sum(v["wall_seconds_subprocess"] for v in result),
            "reused_measurement_hash":PROBE_MEAS_SHA,
            "clean_free_runtime_selection":True,
            "probe_kind":"PROCEDURAL_ENGINEERING_ONLY",
            "prospective_ffhq_cohort_reserved":False,
            "long_gpu_run_authorized":False,
        }
        save_new(OUT/"A1_1B_COMPLETE.json",summary)
        print(json.dumps({"status":summary["status"],"total_wall_seconds":summary["total_wall_seconds"],
                          "sum_gpu_event_seconds":summary["sum_gpu_event_seconds"],
                          "methods_completed":len(result)},sort_keys=True))
        return 0
    except BaseException as e:
        save_new(OUT/"A1_1B_STOP.json",{
            "status":"STOP","error":f"{type(e).__name__}: {e}",
            "methods_completed":result,"gpu":args.physical_gpu})
        raise


if __name__=="__main__":
    try:sys.exit(main())
    except Exception as exc:
        print(f"A1_1B_STOP: {type(exc).__name__}: {exc}",flush=True)
        sys.exit(4)
