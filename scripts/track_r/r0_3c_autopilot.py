#!/usr/bin/env python3
"""Gated one-GPU R0.3c FAST smoke and CPU adjudication autopilot.

This process never starts full R0.4–R0.6 scoring or FAST generation.
It uses one chosen idle GPU, two native roots, and a strict stop receipt.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
CACHE=ROOT/"outputs/.cache/track_r"
FAST_PY=ROOT/"conda-envs/fast_dips_a15_blackwell/bin/python"
R_PY=ROOT/"conda-envs/prdiff_track_r/bin/python"

def save(path,obj):
    part=Path(str(path)+".partial")
    with part.open("w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    part.replace(path)

def checks(head,run):
    if not str(run).startswith(str(BASE)+"/") or run.exists() or run.is_symlink():
        raise RuntimeError("STOP_R03C_BAD_OUTPUT")
    actual=subprocess.check_output(
        ["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(
        ["git","-C",str(WT),"status","--porcelain"],text=True)
    if actual!=head or dirty or not FAST_PY.is_file() or not R_PY.is_file():
        raise RuntimeError("STOP_R03C_BAD_HEAD_TREE_OR_ENV")

def env_base(gpu):
    env=dict(os.environ)
    values={
      "PYTHONDONTWRITEBYTECODE":"1","PYTHONNOUSERSITE":"1",
      "HOME":str(CACHE/"home"),"TMPDIR":str(CACHE/"tmp"),
      "XDG_CACHE_HOME":str(CACHE),"XDG_CONFIG_HOME":str(CACHE/"config"),
      "XDG_DATA_HOME":str(CACHE/"data"),"XDG_STATE_HOME":str(CACHE/"state"),
      "TORCH_HOME":str(CACHE/"torch"),"HF_HOME":str(CACHE/"hf"),
      "PIP_CACHE_DIR":str(CACHE/"pip"),"PYTHONPATH":str(WT),
      "CUDA_VISIBLE_DEVICES":str(gpu),
      "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
      "OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"
    }
    env.update(values)
    for p in (CACHE,CACHE/"home",CACHE/"tmp",CACHE/"config",
              CACHE/"data",CACHE/"state",CACHE/"torch",CACHE/"hf",CACHE/"pip"):
        p.mkdir(parents=True,exist_ok=True)
    return env

def stage(name,args,env,root,timeout):
    log=root/"logs"/(name+".log")
    with log.open("x",encoding="utf-8") as f:
        f.write("COMMAND "+repr(args)+"\n");f.flush()
        started=time.monotonic()
        p=subprocess.Popen(args,cwd=str(WT),env=env,
                           stdout=f,stderr=subprocess.STDOUT)
        (root/"logs"/(name+".pid")).write_text(str(p.pid)+"\n")
        try:
            rc=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            f.write("STOP_R03C_STAGE_TIMEOUT\n");f.flush()
            p.terminate()
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill();p.wait()
            rc=124
    result={"name":name,"rc":rc,"pid":p.pid,
            "seconds":time.monotonic()-started,"log":str(log)}
    print("R03C_STAGE",name,"RC",rc,flush=True)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--physical-gpu",type=int,required=True)
    ap.add_argument("--g0-gpu-receipt",required=True)
    ap.add_argument("--g1-receipt",required=True)
    a=ap.parse_args()
    root=Path(a.out)
    if a.physical_gpu not in range(4):
        raise RuntimeError("STOP_R03C_GPU_ID")
    checks(a.expected_head,root)
    env=env_base(a.physical_gpu)
    root.mkdir(parents=True,exist_ok=False)
    (root/"logs").mkdir()
    state={"schema":"prdiff.track_r.r0.3c.autopilot.v1",
           "status":"RUNNING","head":a.expected_head,
           "physical_gpu":a.physical_gpu,
           "start_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "stages":[],"full_r0_scoring_authorized":False,
           "protected_cohort_access":False}
    save(root/"AUTOPILOT_STATUS.json",state)
    try:
        generation=root/"fast_gpu"
        commands=[
            ("fast_gpu",[str(FAST_PY),"-u",
              "scripts/track_r/r0_3c_fast_smoke_gpu.py",
              "--out",str(generation),
              "--expected-head",a.expected_head,
              "--g0-gpu-receipt",a.g0_gpu_receipt,
              "--physical-gpu",str(a.physical_gpu)],1200),
            ("adjudication",[str(R_PY),"-u",
              "scripts/track_r/r0_3c_fast_adjudicate_cpu.py",
              "--expected-head",a.expected_head,
              "--fast-smoke-root",str(generation),
              "--g0-gpu-receipt",a.g0_gpu_receipt,
              "--g1-receipt",a.g1_receipt,
              "--out",str(root/"adjudication")],600)
        ]
        for name,args,timeout in commands:
            result=stage(name,args,env,root,timeout)
            state["stages"].append(result)
            if result["rc"]:
                state["status"]="STOP_"+name.upper()
                save(root/"AUTOPILOT_STATUS.json",state)
                return result["rc"]
            save(root/"AUTOPILOT_STATUS.json",state)
        adj=json.loads((root/"adjudication"/"R0_3C_ADJUDICATION.json").read_text())
        if adj["status"]!="R0_3C_PASS_READY_FOR_R04_IMPLEMENTATION":
            raise RuntimeError("STOP_R03C_ADJUDICATION_STATUS")
        state["status"]="R0_3C_PASS_FULL_R0_3_SMOKE_COMPLETE"
        state["full_r0_scoring_authorized"]=False
        state["budget"]=adj["budget"]
        save(root/"AUTOPILOT_STATUS.json",state)
        print("R03C_PASS_AND_STOP_BEFORE_FULL_R0",flush=True)
        return 0
    except Exception as exc:
        state["status"]="STOP_R03C_AUTOPILOT_EXCEPTION"
        state["error"]=repr(exc)
        state["traceback"]=traceback.format_exc()[-1800:]
        save(root/"AUTOPILOT_STATUS.json",state)
        print("STOP_R03C_AUTOPILOT",repr(exc),flush=True)
        return 12

if __name__=="__main__":
    sys.exit(main())
