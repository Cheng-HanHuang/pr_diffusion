#!/usr/bin/env python3
"""R0.5 one-GPU eight-root canary; stop before any full 720-root campaign."""
from __future__ import annotations
import argparse
import hashlib
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
PY=ROOT/"conda-envs/prdiff_track_r/bin/python"
FAST=ROOT/"conda-envs/fast_dips_a15_blackwell/bin/python"
RECOVERY=BASE/"r0_4_recovery_20261022T235412Z"

def save(path,data):
    tmp=Path(str(path)+".tmp")
    with tmp.open("w") as f:
        json.dump(data,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for piece in iter(lambda:f.read(4*1024*1024),b""):
            h.update(piece)
    return h.hexdigest()

def environment(gpu):
    env=dict(os.environ)
    values={
        "PYTHONDONTWRITEBYTECODE":"1","PYTHONNOUSERSITE":"1",
        "PYTHONPATH":str(WT),
        "CUDA_VISIBLE_DEVICES":str(gpu) if gpu is not None else "",
        "HOME":str(CACHE/"home"),"TMPDIR":str(CACHE/"tmp"),
        "XDG_CACHE_HOME":str(CACHE),"XDG_CONFIG_HOME":str(CACHE/"config"),
        "XDG_DATA_HOME":str(CACHE/"data"),"XDG_STATE_HOME":str(CACHE/"state"),
        "TORCH_HOME":str(CACHE/"torch"),"HF_HOME":str(CACHE/"hf"),
        "PIP_CACHE_DIR":str(CACHE/"pip"),"OMP_NUM_THREADS":"1",
        "MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"
    }
    env.update(values)
    for p in (CACHE,CACHE/"home",CACHE/"tmp",CACHE/"config",
              CACHE/"data",CACHE/"state",CACHE/"torch",CACHE/"hf",CACHE/"pip"):
        p.mkdir(parents=True,exist_ok=True)
    return env

def run_stage(cmd,name,root,gpu,timeout):
    log=root/"logs"/(name+".log")
    with log.open("x") as f:
        f.write("COMMAND "+repr(cmd)+"\n");f.flush()
        start=time.monotonic()
        proc=subprocess.Popen(cmd,cwd=WT,env=environment(gpu),
                              stdout=f,stderr=subprocess.STDOUT)
        try:rc=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait()
            rc=124
    receipt={"stage":name,"rc":rc,"elapsed_s":time.monotonic()-start,
             "log":str(log),"pid":proc.pid,
             "gpu":gpu}
    print("R05_STAGE",name,"RC",rc,"SECONDS",receipt["elapsed_s"],
          flush=True)
    return receipt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--physical-gpu",type=int,required=True)
    a=ap.parse_args()
    root=Path(a.out)
    if (not str(root).startswith(str(BASE)+"/") or root.exists() or
        root.is_symlink() or a.physical_gpu not in range(4)):
        raise RuntimeError("STOP_R05_CANARY_ARGS")
    if (subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],
          text=True).strip()!=a.expected_head or
        subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],
          text=True) or not FAST.is_file()):
        raise RuntimeError("STOP_R05_CANARY_GIT_OR_ENV")
    root.mkdir(parents=True,exist_ok=False)
    (root/"logs").mkdir()
    state={"schema":"prdiff.track_r.r0.5.canary.v1",
           "status":"RUNNING",
           "head":a.expected_head,"physical_gpu":a.physical_gpu,
           "stages":[],"max_native_roots":8,
           "full_720_generation_authorized":False,
           "no_protected_access":True}
    status=root/"AUTOPILOT_STATUS.json"
    save(status,state)
    try:
        prep=root/"prepare"
        command=[str(PY),"-u","scripts/track_r/r0_5_prepare_cpu.py",
            "--expected-head",a.expected_head,
            "--recovery-run",str(RECOVERY),
            "--out",str(prep)]
        completed=run_stage(command,"cpu_prepare",root,None,1200)
        state["stages"].append(completed)
        save(status,state)
        if completed["rc"]!=0:
            state["status"]="STOP_R05_CPU_PREP"
            save(status,state)
            return completed["rc"]
        prep_receipt=json.loads((prep/"R0_5_CPU_PREP_RECEIPT.json").read_text())
        if prep_receipt["status"]!="R0_5_CPU_PREP_PASS":
            raise RuntimeError("STOP_R05_PREP_BAD_RECEIPT")
        names=prep/"R0_5_GT_FREE_ROOTS.json"
        task_sha=sha(names)
        if task_sha!=prep_receipt["root_sha256"]:
            raise RuntimeError("STOP_R05_TASK_SHA")
        chunk=root/"canary_first_8"
        command=[str(FAST),"-u","scripts/track_r/r0_5_fast_gpu.py",
            "--expected-head",a.expected_head,
            "--roots",str(names),"--roots-sha",task_sha,
            "--out",str(chunk),"--start","0","--count","8",
            "--physical-gpu",str(a.physical_gpu)]
        gpu=run_stage(command,"gpu_canary",root,a.physical_gpu,600)
        state["stages"].append(gpu)
        save(status,state)
        if gpu["rc"]!=0:
            state["status"]="STOP_R05_FAST_CANARY"
            save(status,state)
            return gpu["rc"]
        rec=json.loads((chunk/"R0_5_CHUNK_COMPLETE.json").read_text())
        if rec.get("status")!="R0_5_CHUNK_PASS" or rec.get("count")!=8:
            raise RuntimeError("STOP_R05_NATIVE_CHUNK")
        state["status"]="R0_5_FAST_8_ROOT_CANARY_PASS"
        state["gpu_allocation_seconds"]=gpu["elapsed_s"]
        state["frozen_roots_sha256"]=task_sha
        save(status,state)
        print("R05_FAST_8_ROOT_CANARY_PASS; STOP BEFORE FULL 720",flush=True)
        return 0
    except Exception as exc:
        state["status"]="STOP_R05_CANARY_EXCEPTION"
        state["error"]=repr(exc)
        state["traceback"]=traceback.format_exc()[-2000:]
        save(status,state)
        print("STOP_R05_CANARY_EXCEPTION",repr(exc),flush=True)
        return 18

if __name__=="__main__":
    sys.exit(main())
