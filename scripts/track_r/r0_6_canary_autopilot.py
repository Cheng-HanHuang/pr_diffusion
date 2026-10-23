#!/usr/bin/env python3
"""R0.6 scoring canary: CPU audit 720 FAST roots, score only first eight.

Exits after verified 8/8 diffusion-prior + UnivFD + signed/clipped residual
receipts. The input is 720 hashed PNG candidates; no GT in GPU workers.
"""
from __future__ import annotations
import argparse
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
KNOWN_STAGE_LIMIT_S=4*3600
GLOBAL_R0_LIMIT_S=16*3600

def atomic(path,obj):
    tmp=Path(str(path)+".partial")
    with tmp.open("w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    tmp.replace(path)

def env(gpu):
    result=dict(os.environ)
    result.update({
        "CUDA_VISIBLE_DEVICES":str(gpu) if gpu is not None else "",
        "PYTHONDONTWRITEBYTECODE":"1","PYTHONNOUSERSITE":"1",
        "PYTHONPATH":str(WT),"HOME":str(CACHE/"home"),
        "TMPDIR":str(CACHE/"tmp"),"XDG_CACHE_HOME":str(CACHE),
        "XDG_CONFIG_HOME":str(CACHE/"config"),"XDG_DATA_HOME":str(CACHE/"data"),
        "XDG_STATE_HOME":str(CACHE/"state"),"TORCH_HOME":str(CACHE/"torch"),
        "HF_HOME":str(CACHE/"hf"),"PIP_CACHE_DIR":str(CACHE/"pip"),
        "OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
        "MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"})
    for v in ("home","tmp","config","data","state","torch","hf","pip"):
        (CACHE/v).mkdir(parents=True,exist_ok=True)
    return result

def stage(root,name,args,gpu,timeout):
    log=root/"logs"/(name+".log")
    with log.open("x",encoding="utf-8") as file:
        file.write("COMMAND "+repr(args)+"\n")
        file.flush()
        start=time.monotonic()
        p=subprocess.Popen(args,cwd=WT,env=env(gpu),
                           stdout=file,stderr=subprocess.STDOUT)
        (root/"logs"/(name+".pid")).write_text(str(p.pid)+"\n")
        try:
            rc=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            file.write("STOP_R06_CANARY_TIMEOUT\n");file.flush()
            p.terminate()
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill();p.wait()
            rc=124
    result={"stage":name,"rc":int(rc),"pid":p.pid,
            "elapsed_gpu_seconds":time.monotonic()-start if gpu is not None else 0,
            "elapsed_wall_seconds":time.monotonic()-start,
            "log":str(log)}
    print("R06_CANARY_STAGE",name,"RC",rc,
          "WALL",round(result["elapsed_wall_seconds"],2),flush=True)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--physical-gpu",type=int,required=True)
    a=ap.parse_args()
    root=Path(a.out)
    if (not str(root).startswith(str(BASE)+"/") or
        root.is_symlink() or root.exists() or
        a.physical_gpu not in range(4)):
        raise RuntimeError("STOP_R06_CANARY_OUT_OR_GPU")
    head=subprocess.check_output(
        ["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(
        ["git","-C",str(WT),"status","--porcelain"],text=True)
    if head!=a.expected_head or dirty or not PY.is_file():
        raise RuntimeError("STOP_R06_CANARY_HEAD_ENV")
    root.mkdir(parents=True,exist_ok=False)
    (root/"logs").mkdir()
    state={"schema":"prdiff.track_r.r0.6.canary.v1",
           "status":"RUNNING","head":head,
           "gpu":a.physical_gpu,"stages":[],
           "scored":0,"full_720_authorized":False,
           "gt_in_gpu_worker":False}
    status=root/"AUTOPILOT_STATUS.json"
    atomic(status,state)
    try:
        cp=root/"prepare"
        a0=stage(root,"cpu_prepare",[
            str(PY),"-u","scripts/track_r/r0_6_prepare_cpu.py",
            "--expected-head",head,"--out",str(cp)],None,1500)
        state["stages"].append(a0);atomic(status,state)
        if a0["rc"]:
            state["status"]="STOP_R06_PREP"
            atomic(status,state);return a0["rc"]
        summary=json.loads((cp/"R0_6_CPU_PREP_RECEIPT.json").read_text())
        if summary.get("status")!="R0_6_CPU_PREP_PASS" or summary.get("n_candidates")!=720:
            raise RuntimeError("STOP_R06_CANARY_BAD_PREP")
        if (summary["known_total_gpu_seconds"]>=GLOBAL_R0_LIMIT_S or
            summary["known_total_gpu_seconds"]+KNOWN_STAGE_LIMIT_S>=GLOBAL_R0_LIMIT_S):
            raise RuntimeError("STOP_R06_CANARY_GLOBAL_BUDGET")
        tasks=cp/"R0_6_GT_FREE_TASKS.json"
        chunk=root/"first_8"
        a1=stage(root,"gpu_score8",[
            str(PY),"-u","scripts/track_r/r0_6_score_gpu.py",
            "--expected-head",head,
            "--tasks",str(tasks),"--tasks-sha",summary["tasks_sha256"],
            "--out",str(chunk),"--physical-gpu",str(a.physical_gpu),
            "--batch-size","64","--start","0","--count","8"
        ],a.physical_gpu,600)
        state["stages"].append(a1);atomic(status,state)
        if a1["rc"]:
            state["status"]="STOP_R06_GPU_CANARY"
            atomic(status,state);return a1["rc"]
        rec=json.loads((chunk/"R0_6_SHARD_COMPLETE.json").read_text())
        if rec.get("status")!="R0_6_SHARD_PASS" or rec.get("count")!=8 or rec.get("new")!=8:
            raise RuntimeError("STOP_R06_CANARY_SHARD_RECEIPT")
        from scripts.track_r.r0_6_score_gpu import validate_complete,output_name
        task_json=json.loads(tasks.read_text())
        for row in task_json["tasks"][:8]:
            validate_complete(chunk/"results"/output_name(row),
                              row,summary["tasks_sha256"],head)
        state["status"]="R0_6_EIGHT_CANDIDATE_CANARY_PASS"
        state["scored"]=8
        state["tasks_sha256"]=summary["tasks_sha256"]
        state["gpu_seconds"]=a1["elapsed_gpu_seconds"]
        state["known_r0_gpu_seconds_with_canary"]=(
            summary["known_total_gpu_seconds"]+a1["elapsed_gpu_seconds"])
        atomic(status,state)
        print("R06_CANARY_PASS 8/8; STOP BEFORE 720 SCORE",flush=True)
        return 0
    except Exception as exc:
        state["status"]="STOP_R06_CANARY_EXCEPTION"
        state["error"]=repr(exc)
        state["traceback"]=traceback.format_exc()[-2000:]
        atomic(status,state)
        print("STOP_R06_CANARY",repr(exc),flush=True)
        return 18

if __name__=="__main__":
    sys.exit(main())
