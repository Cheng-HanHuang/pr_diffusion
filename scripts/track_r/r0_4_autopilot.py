#!/usr/bin/env python3
"""R0.4 GT-free single-GPU scored-candidate autopilot with strict stage budget.

First invocation: --max-candidates 16 for canary. Later run with
--resume --max-candidates 2520 in same root after reviewing canary.
One active child GPU worker at a time, each <=64 candidate receipts.
Hard R0.4 allotment 6 GPU-h; total R0 limit remains 16 GPU-h.
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
PY=ROOT/"conda-envs/prdiff_track_r/bin/python"
ALLOTMENT_GPU_H=6.0
CHUNK=64
MAX_CHUNK_SECONDS=900

def write_state(path, obj):
    temporary=Path(str(path)+".partial")
    with temporary.open("w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    temporary.replace(path)

def verify_repo(expected):
    head=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],text=True)
    if head!=expected or dirty or not PY.is_file():
        raise RuntimeError("STOP_R04_GIT_ENV")
    if os.environ.get("PYTHONDONTWRITEBYTECODE")!="1":
        raise RuntimeError("STOP_R04_BYTECODE_GUARD")

def child_environment(gpu):
    result=dict(os.environ)
    vals={
        "PYTHONDONTWRITEBYTECODE":"1","PYTHONNOUSERSITE":"1",
        "PYTHONPATH":str(WT),"CUDA_VISIBLE_DEVICES":str(gpu),
        "HOME":str(CACHE/"home"),"TMPDIR":str(CACHE/"tmp"),
        "XDG_CACHE_HOME":str(CACHE),"XDG_CONFIG_HOME":str(CACHE/"config"),
        "XDG_DATA_HOME":str(CACHE/"data"),"XDG_STATE_HOME":str(CACHE/"state"),
        "TORCH_HOME":str(CACHE/"torch"),"HF_HOME":str(CACHE/"hf"),
        "PIP_CACHE_DIR":str(CACHE/"pip"),
        "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
        "OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1",
    }
    result.update(vals)
    for k in (CACHE,CACHE/"home",CACHE/"tmp",CACHE/"config",
              CACHE/"data",CACHE/"state",CACHE/"torch",CACHE/"hf",CACHE/"pip"):
        k.mkdir(parents=True,exist_ok=True)
    return result

def cpu_prepare(a,env,root,state):
    folder=root/"prepare"
    if folder.exists():
        file=folder/"R0_4_PREPARE_RECEIPT.json"
        tasks=folder/"R0_4_GT_FREE_TASKS.json"
        if not file.is_file() or not tasks.is_file():
            raise RuntimeError("STOP_R04_PARTIAL_PREPARE")
        rec=json.loads(file.read_text())
        if rec.get("status")!="R0_4_TASKS_READY" or rec.get("head")!=a.expected_head:
            raise RuntimeError("STOP_R04_PREPARE_RECEIPT")
        return tasks,rec
    cmd=[str(PY),"scripts/track_r/r0_4_prepare_cpu.py",
         "--expected-head",a.expected_head,
         "--manifest",a.manifest,"--g0-receipt",a.g0_receipt,
         "--g1-receipt",a.g1_receipt,
         "--r03b-receipt",a.r03b_receipt,
         "--r03c-receipt",a.r03c_receipt,"--out",str(folder)]
    proc=subprocess.run(cmd,cwd=WT,env={**env,"CUDA_VISIBLE_DEVICES":""},
                        text=True,capture_output=True,timeout=3600)
    log=root/"logs"/"prepare.log"
    with log.open("x") as f:
        f.write(proc.stdout);f.write(proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"STOP_R04_PREPARE_RC_{proc.returncode} "+str(log))
    return cpu_prepare(a,env,root,state)

def check_gpu(gpu):
    usage={}
    for line in subprocess.check_output(
        ["nvidia-smi","--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],text=True,timeout=30).splitlines():
        i,m=(piece.strip() for piece in line.split(","))
        usage[int(i)]=int(m)
    if usage.get(gpu,999999)>1024:
        raise RuntimeError("STOP_R04_GPU_BUSY "+str(usage))

def run_chunk(argv,env,log,seconds_limit,status_path,state):
    with log.open("x",encoding="utf-8") as f:
        f.write("COMMAND "+repr(argv)+"\n");f.flush()
        start=time.monotonic()
        p=subprocess.Popen(argv,cwd=WT,env=env,stdout=f,stderr=subprocess.STDOUT)
        state["active_attempt"]={"pid":p.pid,"start_unix":time.time(),
                                 "log":str(log)}
        write_state(status_path,state)
        deadline=start+seconds_limit
        while p.poll() is None:
            if time.monotonic()>=deadline:
                f.write("STOP_R04_CHUNK_BUDGET_OR_TIMEOUT\n");f.flush()
                p.terminate()
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    p.kill();p.wait()
                break
            time.sleep(2)
        elapsed=time.monotonic()-start
        rc=p.returncode if p.returncode is not None else 124
    return rc,elapsed,p.pid

def main():
    ap=argparse.ArgumentParser()
    for n in ("out","expected-head","manifest","g0-receipt",
              "g1-receipt","r03b-receipt","r03c-receipt"):
        ap.add_argument("--"+n,required=True)
    ap.add_argument("--physical-gpu",type=int,required=True)
    ap.add_argument("--max-candidates",type=int,default=16)
    ap.add_argument("--resume",action="store_true")
    a=ap.parse_args()
    root=Path(a.out)
    if (not str(root).startswith(str(BASE)+"/") or root.is_symlink() or
        a.physical_gpu not in range(4) or
        not 1<=a.max_candidates<=2520):
        raise RuntimeError("STOP_R04_ARGS")
    verify_repo(a.expected_head)
    env=child_environment(a.physical_gpu)
    if a.resume:
        if not root.is_dir():raise RuntimeError("STOP_R04_RESUME_DIR")
        state=json.loads((root/"AUTOPILOT_STATUS.json").read_text())
        if (state.get("head")!=a.expected_head or
            state.get("physical_gpu")!=a.physical_gpu or
            state.get("active_attempt") is not None or
            state.get("budget_allotment_gpu_hours")!=ALLOTMENT_GPU_H):
            raise RuntimeError("STOP_R04_RESUME_LEDGER_OR_IDENTITY")
        if state.get("status") not in ("CANARY_COMPLETE","PARTIAL_COMPLETE",
                                      "STOP_WORKER_ERROR","STOP_BUDGET_GUARD"):
            raise RuntimeError("STOP_R04_BAD_RESUME_STATUS")
    else:
        if root.exists():raise RuntimeError("STOP_R04_NOT_FRESH")
        root.mkdir(parents=True,exist_ok=False)
        (root/"logs").mkdir()
        (root/"chunks").mkdir()
        state={"schema":"prdiff.track_r.r0.4.autopilot.v1",
               "status":"PREPARING","head":a.expected_head,
               "physical_gpu":a.physical_gpu,
               "budget_allotment_gpu_hours":ALLOTMENT_GPU_H,
               "actual_gpu_seconds":0.0,"active_attempt":None,
               "completed_until":0,"shards":[],
               "gt_access_during_gpu_scoring":False,
               "source_cohort":"S1_TO_S4_EXPOSED_ONLY"}
        write_state(root/"AUTOPILOT_STATUS.json",state)
    status=root/"AUTOPILOT_STATUS.json"
    try:
        task_file,prep=cpu_prepare(a,env,root,state)
        task_sha=prep["tasks_sha256"]
        if prep["n_candidates"]!=2520:
            raise RuntimeError("STOP_R04_NOT_2520")
        chunks=root/"chunks"
        current=int(state["completed_until"])
        if current>a.max_candidates:
            raise RuntimeError("STOP_R04_COMPLETION_BEYOND_REQUEST")
        state["status"]="RUNNING"
        write_state(status,state)
        while current<a.max_candidates:
            check_gpu(a.physical_gpu)
            count=min(CHUNK,a.max_candidates-current)
            remaining=ALLOTMENT_GPU_H*3600.-float(state["actual_gpu_seconds"])
            if remaining<=0:
                state["status"]="STOP_BUDGET_GUARD"
                write_state(status,state)
                return 21
            folder=chunks/(f"batch_{current:04d}_{count:02d}")
            # Keep every attempt's log immutable, including after an
            # interruption or previous unsuccessful retry of this shard.
            attempt=sum(1 for rec in state["shards"]
                        if rec["start"]==current and rec["count"]==count)
            log=root/"logs"/(folder.name+f"_attempt{attempt:03d}.log")
            cmd=[str(PY),"-u","scripts/track_r/r0_4_score_gpu.py",
                 "--expected-head",a.expected_head,
                 "--tasks",str(task_file),"--tasks-sha",task_sha,
                 "--out",str(folder),
                 "--physical-gpu",str(a.physical_gpu),
                 "--batch-size","64","--start",str(current),
                 "--count",str(count)]
            timeout=min(MAX_CHUNK_SECONDS,remaining)
            rc,elapsed,pid=run_chunk(cmd,env,log,timeout,status,state)
            state["actual_gpu_seconds"]+=elapsed
            state["active_attempt"]=None
            state["shards"].append({"start":current,"count":count,
                "rc":rc,"gpu_seconds":elapsed,"pid":pid,"log":str(log)})
            if rc!=0:
                state["status"]="STOP_WORKER_ERROR"
                write_state(status,state)
                return rc
            receipt=folder/"R0_4_SHARD_COMPLETE.json"
            if not receipt.is_file() or json.loads(receipt.read_text()).get("status")!="R0_4_SHARD_PASS":
                raise RuntimeError("STOP_R04_MISSING_COMPLETE_RECEIPT")
            current+=count
            state["completed_until"]=current
            state["status"]="PARTIAL_COMPLETE"
            write_state(status,state)
            print("R04_PROGRESS",current,"/",a.max_candidates,
                  "GPU_H",round(state["actual_gpu_seconds"]/3600,4),flush=True)
        state["status"]=("CANARY_COMPLETE" if a.max_candidates==16
                         else "R0_4_ALL_2520_COMPLETE" if current==2520
                         else "PARTIAL_COMPLETE")
        write_state(status,state)
        print("R04_STATUS",state["status"],"GPU_H",
              state["actual_gpu_seconds"]/3600,flush=True)
        return 0
    except Exception as e:
        state["status"]="STOP_R04_EXCEPTION"
        state["error"]=repr(e)
        state["traceback"]=traceback.format_exc()[-1600:]
        write_state(status,state)
        print("STOP_R04_AUTOPILOT",repr(e),flush=True)
        return 26

if __name__=="__main__":
    sys.exit(main())
