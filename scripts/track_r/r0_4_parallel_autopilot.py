#!/usr/bin/env python3
"""R0.4 full 4-GPU continuation after validated 16-candidate canary.

Sealed GT-free task file from the *original canary* is reused unchanged.
Four distinct idle physical GPUs process disjoint <=64-candidate chunks.
Fail-closed ledger accounts for the sum of all simultaneously active
GPU allocation seconds. No changes to R0 science or historical output.
"""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
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
CANARY=BASE/"r0_4_score_20261022T193238Z"
PRODUCER_HEAD="8aee60d98acfeba836fff4ec824137769b0a4698"
TASK_SHA="38543cc4e4c394ab7c8936303a8d0fee6b46fb81e53f877a5ea6768e151078f7"
STAGE_LIMIT_SECONDS=6*3600
GLOBAL_LIMIT_SECONDS=16*3600
KNOWN_EARLIER_GPU_SECONDS=(0.016863963068074857+0.004520111094720455)*3600+0.01*3600
CHUNK=64
CHUNK_TIMEOUT=900
POLL=2

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()

def json_atomic(path,obj):
    tmp=Path(str(path)+".partial")
    with tmp.open("w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    tmp.replace(path)

def read(path):
    return json.loads(Path(path).read_text())

def verify_git(expected):
    if (subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],
            text=True,timeout=30).strip()!=expected or
        subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],
            text=True,timeout=30)):
        raise RuntimeError("STOP_R04_4GPU_HEAD_OR_DIRTY")
    if not PY.is_file() or os.environ.get("PYTHONDONTWRITEBYTECODE")!="1":
        raise RuntimeError("STOP_R04_4GPU_ENV_OR_BYTECODE")

def check_idle(gpus):
    if len(gpus)!=4 or tuple(sorted(gpus))!=(0,1,2,3):
        raise RuntimeError("STOP_R04_4GPU_NOT_ALL_PHYSICAL")
    usage={}
    for item in subprocess.check_output(
        ["nvidia-smi","--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],text=True,timeout=30).splitlines():
        a,b=[x.strip() for x in item.split(",")]
        usage[int(a)]=int(b)
    busy={g:usage.get(g) for g in gpus if usage.get(g,10**9)>1024}
    if busy:
        raise RuntimeError("STOP_R04_4GPU_BUSY "+str(busy))
    return {str(g):usage[g] for g in gpus}

def env_for(gpu):
    env=dict(os.environ)
    settings={"PYTHONDONTWRITEBYTECODE":"1","PYTHONNOUSERSITE":"1",
       "CUDA_VISIBLE_DEVICES":str(gpu),"PYTHONPATH":str(WT),
       "HOME":str(CACHE/"home"),"TMPDIR":str(CACHE/"tmp"),
       "XDG_CACHE_HOME":str(CACHE),"XDG_CONFIG_HOME":str(CACHE/"config"),
       "XDG_DATA_HOME":str(CACHE/"data"),"XDG_STATE_HOME":str(CACHE/"state"),
       "TORCH_HOME":str(CACHE/"torch"),"HF_HOME":str(CACHE/"hf"),
       "PIP_CACHE_DIR":str(CACHE/"pip"),"OMP_NUM_THREADS":"1",
       "MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
       "NUMEXPR_NUM_THREADS":"1"}
    env.update(settings)
    for p in (CACHE,CACHE/"home",CACHE/"tmp",CACHE/"config",CACHE/"data",
              CACHE/"state",CACHE/"torch",CACHE/"hf",CACHE/"pip"):
        p.mkdir(parents=True,exist_ok=True)
    return env

def validate_canary():
    from scripts.track_r.r0_4_score_gpu import (
        validate_complete, output_name, validate_input,
        candidate_tensor,measurement_tensor)
    state=read(CANARY/"AUTOPILOT_STATUS.json")
    if (state.get("status")!="CANARY_COMPLETE" or
        state.get("head")!=PRODUCER_HEAD or
        state.get("completed_until")!=16 or
        state.get("active_attempt") is not None or
        len(state.get("shards",[]))!=1 or
        state["shards"][0].get("rc")!=0 or
        state["shards"][0].get("count")!=16):
        raise RuntimeError("STOP_R04_4GPU_CANARY_STATE")
    tasks_path=CANARY/"prepare/R0_4_GT_FREE_TASKS.json"
    if tasks_path.is_symlink() or sha(tasks_path)!=TASK_SHA:
        raise RuntimeError("STOP_R04_4GPU_CANARY_TASKS_SHA")
    tasks=validate_input(read(tasks_path),PRODUCER_HEAD)
    if len({r["uid"] for r in tasks})!=2520:
        raise RuntimeError("STOP_R04_4GPU_DUPLICATE_UID")
    folder=CANARY/"chunks/batch_0000_16"
    meta=folder/"R0_4_SHARD_META.json"
    complete=read(folder/"R0_4_SHARD_COMPLETE.json")
    if (complete.get("status")!="R0_4_SHARD_PASS" or
        complete.get("meta_sha256")!=sha(meta) or
        complete.get("count")!=16 or
        read(meta).get("task_file_sha256")!=TASK_SHA):
        raise RuntimeError("STOP_R04_4GPU_CANARY_SHARD")
    results=folder/"results"
    for row in tasks[:16]:
        validate_complete(results/output_name(row),row,TASK_SHA,PRODUCER_HEAD)
        r=read(results/output_name(row))
        if (r.get("gt_access") is not False or
            r.get("univfd_forwards")!=9 or
            r["s_star"] not in r["prior"]["S"] or
            abs(r["S_repaired"]-r["prior"]["S"][r["s_star"]])>1e-8 or
            abs(r["rho_signed_raw"]-r["rho_signed_repaired"])>1e-4 or
            abs(r["rho_clipped_raw"]-r["rho_clipped_repaired"])>1e-4):
            raise RuntimeError("STOP_R04_4GPU_CANARY_CONTENT "+row["uid"])
    # Heterogeneous source formats checked without GT, and no
    # historical data or source worktrees are modified.
    for i in (200,600,2040,2360,2488):
        img=candidate_tensor(tasks[i])
        y=measurement_tensor(tasks[i],{})
        if tuple(img.shape)!=(1,3,256,256) or tuple(y.shape)!=(1,3,384,384):
            raise RuntimeError("STOP_R04_4GPU_HETEROGENEOUS_SOURCE "+str(i))
    return tasks,TASK_SHA,float(state["actual_gpu_seconds"])

def chunks():
    return [{"start":i,"count":min(CHUNK,2520-i)}
            for i in range(16,2520,CHUNK)]

def result_dir(root,item):
    return root/"chunks"/("batch_%04d_%02d"%(item["start"],item["count"]))

def validate_shard(root,chunk,task_sha,head,tasks):
    from scripts.track_r.r0_4_score_gpu import validate_complete,output_name
    folder=result_dir(root,chunk)
    receipt=read(folder/"R0_4_SHARD_COMPLETE.json")
    meta=folder/"R0_4_SHARD_META.json"
    if (receipt.get("status")!="R0_4_SHARD_PASS" or
        receipt.get("meta_sha256")!=sha(meta) or
        receipt.get("task_file_sha256")!=task_sha or
        receipt.get("start")!=chunk["start"] or
        receipt.get("count")!=chunk["count"]):
        raise RuntimeError("STOP_R04_4GPU_SHARD_RECEIPT "+str(chunk))
    for row in tasks[chunk["start"]:chunk["start"]+chunk["count"]]:
        validate_complete(folder/"results"/output_name(row),row,task_sha,head)

def active_seconds(active,now):
    return sum(max(0.,now-item["start_monotonic"]]) for item in active.values())

def valid_budget(ledger,active,now,canary_seconds):
    running=active_seconds(active,now)
    spent=canary_seconds+ledger+running
    return spent < STAGE_LIMIT_SECONDS and spent+KNOWN_EARLIER_GPU_SECONDS < GLOBAL_LIMIT_SECONDS

def stage_deadline_remaining(ledger,active,now,canary_seconds):
    running=active_seconds(active,now)
    remaining=STAGE_LIMIT_SECONDS-(canary_seconds+ledger+running)
    global_remaining=GLOBAL_LIMIT_SECONDS-(KNOWN_EARLIER_GPU_SECONDS+canary_seconds+ledger+running)
    return min(remaining,global_remaining)

def update_status(path,state,active,charged,canary_seconds):
    now=time.monotonic()
    running=active_seconds(active,now)
    state["active_attempts"]=[
        {"start":v["start"],"count":v["count"],"gpu":g,
         "pid":v["proc"].pid,"log":v["log"],"start_unix":v["start_unix"],
         "elapsed_seconds_so_far":max(0.,now-v["start_monotonic"])}
        for g,v in sorted(active.items())]
    state["total_gpu_seconds_including_canary"]=canary_seconds+charged+running
    state["gpu_budget_remaining_seconds"]=min(
        STAGE_LIMIT_SECONDS-(canary_seconds+charged+running),
        GLOBAL_LIMIT_SECONDS-(KNOWN_EARLIER_GPU_SECONDS+canary_seconds+charged+running))
    state["completed_count"]=16+sum(c["count"] for c in state["completed"])
    json_atomic(path,state)

def launch_one(gpu,chunk,a,root,state,active,charged,canary_seconds):
    # Avoid accidental interference if an unrelated process took this GPU.
    check_single_gpu(gpu)
    folder=result_dir(root,chunk)
    attempts=sum(1 for x in state["attempts"]
                 if x["start"]==chunk["start"] and x["count"]==chunk["count"])
    log=root/"logs"/(folder.name+"_gpu%d_attempt%03d.log"%(gpu,attempts))
    args=[str(PY),"-u","scripts/track_r/r0_4_score_gpu.py",
          "--expected-head",a.expected_head,
          "--tasks",str(CANARY/"prepare/R0_4_GT_FREE_TASKS.json"),
          "--tasks-sha",TASK_SHA,
          "--tasks-source-head",PRODUCER_HEAD,
          "--out",str(folder),
          "--physical-gpu",str(gpu),"--batch-size","64",
          "--start",str(chunk["start"]),"--count",str(chunk["count"])]
    f=log.open("x",encoding="utf-8")
    f.write("COMMAND "+repr(args)+"\n");f.flush()
    try:
        proc=subprocess.Popen(args,cwd=str(WT),env=env_for(gpu),
                              stdout=f,stderr=subprocess.STDOUT)
    except BaseException:
        f.close();raise
    task={"proc":proc,"file":f,"log":str(log),"start":chunk["start"],
          "count":chunk["count"],"gpu":gpu,
          "start_unix":time.time(),"start_monotonic":time.monotonic(),
          "timeout":CHUNK_TIMEOUT,"forced_stop":False}
    active[gpu]=task
    state["attempts"].append({
        "start":chunk["start"],"count":chunk["count"],
        "gpu":gpu,"pid":proc.pid,"log":str(log),"rc":None})
    update_status(root/"AUTOPILOT_STATUS.json",state,active,charged,canary_seconds)
    print("R04_4GPU_START",gpu,chunk["start"],chunk["count"],"PID",proc.pid,flush=True)

def check_single_gpu(gpu):
    usages={}
    for line in subprocess.check_output(
        ["nvidia-smi","--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],text=True,timeout=30).splitlines():
        s,m=[v.strip() for v in line.split(",")]
        usages[int(s)]=int(m)
    if usages.get(gpu,999999)>1024:
        raise RuntimeError("STOP_R04_4GPU_GPU_TAKEN "+str(gpu))

def stop_owned(active,reason):
    for v in active.values():
        if v["proc"].poll() is None:
            v["forced_stop"]=True
            v["file"].write("STOP_R04_4GPU_"+reason+"\n")
            v["file"].flush()
            v["proc"].terminate()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--resume",action="store_true")
    a=ap.parse_args()
    root=Path(a.out)
    if not str(root).startswith(str(BASE)+"/") or root.is_symlink():
        raise RuntimeError("STOP_R04_4GPU_OUTPUT_NAMESPACE")
    verify_git(a.expected_head)
    tasks,task_sha,canary_seconds=validate_canary()
    if a.resume:
        if not root.is_dir():raise RuntimeError("STOP_R04_4GPU_RESUME_ROOT")
    else:
        if root.exists():raise RuntimeError("STOP_R04_4GPU_OUTPUT_OCCUPIED")
        check_idle((0,1,2,3))
        root.mkdir(parents=True,exist_ok=False)
        (root/"logs").mkdir()
        (root/"chunks").mkdir()
    lockfile=root/"CONTROLLER.lock"
    with lockfile.open("a+") as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("STOP_R04_4GPU_CONTROLLER_ALREADY_RUNNING")
        status=root/"AUTOPILOT_STATUS.json"
        if a.resume:
            state=read(status)
            if (state.get("schema")!="prdiff.track_r.r0.4.four_gpu.v1" or
                state.get("head")!=a.expected_head or
                state.get("task_sha256")!=TASK_SHA or
                state.get("source_canary")!=str(CANARY) or
                state.get("active_attempts") or
                state.get("status") not in
                    ("STOP_WORKER","STOP_GPU_BUDGET","STOP_CONTROLLER_EXCEPTION")):
                raise RuntimeError("STOP_R04_4GPU_UNRECONCILED_RESUME")
            for c in state["completed"]:
                validate_shard(root,c,task_sha,a.expected_head,tasks)
            # All recorded prior GPU attempts remain charged, including failed.
            prior=float(state["charged_parallel_gpu_seconds"])
            if prior<0:raise RuntimeError("STOP_R04_4GPU_BAD_BUDGET_LEDGER")
            check_idle((0,1,2,3))
        else:
            state={
                "schema":"prdiff.track_r.r0.4.four_gpu.v1",
                "head":a.expected_head,"source_canary":str(CANARY),
                "task_sha256":TASK_SHA,"status":"RUNNING",
                "physical_gpus":[0,1,2,3],
                "gpu_hour_limit_stage":6.,
                "gpu_hour_limit_all_r0":16.,
                "historical_canary_gpu_seconds":canary_seconds,
                "historical_smoke_gpu_seconds":KNOWN_EARLIER_GPU_SECONDS,
                "charged_parallel_gpu_seconds":0.,
                "attempts":[],"completed":[],
                "active_attempts":[],
                "no_gt_gpu_inputs":True,"protected_cohort_access":False,
                "start_utc":dt.datetime.now(dt.timezone.utc).isoformat()}
            prior=0.
            json_atomic(status,state)
        pending=[c for c in chunks() if (c["start"],c["count"]) not in
                 {(x["start"],x["count"]) for x in state["completed"]}]
        if len(pending)+len(state["completed"])!=40:
            raise RuntimeError("STOP_R04_4GPU_SHARD_PARTITION")
        state["status"]="RUNNING_4_GPU"
        state.pop("error",None)
        state.pop("traceback",None)
        active={}
        charged=prior
        failing=False
        fatal_reason=None
        check_every=time.monotonic()
        try:
            while pending or active:
                now=time.monotonic()
                remaining=stage_deadline_remaining(charged,active,now,canary_seconds)
                if remaining<=0 and not failing:
                    failing=True;fatal_reason="GPU_BUDGET"
                    stop_owned(active,"GPU_BUDGET")
                if not failing:
                    for gpu in range(4):
                        if gpu not in active and pending:
                            # Critical: avoid starting four simultaneous
                            # chunks if remaining budget cannot cover them.
                            if stage_deadline_remaining(charged,active,
                                time.monotonic(),canary_seconds)<max(120.,CHUNK*0.7):
                                failing=True;fatal_reason="GPU_BUDGET"
                                stop_owned(active,"GPU_BUDGET")
                                break
                            c=pending.pop(0)
                            launch_one(gpu,c,a,root,state,active,
                                       charged,canary_seconds)
                for gpu,v in list(active.items()):
                    elapsed=time.monotonic()-v["start_monotonic"]
                    if elapsed>v["timeout"] and v["proc"].poll() is None:
                        failing=True
                        fatal_reason=fatal_reason or "CHUNK_TIMEOUT"
                        stop_owned(active,"CHUNK_TIMEOUT")
                    rc=v["proc"].poll()
                    if rc is None:
                        continue
                    v["file"].close()
                    elapsed=time.monotonic()-v["start_monotonic"]
                    charged+=elapsed
                    attempt=next(x for x in reversed(state["attempts"])
                                 if x["pid"]==v["proc"].pid and x["rc"] is None)
                    attempt["rc"]=int(rc)
                    attempt["elapsed_gpu_seconds"]=elapsed
                    c={"start":v["start"],"count":v["count"],"gpu":gpu}
                    del active[gpu]
                    if rc==0 and not v["forced_stop"]:
                        try:
                            validate_shard(root,c,task_sha,a.expected_head,tasks)
                        except Exception as e:
                            failing=True
                            fatal_reason="SHARD_VALIDATION"
                            attempt["validation_error"]=repr(e)
                        else:
                            state["completed"].append(c)
                            print("R04_4GPU_SHARD_PASS",gpu,
                                  c["start"],c["count"],flush=True)
                    else:
                        failing=True
                        fatal_reason=fatal_reason or "WORKER"
                        print("R04_4GPU_SHARD_STOP",gpu,
                              c["start"],c["count"],"RC",rc,flush=True)
                if failing and active and fatal_reason!="GPU_BUDGET":
                    # Existing owned workers finish their in-flight shards;
                    # do not launch another shard after any failure.
                    pass
                state["charged_parallel_gpu_seconds"]=charged
                if time.monotonic()-check_every>8 or failing or not active:
                    update_status(status,state,active,charged,canary_seconds)
                    check_every=time.monotonic()
                if failing and not active:
                    state["status"]="STOP_GPU_BUDGET" if fatal_reason=="GPU_BUDGET" else "STOP_WORKER"
                    update_status(status,state,active,charged,canary_seconds)
                    return 21
                time.sleep(POLL)
            state["charged_parallel_gpu_seconds"]=charged
            if 16+sum(c["count"] for c in state["completed"])!=2520:
                raise RuntimeError("STOP_R04_4GPU_INCOMPLETE_SHARD_TOTAL")
            if len(state["completed"])!=40:
                raise RuntimeError("STOP_R04_4GPU_INCOMPLETE_PARTITION")
            state["status"]="R0_4_ALL_2520_COMPLETE_4GPU"
            update_status(status,state,active,charged,canary_seconds)
            print("R04_4GPU_ALL_2520_COMPLETE",state["completed_count"],
                  "AGG_GPU_H",state["total_gpu_seconds_including_canary"]/3600,
                  flush=True)
            return 0
        except BaseException as e:
            stop_owned(active,"CONTROLLER_EXCEPTION")
            # Account for active work on exit; never silently abandon owned
            # GPU workers or lose their allocation from the durable ledger.
            for gpu,v in list(active.items()):
                try:v["proc"].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    v["proc"].kill();v["proc"].wait()
                charged+=time.monotonic()-v["start_monotonic"]
                v["file"].close()
            active.clear()
            state["charged_parallel_gpu_seconds"]=charged
            state["status"]="STOP_CONTROLLER_EXCEPTION"
            state["error"]=repr(e)
            state["traceback"]=traceback.format_exc()[-2000:]
            update_status(status,state,active,charged,canary_seconds)
            print("STOP_R04_4GPU_EXCEPTION",repr(e),flush=True)
            return 26

if __name__=="__main__":
    sys.exit(main())
