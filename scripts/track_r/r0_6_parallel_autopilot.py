#!/usr/bin/env python3
"""R0.6 four-GPU scoring of the remaining 712 FAST roots.

Preserves the sealed R0.6 8-root score canary and its output receipts.
GPU workers receive GT-free candidate paths and frozen signed observations.
Every shard has immutable per-candidate receipts; four concurrently active
physical GPUs count as four GPU-seconds per elapsed wall-clock second.
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
CANARY=BASE/"r0_6_canary_20261023T020016Z"
PRODUCER_HEAD="bc62488aa088ae681cbedd4264fa3a099bed82bb"
TASK_SHA="f268e29b482da10aee31f79de43a6bf19d460f4203f7813c46dbee83b0591596"
R05=BASE/"r0_5_parallel_20261023T012732Z"
R05_HEAD="2a65698a3855ddcc65071af90f69f287c824fa13"
STAGE_CAP_SECONDS=4*3600
GLOBAL_CAP_SECONDS=16*3600
CHUNK=64
CHUNK_TIMEOUT=900
POLL=2

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def atomic(path,obj):
    tmp=Path(str(path)+".partial")
    with tmp.open("w",encoding="utf-8") as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    tmp.replace(path)

def verify_git(expected):
    head=subprocess.check_output(
        ["git","-C",str(WT),"rev-parse","HEAD"],text=True,timeout=30).strip()
    dirty=subprocess.check_output(
        ["git","-C",str(WT),"status","--porcelain"],
        text=True,timeout=30)
    if head!=expected or dirty:
        raise RuntimeError("STOP_R06_PARALLEL_WRONG_HEAD_OR_DIRTY")
    if not PY.is_file() or os.environ.get("PYTHONDONTWRITEBYTECODE")!="1":
        raise RuntimeError("STOP_R06_PARALLEL_ENV")

def sealed_canary():
    from scripts.track_r.r0_6_score_gpu import (
        validate_input,validate_complete,output_name)
    previous=read(R05/"AUTOPILOT_STATUS.json")
    if (previous.get("status")!="R0_5_ALL_720_FAST_ROOTS_COMPLETE_4GPU" or
        previous.get("head")!=R05_HEAD or
        previous.get("completed_roots")!=720 or
        previous.get("active_attempts") or
        len(previous.get("completed",[]))!=30 or
        not 0<float(previous.get("all_r0_known_gpu_seconds",0))<GLOBAL_CAP_SECONDS):
        raise RuntimeError("STOP_R06_R05_SOURCE_LEDGER")
    prior_s=float(previous["all_r0_known_gpu_seconds"])
    c=read(CANARY/"AUTOPILOT_STATUS.json")
    prep=read(CANARY/"prepare/R0_6_CPU_PREP_RECEIPT.json")
    if (c.get("status")!="R0_6_EIGHT_CANDIDATE_CANARY_PASS" or
        c.get("head")!=PRODUCER_HEAD or c.get("scored")!=8 or
        c.get("full_720_authorized") is not False or
        c.get("tasks_sha256")!=TASK_SHA or
        len(c.get("stages",[]))!=2 or
        [x.get("rc") for x in c["stages"]]!=[0,0] or
        c.get("gpu")!=1 or
        prep.get("status")!="R0_6_CPU_PREP_PASS" or
        prep.get("head")!=PRODUCER_HEAD or
        prep.get("tasks_sha256")!=TASK_SHA or
        prep.get("n_candidates")!=720 or
        abs(float(prep.get("known_total_gpu_seconds",0))-prior_s)>1e-3):
        raise RuntimeError("STOP_R06_CANARY_AND_PREP_RECEIPTS")
    canary_s=float(c["gpu_seconds"])
    if (not 0<canary_s<600 or
        abs(canary_s-float(c["stages"][1]["elapsed_gpu_seconds"]))>1e-3 or
        abs(float(c["known_r0_gpu_seconds_with_canary"])-
            (prior_s+canary_s))>1e-3):
        raise RuntimeError("STOP_R06_CANARY_GPU_LEDGER")
    task_path=CANARY/"prepare/R0_6_GT_FREE_TASKS.json"
    if task_path.is_symlink() or sha(task_path)!=TASK_SHA:
        raise RuntimeError("STOP_R06_GT_FREE_TASK_SHA")
    rows=validate_input(read(task_path),PRODUCER_HEAD)
    if len(rows)!=720 or len({r["uid"] for r in rows})!=720 or any(
        r.get("source")!="S5_r0_fast" or r.get("family")!="FAST" or
        r.get("run_index")!=i%4 or
        r.get("measurement_source") not in ("S1_b22","S2_b24_dev80")
        for i,r in enumerate(rows)):
        raise RuntimeError("STOP_R06_SOURCE_OR_ORDER")
    src=CANARY/"first_8"
    meta=src/"R0_6_SHARD_META.json"
    final=read(src/"R0_6_SHARD_COMPLETE.json")
    conf=read(meta)
    if (final.get("status")!="R0_6_SHARD_PASS" or
        final.get("start")!=0 or final.get("count")!=8 or
        final.get("new")!=8 or
        final.get("meta_sha256")!=sha(meta) or
        conf.get("executor_head")!=PRODUCER_HEAD or
        conf.get("task_file_sha256")!=TASK_SHA or
        conf.get("gpu")!=1 or
        conf.get("uids")!=[r["uid"] for r in rows[:8]]):
        raise RuntimeError("STOP_R06_CANARY_SHARD_VALIDATION")
    for r in rows[:8]:
        path=src/"results"/output_name(r)
        validate_complete(path,r,TASK_SHA,PRODUCER_HEAD)
        result=read(path)
        if (result.get("gt_access") is not False or
            result.get("s_star") not in result.get("p_fake_orbit",{}) or
            abs(float(result["p_fake_repaired"])-
                float(result["p_fake_orbit"][result["s_star"]]))>1e-5 or
            abs(float(result["rho_signed_raw"])-
                float(result["rho_signed_repaired"]))>1e-4 or
            abs(float(result["rho_clipped_raw"])-
                float(result["rho_clipped_repaired"]))>1e-4):
            raise RuntimeError("STOP_R06_CANARY_SCORE_CONTENT "+r["uid"])
    print("R06_CANARY_8_AUDITED",TASK_SHA,
          "PRIOR_R0_GPU_H",round(prior_s/3600,6),
          "CANARY_GPU_SECONDS",round(canary_s,3),flush=True)
    return rows,prior_s,canary_s

def shards():
    return [{"start":i,"count":min(CHUNK,720-i)}
            for i in range(8,720,CHUNK)]

def location(root,item):
    return root/"chunks"/f"batch_{item['start']:04d}_{item['count']:02d}"

def validate_shard(root,item,rows,head):
    from scripts.track_r.r0_6_score_gpu import validate_complete,output_name
    folder=location(root,item)
    meta=folder/"R0_6_SHARD_META.json"
    receipt=read(folder/"R0_6_SHARD_COMPLETE.json")
    info=read(meta)
    selection=rows[item["start"]:item["start"]+item["count"]]
    if (receipt.get("status")!="R0_6_SHARD_PASS" or
        receipt.get("start")!=item["start"] or
        receipt.get("count")!=item["count"] or
        receipt.get("meta_sha256")!=sha(meta) or
        receipt.get("task_file_sha256")!=TASK_SHA or
        info.get("executor_head")!=head or
        info.get("task_file_sha256")!=TASK_SHA or
        info.get("start")!=item["start"] or
        info.get("count")!=item["count"] or
        info.get("gpu")!=item["gpu"] or
        info.get("uids")!=[r["uid"] for r in selection]):
        raise RuntimeError("STOP_R06_SHARD_INTEGRITY "+str(item))
    for r in selection:
        validate_complete(folder/"results"/output_name(r),r,TASK_SHA,head)

def gpu_memory():
    info={}
    output=subprocess.check_output(
        ["nvidia-smi","--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],text=True,timeout=30)
    for line in output.splitlines():
        i,v=[k.strip() for k in line.split(",")]
        info[int(i)]=int(v)
    return info

def check_idle(devices=(0,1,2,3)):
    if any(i not in (0,1,2,3) for i in devices):
        raise RuntimeError("STOP_R06_GPU_INDEX")
    usage=gpu_memory()
    occupied={i:usage.get(i) for i in devices if usage.get(i,999999)>1024}
    if occupied:
        raise RuntimeError("STOP_R06_GPU_BUSY "+str(occupied))

def environment(gpu):
    result=dict(os.environ)
    result.update({
        "CUDA_VISIBLE_DEVICES":str(gpu),
        "PYTHONDONTWRITEBYTECODE":"1","PYTHONNOUSERSITE":"1",
        "PYTHONPATH":str(WT),"HOME":str(CACHE/"home"),
        "TMPDIR":str(CACHE/"tmp"),"XDG_CACHE_HOME":str(CACHE),
        "XDG_CONFIG_HOME":str(CACHE/"config"),
        "XDG_DATA_HOME":str(CACHE/"data"),
        "XDG_STATE_HOME":str(CACHE/"state"),
        "TORCH_HOME":str(CACHE/"torch"),
        "HF_HOME":str(CACHE/"hf"),
        "PIP_CACHE_DIR":str(CACHE/"pip"),
        "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
        "OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"})
    for name in ("home","tmp","config","data","state","torch","hf","pip"):
        (CACHE/name).mkdir(parents=True,exist_ok=True)
    return result

def active_seconds(active,now):
    return sum(max(0.,now-v["began"]) for v in active.values())

def remaining_budget(charged,active,now,previous,canary):
    current=canary+charged+active_seconds(active,now)
    return min(STAGE_CAP_SECONDS-current,
               GLOBAL_CAP_SECONDS-previous-current)

def status_sync(path,state,active,charged,previous,canary):
    now=time.monotonic()
    running=active_seconds(active,now)
    state["active_attempts"]=[
        {"gpu":g,"pid":v["proc"].pid,"start":v["start"],"count":v["count"],
         "log":v["log"],"start_unix":v["start_unix"],
         "elapsed_seconds":max(0.,now-v["began"])}
        for g,v in sorted(active.items())]
    state["charged_new_gpu_seconds"]=charged
    state["total_r06_gpu_seconds_including_canary"]=canary+charged+running
    state["total_r0_gpu_seconds"]=previous+canary+charged+running
    state["gpu_budget_remaining_seconds"]=remaining_budget(
        charged,active,now,previous,canary)
    state["completed_scores"]=8+sum(x["count"] for x in state["completed"])
    atomic(path,state)

def compatible_next(pending,gpu,root):
    for i,c in enumerate(pending):
        meta=location(root,c)/"R0_6_SHARD_META.json"
        if meta.is_file():
            old=read(meta)
            if old.get("gpu")!=gpu:
                continue
            if old.get("start")!=c["start"] or old.get("count")!=c["count"]:
                raise RuntimeError("STOP_R06_RETRY_METADATA_DRIFT")
        return pending.pop(i)
    return None

def launch_one(gpu,item,args,root,state,active,charged,previous,canary):
    check_idle((gpu,))
    folder=location(root,item)
    ordinal=sum(1 for x in state["attempts"] if
                x["start"]==item["start"] and x["count"]==item["count"])
    log=root/"logs"/f"batch_{item['start']:04d}_gpu{gpu}_attempt{ordinal:03d}.log"
    cmd=[
        str(PY),"-u","scripts/track_r/r0_6_score_gpu.py",
        "--expected-head",args.expected_head,
        "--tasks",str(CANARY/"prepare/R0_6_GT_FREE_TASKS.json"),
        "--tasks-sha",TASK_SHA,"--tasks-source-head",PRODUCER_HEAD,
        "--out",str(folder),
        "--physical-gpu",str(gpu),
        "--batch-size","64","--start",str(item["start"]),
        "--count",str(item["count"])]
    f=log.open("x",encoding="utf-8")
    f.write("COMMAND "+repr(cmd)+"\n");f.flush()
    try:
        proc=subprocess.Popen(cmd,cwd=WT,env=environment(gpu),
                              stdout=f,stderr=subprocess.STDOUT)
    except BaseException:
        f.close()
        raise
    active[gpu]={"proc":proc,"f":f,"log":str(log),
                 "start":item["start"],"count":item["count"],
                 "began":time.monotonic(),"start_unix":time.time(),
                 "signaled":False}
    state["attempts"].append({"start":item["start"],
        "count":item["count"],"gpu":gpu,"pid":proc.pid,
        "log":str(log),"rc":None})
    status_sync(root/"AUTOPILOT_STATUS.json",state,active,
                charged,previous,canary)
    print("R06_4GPU_START",gpu,item["start"],item["count"],
          "PID",proc.pid,flush=True)

def stop_owned(active,reason):
    for v in active.values():
        if v["proc"].poll() is None and not v["signaled"]:
            v["signaled"]=True
            v["f"].write("STOP_R06_"+reason+"\n");v["f"].flush()
            v["proc"].terminate()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--resume",action="store_true")
    args=ap.parse_args()
    root=Path(args.out)
    if not str(root).startswith(str(BASE)+"/") or root.is_symlink():
        raise RuntimeError("STOP_R06_PARALLEL_OUTPUT_NAMESPACE")
    verify_git(args.expected_head)
    rows,previous,canary=sealed_canary()
    if args.resume:
        if not root.is_dir():
            raise RuntimeError("STOP_R06_RESUME_MISSING")
    else:
        if root.exists():
            raise RuntimeError("STOP_R06_OUTPUT_EXISTS")
        check_idle()
        root.mkdir(parents=True,exist_ok=False)
        (root/"logs").mkdir()
        (root/"chunks").mkdir()
    with (root/"CONTROLLER.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("STOP_R06_OTHER_CONTROLLER_ACTIVE")
        path=root/"AUTOPILOT_STATUS.json"
        if args.resume:
            state=read(path)
            if (state.get("schema")!="prdiff.track_r.r0.6.four_gpu.v1" or
                state.get("head")!=args.expected_head or
                state.get("canary_run")!=str(CANARY) or
                state.get("tasks_sha256")!=TASK_SHA or
                state.get("active_attempts") or
                state.get("status") not in
                    ("STOP_WORKER","STOP_BUDGET","STOP_CONTROLLER_EXCEPTION")):
                raise RuntimeError("STOP_R06_UNRECONCILED_RESTART")
            if abs(float(state["earlier_r0_gpu_seconds"])-previous)>1e-3:
                raise RuntimeError("STOP_R06_PREVIOUS_LEDGER_DRIFT")
            if abs(float(state["canary_gpu_seconds"])-canary)>1e-3:
                raise RuntimeError("STOP_R06_CANARY_LEDGER_DRIFT")
            seen=set()
            for item in state["completed"]:
                r=set(range(item["start"],item["start"]+item["count"]))
                if r & seen:
                    raise RuntimeError("STOP_R06_RESUME_OVERLAPPING_SHARDS")
                seen.update(r)
                validate_shard(root,item,rows,args.expected_head)
            charged=float(state["charged_new_gpu_seconds"])
            if not math.isfinite(charged) or charged<0:
                raise RuntimeError("STOP_R06_RESUME_GPU_LEDGER")
            check_idle()
        else:
            state={
                "schema":"prdiff.track_r.r0.6.four_gpu.v1",
                "status":"RUNNING","head":args.expected_head,
                "canary_run":str(CANARY),"tasks_sha256":TASK_SHA,
                "physical_gpus":[0,1,2,3],"earlier_r0_gpu_seconds":previous,
                "canary_gpu_seconds":canary,
                "stage_limit_gpu_hours":STAGE_CAP_SECONDS/3600,
                "global_limit_gpu_hours":GLOBAL_CAP_SECONDS/3600,
                "attempts":[],"completed":[],"active_attempts":[],
                "charged_new_gpu_seconds":0.,
                "gt_read_by_gpu_workers":False,
                "protected_cohort_access":False,
                "start_utc":dt.datetime.now(dt.timezone.utc).isoformat()}
            charged=0.
            atomic(path,state)
        pending=[x for x in shards() if
                 (x["start"],x["count"]) not in
                 {(y["start"],y["count"]) for y in state["completed"]}]
        if len(pending)+len(state["completed"])!=12:
            raise RuntimeError("STOP_R06_EXPECTED_12_SHARDS")
        active={}
        failing=False
        reason=None
        try:
            state["status"]="RUNNING"
            state.pop("error",None)
            state.pop("traceback",None)
            status_sync(path,state,active,charged,previous,canary)
            while pending or active:
                now=time.monotonic()
                if remaining_budget(charged,active,now,previous,canary)<=0:
                    failing=True;reason="BUDGET"
                    stop_owned(active,reason)
                if not failing:
                    for gpu in range(4):
                        if gpu in active or not pending:
                            continue
                        if remaining_budget(charged,active,time.monotonic(),
                                            previous,canary)<150:
                            failing=True;reason="BUDGET"
                            stop_owned(active,reason)
                            break
                        c=compatible_next(pending,gpu,root)
                        if c is None:
                            continue
                        launch_one(gpu,c,args,root,state,active,
                                   charged,previous,canary)
                for gpu,v in list(active.items()):
                    elapsed=time.monotonic()-v["began"]
                    if elapsed>CHUNK_TIMEOUT and v["proc"].poll() is None:
                        failing=True;reason=reason or "CHUNK_TIMEOUT"
                        stop_owned(active,reason)
                    rc=v["proc"].poll()
                    if rc is None:
                        continue
                    v["f"].close()
                    elapsed=time.monotonic()-v["began"]
                    charged+=elapsed
                    rec=next(x for x in reversed(state["attempts"])
                             if x["pid"]==v["proc"].pid and x["rc"] is None)
                    rec["rc"]=int(rc)
                    rec["elapsed_gpu_seconds"]=elapsed
                    item={"start":v["start"],"count":v["count"],"gpu":gpu}
                    del active[gpu]
                    if rc==0 and not v["signaled"]:
                        try:
                            validate_shard(root,item,rows,args.expected_head)
                        except Exception as e:
                            failing=True;reason=reason or "RESULT_VALIDATION"
                            rec["validation_error"]=repr(e)
                        else:
                            state["completed"].append(item)
                            print("R06_4GPU_SHARD_PASS",gpu,item["start"],
                                  item["count"],flush=True)
                    else:
                        failing=True;reason=reason or "WORKER"
                        print("R06_4GPU_SHARD_STOP",gpu,item["start"],
                              item["count"],"RC",rc,flush=True)
                status_sync(path,state,active,charged,previous,canary)
                if failing and not active:
                    state["status"]="STOP_BUDGET" if reason=="BUDGET" else "STOP_WORKER"
                    status_sync(path,state,active,charged,previous,canary)
                    return 22
                time.sleep(POLL)
            if state["completed_scores"]!=720 or len(state["completed"])!=12:
                raise RuntimeError("STOP_R06_INCOMPLETE_FINAL_COVERAGE")
            # Re-validate ALL shards and their individual receipts before
            # recording a full-stage completion status.
            for item in state["completed"]:
                validate_shard(root,item,rows,args.expected_head)
            state["status"]="R0_6_ALL_720_FAST_SCORES_COMPLETE_4GPU"
            status_sync(path,state,active,charged,previous,canary)
            print("R06_720_ALL_FAST_SCORES_COMPLETE",
                  "R06_AGG_GPU_H",round(state["total_r06_gpu_seconds_including_canary"]/3600,6),
                  "R0_ALL_GPU_H",round(state["total_r0_gpu_seconds"]/3600,6),
                  flush=True)
            return 0
        except BaseException as e:
            stop_owned(active,"CONTROLLER_EXCEPTION")
            for gpu,v in list(active.items()):
                try:
                    v["proc"].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    v["proc"].kill();v["proc"].wait()
                elapsed=time.monotonic()-v["began"]
                charged+=elapsed
                v["f"].close()
                rec=next((x for x in reversed(state["attempts"])
                          if x["pid"]==v["proc"].pid and x["rc"] is None),None)
                if rec is not None:
                    rec["rc"]=v["proc"].returncode
                    rec["elapsed_gpu_seconds"]=elapsed
            active.clear()
            state["status"]="STOP_CONTROLLER_EXCEPTION"
            state["error"]=repr(e)
            state["traceback"]=traceback.format_exc()[-2400:]
            status_sync(path,state,active,charged,previous,canary)
            print("STOP_R06_CONTROLLER",repr(e),flush=True)
            return 27

if __name__=="__main__":
    sys.exit(main())
