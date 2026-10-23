#!/usr/bin/env python3
"""R0.5 4-GPU native FAST generation, reusing eight validated canary roots.

Freeze immutable GT-free input and exact old canary SHA. Generate the
remaining 712 roots in disjoint image-aligned chunks of <=24 roots.
Aggregate GPU allocation charged across all concurrent workers.
Failure stops new starts; old and completed receipts remain immutable.
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
FAST=ROOT/"conda-envs/fast_dips_a15_blackwell/bin/python"
CANARY=BASE/"r0_5_canary_20261023T004927Z"
CANARY_HEAD="c3170587e58e9610c97c6f4ae48964eab2db1b46"
ROOT_SHA="2291aa793b612c973ae76ab3d36b1f20263cd747f3b23cf593e51517a26c3bfe"
PREVIOUS_R04=BASE/"r0_4_recovery_20261022T235412Z"
R04_HEAD="62eaaafe3e22cd6222fc66858cad6dcef532a2cd"
PRIOR_SMOKE_SECONDS=112.98266698606312
R05_STAGE_CAP_SECONDS=4*3600
GLOBAL_CAP_SECONDS=16*3600
CHUNK=24
TIMEOUT=600
POLL=2

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(4*1024*1024),b""):
            h.update(part)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def save(path,data):
    tmp=Path(str(path)+".partial")
    with tmp.open("w",encoding="utf-8") as f:
        json.dump(data,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
    tmp.replace(path)

def git_head(path):
    return subprocess.check_output(
        ["git","-C",str(path),"rev-parse","HEAD"],
        text=True,timeout=30).strip()

def git_preflight(expected):
    if (git_head(WT)!=expected or subprocess.check_output(
         ["git","-C",str(WT),"status","--porcelain"],
         text=True,timeout=30)):
        raise RuntimeError("STOP_R05_PARALLEL_GIT_HEAD_OR_TREE")
    if (not FAST.is_file() or
        os.environ.get("PYTHONDONTWRITEBYTECODE")!="1"):
        raise RuntimeError("STOP_R05_PARALLEL_PINNED_ENV")

def frozen_input():
    from scripts.track_r.r0_5_fast_gpu import (
        frozen_roots,validate_complete)
    r04=load(PREVIOUS_R04/"AUTOPILOT_STATUS.json")
    if (r04.get("status")!="R0_4_ALL_2520_COMPLETE_RECOVERED_4GPU" or
        r04.get("head")!=R04_HEAD or r04.get("completed_count")!=2520 or
        r04.get("active_attempts") or
        not 0<float(r04.get("total_gpu_seconds_including_canary",0))<6*3600):
        raise RuntimeError("STOP_R05_R04_NOT_COMPLETED")
    r04_seconds=float(r04["total_gpu_seconds_including_canary"])
    c=load(CANARY/"AUTOPILOT_STATUS.json")
    if (c.get("status")!="R0_5_FAST_8_ROOT_CANARY_PASS" or
        c.get("head")!=CANARY_HEAD or
        c.get("max_native_roots")!=8 or
        c.get("full_720_generation_authorized") is not False or
        c.get("frozen_roots_sha256")!=ROOT_SHA or
        len(c.get("stages",[]))!=2 or
        [x.get("rc") for x in c["stages"]]!=[0,0]):
        raise RuntimeError("STOP_R05_UNVERIFIED_CANARY")
    measured=float(c["gpu_allocation_seconds"])
    if not 0<measured<600 or abs(measured-c["stages"][1]["elapsed_s"])>1e-5:
        raise RuntimeError("STOP_R05_CANARY_LEDGER")
    manifest=CANARY/"prepare/R0_5_GT_FREE_ROOTS.json"
    if manifest.is_symlink() or sha(manifest)!=ROOT_SHA:
        raise RuntimeError("STOP_R05_FROZEN_ROOT_SHA")
    roots=frozen_roots(load(manifest),CANARY_HEAD)
    if len(roots)!=720 or len({r["uid"] for r in roots})!=720:
        raise RuntimeError("STOP_R05_ROOT_COUNT")
    from scripts.track_r.r0_5_fast_gpu import root_folder
    folder=CANARY/"canary_first_8"
    complete=load(folder/"R0_5_CHUNK_COMPLETE.json")
    meta=folder/"R0_5_CHUNK_META.json"
    if (complete.get("status")!="R0_5_CHUNK_PASS" or
        complete.get("count")!=8 or complete.get("start")!=0 or
        complete.get("meta_sha256")!=sha(meta) or
        load(meta).get("head")!=CANARY_HEAD or
        load(meta).get("gpu")!=1):
        raise RuntimeError("STOP_R05_BAD_CANARY_CHUNK")
    for r in roots[:8]:
        data=validate_complete(root_folder(folder,r),r,CANARY_HEAD,ROOT_SHA)
        if (data.get("gt_access") is not False or
            not math.isfinite(data["native_sample_wall_seconds"]) or
            data["native_sample_wall_seconds"]<=0):
            raise RuntimeError("STOP_R05_BAD_CANARY_ROOT")
    expected=set()
    for i,r in enumerate(roots):
        if r["rep"]!=i%4 or r["source"]!="S5_r0_fast":
            raise RuntimeError("STOP_R05_UNEXPECTED_ROOT_ORDER")
        key=(r["measurement_source"],r["image_id"],r["rep"])
        if key in expected:
            raise RuntimeError("STOP_R05_DUPLICATE_FAST_ROOT")
        expected.add(key)
    print("R05_PRIOR_CANARY_AUDITED",8,"R04_GPU_H",
          round(r04_seconds/3600,5),"CANARY_GPU_S",round(measured,3),
          flush=True)
    return roots,r04_seconds,measured

def chunks():
    return [{"start":s,"count":min(CHUNK,720-s)}
            for s in range(8,720,CHUNK)]

def shard_path(root,item):
    return root/"chunks"/f"batch_{item['start']:04d}_{item['count']:02d}"

def validate_chunk(root,item,roots,head):
    from scripts.track_r.r0_5_fast_gpu import (
        validate_complete,root_folder)
    folder=shard_path(root,item)
    complete=load(folder/"R0_5_CHUNK_COMPLETE.json")
    meta=folder/"R0_5_CHUNK_META.json"
    conf=load(meta)
    if (complete.get("status")!="R0_5_CHUNK_PASS" or
        complete.get("start")!=item["start"] or
        complete.get("count")!=item["count"] or
        complete.get("meta_sha256")!=sha(meta) or
        conf.get("head")!=head or
        conf.get("root_sha256")!=ROOT_SHA or
        conf.get("uids")!=[x["uid"] for x in roots[
            item["start"]:item["start"]+item["count"]]]):
        raise RuntimeError("STOP_R05_INVALID_CHUNK_RECEIPT "+str(item))
    for r in roots[item["start"]:item["start"]+item["count"]]:
        validate_complete(root_folder(folder,r),r,head,ROOT_SHA)

def gpu_usage():
    usage={}
    for line in subprocess.check_output(
        ["nvidia-smi","--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],text=True,timeout=30).splitlines():
        i,m=(x.strip() for x in line.split(","))
        usage[int(i)]=int(m)
    return usage

def check_idle(devices=(0,1,2,3)):
    if tuple(devices)!=(0,1,2,3) and len(devices)>1:
        raise RuntimeError("STOP_R05_UNEXPECTED_GPU_SET")
    usage=gpu_usage()
    busy={g:usage.get(g) for g in devices if usage.get(g,999999)>1024}
    if busy:
        raise RuntimeError("STOP_R05_GPU_NOT_IDLE "+str(busy))

def environment(gpu):
    env=dict(os.environ)
    values={
        "CUDA_VISIBLE_DEVICES":str(gpu),"PYTHONDONTWRITEBYTECODE":"1",
        "PYTHONNOUSERSITE":"1","PYTHONPATH":str(WT),
        "HOME":str(CACHE/"home"),"TMPDIR":str(CACHE/"tmp"),
        "XDG_CACHE_HOME":str(CACHE),"XDG_CONFIG_HOME":str(CACHE/"config"),
        "XDG_DATA_HOME":str(CACHE/"data"),"XDG_STATE_HOME":str(CACHE/"state"),
        "TORCH_HOME":str(CACHE/"torch"),"HF_HOME":str(CACHE/"hf"),
        "PIP_CACHE_DIR":str(CACHE/"pip"),"OMP_NUM_THREADS":"1",
        "MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
    env.update(values)
    for p in (CACHE,CACHE/"home",CACHE/"tmp",CACHE/"config",
              CACHE/"data",CACHE/"state",CACHE/"torch",CACHE/"hf",CACHE/"pip"):
        p.mkdir(parents=True,exist_ok=True)
    return env

def active_seconds(active,now):
    return sum(max(0.,now-v["started"]) for v in active.values())

def budget_remaining(charged,active,now,r04,canary):
    running=active_seconds(active,now)
    r05=canary+charged+running
    return min(R05_STAGE_CAP_SECONDS-r05,
        GLOBAL_CAP_SECONDS-PRIOR_SMOKE_SECONDS-r04-r05)

def reconcile(path,state,active,charged,r04,canary):
    now=time.monotonic()
    total=canary+charged+active_seconds(active,now)
    state["active_attempts"]=[
        {"gpu":gpu,"pid":v["p"].pid,"start":v["start"],
         "count":v["count"],"log":v["log"],
         "start_unix":v["start_unix"],"elapsed_s":max(0,now-v["started"])}
        for gpu,v in sorted(active.items())]
    state["charged_generation_gpu_seconds"]=charged
    state["total_r05_gpu_seconds_including_canary"]=total
    state["all_r0_known_gpu_seconds"]=PRIOR_SMOKE_SECONDS+r04+total
    state["remaining_budget_gpu_seconds"]=budget_remaining(
        charged,active,now,r04,canary)
    state["completed_roots"]=8+sum(c["count"] for c in state["completed"])
    save(path,state)

def compatible_next(pending,gpu,root):
    for i,c in enumerate(pending):
        meta=shard_path(root,c)/"R0_5_CHUNK_META.json"
        if meta.exists():
            prev=load(meta)
            if prev.get("gpu")!=gpu:
                continue
            if (prev.get("start")!=c["start"] or
                prev.get("count")!=c["count"]):
                raise RuntimeError("STOP_R05_PARTIAL_META_DRIFT")
        return pending.pop(i)
    return None

def start_worker(gpu,c,a,root,state,active,charged,r04,canary):
    check_idle((gpu,))
    folder=shard_path(root,c)
    resume=folder.exists()
    attempt=sum(1 for x in state["attempts"]
                if x["start"]==c["start"] and x["count"]==c["count"])
    log=root/"logs"/f"batch_{c['start']:04d}_gpu{gpu}_attempt{attempt:03d}.log"
    cmd=[
        str(FAST),"-u","scripts/track_r/r0_5_fast_gpu.py",
        "--expected-head",a.expected_head,
        "--roots",str(CANARY/"prepare/R0_5_GT_FREE_ROOTS.json"),
        "--roots-sha",ROOT_SHA,"--roots-source-head",CANARY_HEAD,
        "--out",str(folder),"--start",str(c["start"]),
        "--count",str(c["count"]),"--physical-gpu",str(gpu)]
    if resume:cmd.append("--resume")
    f=log.open("x",encoding="utf-8")
    f.write("COMMAND "+repr(cmd)+"\n");f.flush()
    try:
        p=subprocess.Popen(cmd,cwd=WT,env=environment(gpu),
                           stdout=f,stderr=subprocess.STDOUT)
    except BaseException:
        f.close()
        raise
    active[gpu]={"p":p,"f":f,"log":str(log),
                 "start":c["start"],"count":c["count"],
                 "started":time.monotonic(),"start_unix":time.time(),
                 "signaled":False}
    state["attempts"].append({"start":c["start"],"count":c["count"],
                             "gpu":gpu,"pid":p.pid,"log":str(log),"rc":None})
    reconcile(root/"AUTOPILOT_STATUS.json",state,active,charged,r04,canary)
    print("R05_FAST_START_GPU",gpu,"INDEX",c["start"],
          "COUNT",c["count"],"PID",p.pid,flush=True)

def signal_owned(active,reason):
    for gpu,v in active.items():
        if v["p"].poll() is None and not v["signaled"]:
            v["signaled"]=True
            v["f"].write("STOP_R05_"+reason+"\n");v["f"].flush()
            v["p"].terminate()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--resume",action="store_true")
    a=ap.parse_args()
    root=Path(a.out)
    if not str(root).startswith(str(BASE)+"/") or root.is_symlink():
        raise RuntimeError("STOP_R05_OUTPUT_NAMESPACE")
    git_preflight(a.expected_head)
    roots,r04,canary=frozen_input()
    if a.resume:
        if not root.is_dir():
            raise RuntimeError("STOP_R05_RESUME_ROOT_MISSING")
    else:
        if root.exists():
            raise RuntimeError("STOP_R05_NEW_OUTPUT_MUST_BE_FRESH")
        check_idle()
        root.mkdir(parents=True,exist_ok=False)
        (root/"logs").mkdir()
        (root/"chunks").mkdir()
    with (root/"CONTROLLER.lock").open("a+") as locked:
        try:
            fcntl.flock(locked.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("STOP_R05_CONTROLLER_ALREADY_RUNNING")
        path=root/"AUTOPILOT_STATUS.json"
        if a.resume:
            state=load(path)
            if (state.get("schema")!="prdiff.track_r.r0.5.four_gpu.v1" or
                state.get("head")!=a.expected_head or
                state.get("status") not in ("STOP_WORKER","STOP_BUDGET",
                                            "STOP_CONTROLLER_EXCEPTION") or
                state.get("active_attempts") or
                state.get("root_sha256")!=ROOT_SHA or
                state.get("canary_run")!=str(CANARY)):
                raise RuntimeError("STOP_R05_RESUME_NEEDS_RECONCILIATION")
            for c in state["completed"]:
                validate_chunk(root,c,roots,a.expected_head)
            spent=float(state["charged_generation_gpu_seconds"])
            if not math.isfinite(spent) or spent<0:
                raise RuntimeError("STOP_R05_BAD_LEDGER")
            check_idle()
        else:
            state={
                "schema":"prdiff.track_r.r0.5.four_gpu.v1",
                "status":"RUNNING","head":a.expected_head,
                "canary_run":str(CANARY),"root_sha256":ROOT_SHA,
                "physical_gpus":[0,1,2,3],
                "r04_gpu_seconds":r04,"prior_smoke_gpu_seconds":PRIOR_SMOKE_SECONDS,
                "r05_canary_gpu_seconds":canary,
                "stage_cap_gpu_hours":R05_STAGE_CAP_SECONDS/3600,
                "global_cap_gpu_hours":16,
                "attempts":[],"completed":[],"active_attempts":[],
                "charged_generation_gpu_seconds":0.,
                "no_gt_in_gpu_worker":True,
                "confirmation305_untouched":True,
                "start_utc":dt.datetime.now(dt.timezone.utc).isoformat()}
            spent=0.
            save(path,state)
        pending=[c for c in chunks()
                 if (c["start"],c["count"]) not in
                    {(x["start"],x["count"]) for x in state["completed"]}]
        if len(pending)+len(state["completed"])!=30:
            raise RuntimeError("STOP_R05_720_PARTITION")
        active={}
        failing=False
        reason=None
        try:
            state["status"]="RUNNING"
            reconcile(path,state,active,spent,r04,canary)
            while pending or active:
                now=time.monotonic()
                if budget_remaining(spent,active,now,r04,canary)<=0:
                    failing=True
                    reason="BUDGET"
                    signal_owned(active,reason)
                if not failing:
                    for gpu in range(4):
                        if gpu in active or not pending:
                            continue
                        if budget_remaining(spent,active,time.monotonic(),
                                            r04,canary)<150:
                            failing=True
                            reason="BUDGET"
                            signal_owned(active,reason)
                            break
                        next_chunk=compatible_next(pending,gpu,root)
                        if next_chunk is None:
                            continue
                        start_worker(gpu,next_chunk,a,root,state,active,
                                     spent,r04,canary)
                for gpu,v in list(active.items()):
                    elapsed=time.monotonic()-v["started"]
                    if elapsed>TIMEOUT and v["p"].poll() is None:
                        failing=True
                        reason=reason or "SHARD_TIMEOUT"
                        signal_owned(active,reason)
                    rc=v["p"].poll()
                    if rc is None:
                        continue
                    v["f"].close()
                    elapsed=time.monotonic()-v["started"]
                    spent+=elapsed
                    entry=next(x for x in reversed(state["attempts"])
                        if x["pid"]==v["p"].pid and x["rc"] is None)
                    entry["rc"]=int(rc)
                    entry["elapsed_gpu_seconds"]=elapsed
                    c={"start":v["start"],"count":v["count"],"gpu":gpu}
                    del active[gpu]
                    if rc==0 and not v["signaled"]:
                        try:
                            validate_chunk(root,c,roots,a.expected_head)
                        except Exception as e:
                            failing=True
                            reason=reason or "RESULT_VALIDATION"
                            entry["validation_error"]=repr(e)
                        else:
                            state["completed"].append(c)
                            print("R05_FAST_SHARD_PASS",gpu,c["start"],
                                  c["count"],flush=True)
                    else:
                        failing=True
                        reason=reason or "WORKER"
                        print("R05_FAST_SHARD_STOP",gpu,c["start"],
                              c["count"],"RC",rc,flush=True)
                reconcile(path,state,active,spent,r04,canary)
                if failing and not active:
                    state["status"]="STOP_BUDGET" if reason=="BUDGET" else "STOP_WORKER"
                    reconcile(path,state,active,spent,r04,canary)
                    return 22
                time.sleep(POLL)
            if state["completed_roots"]!=720 or len(state["completed"])!=30:
                raise RuntimeError("STOP_R05_ROOT_COUNT_INCOMPLETE")
            state["status"]="R0_5_ALL_720_FAST_ROOTS_COMPLETE_4GPU"
            reconcile(path,state,active,spent,r04,canary)
            print("R05_720_ALL_FAST_ROOTS_COMPLETE","GPUH",
                  round(state["total_r05_gpu_seconds_including_canary"]/3600,5),
                  flush=True)
            return 0
        except BaseException as e:
            signal_owned(active,"CONTROLLER_EXCEPTION")
            for gpu,v in list(active.items()):
                try:v["p"].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    v["p"].kill()
                    v["p"].wait()
                spent+=time.monotonic()-v["started"]
                v["f"].close()
                entry=next(x for x in reversed(state["attempts"])
                           if x["pid"]==v["p"].pid and x["rc"] is None)
                entry["rc"]=v["p"].returncode
                entry["elapsed_gpu_seconds"]=time.monotonic()-v["started"]
            active.clear()
            state["status"]="STOP_CONTROLLER_EXCEPTION"
            state["error"]=repr(e)
            state["traceback"]=traceback.format_exc()[-2200:]
            reconcile(path,state,active,spent,r04,canary)
            print("STOP_R05_4GPU_CONTROLLER",repr(e),flush=True)
            return 27

if __name__=="__main__":
    sys.exit(main())
