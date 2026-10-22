#!/usr/bin/env python3
"""One-command fail-closed R0.3b dual-GPU smoke autopilot.

CPU selection -> two idle Blackwell GPUs batch 8/64 -> CPU determinism
adjudication. Bounded to 8 candidates, no FAST generation and no R0.4–6.
"""
from __future__ import annotations
import argparse
import datetime
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


def save(path,obj):
    t=Path(str(path)+".partial")
    with t.open("w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    t.replace(path)


def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def check_gpu(gpus):
    lines=subprocess.check_output(
        ["nvidia-smi","--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],text=True,timeout=30).splitlines()
    usage={}
    for row in lines:
        a,b=(v.strip() for v in row.split(","))
        usage[int(a)]=int(b)
    for g in gpus:
        if g not in usage or usage[g]>1024:
            raise RuntimeError("STOP_R0_3B_GPU_BUSY "+str(usage))
    return {str(g):usage[g] for g in gpus}


def env_base():
    env=dict(os.environ)
    for k,v in {
        "PYTHONDONTWRITEBYTECODE":"1","PYTHONNOUSERSITE":"1",
        "HOME":str(CACHE/"home"),"TMPDIR":str(CACHE/"tmp"),
        "XDG_CACHE_HOME":str(CACHE),
        "XDG_CONFIG_HOME":str(CACHE/"config"),
        "XDG_DATA_HOME":str(CACHE/"data"),
        "XDG_STATE_HOME":str(CACHE/"state"),
        "HF_HOME":str(CACHE/"hf"),"TORCH_HOME":str(CACHE/"torch"),
        "PIP_CACHE_DIR":str(CACHE/"pip"),"PYTHONPATH":str(WT),
        "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
        "OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1",
    }.items():env[k]=v
    for p in (CACHE,CACHE/"home",CACHE/"tmp",CACHE/"config",
              CACHE/"data",CACHE/"state",CACHE/"hf",CACHE/"torch",CACHE/"pip"):
        p.mkdir(parents=True,exist_ok=True)
    return env


def spawn(argv,name,logs,env):
    f=(logs/(name+".log")).open("x",encoding="utf-8")
    f.write("COMMAND "+repr(argv)+"\n")
    f.flush()
    p=subprocess.Popen(argv,cwd=str(WT),env=env,stdout=f,stderr=subprocess.STDOUT)
    (logs/(name+".pid")).write_text(str(p.pid)+"\n")
    return (p,f,time.monotonic())


def wait_one(item,timeout):
    p,f,start=item
    try:
        rc=p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        f.write("STOP_R0_3B_CHILD_TIMEOUT\n");f.flush()
        p.terminate()
        try:p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill();p.wait()
        rc=124
    elapsed=time.monotonic()-start
    f.close()
    return rc,elapsed


def identity(head):
    actual=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],text=True)
    if head!=actual or dirty:raise RuntimeError("STOP_R0_3B_HEAD_OR_TREE")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--g0-receipt",required=True)
    ap.add_argument("--g1-receipt",required=True)
    ap.add_argument("--gpu8",type=int,required=True)
    ap.add_argument("--gpu64",type=int,required=True)
    a=ap.parse_args()
    root=Path(a.out)
    if not str(root).startswith(str(BASE)+"/") or root.exists():
        raise RuntimeError("STOP_R0_3B_OUTPUT_NOT_FRESH")
    if a.gpu8==a.gpu64 or a.gpu8 not in range(4) or a.gpu64 not in range(4):
        raise RuntimeError("STOP_R0_3B_GPU_IDS")
    if not PY.is_file():raise RuntimeError("STOP_R0_3B_ENV_MISSING")
    identity(a.expected_head)
    usage=check_gpu((a.gpu8,a.gpu64))
    env=env_base()
    root.mkdir(parents=True,exist_ok=False)
    logs=root/"logs";logs.mkdir()
    state={"schema":"prdiff.track_r.r0.3b.autopilot.v1",
           "status":"RUNNING","head":a.expected_head,
           "physical_gpus":{"batch8":a.gpu8,"batch64":a.gpu64},
           "gpu_usage_before_mib":usage,
           "start_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "stages":[],"gpu_ceiling_hours":16,"full_r0_scoring_authorized":False,
           "protected_cohort_access":False}
    save(root/"AUTOPILOT_STATUS.json",state)
    def stage(name,cmd,timeout,environ):
        item=spawn(cmd,name,logs,environ)
        result=wait_one(item,timeout)
        state["stages"].append({"name":name,"rc":result[0],"elapsed_s":result[1],
                                "log":str(logs/(name+".log"))})
        save(root/"AUTOPILOT_STATUS.json",state)
        print("R0_3B_STAGE",name,"RC",result[0],"S",round(result[1],2),flush=True)
        return result[0]==0
    try:
        sel=root/"selection"
        if not stage("selection",
            [str(PY),"scripts/track_r/r0_3b_select_smoke_cpu.py",
             "--repo",str(WT),"--expected-head",a.expected_head,
             "--manifest",a.manifest,"--g0-receipt",a.g0_receipt,
             "--g1-receipt",a.g1_receipt,"--out",str(sel)],600,env):
            state["status"]="STOP_SELECTION";save(root/"AUTOPILOT_STATUS.json",state)
            return 11
        selected=sel/"R0_3B_SMOKE_SELECTION.json"
        digest=sha(selected)
        check_gpu((a.gpu8,a.gpu64))
        workers=[]
        for batch,gpu,with_fd in ((8,a.gpu8,True),(64,a.gpu64,False)):
            name=f"gpu_batch{batch}"
            out=root/name
            args=[str(PY),"-u","scripts/track_r/r0_3b_gpu_worker.py",
                  "--out",str(out),"--selection",str(selected),
                  "--selection-sha",digest,
                  "--expected-head",a.expected_head,
                  "--physical-gpu",str(gpu),
                  "--batch-size",str(batch)]
            if with_fd:args.append("--univfd")
            e=dict(env);e["CUDA_VISIBLE_DEVICES"]=str(gpu)
            workers.append((name,spawn(args,name,logs,e)))
        failed=False
        # Workers are independent, run concurrently. On a child failure,
        # let its peer terminate normally; never kill unrelated processes.
        for name,item in workers:
            rc,elapsed=wait_one(item,1200)
            state["stages"].append({"name":name,"rc":rc,"elapsed_s":elapsed,
                                   "pid":item[0].pid,
                                   "log":str(logs/(name+".log"))})
            save(root/"AUTOPILOT_STATUS.json",state)
            print("R0_3B_STAGE",name,"RC",rc,"S",round(elapsed,2),flush=True)
            if rc:failed=True
        if failed:
            state["status"]="STOP_GPU_WORKER"
            save(root/"AUTOPILOT_STATUS.json",state)
            return 12
        cmp=root/"determinism"
        if not stage("determinism",
            [str(PY),"scripts/track_r/r0_3b_compare_cpu.py",
             "--gpu8-root",str(root/"gpu_batch8"),
             "--gpu64-root",str(root/"gpu_batch64"),
             "--expected-head",a.expected_head,"--out",str(cmp)],600,env):
            state["status"]="STOP_G0_DETERMINISM"
            save(root/"AUTOPILOT_STATUS.json",state)
            return 13
        receipt=json.loads((cmp/"R0_3B_DETERMINISM_RECEIPT.json").read_text())
        if receipt["status"]!="G0_FULL_PASS_R0_3B_NOT_FULL_R0_3":
            raise RuntimeError("STOP_G0_NOT_PASS")
        state["status"]="G0_FULL_PASS_PENDING_FAST_SMOKE_AND_BUDGET"
        state["measured_smoke_gpu_hours"]=receipt["measured_smoke_gpu_hours"]
        state["full_r0_scoring_authorized"]=False
        save(root/"AUTOPILOT_STATUS.json",state)
        print("R0_3B_G0_FULL_PASS; STOP_BEFORE_FAST_OR_FULL_SCORING",flush=True)
        return 0
    except Exception as e:
        state["status"]="STOP_R0_3B_EXCEPTION"
        state["error"]=repr(e)
        state["traceback"]=traceback.format_exc()[-2500:]
        save(root/"AUTOPILOT_STATUS.json",state)
        print("STOP_R0_3B_EXCEPTION",repr(e),flush=True)
        return 15

if __name__=="__main__":
    sys.exit(main())
