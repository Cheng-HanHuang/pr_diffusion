#!/usr/bin/env python3
"""R0.7 CPU-only complete offline GT evaluation autopilot.

Step 1 audits and freezes 3240 source/candidate/score/GT join receipts.
Step 2 evaluates frozen P1–P4 and frozen secondary diagnostics.
Never execute GPUs or open B24 protected cohorts.
"""
from __future__ import annotations
import argparse
import datetime as dt
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

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def atomic(path,obj):
    intermediate=Path(str(path)+".partial")
    with intermediate.open("w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    intermediate.replace(path)

def run_stage(root,name,cmd,environment,timeout,save_state,state):
    log=root/"logs"/(name+".log")
    with log.open("x",encoding="utf-8") as f:
        f.write("COMMAND "+repr(cmd)+"\n")
        f.flush()
        start=time.monotonic()
        proc=subprocess.Popen(cmd,cwd=WT,env=environment,
                              stdout=f,stderr=subprocess.STDOUT)
        state["active_process"]={
            "stage":name,"pid":proc.pid,"log":str(log),
            "start_unix":time.time()}
        atomic(save_state,state)
        try:
            rc=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            f.write("STOP_R07_CPU_STAGE_TIME_LIMIT\n");f.flush()
            proc.terminate()
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait()
            rc=124
    state["active_process"]=None
    entry={"name":name,"rc":int(rc),"wall_seconds":time.monotonic()-start,
           "log":str(log),"pid":proc.pid,"cpu_only":True}
    state["stages"].append(entry)
    atomic(save_state,state)
    print("R07_STAGE",name,"RC",rc,
          "WALL_SECONDS",round(entry["wall_seconds"],2),
          "LOG",str(log),flush=True)
    return int(rc)

def environment():
    env=dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1",
        "PYTHONNOUSERSITE":"1","PYTHONPATH":str(WT),
        "HOME":str(CACHE/"home"),"TMPDIR":str(CACHE/"tmp"),
        "XDG_CACHE_HOME":str(CACHE),"XDG_CONFIG_HOME":str(CACHE/"config"),
        "XDG_DATA_HOME":str(CACHE/"data"),"XDG_STATE_HOME":str(CACHE/"state"),
        "TORCH_HOME":str(CACHE/"torch"),"HF_HOME":str(CACHE/"hf"),
        "PIP_CACHE_DIR":str(CACHE/"pip"),
        "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1",
        "OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"})
    for part in ("home","tmp","config","data","state","torch","hf","pip"):
        (CACHE/part).mkdir(parents=True,exist_ok=True)
    return env

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-head",required=True)
    parser.add_argument("--out",required=True)
    a=parser.parse_args()
    root=Path(a.out)
    if (not str(root).startswith(str(BASE)+"/") or root.exists() or
        root.is_symlink() or not PY.is_file()):
        raise RuntimeError("STOP_R07_OUTPUT_OR_ENV")
    head=subprocess.check_output(
        ["git","-C",str(WT),"rev-parse","HEAD"],text=True,timeout=30).strip()
    dirty=subprocess.check_output(
        ["git","-C",str(WT),"status","--porcelain"],
        text=True,timeout=30)
    if head!=a.expected_head or dirty:
        raise RuntimeError("STOP_R07_GIT_HEAD")
    root.mkdir(parents=True,exist_ok=False)
    (root/"logs").mkdir()
    status=root/"AUTOPILOT_STATUS.json"
    state={"schema":"prdiff.track_r.r0.7.cpu-autopilot.v1",
        "status":"CPU_PREP","head":head,
        "start_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "cuda_visible_devices":"","gpu_calls":0,
        "protected_cohort_access":False,
        "offline_gt_used_only_in_cpu":True,
        "active_process":None,"stages":[]}
    atomic(status,state)
    env=environment()
    try:
        prepared=root/"prepare"
        rc=run_stage(root,"cpu_join",[
            str(PY),"-u","scripts/track_r/r0_7_prepare_cpu.py",
            "--expected-head",head,"--out",str(prepared)
        ],env,7200,status and status,state)
        if rc!=0:
            state["status"]="STOP_R07_JOIN"
            atomic(status,state)
            return rc
        receipt=json.loads((prepared/"R0_7_JOIN_RECEIPT.json").read_text())
        catalog=prepared/"R0_7_OFFLINE_GT_CATALOG.json"
        if (receipt.get("status")!="R0_7_CPU_JOIN_PASS" or
            receipt.get("n_candidates")!=3240 or
            receipt.get("catalog_sha256")!=sha(catalog) or
            receipt.get("gpu_calls")!=0):
            raise RuntimeError("STOP_R07_JOIN_RECEIPT")
        state["catalog_sha256"]=receipt["catalog_sha256"]
        state["status"]="CPU_ANALYSIS"
        atomic(status,state)
        analysis=root/"analysis"
        rc=run_stage(root,"cpu_analysis",[
            str(PY),"-u","scripts/track_r/r0_7_analyze_cpu.py",
            "--expected-head",head,
            "--catalog",str(catalog),
            "--catalog-sha",receipt["catalog_sha256"],
            "--out",str(analysis)
        ],env,14400,status,state)
        if rc!=0:
            state["status"]="STOP_R07_ANALYSIS"
            atomic(status,state)
            return rc
        final=json.loads((analysis/"R0_7_COMPLETE_RECEIPT.json").read_text())
        if (final.get("status") not in
              ("R0_7_ANALYSIS_COMPLETE","R0_7_PARTIAL_TAU_UNAVAILABLE") or
            final.get("n_scored")!=3240 or final.get("gpu_calls")!=0 or
            final.get("planner_decision_pending") is not True or
            final.get("analysis_sha256")!=sha(analysis/"R0_7_ANALYSIS.json") or
            final.get("report_sha256")!=sha(analysis/"R0_7_REPORT.md")):
            raise RuntimeError("STOP_R07_ANALYSIS_RECEIPT")
        state["status"]=final["status"]
        state["n_candidates"]=3240
        state["analysis_sha256"]=final["analysis_sha256"]
        state["report_sha256"]=final["report_sha256"]
        state["reported_r0_gpu_hours"]=final["all_r0_gpu_hours"]
        state["tau_unavailable"]=final["n_tau_unavailable"]
        state["psnr_anomalies"]=final["n_recorded_psnr_anomalies"]
        state["report_path"]=str(analysis/"R0_7_REPORT.md")
        atomic(status,state)
        print("R07_CPU_OFFLINE_FINAL",state["status"],
              "N",state["n_candidates"],"REPORT",state["report_path"],flush=True)
        return 0
    except Exception as exc:
        state["status"]="STOP_R07_EXCEPTION"
        state["error"]=repr(exc)
        state["traceback"]=traceback.format_exc()[-2500:]
        atomic(status,state)
        print("STOP_R07_AUTOPILOT",repr(exc),flush=True)
        return 24

if __name__=="__main__":
    sys.exit(main())
