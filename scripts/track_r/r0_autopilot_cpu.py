#!/usr/bin/env python3
"""Safe overnight R0 autostaging, CPU portion only until approved GPU gates.

Runs verified R0.1 unit tests, R0.2 manifest and G0 CPU validity sequentially.
It NEVER initiates GPU work or infers success on a skipped/missing gate.
All new files belong to a fresh Track R run; past research is read-only.
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
PREFIX=ROOT/"conda-envs/prdiff_track_r"
OUTROOT=ROOT/"outputs/pr_diffusion/track_r"
CACHE=ROOT/"outputs/.cache/track_r"

def record(run,obj):
    target=run/"AUTOPILOT_STATUS.json"
    temporary=run/"AUTOPILOT_STATUS.json.partial"
    with temporary.open("w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    temporary.replace(target)

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def command(argv,cwd,env,output,timeout):
    with output.open("x",encoding="utf-8") as f:
        f.write("COMMAND "+repr(argv)+"\\n")
        f.flush()
        start=time.monotonic()
        p=subprocess.Popen(argv,cwd=cwd,env=env,
                           stdout=f,stderr=subprocess.STDOUT)
        # Every long-running child has an exact recorded PID.
        pidfile=output.with_suffix(".pid")
        pidfile.write_text(str(p.pid)+"\\n",encoding="utf-8")
        try:
            rc=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            f.write("\\nSTOP_AUTOPILOT_STAGE_TIMEOUT\\n")
            f.flush()
            p.terminate()
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            rc=124
        return {"rc":rc,"pid":p.pid,
                "elapsed_s":round(time.monotonic()-start,3),
                "log":str(output)}


def check_identity(expected):
    if not WT.is_dir() or not (WT/".git").is_dir():
        raise RuntimeError("STOP_R0_AUTOPILOT_NO_TRACK_R_CHECKOUT")
    head=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],
                                 text=True,timeout=30).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],
                                  text=True,timeout=30)
    if head!=expected or dirty:
        raise RuntimeError("STOP_R0_AUTOPILOT_GIT_HEAD_OR_DIRTY "+head)
    if not (PREFIX/"bin/python").is_file():
        raise RuntimeError("STOP_R0_AUTOPILOT_ENV_MISSING")
    return head

def verify_sources_and_models():
    config=json.loads((WT/"configs/track_r/pac_paths_track_r_v1.json").read_text())
    files=(
        (config["models"]["ffhq_ddpm"]["path"],config["models"]["ffhq_ddpm"]["sha256"]),
        (config["models"]["clip_vit_l14"]["path"],config["models"]["clip_vit_l14"]["sha256"]),
        (str(Path(config["external"]["univfd"]["path"])/"pretrained_weights"/"fc_weights.pth"),
         config["external"]["univfd"]["fc_weights_sha256"]),
        (config["sources"]["S1_b22"]["b22_full_manifest"],
         config["sources"]["S1_b22"]["b22_full_manifest_sha256"]),
        (str(Path(config["sources"]["S3_a12"]["inputs_root"])/"A1_2_INPUT_FREEZE_COMPLETE.json"),
         config["sources"]["S3_a12"]["input_freeze_manifest_sha256"]),
    )
    for path,want in files:
        if "confirmation" in path.lower() or "confirm305" in path.lower():
            raise RuntimeError("STOP_PROTECTED_COHORT")
        p=Path(path)
        if not p.is_file() or "confirmation" in str(p.resolve()).lower() or "confirm305" in str(p.resolve()).lower():
            raise RuntimeError("STOP_AUTOPILOT_PINNED_INPUT_ABSENT "+str(p))
        if sha256(p)!=want:
            raise RuntimeError("STOP_AUTOPILOT_PINNED_INPUT_SHA "+str(p))
    univ=Path(config["external"]["univfd"]["path"])
    got=subprocess.check_output(["git","-C",str(univ),"rev-parse","HEAD"],text=True,timeout=30).strip()
    if got!=config["external"]["univfd"]["commit"]:
        raise RuntimeError("STOP_UNIVFD_HEAD_DRIFT")
    return {"sha_verified_files":len(files),"univfd_head":got}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True)
    ap.add_argument("--expected-head",required=True)
    args=ap.parse_args()
    out=Path(args.out)
    if not str(out).startswith(str(OUTROOT)+os.sep) or out.exists():
        raise RuntimeError("STOP_AUTOPILOT_OUTPUT_FRESHNESS")
    env=dict(os.environ)
    cache=str(CACHE)
    for key,value in {
        "CUDA_VISIBLE_DEVICES":"","PYTHONNOUSERSITE":"1",
        "PYTHONDONTWRITEBYTECODE":"1","HOME":cache+"/home",
        "XDG_CACHE_HOME":cache,"XDG_CONFIG_HOME":cache+"/config",
        "XDG_DATA_HOME":cache+"/data","XDG_STATE_HOME":cache+"/state",
        "PIP_CACHE_DIR":cache+"/pip","TMPDIR":cache+"/tmp",
        "TORCH_HOME":cache+"/torch","HF_HOME":cache+"/hf",
        "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
        "NUMEXPR_NUM_THREADS":"1","PYTHONPATH":str(WT),
        }.items():env[key]=value
    for p in (CACHE,CACHE/"home",CACHE/"config",CACHE/"data",
              CACHE/"state",CACHE/"tmp",CACHE/"pip",CACHE/"torch",CACHE/"hf"):
        p.mkdir(parents=True,exist_ok=True)
    head=check_identity(args.expected_head)
    verification=verify_sources_and_models()
    out.mkdir(parents=True,exist_ok=False)
    logs=out/"logs"
    logs.mkdir()
    state={
        "schema":"prdiff.track_r.r0.autopilot.cpu.v1",
        "status":"RUNNING_CPU_STAGES",
        "run_root":str(out),
        "head":head,
        "start_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pid":os.getpid(),
        "source_verification":verification,
        "gpu_work_authorized":False,
        "gpu_forward_calls":0,
        "gpu_hour_ceiling":16.0,
        "used_gpu_hours":0.0,
        "stages":[],
        "protected_cohort_access":False,
    }
    record(out,state)
    python=str(PREFIX/"bin/python")
    def stage(name,argv,timeout):
        print("START",name,flush=True)
        result=command(argv,str(WT),env,logs/(name+".log"),timeout)
        result["name"]=name
        state["stages"].append(result)
        if result["rc"]!=0:
            state["status"]="STOP_"+name.upper()+"_RC_"+str(result["rc"])
            print("STOP",name,"RC",result["rc"],"LOG",result["log"],flush=True)
        else:
            print("PASS",name,"ELAPSED",result["elapsed_s"],flush=True)
        record(out,state)
        return result["rc"]==0
    try:
        if not stage("r0_1_cpu",
                     [python,"-m","pytest","-q","-p","no:cacheprovider",
                      "tests/track_r"],3600):return 11
        man=out/"r0_2_manifest"
        if not stage("r0_2_manifest",
                     [python,"scripts/track_r/r0_2_build_manifest.py",
                      "--repo-root",str(WT),"--expected-head",head,
                      "--out",str(man)],3600):return 12
        pre=out/"r0_2_g0"
        if not stage("r0_2_g0_cpu",
                     [python,"scripts/track_r/r0_2_g0_cpu.py",
                      "--repo-root",str(WT),"--expected-head",head,
                      "--manifest",str(man/"R0_CANDIDATE_MANIFEST.csv"),
                      "--out",str(pre)],14400):return 13
        receipt=json.loads((pre/"R0_2_G0_CPU_RECEIPT.json").read_text())
        if receipt.get("status")!="G0_CPU_PRECHECK_PASS_PENDING_DETERMINISM":
            raise RuntimeError("STOP_R0_2_CPU_NOT_PASS")
        # G0 determinism, G1 and full R0.3–R0.7 are NOT implemented.
        # They must not be invented in an unattended background process.
        state["status"]="PAUSED_GPU_G0_DETERMINISM_AND_R0_3_EXECUTOR_NOT_IMPLEMENTED"
        state["next_required"]="R0.3 verified GPU executor including G1 and dual-GPU batch determinism; enforce 16 aggregate GPU-hour ceiling before R0.4–R0.6"
        record(out,state)
        print("PAUSE_GPU_GATES_NOT_IMPLEMENTED NO_GPU_RUNS_PERFORMED",flush=True)
        return 0
    except Exception as e:
        state["status"]="STOP_AUTOPILOT_EXCEPTION"
        state["error"]=repr(e)
        state["traceback"]=traceback.format_exc()[-2500:]
        record(out,state)
        print("STOP_AUTOPILOT_EXCEPTION",repr(e),flush=True)
        return 90

if __name__=="__main__":
    sys.exit(main())
