#!/usr/bin/env python3
"""Conservative GPU-supervised FAST replay; never launches an XP scaling batch.

Wait/rescan VRAM, measure co-tenancy at every poll, perform exact 8-root
frozen replay using the official Blackwell FAST environment, then run GT-only
CPU evaluation AFTER CUDA_VISIBLE_DEVICES is cleared.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from prdiffusion.track_r.xp_gpu_telemetry import (
    collect, choose_gpu, classify_timing, event)

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
CACHE=ROOT/"outputs/.cache/track_r"
FAST=ROOT/"conda-envs/fast_dips_a15_blackwell/bin/python"
CPU=ROOT/"conda-envs/prdiff_track_r/bin/python"

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def atomic(p,obj):
    target=Path(p)
    tmp=target.with_name("."+target.name+"."+str(os.getpid())+".tmp")
    with tmp.open("x",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    os.replace(tmp,target)

def append(p,value):
    with Path(p).open("a",encoding="utf-8") as f:
        f.write(json.dumps(value,sort_keys=True,allow_nan=False)+"\n")

def main(a):
    out=Path(a.out)
    if not str(out).startswith(str(BASE/"xp_fast_replay_")) or not out.is_dir() or out.is_symlink():
        raise RuntimeError("STOP_REPLAY_OUTPUT_ROOT")
    if str(WT.resolve())!=str(Path(__file__).resolve().parents[2]):
        raise RuntimeError("STOP_WRONG_WORKTREE")
    h=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],text=True).strip()
    if h!=a.expected_head or dirty:
        raise RuntimeError("STOP_REPLAY_WRONG_HEAD_OR_DIRTY")
    from scripts.track_r.xp_fast_replay_cpu import sha as safe_sha
    manifest=out/"XP_FAST_REPLAY_TASKS.json"
    prep=json.loads((out/"XP_FAST_REPLAY_PREP_RECEIPT.json").read_text())
    if (prep.get("status")!="XP_FAST_REPLAY_PREP_PASS" or
        prep.get("head")!=h or prep.get("n_roots")!=8 or
        safe_sha(manifest)!=prep.get("manifest_sha256")):
        raise RuntimeError("STOP_REPLAY_MANIFEST_NOT_FROZEN")
    for file in ("XP_FAST_REPLAY_STATUS.json","GPU_EVENTS.jsonl","XP_FAST_REPLAY_TIMING.json"):
        if (out/file).exists():
            raise RuntimeError("STOP_REPLAY_ALREADY_LAUNCHED")
    try:
        with (out/"XP_FAST_REPLAY_PID").open("x") as f:
            f.write(str(os.getpid())+"\n")
    except FileExistsError:
        raise RuntimeError("STOP_REPLAY_PID_ALREADY_EXISTS")
    stage="XP_FAST_R05_REPLAY_8"
    status_path=out/"XP_FAST_REPLAY_STATUS.json"
    events=out/"GPU_EVENTS.jsonl"
    info={"state":"WAIT_VRAM","head":h,"pid":os.getpid(),"n_roots":8,
          "gpu":None,"model":"ffhq","run_index":[0,1],
          "gpu_hours":0,"reason":None}
    atomic(status_path,info)
    poll=a.poll_seconds
    jobmem=a.estimated_vram_mib
    selected=None
    while selected is None:
        if (out/"STOP_AFTER_ACTIVE").exists():
            info["state"]="STOP_REQUESTED"
            atomic(status_path,info)
            print("ACTION NEEDED: REPLAY_STOP_REQUESTED")
            return 8
        try:
            cards=collect()
            selected,_=choose_gpu(cards,jobmem,safety_margin_mib=2048)
            append(events,event(cards,stage=stage,job_id="fast_replay8",
                                action="WAIT_POLL",selected_gpu=selected))
        except Exception as exc:
            info["reason"]="GPU_TELEMETRY "+str(exc)[:150]
            atomic(status_path,info)
            time.sleep(poll)
            continue
        if selected is None:
            info["state"]="WAIT_VRAM"
            atomic(status_path,info)
            time.sleep(poll)
    chunk=out/"gpu"
    env=dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"]="1"
    env["PYTHONNOUSERSITE"]="1"
    env["PYTHONPATH"]=str(WT)
    env["CUDA_VISIBLE_DEVICES"]=str(selected)
    env["TMPDIR"]=str(CACHE/"tmp")
    env["XDG_CACHE_HOME"]=str(CACHE)
    command=[str(FAST),str(WT/"scripts/track_r/xp_fast_gpu.py"),
        "--expected-head",h,
        "--roots",str(manifest),
        "--roots-sha",prep["manifest_sha256"],
        "--out",str(chunk),
        "--start","0","--count","8",
        "--physical-gpu",str(selected),
        "--dataset","ffhq","--oversample","2.0"]
    print("XP_FAST_NATIVE_PREFLIGHT_START gpu="+str(selected),flush=True)
    validate=subprocess.run(command+["--preflight-only"],env=env,cwd=str(WT),
                            capture_output=True,text=True,timeout=600)
    (out/"XP_FAST_PREFLIGHT.log").write_text(validate.stdout+validate.stderr)
    if validate.returncode:
        info.update(state="FAILED_PREFLIGHT",gpu=selected,
                    reason="FAST_NATIVE_PREFLIGHT_RC_"+str(validate.returncode))
        atomic(status_path,info)
        print("ACTION NEEDED: FAST_NATIVE_PREFLIGHT_FAILED",flush=True)
        return 8
    info.update(state="RUNNING_GPU",gpu=selected,reason=None)
    atomic(status_path,info)
    first=collect()
    samples=[first]
    append(events,event(first,stage=stage,job_id="fast_replay8",
                        action="PRE_GPU",selected_gpu=selected))
    t0=time.monotonic()
    logfile=(out/"XP_FAST_GPU.log").open("xb")
    try:
        proc=subprocess.Popen(command,env=env,cwd=str(WT),
                              stdout=logfile,stderr=subprocess.STDOUT,
                              start_new_session=True)
    finally:
        logfile.close()
    info["worker_pid"]=proc.pid
    atomic(status_path,info)
    while proc.poll() is None:
        time.sleep(poll)
        try:
            snap=collect([proc.pid])
            samples.append(snap)
            append(events,event(snap,stage=stage,job_id="fast_replay8",
                                action="HEARTBEAT",selected_gpu=selected,
                                own_roots=[proc.pid]))
        except Exception as exc:
            append(events,{"action":"TELEMETRY_UNAVAILABLE",
                           "gpu":selected,"error":str(exc)[:150]})
    secs=time.monotonic()-t0
    try:
        last=collect([proc.pid]);samples.append(last)
        append(events,event(last,stage=stage,job_id="fast_replay8",
                            action="POST_GPU",selected_gpu=selected))
    except Exception as exc:
        append(events,{"action":"POST_TELEMETRY_UNAVAILABLE","error":str(exc)[:150]})
    timing=classify_timing(samples,selected)
    info["gpu_hours"]=secs/3600
    info["worker_returncode"]=proc.returncode
    timing_doc={"status":"XP_FAST_REPLAY_GPU_TIMING_RECORDED",
      "head":h,"seconds":secs,"gpu_hours":secs/3600,
      "selected_gpu":selected,"worker_returncode":proc.returncode,
      "timing":timing,"clean_timing_eligible":bool(timing["timing_valid"]),
      "quality_eligible_if_replay_passes":True}
    atomic(out/"XP_FAST_REPLAY_TIMING.json",timing_doc)
    if proc.returncode:
        info["state"]="FAILED_GPU"
        info["reason"]="GPU_WORKER_RC_"+str(proc.returncode)
        atomic(status_path,info)
        print("ACTION NEEDED: REPLAY_GPU_WORKER_RC="+str(proc.returncode))
        return 8
    info["state"]="CPU_SCORING"
    atomic(status_path,info)
    env["CUDA_VISIBLE_DEVICES"]=""
    # Use offline scorer from same pinned immutable checkout.
    res=subprocess.run([str(CPU),str(WT/"scripts/track_r/xp_fast_replay_cpu.py"),
        "score","--expected-head",h,"--out",str(out)],
        cwd=str(WT),env=env,capture_output=True,text=True,timeout=3600)
    (out/"XP_FAST_REPLAY_SCORE.log").write_text(res.stdout+res.stderr)
    info["cpu_score_rc"]=res.returncode
    if res.returncode:
        info["state"]="REPLAY_GATE_FAILED"
        info["reason"]="SCORER_RC_"+str(res.returncode)
        atomic(status_path,info)
        print("ACTION NEEDED: FAST_REPLAY_GATE_FAILED")
        return 8
    offline=json.loads((out/"XP_FAST_REPLAY_OFFLINE_RECEIPT.json").read_text())
    if offline.get("status")!="XP_FAST_REPLAY_PASS":
        raise RuntimeError("STOP_REPLAY_SCIENCE_NOT_PASS")
    info["state"]="PASS"
    info["max_abs_psnr_delta_db"]=offline["max_abs_delta_db"]
    info["exact_png"]=offline["n_png_exact"]
    info["timing_class"]=timing["timing_class"]
    atomic(status_path,info)
    print("XP_FAST_REPLAY_PASS n=8 max_db="+str(offline["max_abs_delta_db"])+
          " gpu="+str(selected)+" timing="+timing["timing_class"])
    print("OK")
    return 0

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-head",required=True)
    parser.add_argument("--out",required=True)
    parser.add_argument("--poll-seconds",type=int,default=10)
    parser.add_argument("--estimated-vram-mib",type=int,default=32000)
    try:
        args=parser.parse_args()
        if not 5<=args.poll_seconds<=60 or not 16000<=args.estimated_vram_mib<=64000:
            raise RuntimeError("STOP_REPLAY_SETTINGS")
        sys.exit(main(args))
    except Exception as e:
        print("ACTION NEEDED: XP_FAST_REPLAY_SUPERVISOR "+str(e))
        traceback.print_exc()
        sys.exit(8)
