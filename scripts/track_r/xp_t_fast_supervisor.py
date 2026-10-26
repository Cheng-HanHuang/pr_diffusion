#!/usr/bin/env python3
"""Bounded XP-T FAST FFHQ batch-1 timing pilot (3 images x 3 new seeds).

One GPU, GT-free. Safe co-tenancy: pause until >=32 GiB allocation fits within
80% device memory, preserve timing contamination/visibility, never kill others.
Successful quality receipts are independent of timing cleanliness.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from prdiffusion.track_r.xp_gpu_telemetry import collect,choose_gpu,classify_timing,event

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
CACHE=ROOT/"outputs/.cache/track_r"
FAST=ROOT/"conda-envs/fast_dips_a15_blackwell/bin/python"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")

def sha(p):
    p=Path(p)
    if any(w in str(p).lower() for w in BAD):
        raise RuntimeError("STOP_XPT_FORBIDDEN_SHA")
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def new_json(path,data):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(data,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")

def update_json(path,data):
    dest=Path(path)
    tmp=dest.with_name("."+dest.name+"."+str(os.getpid()))
    with tmp.open("x",encoding="utf-8") as f:
        json.dump(data,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    os.replace(tmp,dest)

def record(path,what):
    with Path(path).open("a",encoding="utf-8") as f:
        f.write(json.dumps(what,sort_keys=True,allow_nan=False)+"\n")

def summarize(root,head,roots_path,roots_sha,device,timings,cards,secs):
    from scripts.track_r.xp_fast_gpu import validate_complete,root_folder,frozen_roots
    doc=json.loads(roots_path.read_text())
    tasks=frozen_roots(doc,head)
    chunk=root/"gpu"
    m=json.loads((chunk/"R0_5_CHUNK_META.json").read_text())
    complete=json.loads((chunk/"R0_5_CHUNK_COMPLETE.json").read_text())
    if (m.get("head")!=head or complete.get("head")!=head or
        complete.get("status")!="R0_5_CHUNK_PASS" or
        complete.get("new")!=9 or complete.get("reused")!=0 or
        complete.get("count")!=9 or
        complete.get("meta_sha256")!=sha(chunk/"R0_5_CHUNK_META.json") or
        m.get("root_sha256")!=roots_sha):
        raise RuntimeError("STOP_XPT_GPU_CHUNK_NOT_COMPLETE")
    root_rows=[]
    for task in tasks:
        native=validate_complete(root_folder(chunk,task),task,head,roots_sha)
        if native.get("method")!="UNMODIFIED_NATIVE_FAST_T150_ADMM2_S5" or native.get("dataset")!="ffhq":
            raise RuntimeError("STOP_XPT_NATIVE_CONFIG")
        root_rows.append({"uid":task["uid"],"image_id":task["image_id"],
                          "run_index":task["rep"],"seed":task["seed"],
                          "sampling_wall_s":native["native_sample_wall_seconds"],
                          "sampling_cuda_event_s":native["sample_cuda_event_seconds"],
                          "sampling_native_steps_s":native["native_step_sum_seconds"],
                          "measurement_sha256":task["measurement_sha256"],
                          "candidate_png_sha256":native["reconstruction_png_sha256"]})
    samples=[g for snap in cards for g in snap if g.index==device]
    peak_used=max((s.used_mib for s in samples),default=None)
    outcome={
        "status":"XP_T_FAST_FFHQ_BATCH1_PASS",
        "stage":"XP-T","head":head,"dataset":"ffhq","family":"FAST",
        "gpu_index":device,"n_images":3,"runs_per_image":3,"new_roots":9,
        "batch_size":1,"model":"ffhq256ddpm","oversample":2.0,
        "sampling_cuda_event_s_median":float(sorted(r["sampling_cuda_event_s"] for r in root_rows)[4]),
        "sampling_wall_s_median":float(sorted(r["sampling_wall_s"] for r in root_rows)[4]),
        "sampling_wall_s_mean":sum(r["sampling_wall_s"] for r in root_rows)/9,
        "gpu_worker_wall_s":secs,"gpu_hours":secs/3600,
        "native_chunk_sha256":sha(chunk/"R0_5_CHUNK_COMPLETE.json"),
        "manifest_sha256":roots_sha,
        "peak_used_mib_sampled":peak_used,
        "timing":timings,
        "timing_clean_eligible":bool(timings["timing_valid"]),
        "historical_replay_gate":"XP_FAST_REPLAY_RECOVERY_PASS",
        "quality_output_gt_free":True,
        "per_run":root_rows,
        "no_gt_access_for_generation":True,
        "evaluation_outcomes_embargo":"NOT_SCORED",
    }
    new_json(root/"XP_T_FAST_COMPLETE.json",outcome)
    return outcome

def main(a):
    root=Path(a.out)
    if any(t in str(root).lower() for t in BAD):
        raise RuntimeError("STOP_XPT_PROTECTED_OUTPUT")
    if not str(root).startswith(str(BASE/"xp_t_fast_")) or not root.is_dir():
        raise RuntimeError("STOP_XPT_OUT_ROOT")
    head=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],text=True).strip()
    if a.expected_head!=head or dirty:
        raise RuntimeError("STOP_XPT_CHECKOUT")
    proof=BASE/"xp_fast_replay_20261026T203415Z/XP_FAST_REPLAY_RECOVERY_STATUS.json"
    replay=json.loads(proof.read_text())
    if (replay.get("status")!="XP_FAST_REPLAY_RECOVERY_PASS" or
        replay.get("r07_label_matches")!=8 or
        replay.get("n_roots")!=8 or replay.get("gpu_rerun_performed") is not False or
        float(replay.get("max_abs_delta_db",1e6))>0.1):
        raise RuntimeError("STOP_XPT_FAST_REPLAY_NOT_PASS")
    prep=json.loads((root/"XP_T_FAST_PREP.json").read_text())
    roots=root/"XP_T_FAST_ROOTS.json"
    if (prep.get("status")!="XP_T_FAST_PREP_PASS" or prep.get("head")!=head
        or sha(roots)!=prep["manifest_sha256"] or
        prep.get("roots")!=9):
        raise RuntimeError("STOP_XPT_PREP_MISMATCH")
    for filename in ("XP_T_FAST_STATUS.json","GPU_EVENTS.jsonl","XP_T_FAST_COMPLETE.json"):
        if (root/filename).exists():
            raise RuntimeError("STOP_XPT_ALREADY_LAUNCHED")
    if not 5<=a.poll_seconds<=60 or not 16000<=a.estimated_vram_mib<=64000:
        raise RuntimeError("STOP_XPT_SETTINGS")
    with (root/"XP_T_FAST_PID").open("x",encoding="utf-8") as f:
        f.write(str(os.getpid())+"\n")
    status=root/"XP_T_FAST_STATUS.json"
    events=root/"GPU_EVENTS.jsonl"
    state={"state":"WAIT_VRAM","head":head,"n_roots":9,"pid":os.getpid(),
           "started_utc":stamp(),"gpu":None,"gpu_hours":0.0,"reason":None}
    update_json(status,state)
    gpu=None
    while gpu is None:
        if (root/"STOP_AFTER_ACTIVE").is_file():
            state["state"]="STOP_REQUESTED";update_json(status,state)
            print("ACTION NEEDED: XPT_STOP_REQUESTED");return 8
        try:
            cards=collect()
            gpu,_=choose_gpu(cards,a.estimated_vram_mib,
                             safety_margin_mib=2048,fraction=0.8)
            record(events,event(cards,stage="XP-T",job_id="FAST_FFHQ_BATCH1",
                                action="ADMISSION",selected_gpu=gpu))
        except Exception as e:
            state["reason"]="GPU_TELEMETRY "+str(e)[:140]
            update_json(status,state)
            time.sleep(a.poll_seconds);continue
        if gpu is None:
            state["state"]="WAIT_VRAM"
            update_json(status,state)
            time.sleep(a.poll_seconds)
    env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(WT),
               PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",
               TMPDIR=str(CACHE/"tmp"),XDG_CACHE_HOME=str(CACHE))
    cmd=[str(FAST),str(WT/"scripts/track_r/xp_fast_gpu.py"),
         "--expected-head",head,"--roots",str(roots),
         "--roots-sha",prep["manifest_sha256"],
         "--out",str(root/"gpu"),"--start","0","--count","9",
         "--physical-gpu",str(gpu),"--dataset","ffhq",
         "--oversample","2.0"]
    preflight=subprocess.run(cmd+["--preflight-only"],env=env,cwd=str(WT),
                            capture_output=True,text=True,timeout=600)
    (root/"XP_T_FAST_PREFLIGHT.log").write_text(preflight.stdout+preflight.stderr)
    if preflight.returncode!=0:
        state.update(state="FAILED_PREFLIGHT",gpu=gpu,reason="NATIVE_CONFIG_OR_SHA")
        update_json(status,state)
        print("ACTION NEEDED: XP_T_FAST_PREFLIGHT_RC="+str(preflight.returncode))
        return 8
    state.update(state="RUNNING_GPU",gpu=gpu,reason=None)
    update_json(status,state)
    samples=[]
    try:
        first=collect()
        samples.append(first)
        record(events,event(first,stage="XP-T",job_id="FAST_FFHQ_BATCH1",
                            action="PRE_GPU",selected_gpu=gpu))
    except Exception as e:
        record(events,{"utc":stamp(),"action":"PRE_TELEMETRY_ERROR","error":str(e)[:200]})
    begin=time.monotonic()
    with (root/"XP_T_FAST_GPU.log").open("xb") as log:
        worker=subprocess.Popen(cmd,env=env,cwd=str(WT),stdout=log,
                                stderr=subprocess.STDOUT,start_new_session=True)
    state["worker_pid"]=worker.pid
    update_json(status,state)
    missing=0
    while worker.poll() is None:
        time.sleep(a.poll_seconds)
        try:
            cards=collect([worker.pid])
            samples.append(cards)
            record(events,event(cards,stage="XP-T",job_id="FAST_FFHQ_BATCH1",
                                action="HEARTBEAT",selected_gpu=gpu,
                                own_roots=[worker.pid]))
        except Exception as e:
            missing+=1
            record(events,{"utc":stamp(),"action":"TELEMETRY_ERROR","error":str(e)[:200]})
    wall=time.monotonic()-begin
    try:
        last=collect([worker.pid])
        samples.append(last)
        record(events,event(last,stage="XP-T",job_id="FAST_FFHQ_BATCH1",
                            action="POST_GPU",selected_gpu=gpu))
    except Exception as e:
        missing+=1
        record(events,{"utc":stamp(),"action":"POST_TELEMETRY_ERROR","error":str(e)[:200]})
    timing=classify_timing(samples,gpu)
    if missing:
        timing["timing_class"]="UNVERIFIED"
        timing["timing_valid"]=False
        timing["reasons"].append("telemetry_samples_missing")
    state["gpu_hours"]=wall/3600
    state["worker_rc"]=worker.returncode
    state["timing"]=timing
    if worker.returncode!=0:
        state.update(state="FAILED_GPU",reason="WORKER_RC_"+str(worker.returncode))
        update_json(status,state)
        print("ACTION NEEDED: XPT_GPU_RC="+str(worker.returncode))
        return 8
    try:
        done=summarize(root,head,roots,prep["manifest_sha256"],gpu,timing,samples,wall)
    except Exception as e:
        state.update(state="FAILED_ANALYSIS",reason=str(e)[:200])
        update_json(status,state)
        raise
    state.update(state="PASS",reason=None,
                 sampling_wall_s_median=done["sampling_wall_s_median"],
                 peak_used_mib=done["peak_used_mib_sampled"])
    update_json(status,state)
    print("XP_T_FAST_FFHQ_BATCH1_PASS roots=9 gpu="+str(gpu)+
          " median_s="+format(done["sampling_wall_s_median"],".3f")+
          " wall_s="+format(wall,".3f")+
          " timing="+timing["timing_class"])
    print("XP_T_FAST_SHA256="+sha(root/"XP_T_FAST_COMPLETE.json"))
    print("OK")
    return 0

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--poll-seconds",type=int,default=10)
    p.add_argument("--estimated-vram-mib",type=int,default=32000)
    try:
        rc=main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: XP_T_FAST "+str(e),flush=True)
        rc=8
    raise SystemExit(rc)
