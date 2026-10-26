#!/usr/bin/env python3
"""Track R XP GPU queue with opportunistic overlap and remeasurement.

This controller is a framework: it executes ONLY committed, preregistered
XP GPU worker commands, AFTER manifests and replay/physics gates exist.
It never creates an XP job or authorizes a missing scientific gate by itself.
Only one scheduler should own an output root (O_EXCL PID lock).

Quality outputs are independent of timing cleanliness. GPU contention is
recorded on every job, at launch, during progress and at completion.
When memory is tight, wait/rescan; never kill foreign processes.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from dataclasses import asdict

from prdiffusion.track_r.xp_gpu_telemetry import (
    collect,choose_gpu,classify_timing,event,
)

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
BAD=("confirmation","confirm305","test-ffhq","demo-ffhq")
LANES=("CRITICAL","BACKGROUND")
CAP_GPU_H=400.0
APPROVED_WORKERS={
    "xp_fast_gpu.py", "xp_sitcom_gpu.py", "xp_daps_gpu.py",
    "xp_dps_gpu.py", "xp_pcdaps_gpu.py", "xp_difffpr_gpu.py"}
STOP=False

def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def atomic_json(path,obj):
    p=Path(path)
    tmp=p.with_name("."+p.name+".tmp."+str(os.getpid()))
    with tmp.open("x",encoding="utf-8") as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def append_event(root,item):
    with (root/"GPU_EVENTS.jsonl").open("a",encoding="utf-8") as f:
        f.write(json.dumps(item,sort_keys=True,allow_nan=False)+"\n")

def guarded(path):
    p=Path(path)
    if any(word in str(p).lower() for word in BAD):
        raise RuntimeError("STOP_PROTECTED_PATH")
    r=p.resolve(strict=False)
    if any(word in str(r).lower() for word in BAD):
        raise RuntimeError("STOP_PROTECTED_TARGET")
    base=ROOT.resolve(strict=False)
    if base!=r and base not in r.parents:
        raise RuntimeError("STOP_OUTSIDE_PAC")
    return p

def head():
    return subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],
                                   text=True,timeout=30).strip()

def verify_checkout(expected):
    if head()!=expected or subprocess.check_output(
        ["git","-C",str(WT),"status","--porcelain"],text=True,timeout=30).strip():
        raise RuntimeError("STOP_UNCLEAN_OR_MISMATCHED_HEAD")

def valid_job(raw,expected):
    required=("id","stage","lane","family","kind","expected_vram_mib",
              "projected_sec","argv","worker_receipt","expected_measurement_sha256")
    if any(k not in raw for k in required):
        raise RuntimeError("STOP_JOB_SCHEMA")
    jid=raw["id"]
    if (not isinstance(jid,str) or not jid or len(jid)>100 or
        any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
            for c in jid)):
        raise RuntimeError("STOP_JOB_ID")
    for key in ("stage","family"):
        x=raw[key]
        if (not isinstance(x,str) or not x or len(x)>100 or
            any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."
                for c in x)):
            raise RuntimeError("STOP_JOB_FIELD_"+key)
    if raw["lane"] not in LANES or raw["kind"] not in ("quality","timing"):
        raise RuntimeError("STOP_JOB_LANE_OR_KIND")
    mem=int(raw["expected_vram_mib"])
    sec=float(raw["projected_sec"])
    if not (1024<=mem<=95000 and 1<=sec<=86400):
        raise RuntimeError("STOP_JOB_RESOURCE_HINTS")
    argv=raw["argv"]
    if not isinstance(argv,list) or len(argv)<3 or not all(isinstance(x,str) for x in argv):
        raise RuntimeError("STOP_JOB_ARGV")
    for item in argv:
        if any(x in item.lower() for x in BAD):
            raise RuntimeError("STOP_JOB_PROTECTED_ARG")
    program=guarded(argv[0])
    runner=guarded(argv[1])
    expected_runners=WT/"scripts"/"track_r"
    if (not program.is_file() or runner.parent.resolve()!=expected_runners.resolve() or
        runner.name not in APPROVED_WORKERS or not runner.is_file()):
        raise RuntimeError("STOP_UNAPPROVED_GPU_WORKER")
    if raw.get("code_head")!=expected:
        raise RuntimeError("STOP_MANIFEST_HEAD")
    if "{attempt_dir}" not in argv:
        raise RuntimeError("STOP_NO_ATTEMPT_DIRECTORY")
    receipt=Path(raw["worker_receipt"])
    if receipt.is_absolute() or ".." in receipt.parts or any(w in str(receipt).lower() for w in BAD):
        raise RuntimeError("STOP_WORKER_RECEIPT_PATH")
    if raw["expected_measurement_sha256"] and (
        len(raw["expected_measurement_sha256"])!=64 or
        any(c not in "0123456789abcdef" for c in raw["expected_measurement_sha256"])):
        raise RuntimeError("STOP_MEASUREMENT_DIGEST")
    return dict(raw,expected_vram_mib=mem,projected_sec=sec)

def load_jobs(spec,expected):
    doc=json.loads(guarded(spec).read_text())
    if doc.get("status")!="XP_MANIFEST_FROZEN" or doc.get("head")!=expected or doc.get("gates_pass") is not True:
        raise RuntimeError("STOP_MANIFEST_NOT_FROZEN_OR_GATES")
    jobs=[valid_job(x,expected) for x in doc.get("jobs",[])]
    if not jobs or len({x["id"] for x in jobs})!=len(jobs):
        raise RuntimeError("STOP_JOB_IDS")
    # Reject a job queue that alone exceeds the total GPU-hour budget.
    if sum(x["projected_sec"] for x in jobs)/3600>CAP_GPU_H:
        raise RuntimeError("STOP_BUDGET_400_GPUH")
    return jobs

def receipt_valid(job,path,expected):
    """Only a full, pinned, generated receipt can advance a job to DONE."""
    p=guarded(path)
    if not p.is_file() or p.is_symlink():return False
    try:d=json.loads(p.read_text())
    except (json.JSONDecodeError,OSError):return False
    return (d.get("status")=="XP_JOB_PASS" and
            d.get("job_id")==job["id"] and
            d.get("head")==expected and
            d.get("measurement_sha256")==job["expected_measurement_sha256"] and
            d.get("gt_used_during_generation") is False)

def next_attempt_dir(root,job_id):
    p=root/"attempts"/job_id
    p.mkdir(parents=True,exist_ok=True)
    for i in range(1,50):
        candidate=p/("attempt_%02d"%i)
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise RuntimeError("STOP_TOO_MANY_ATTEMPTS")

def review(state,job):
    """Stable job states; timing dirtiness never invalidates quality."""
    x=state.get(job["id"],{})
    if x.get("state") in ("DONE","TIMING_CLEAN"):
        return False
    if x.get("state")=="FAILED":
        return False
    if x.get("state")=="TIMING_REMEASURE_PENDING":
        return True
    return True

def choose_pending(jobs,state):
    def order(job):
        old=state.get(job["id"],{})
        remeasure=old.get("state")=="TIMING_REMEASURE_PENDING"
        return (0 if job["lane"]=="CRITICAL" else 1,
                1 if remeasure else 0,
                job["stage"],job["id"])
    return sorted((j for j in jobs if review(state,j)),key=order)

def record_snapshot(root,job,gpu_idx,action,own_roots):
    try:
        cards=collect(own_roots)
        append_event(root,event(cards,stage=job["stage"],job_id=job["id"],
                                action=action,selected_gpu=gpu_idx,own_roots=own_roots))
        return cards
    except Exception as exc:
        append_event(root,{"utc":timestamp(),"job_id":job["id"],"action":action,
                           "selected_gpu":gpu_idx,"telemetry_error":str(exc)[:250]})
        return []

def checkpoint(root,state, running, waiting, gpu_hours,stage_failures,headsha):
    payload={"state":"RUNNING" if running else ("WAIT_VRAM_OR_CLEAN" if waiting else "READY"),
             "head":headsha,"at":timestamp(),
             "lanes":{},"gpu_hours":gpu_hours,"projected_gpu_hours":None,
             "waiting":waiting,"stop_flags":sorted(stage_failures),
             "active_jobs":{k:{"gpu":v["gpu"],"pid":v["proc"].pid,"started":v["utc"],
                               "kind":v["job"]["kind"]} for k,v in running.items()}}
    for lane in LANES:
        lane_ids=[j for j in state if state[j].get("lane")==lane]
        tally={k:0 for k in ("done","running","failed","queued")}
        for j in lane_ids:
            st=state[j]["state"]
            if st in ("DONE","TIMING_CLEAN"):tally["done"]+=1
            elif st=="RUNNING":tally["running"]+=1
            elif st=="FAILED":tally["failed"]+=1
            else:tally["queued"]+=1
        payload["lanes"][lane]=tally
    atomic_json(root/"XP_CONTROLLER_STATUS.json",payload)

def ctrl_stop(_sig,_frame):
    global STOP
    STOP=True

def start(job,gpu,root,expected):
    attempt=next_attempt_dir(root,job["id"])
    attempt.mkdir(parents=True,exist_ok=False)
    cmd=[str(attempt) if v=="{attempt_dir}" else
         str(gpu) if v=="{gpu}" else v for v in job["argv"]]
    if any("{" in v or "}" in v for v in cmd):
        raise RuntimeError("STOP_UNSUBSTITUTED_ARG")
    env=dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    env["PYTHONDONTWRITEBYTECODE"]="1"
    env["PYTHONNOUSERSITE"]="1"
    env["TRACK_R_XP_HEAD"]=expected
    env["TRACK_R_XP_JOB_ID"]=job["id"]
    logfile=(attempt/"gpu_worker.log").open("xb")
    try:
        p=subprocess.Popen(cmd,cwd=str(WT),env=env,stdout=logfile,stderr=subprocess.STDOUT,
                           start_new_session=True)
    finally:logfile.close()
    return {"proc":p,"gpu":gpu,"attempt":attempt,"job":job,
            "utc":timestamp(),"start":time.monotonic(),
            "samples":[],"contaminated_seconds":0.0,
            "last_sample":time.monotonic()}

def finish(root, state, item, expected, memory_retry=2):
    job=item["job"]
    jid=job["id"]
    p=item["proc"]
    rc=p.poll()
    samples=item["samples"]
    timing=classify_timing(samples,item["gpu"])
    attempt=item["attempt"]
    cost=(time.monotonic()-item["start"])/3600
    receipt=attempt/job["worker_receipt"]
    valid=(rc==0 and receipt_valid(job,receipt,expected))
    prior=state[jid].get("attempts",0)
    previous=state[jid].get("gpu_hours",0.0)
    # Recovery from OOM is permitted only via a brand new immutable attempt.
    oom=False
    if not valid:
        log=attempt/"gpu_worker.log"
        if log.is_file():
            with log.open("rb") as f:
                tail=f.read()[-16384:].decode("utf-8","replace").lower()
            oom=("out of memory" in tail or "cuda error: memory allocation" in tail)
    if valid:
        if job["kind"]=="timing" and not timing["timing_valid"]:
            state_name="TIMING_REMEASURE_PENDING" if prior<3 else "FAILED"
        else:
            state_name="TIMING_CLEAN" if job["kind"]=="timing" else "DONE"
    elif oom and prior<memory_retry+1:
        state_name="WAIT_VRAM"
    else:
        state_name="FAILED"
    row=dict(id=jid,stage=job["stage"],family=job["family"],lane=job["lane"],
             state=state_name,attempts=prior+1,exit_code=rc,gpu=item["gpu"],
             gpu_hours=previous+cost,worker_receipt_valid=valid,
             measured_sec=cost*3600,oom_detected=oom,
             timing=timing,at=timestamp(),attempt_directory=str(attempt))
    atomic_json(attempt/"XP_ATTEMPT_ACCOUNTING.json",row)
    state[jid]=row
    append_event(root,{"utc":timestamp(),"job_id":jid,"action":"FINISH",
                       "gpu":item["gpu"],"exit_code":rc,"state":state_name,
                       "gpu_hours":cost,"timing":timing})
    return cost

def main(a):
    global STOP
    if a.poll_seconds<1 or a.poll_seconds>120 or a.memory_margin_mib<512:
        raise RuntimeError("STOP_CONTROLLER_SETTINGS")
    verify_checkout(a.expected_head)
    root=guarded(a.root)
    if not str(root).startswith(str(BASE/"xp_controller_")):
        raise RuntimeError("STOP_CONTROLLER_NAMESPACE")
    root.mkdir(parents=True,exist_ok=True)
    lock=root/"XP_CONTROLLER_PID"
    try:
        fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:
        raise RuntimeError("STOP_CONTROLLER_ALREADY_OWNED")
    with os.fdopen(fd,"w") as f:
        f.write(str(os.getpid())+"\n")
    signal.signal(signal.SIGTERM,ctrl_stop)
    signal.signal(signal.SIGINT,ctrl_stop)
    try:
        state_file=root/"XP_CONTROLLER_JOBS.json"
        state=json.loads(state_file.read_text()) if state_file.is_file() else {}
        total_used=sum(float(v.get("gpu_hours",0)) for v in state.values())
        running={}
        failures=set()
        while True:
            verify_checkout(a.expected_head)
            jobs=load_jobs(a.manifest,a.expected_head)
            for job in jobs:
                entry=state.setdefault(job["id"],{"state":"QUEUED","attempts":0,
                                      "gpu_hours":0.0,"lane":job["lane"]})
                if entry.get("lane")!=job["lane"]:
                    raise RuntimeError("STOP_JOB_LANE_CHANGED")
            # Crash recovery of an unrecorded previous active worker is a
            # hard STOP: a human must reconcile GPU processes/receipts first.
            if not running and any(v["state"]=="RUNNING" for v in state.values()):
                raise RuntimeError("STOP_ORPHAN_RUNNING_JOB_RECONCILE")
            for jid,item in list(running.items()):
                item["samples"].append(record_snapshot(root,item["job"],item["gpu"],
                                                        "HEARTBEAT",(item["proc"].pid,)))
                if item["proc"].poll() is not None:
                    item["samples"].append(record_snapshot(root,item["job"],item["gpu"],
                                                            "POST",(item["proc"].pid,)))
                    total_used+=finish(root,state,item,a.expected_head)
                    del running[jid]
                else:
                    # Clean timing projection is bounded; foreign overlap
                    # expands the allowance for timing/quality completion.
                    card=next((c for c in item["samples"][-1] if c.index==item["gpu"]),None)
                    current=time.monotonic()
                    if card is None or card.foreign_pids or card.visibility!="known":
                        item["contaminated_seconds"]+=current-item["last_sample"]
                    item["last_sample"]=current
                    budget=3*item["job"]["projected_sec"]
                    uncontaminated=current-item["start"]-item["contaminated_seconds"]
                    if uncontaminated>budget:
                        # Never kill existing healthy work: record overrun,
                        # but preserve user-authorized completion.
                        item["overtime_sec"]=uncontaminated-budget
            if total_used>=CAP_GPU_H:
                failures.add("STOP_400_GPU_HOUR_CEILING")
                STOP=True
            stop_requested=(root/"STOP_AFTER_ACTIVE").is_file()
            if STOP or stop_requested:
                checkpoint(root,state,running,[],total_used,failures,a.expected_head)
                if not running:
                    break
                time.sleep(a.poll_seconds)
                continue
            pending=choose_pending(jobs,state)
            if not pending and not running:
                break
            waiting=[]
            try:cards=collect(tuple(x["proc"].pid for x in running.values()))
            except Exception as exc:
                failures.add("STOP_GPU_TELEMETRY "+str(exc)[:120])
                checkpoint(root,state,running,["NO_TELEMETRY"],total_used,failures,a.expected_head)
                time.sleep(a.poll_seconds)
                continue
            reservation={}
            for item in running.values():
                g=item["gpu"]
                reservation[g]=reservation.get(g,0)+item["job"]["expected_vram_mib"]
            for job in pending:
                if job["id"] in running:continue
                old=state[job["id"]]
                clean_only=(old["state"]=="TIMING_REMEASURE_PENDING")
                gpu,why=choose_gpu(cards,job["expected_vram_mib"],
                    reservations=reservation,
                    safety_margin_mib=a.memory_margin_mib,
                    clean_only=clean_only)
                if gpu is None:
                    tag="WAIT_CLEAN_TIMING" if clean_only else "WAIT_VRAM"
                    waiting.append(job["id"]+":"+tag)
                    if old["state"] not in ("TIMING_REMEASURE_PENDING","WAIT_VRAM"):
                        old["state"]=tag
                    continue
                # No fabricated accepted receipts or test bypass.
                if total_used+sum(x["projected_sec"] for x in jobs if
                    state[x["id"]]["state"] not in ("DONE","TIMING_CLEAN"))/3600>CAP_GPU_H:
                    failures.add("STOP_PROJECTED_400_GPU_HOURS")
                    break
                item=start(job,gpu,root,a.expected_head)
                item["samples"].append(record_snapshot(root,job,gpu,
                                                         "START",(item["proc"].pid,)))
                running[job["id"]]=item
                state[job["id"]]=dict(old,state="RUNNING",
                                       lane=job["lane"],gpu=gpu)
                reservation[gpu]=reservation.get(gpu,0)+job["expected_vram_mib"]
                append_event(root,{"utc":timestamp(),"job_id":job["id"],
                                   "action":"START","gpu":gpu,
                                   "memory_reserve_mib":job["expected_vram_mib"]})
            checkpoint(root,state,running,waiting,total_used,failures,a.expected_head)
            atomic_json(state_file,state)
            if not running and not pending:
                break
            if not running and any(x.startswith("STOP_PROJECTED") for x in failures):
                break
            time.sleep(a.poll_seconds)
        atomic_json(state_file,state)
        checkpoint(root,state,{},[],total_used,failures,a.expected_head)
        if not failures:
            print("XP_GPU_CONTROLLER_STOPPED gpu_h="+format(total_used,".3f")+" OK")
        else:
            print("ACTION NEEDED: "+";".join(sorted(failures)))
    finally:
        if lock.is_file() and lock.read_text().strip()==str(os.getpid()):
            lock.unlink()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-head",required=True)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--root",required=True)
    parser.add_argument("--poll-seconds",type=int,default=10)
    parser.add_argument("--memory-margin-mib",type=int,default=2048)
    try:main(parser.parse_args())
    except Exception as e:
        print("ACTION NEEDED: XP_GPU_CONTROLLER "+str(e),flush=True)
        raise SystemExit(8)
