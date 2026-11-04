#!/usr/bin/env python3
"""A23 G7 GT-only S_mid global2 280-image co-tenant controller, 1 worker/GPU.

Reuses frozen 0,180 A33 pilot outputs (does not rerun them). Charges pilot
GPU time and per-physical-GPU union worker time, audits live foreign memory
and preserves results/checkpoints on budget/communication stops.
No scientific parameter selection, no new solver generation, no XP-K.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

from prdiffusion.track_r.xp_gpu_telemetry import collect
from scripts.track_r.xp_a29_a21_controller import total_union
from scripts.track_r.xp_a34_a23_g7_smid_prod_gpu import (
    ROOT, BASE, WT, LOCK, FREEZE, N, PILOT_HEAD, SOURCE_SHA, C2_SOURCE_SHA,
    RESULT_STATUS, gate, sha, jsonread, git,
)

PILOT_DIR = BASE/"xp_a33_a23_g7_smid_pilot_20261104T070359Z"
SOURCE_COMMIT = "9090a50a4d4ffe922bdb43c5ee9e877c059a7dd1"
C2_STATUS = BASE/"xp_a12_g7_prod_a22_20261104T025355Z/XP_A22_G7_C2_FINAL_STATUS.json"
PROJECT_BASE = 244.45709476042322
C2_USED = 2.6265298946173345
G7_CAP_H = 10.
PROJECT_CAP_H = 400.
MAX_NEW_WORK_H = 6.70  # below 7.355 h after pilot, safe headroom
MAX_WORKER_S = 1800.
MAX_STAGE_WALL_S = 7.*3600.
MIN_FREE_MIB = 28672 # >= 16GiB*1.25+8GiB, conservative B8
LOCK_FILE = BASE/"XP_A12_CAMPAIGN_CONTROLLER.lock"
WORKER = WT/"scripts/track_r/xp_a34_a23_g7_smid_prod_gpu.py"
PY = ROOT/"conda-envs/daps/bin/python"
STOP = False

def stop(*args):
    global STOP
    STOP = True

def fail(ok,code):
    if not ok:
        raise RuntimeError("STOP_A34_G7_CONTROL_"+code)

def write(path,obj):
    p=Path(path)
    temp=p.with_name("."+p.name+".tmp")
    with temp.open("x") as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(temp,p)

def validated_pilot():
    specs=jsonread(LOCK)
    source_sha=sha(LOCK)
    frozen=jsonread(FREEZE/"XP_C_GT_IMAGES.json")["images"]
    res={}
    cost=0.
    for i in (0,180):
        p=PILOT_DIR/f"G7_S_MID_GLOBAL2_PILOT_{i:03d}.json"
        fail(p.is_file() and not p.is_symlink(),"A33_PILOT_RECEIPT_MISSING")
        r=jsonread(p)
        entry=frozen[i]
        uid=f'XPCGT|{entry["dataset"]}|{entry["basename"]}'
        score=r["B8_fixed_t_batched"]
        fail(r["status"]=="A23_G7_S_MID_GLOBAL2_TWO_IMAGE_PILOT_PASS"
             and r["code_head"]==PILOT_HEAD
             and r["source_lock_sha256"]==source_sha
             and r["source_manifest_sha256"]==SOURCE_SHA
             and r["source_sha256"]==entry["source_sha256"]
             and r["dataset"]==entry["dataset"]
             and r["basename"]==entry["basename"]
             and r["B2_B8_choice_equal"] is True
             and r["max_abs_loss_or_margin_diff"]<=1e-5
             and score["denoiser_examples"]==32
             and score["prior_choice"] in ("000","111")
             and r["gpu_occupancy_h"]>0,
             "A33_PILOT_SOURCE_OR_EQUIVALENCE")
        cost+=float(r["gpu_occupancy_h"])
        res[i]=dict(
           status=RESULT_STATUS,index=i,dataset=entry["dataset"],
           basename=entry["basename"],source_sha256=entry["source_sha256"],
           source_manifest_sha256=SOURCE_SHA,
           C2_manifest_sha256=C2_SOURCE_SHA,uid=uid,
           source_lock_sha256=source_sha,source_pilot_head=PILOT_HEAD,
           implementation_head=PILOT_HEAD,reused_A33_pilot=True,
           original_A33_pilot_receipt_sha256=sha(p),
           B8_fixed_t_native_score=score,
           selected_orientation=score["prior_choice"],
           upright_kept=(score["prior_choice"]=="000"),
           denoiser_examples=32,GT_only_diagnostic=True,
           GT_not_passed_to_inverse_solver=True,
           original_XPC3_A20_estimator_preserved=True,
           A21_B8_equivalence_pilot_previously_passed=True,
           physical_gpu=r["physical_gpu"],peak_reserved_mib=r["peak_reserved_mib"],
           not_clean_timing=True,no_main_paper_claim=True)
    fail(0.015<cost<.05,"A33_PILOT_BUDGET")
    return res,cost

def check_ledger(pilot_cost):
    s=jsonread(C2_STATUS)
    fail(s["status"]=="A22_G7_C2_280_COMPLETE"
         and not s["active"]
         and len(s["completed"])==280
         and abs(float(s["gpu_occupancy_h"])-C2_USED)<1e-8
         and abs(float(s["project_gpu_h"])-PROJECT_BASE)<1e-8,
         "A22_G7_LEDGER_FROZEN")
    # Detect later project charges in conventional Track R stage receipts.
    # This does not replace checking all in-flight controllers (global lock).
    for f in BASE.glob("xp_*/XP_*FINAL_STATUS.json"):
        try:
            d=jsonread(f)
        except (OSError,ValueError):
            continue
        if isinstance(d.get("project_gpu_h"),(int,float)):
            fail(float(d["project_gpu_h"])<=PROJECT_BASE+1e-8,
                 "NEWER_UNRECONCILED_PROJECT_LEDGER")
    fail(C2_USED+pilot_cost+MAX_NEW_WORK_H<G7_CAP_H
         and PROJECT_BASE+pilot_cost+MAX_NEW_WORK_H<PROJECT_CAP_H,
         "GLOBAL_AND_G7_BUDGET")
    return s

def valid_result(p,index):
    if not p.is_file() or p.is_symlink():return False
    s=jsonread(p)
    source=jsonread(FREEZE/"XP_C_GT_IMAGES.json")["images"][index]
    sc=s["B8_fixed_t_native_score"]
    return (s["status"]==RESULT_STATUS
        and s["index"]==index
        and s["dataset"]==source["dataset"]
        and s["basename"]==source["basename"]
        and s["source_sha256"]==source["source_sha256"]
        and s["source_manifest_sha256"]==SOURCE_SHA
        and s["C2_manifest_sha256"]==C2_SOURCE_SHA
        and s["source_lock_sha256"]==sha(LOCK)
        and s["GT_only_diagnostic"] is True
        and s["GT_not_passed_to_inverse_solver"] is True
        and s["selected_orientation"]==sc["prior_choice"]
        and s["upright_kept"]==(sc["prior_choice"]=="000")
        and sc["denoiser_examples"]==32
        and abs(float(sc["L000"])-float(sc["L111"])-
                float(sc["prior_margin"]))<1e-6
        and (float(sc["prior_margin"])>1e-7)==
                (sc["prior_choice"]=="111"))

def summary(run,head,claimed,spent,events,pilot_h,stage_status):
    images=jsonread(FREEZE/"XP_C_GT_IMAGES.json")["images"]
    data=[]
    for i in range(N):
        p=run/"results"/f"{i:03d}.json"
        if p.is_file():
            fail(valid_result(p,i),"ACCEPTED_RESULT_INTEGRITY_"+str(i))
            data.append(jsonread(p))
    subset=Counter(x["dataset"] for x in data)
    state=dict(status=stage_status,head=head,
        frozen_source_manifest_sha256=SOURCE_SHA,
        frozen_lock_sha256=sha(LOCK),
        original_pilot_head=PILOT_HEAD,
        pilot_reused_images=[0,180],
        n_complete=len(data),n_expected=N,
        by_dataset=dict(subset),completed_indices=sorted(x["index"] for x in data),
        worker_events=events,
        workers_active=[dict(pid=p,gpu=v["gpu"],index=v["index"])
                        for p,v in claimed.items()],
        A33_pilot_charged_gpu_h=pilot_h,
        new_production_gpu_occupancy_h=spent,
        G7_total_charged_gpu_h=C2_USED+pilot_h+spent,
        G7_total_cap_gpu_h=G7_CAP_H,
        project_charged_gpu_h=PROJECT_BASE+pilot_h+spent,
        project_gpu_cap_h=PROJECT_CAP_H,
        no_new_inverse_solver_runs=True,diagnostic_GT_only=True,
        all_280_required_to_claim_complete=True,
        no_main_paper_data_update=True,no_A21_10=True)
    write(run/"XP_A34_G7_S_MID_STATE.json",state)
    return state

def main(a):
    fail(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and
         git("rev-parse","HEAD")==a.expected_head and
         not git("status","--porcelain"),
         "CLEAN_PINNED_CPU_CONTROLLER")
    fail(Path(a.run).parent==BASE
         and Path(a.run).name.startswith("xp_a34_a23_g7_smid_prod_")
         and not Path(a.run).is_symlink(),
         "RUN_NAMESPACE")
    fail(sha(FREEZE/"XP_C_GT_IMAGES.json")==SOURCE_SHA
         and sha(FREEZE/"XP_C2_EQUIVARIANCE_TASKS.json")==C2_SOURCE_SHA,
         "FROZEN_GT_SOURCE")
    pilot,pilot_h=validated_pilot()
    check_ledger(pilot_h)
    if a.preflight_only:
        print("A34_G7_GT_SOURCE_PILOT_LEDGER_CPU_PREFLIGHT_PASS",
              "pilot_h",pilot_h,"G7_new_cap_h",MAX_NEW_WORK_H,flush=True)
        return

    with LOCK_FILE.open("a+") as fd:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run=Path(a.run)
        fail(not run.exists() or a.resume,"NEW_OR_RESUME")
        if a.resume:
            fail(run.is_dir() and (run/"XP_A34_G7_S_MID_STATE.json").is_file(),
                 "RESUME_STATE_PRESENT")
            old=jsonread(run/"XP_A34_G7_S_MID_STATE.json")
            fail(old["head"]==a.expected_head
                 and old["frozen_source_manifest_sha256"]==SOURCE_SHA
                 and old["workers_active"]==[]
                 and old["status"] in (
                     "PARTIAL_RESOURCE_STOP","PARTIAL_SAFE_BUDGET_STOP",
                     "PARTIAL_USER_STOP"),
                 "RESUME_AUTHORITY")
            spent0=float(old["new_production_gpu_occupancy_h"])
            fail(0<=spent0<MAX_NEW_WORK_H,"RESUME_LEDGER")
            events=list(old["worker_events"])
        else:
            run.mkdir(mode=0o700)
            for name in ("results","logs","pycache"):
                (run/name).mkdir()
            spent0=0.
            events=[]
            for i,x in pilot.items():
                write(run/"results"/f"{i:03d}.json",x)
        # Verify every existing receipt, without rewriting it.
        completed=set()
        for p in (run/"results").glob("*.json"):
            fail(p.stem.isdigit() and 0<=int(p.stem)<N
                 and valid_result(p,int(p.stem)),
                 "PREEXISTING_RESULT_NOT_ACCEPTED_"+p.name)
            completed.add(int(p.stem))
        fail({0,180}.issubset(completed),"PILOT_REUSE_BOTH")
        fail(spent0 <= MAX_NEW_WORK_H,"BUDGET_PRERUN")
        start=time.monotonic()
        intervals=[];active={};n_initial=len(completed)
        status="A34_G7_RUNNING"
        STOPPED=False
        def charged():
            return spent0+sum(total_union(intervals,active).values())
        def next_index():
            busy={z["index"] for z in active.values()}
            return next((i for i in range(N) if i not in completed and i not in busy),None)
        def poll():
            nonlocal STOPPED
            for pid,v in list(active.items()):
                rc=v["proc"].poll()
                if rc is None:continue
                end=time.monotonic()
                del active[pid]
                intervals.append((v["gpu"],v["start"],end))
                v["proc"].wait()
                ix=v["index"]
                path=run/"results"/f"{ix:03d}.json"
                ok=(rc==0 and valid_result(path,ix))
                events.append(dict(index=ix,physical_gpu=v["gpu"],
                        worker_pid=pid,rc=rc,valid=ok,
                        elapsed_seconds=end-v["start"]))
                if ok:
                    completed.add(ix)
                else:
                    STOPPED=True
                    print("ACTION NEEDED: A34_G7_NATIVE_IMAGE_FAILED",ix,
                          "rc",rc,"log",v["log"],flush=True)
        def emit(x):
            return summary(run,a.expected_head,active,charged(),events,pilot_h,x)
        def kill_own():
            # Only children started by this controller; never foreign jobs.
            for pid,v in list(active.items()):
                if v["proc"].poll() is None:
                    try: os.killpg(pid,signal.SIGTERM)
                    except ProcessLookupError: pass
            deadline=time.monotonic()+15
            while active and time.monotonic()<deadline:
                poll();time.sleep(.25)
            for pid,v in list(active.items()):
                if v["proc"].poll() is None:
                    try:os.killpg(pid,signal.SIGKILL)
                    except ProcessLookupError:pass
            while active:
                poll();time.sleep(.25)

        try:
            emit(status)
            while len(completed)<N and not STOP and not STOPPED:
                poll()
                used=charged()
                if used>=MAX_NEW_WORK_H-0.05 or C2_USED+pilot_h+used>=G7_CAP_H-.4:
                    status="PARTIAL_SAFE_BUDGET_STOP";break
                if time.monotonic()-start>=MAX_STAGE_WALL_S:
                    status="PARTIAL_RESOURCE_STOP";break
                cards={c.index:c for c in collect(own_roots=list(active))}
                now=time.monotonic()
                for gpu in range(4):
                    c=cards.get(gpu)
                    if not c or c.visibility!="known":continue
                    if any(v["gpu"]==gpu for v in active.values()):continue
                    if c.free_mib<MIN_FREE_MIB:continue
                    if c.used_mib+MIN_FREE_MIB>int(.8*c.total_mib):continue
                    if len(completed)+len(active)>=N:continue
                    # Guard worst-case one 1800s image per four GPUs + 0.4h
                    # overrun headroom; stop admission before exhausting stage.
                    projected=used+(len(active)+1)*MAX_WORKER_S/3600.
                    if projected>MAX_NEW_WORK_H-.15:continue
                    ix=next_index()
                    if ix is None:break
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),
                          PYTHONPATH=str(WT),PYTHONNOUSERSITE="1",
                          PYTHONDONTWRITEBYTECODE="1",
                          PYTHONPYCACHEPREFIX=str(run/"pycache"/f"I{ix:03d}"),
                          OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",
                          OPENBLAS_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4")
                    log=run/"logs"/f"I{ix:03d}.log"
                    with log.open("x") as f:
                        proc=subprocess.Popen([str(PY),str(WORKER),
                            "--expected-head",a.expected_head,
                            "--run",str(run),
                            "--index",str(ix),"--gpu",str(gpu)],
                            cwd=str(WT),env=env,stdout=f,
                            stderr=subprocess.STDOUT,start_new_session=True)
                    active[proc.pid]=dict(proc=proc,start=time.monotonic(),
                                        gpu=gpu,index=ix,log=str(log))
                # Do not hang forever if zero cards are admissible.
                if not active and len(completed)<N and now-start>180:
                    status="PARTIAL_RESOURCE_STOP";break
                if int(now-start)%30<3:
                    print("A34_G7_PROGRESS",len(completed),"/",N,
                          "new_gpu_h",round(used,5),
                          "G7_total_h",round(C2_USED+pilot_h+used,5),
                          "running",len(active),flush=True)
                    emit(status)
                time.sleep(3)
            if STOP:status="PARTIAL_USER_STOP"
            elif STOPPED:status="PARTIAL_NATIVE_ERROR_STOP"
            elif len(completed)==N:status="A34_G7_S_MID_GLOBAL2_280_COMPLETE"
        except BaseException:
            status="STOP_A34_G7_CONTROLLER_EXCEPTION"
            raise
        finally:
            # Active worker time is included in the terminal ledger.
            if active:kill_own()
            poll()
            if len(completed)==N and status=="A34_G7_RUNNING":
                status="A34_G7_S_MID_GLOBAL2_280_COMPLETE"
            s=emit(status)
            if status=="A34_G7_S_MID_GLOBAL2_280_COMPLETE":
                fail(s["n_complete"]==N and
                     s["G7_total_charged_gpu_h"]<G7_CAP_H and
                     s["project_charged_gpu_h"]<PROJECT_CAP_H,
                     "PRODUCTION_COMPLETION_BUDGET")
                write(run/"XP_A34_G7_S_MID_FINAL_STATUS.json",s)
            print("A34_G7_S_MID_GLOBAL2_TERMINAL",status,
                  "complete",s["n_complete"],
                  "G7_TOTAL_GPU_H",round(s["G7_total_charged_gpu_h"],6),
                  "project_gpu_h",round(s["project_charged_gpu_h"],6),
                  "run",run,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--run",required=True)
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--resume",action="store_true")
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A34_G7_S_MID_CONTROLLER",repr(e),flush=True)
        sys.exit(8)
