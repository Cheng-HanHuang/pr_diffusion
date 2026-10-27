#!/usr/bin/env python3
"""A7 SAME-RUN 1645/1645 immutable reconstruction and resume-state audit.

Read-only CPU stage. Does NOT touch GPU, outcomes, or frozen manifests.
Must run from a separate A7 checkout. Original A6 worker code remains b7.
"""
from __future__ import annotations
import argparse,collections,json,os,subprocess,sys,traceback
from pathlib import Path
from scripts.track_r import xp_a7_resume_supervisor as sup

RUN=sup.BASE/"xp_a6_20261027T165112Z"
GENERATION_HEAD="b7af2f7135beb6e8367d83f028e0fbaf163bc7bd"
ALLOWED_STATUS="STOP_NO_CLEAN_GPU_FOR_30_MIN"
COMPLETED={"D1_P2":1200,"D3":400,"EPRIME":45}

def verify(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_A7_AUDIT_GPU_VISIBLE")
    author=Path(__file__).resolve().parents[2]
    if (subprocess.check_output(["git","-C",str(author),"rev-parse","HEAD"],
         text=True).strip()!=a.a7_head
        or subprocess.check_output(["git","-C",str(author),"status","--porcelain"],
                                   text=True).strip()):
        raise RuntimeError("STOP_A7_AUDIT_BRANCH_OR_DIRTY")
    if not (RUN.is_dir() and not RUN.is_symlink()):
        raise RuntimeError("STOP_A7_AUDIT_RUN_ROOT")
    sup.verify_preparation(RUN,GENERATION_HEAD)
    state=json.loads((RUN/"XP_A6_CONTROLLER_STATE.json").read_text())
    smoke=json.loads((RUN/"XP_A6_SMOKE_COMPLETE.json").read_text())
    if (state.get("status")!=ALLOWED_STATUS
        or state.get("head")!=GENERATION_HEAD
        or state.get("active_jobs")!=[]
        or state.get("outcomes_scored") is not False
        or state.get("K_gt4_results_sealed") is not True
        or state.get("gpu_hours",999)>=40
        or smoke.get("status")!="XP_A6_ALL_SMOKES_PASS"
        or smoke.get("head")!=GENERATION_HEAD):
        raise RuntimeError("STOP_A7_AUDIT_CONTROLLER_STATE")
    owner=RUN/"XP_A6_ACTIVE_OWNER.json"
    if owner.exists() and sup.pid_alive(json.loads(owner.read_text()).get("pid",-1)):
        raise RuntimeError("STOP_A7_AUDIT_LIVE_OLD_CONTROLLER")
    tasks={j["id"]:j for j in sup.jobs()}
    done=state.get("done",[])
    if len(done)!=len(set(done)):
        raise RuntimeError("STOP_A7_AUDIT_DUPLICATE_DONE")
    counts=collections.Counter()
    for i,jid in enumerate(done):
        if jid not in tasks:
            raise RuntimeError("STOP_A7_AUDIT_UNEXPECTED_JOB")
        job=tasks[jid]
        sampled=sup.validate(RUN,GENERATION_HEAD,job)
        if len(sampled)!=job["count"]:
            raise RuntimeError("STOP_A7_AUDIT_ROOT_RECEIPTS")
        counts[job["stage"]]+=job["count"]
        if (i+1)%100==0:
            print("A7_AUDIT_RECEIPTS",i+1,"/",len(done),flush=True)
    if dict(counts)!=COMPLETED or sum(counts.values())!=1645:
        raise RuntimeError("STOP_A7_AUDIT_COUNT_MISMATCH_"+str(counts))
    print("A7_ORIGINAL_RECEIPTS_PASS",dict(counts),flush=True)
    print("A7_EXPECTED_GENERATION_HEAD="+GENERATION_HEAD,flush=True)
    print("A7_AVAILABLE_GPU_HOURS="+str(40-float(state["gpu_hours"])),flush=True)
    print("A7_REMAINING_EPRIME=675",flush=True)
    print("A7_K_GT4_OUTCOMES_STILL_SEALED=true",flush=True)
    print("A7_AUDIT_PASS",flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--a7-head",required=True)
    try:verify(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: A7_RESUME_AUDIT "+str(e),flush=True)
        sys.exit(8)
