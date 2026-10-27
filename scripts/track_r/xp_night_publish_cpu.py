#!/usr/bin/env python3
"""CPU-only immutable operational XP overnight publisher, embargo-safe.

Independently re-hashes every generated candidate and validates each
native receipt through the frozen GT-free generator validation contract.
Publishes only a small aggregate summary: NO GT, PSNR, images, selection,
per-run rho, predictor outcomes, or candidate files. Never changes PAC
generation outputs; never touches a live GPU job.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

from scripts.track_r import xp_night_supervisor as night

ROOT=night.ROOT
WT=night.WT
BASE=night.BASE
GEN_HEAD="8d9bfed0fa62735791987a24791cfb97123540d8"
RUN=BASE/"xp_overnight_20261027T045215Z"
DEST=WT/"docs/track_r/executors/xp/returns/XP_NIGHT_20261027T045215Z"
EXPECTED_STAGE_COUNTS={"D1_P1":400,"A64":8640,"D2":400}
EXPECTED_PREP_HASH="bffd29a095cda4b8575a08bde48cca152d50cf3af84e799f080619c8c8e409e4"
EXPECTED_MANIFEST_HASHES={
    "D1_P1":"1716a8cf3be6868e2341e26f956fdf009436625a8052b9f08b16ed35a9dcd3d9",
    "D1_P2":"3107d40c495886c52ea2a407f4d8f74079278fffffaa93f4f55eff1edb5272f7",
    "A64":"bbfe0dbd12d963b24ce84d6a24e9b6129c53b46654d423bf630dc2d0976bf848",
    "D2":"9b08c8527c3388cfb44b56f934f62b3e48d254ea433867b1c4d87f5ce8335f44",
}
BANNED=("confirmation","confirm305","test-ffhq","demo-ffhq")
SOURCE_PATHS=(
 "scripts/track_r/xp_night_supervisor.py",
 "scripts/track_r/xp_night_prepare_cpu.py",
 "scripts/track_r/xp_night_d2_worker.py",
 "scripts/track_r/xp_fast_gpu.py",
 "scripts/track_r/xp_t_daps4k_worker_gpu.py",
)

def safe(path):
    p=Path(path)
    if (p.is_symlink() or not p.is_file()
        or ROOT.resolve() not in p.resolve(strict=False).parents
        or any(token in str(p.resolve(strict=False)).lower() for token in BANNED)):
        raise RuntimeError("STOP_XP_NIGHT_PUBLISH_PROTECTED_OR_MISSING_SOURCE")
    return p

def read(path):
    return json.loads(safe(path).read_text())

def digest(path):
    return night.sha(safe(path))

def to_json_x(path,obj):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")

def git(*cmd):
    return subprocess.check_output(
        ["git","-C",str(WT),*cmd],timeout=60)

def verify_source(expected_head):
    if (git("rev-parse","HEAD").decode().strip()!=expected_head
        or git("status","--porcelain").strip()):
        raise RuntimeError("STOP_XP_NIGHT_PUBLISH_GIT")
    # Critical validation/producer code MUST be byte-identical to the
    # actual launched commit despite the new publication commit.
    for relative in SOURCE_PATHS:
        original=git("show",GEN_HEAD+":"+relative)
        current=safe(WT/relative).read_bytes()
        if original!=current:
            raise RuntimeError("STOP_XP_NIGHT_SOURCE_DRIFT_"+relative)
    return True

def verify_state(state,smoke,prep,all_jobs):
    if not (prep.get("status")=="XP_NIGHT_INPUTS_FROZEN"
        and prep.get("head")==GEN_HEAD
        and prep.get("total_new_roots")==10640
        and prep.get("gpu_work_performed") is False
        and prep.get("gt_paths_supplied") is False
        and prep.get("new_quality_scores_generated") is False
        and prep.get("prediction_freeze_required_for_scoring_D1_P2_A64") is True):
        raise RuntimeError("STOP_XP_NIGHT_PREP_CONTRACT")
    if (smoke.get("status")!="XP_NIGHT_ALL_SMOKES_PASS"
        or smoke.get("head")!=GEN_HEAD
        or set(smoke.get("stages",{}))!=set(EXPECTED_STAGE_COUNTS)
        or smoke.get("no_gt_access") is not True
        or smoke.get("outcomes_scored") is not False):
        raise RuntimeError("STOP_XP_NIGHT_SMOKE_CONTRACT")
    for name,count in {"D1_P1":8,"A64":96,"D2":8}.items():
        info=smoke["stages"][name]
        if info.get("n_new_roots")!=count or info.get("source_outcome_inspected") is not False:
            raise RuntimeError("STOP_XP_NIGHT_SMOKE_STAGE_"+name)
    if not (state.get("status")=="MAIN_PASS"
        and state.get("mode")=="main"
        and state.get("head")==GEN_HEAD
        and state.get("completed_roots")==9440
        and state.get("active_jobs")==[]
        and state.get("smoke_pass") is True
        and state.get("outcomes_scored") is False
        and state.get("sealed_outcomes_not_inspected") is True
        and set(state.get("critical_stages_completed",[]))==set(EXPECTED_STAGE_COUNTS)):
        raise RuntimeError("STOP_XP_NIGHT_FINAL_CONTROLLER_STATE")
    total_gpu_h=float(state.get("gpu_hours",-1))
    if not math.isfinite(total_gpu_h) or not 0<total_gpu_h<=46.1:
        raise RuntimeError("STOP_XP_NIGHT_GPU_HOURS")
    jobs=night.jobs()
    expected={j["id"] for j in jobs if j["stage"] in EXPECTED_STAGE_COUNTS}
    done=state.get("done")
    if (not isinstance(done,list) or len(done)!=len(expected)
        or len(done)!=len(set(done)) or set(done)!=expected):
        raise RuntimeError("STOP_XP_NIGHT_CLOSED_JOB_COVERAGE")
    if (state.get("deferred_prediction_gate") not in (["D1_P2"],None)
        or state.get("background_ready") not in ([],None)):
        raise RuntimeError("STOP_XP_NIGHT_DEFERRED_GATE_MUTATION")
    times=state.get("per_job_timing",{})
    if (not isinstance(times,dict) or set(times)!=expected):
        raise RuntimeError("STOP_XP_NIGHT_TIMING_COVERAGE")
    unverified=sum(not v.get("timing_valid",False) for v in times.values())
    if state.get("timing_unverified_jobs")!=unverified:
        raise RuntimeError("STOP_XP_NIGHT_TIMING_ACCOUNTING")
    if len(all_jobs)!=len(jobs) or {j["id"] for j in all_jobs}!={j["id"] for j in jobs}:
        raise RuntimeError("STOP_XP_NIGHT_JOB_PARTITION_CHANGED")
    return {"expected_jobs":len(expected),"gpu_hours":total_gpu_h,
            "timing_unverified_jobs":unverified,
            "timing_clean_jobs":len(expected)-unverified}

def verify_manifests(prep):
    if (digest(RUN/"XP_NIGHT_PREP.json")!=EXPECTED_PREP_HASH
        or set(prep.get("stages",{}))!=set(EXPECTED_MANIFEST_HASHES)):
        raise RuntimeError("STOP_XP_NIGHT_PREP_SHA")
    for name, expected in EXPECTED_MANIFEST_HASHES.items():
        record=prep["stages"][name]
        p=RUN/(name+".json")
        if (record.get("path")!=str(p)
            or record.get("sha256")!=expected
            or digest(p)!=expected or
            record.get("rows")!=night.COUNTS[name]):
            raise RuntimeError("STOP_XP_NIGHT_STAGE_SHA_"+name)
        mf=read(p)
        if (mf.get("head")!=GEN_HEAD
            or mf.get("n_roots")!=night.COUNTS[name]
            or mf.get("gt_paths_supplied") is not False
            or mf.get("stage")!=name
            or mf.get("outcome_embargo") is not (name in ("D1_P2","A64"))):
            raise RuntimeError("STOP_XP_NIGHT_STAGE_EMBARGO_"+name)
    return True

def verify_all_outputs(jobs,verify_job):
    counts=Counter()
    chunk_count=Counter()
    candidate_uids=set()
    for job in jobs:
        if job["stage"] not in EXPECTED_STAGE_COUNTS:
            continue
        # This has no GT read; validates FAST original chunk + all per-root
        # file checksums; DAPS original strict-load receipt and two hashes.
        samples=verify_job(job)
        if (len(samples)!=job["count"] or any(
                not math.isfinite(x) or x<=0 for x in samples)):
            raise RuntimeError("STOP_XP_NIGHT_CANDIDATE_VALIDATION_"+job["id"])
        counts[job["stage"]]+=len(samples)
        chunk_count[job["stage"]]+=1
    if dict(counts)!=EXPECTED_STAGE_COUNTS:
        raise RuntimeError("STOP_XP_NIGHT_ROOT_COVERAGE")
    return {"per_stage_candidates":dict(counts),
            "per_stage_jobs":dict(chunk_count),
            "total_validated_candidates":sum(counts.values()),
            "reconstruction_files_sha256_verified":2*sum(counts.values()),
            "gt_read":False,"new_outcomes_scored":False}

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_XP_NIGHT_PUBLISH_CPU_ONLY")
    verify_source(a.expected_head)
    if not RUN.is_dir() or RUN.is_symlink():
        raise RuntimeError("STOP_XP_NIGHT_RUN_ROOT")
    if DEST.exists() or DEST.is_symlink():
        raise RuntimeError("STOP_XP_NIGHT_ALREADY_PUBLISHED")
    prep=read(RUN/"XP_NIGHT_PREP.json")
    smoke=read(RUN/"XP_NIGHT_SMOKE_COMPLETE.json")
    state=read(RUN/"XP_NIGHT_CONTROLLER_STATE.json")
    verify_manifests(prep)
    all_jobs=night.jobs()
    verified=verify_state(state,smoke,prep,all_jobs)
    result=verify_all_outputs(
        all_jobs,lambda j:night.validate_root(RUN,j,GEN_HEAD))
    if (RUN/"gpu"/"D1_P2").exists() or (RUN/"gpu"/"D1_P2").is_symlink():
        raise RuntimeError("STOP_XP_NIGHT_D1_P2_NOT_SEALED")
    if result["reconstruction_files_sha256_verified"]!=18880:
        raise RuntimeError("STOP_XP_NIGHT_RECONSTRUCTION_HASHES")
    receipt={
       "status":"XP_NIGHT_OPERATIONAL_EVIDENCE_PASS",
       "generation_head":GEN_HEAD,
       "publication_head":a.expected_head,
       "source_code_identical_to_generator_commit":True,
       "frozen_prep_sha256":EXPECTED_PREP_HASH,
       "stage_manifest_sha256":EXPECTED_MANIFEST_HASHES,
       "state_receipt_sha256":digest(RUN/"XP_NIGHT_CONTROLLER_STATE.json"),
       "smoke_receipt_sha256":digest(RUN/"XP_NIGHT_SMOKE_COMPLETE.json"),
       "controller":verified,
       "audit":result,
       "deferred_stage":"D1_P2",
       "theory_freeze_required_for_scoring_D1_P2_A64":True,
       "gt_read":False,
       "reconstruction_images_copied":False,
       "candidate_quality_inspected":False,
       "prediction_registry_updated":False,
       "other_gpu_jobs_interrupted":False,
    }
    DEST.mkdir(parents=True,exist_ok=False)
    target=DEST/"XP_NIGHT_OPERATIONAL_EVIDENCE.json"
    to_json_x(target,receipt)
    to_json_x(DEST/"XP_NIGHT_SMOKE_COMPLETE.json",smoke)
    with (DEST/"SHA256SUMS.txt").open("x") as f:
        for name in ("XP_NIGHT_OPERATIONAL_EVIDENCE.json",
                     "XP_NIGHT_SMOKE_COMPLETE.json"):
            f.write(digest(DEST/name)+"  "+name+"\n")
    print("XP_NIGHT_OPERATIONAL_EVIDENCE_STAGE_PASS")
    print("ROOTS=9440 FAST_D1P1=400 FAST_A64=8640 DAPS_D2=400")
    print("CHUNKS_AND_JOBS="+str(verified["expected_jobs"]))
    print("GPU_HOURS="+str(verified["gpu_hours"]))
    print("HASHED_RECONSTRUCTION_FILES=18880")
    print("TIMING_UNVERIFIED_JOBS="+str(verified["timing_unverified_jobs"]))
    print("DEST="+str(DEST))
    print("EMBARGO=D1P2_A64_XPA16_STILL_ACTIVE")
    print("OK")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    try: main(p.parse_args())
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: NIGHT_PUBLISH "+str(exc),flush=True)
        sys.exit(8)
