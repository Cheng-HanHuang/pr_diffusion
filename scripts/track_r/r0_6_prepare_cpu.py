#!/usr/bin/env python3
"""R0.6 CPU preparation: audit all 720 completed native FAST roots, freeze GT-free score tasks.

Only the sealed R0.5 canary and its fully completed four-GPU follow-up are
sources. No new observations, GT, historical PSNR, or protected cohorts read.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from prdiffusion.track_r.manifest import assert_no_forbidden
from scripts.track_r.r0_5_fast_gpu import (
    CANARY_HEAD, CANARY_TASK_SHA, frozen_roots, root_folder, validate_complete)
from prdiffusion.track_r.physics import load_png_rgb8, to_rgb8

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
CANARY=BASE/"r0_5_canary_20261023T004927Z"
R05=BASE/"r0_5_parallel_20261023T012732Z"
R05_HEAD="2a65698a3855ddcc65071af90f69f287c824fa13"
GPU_STAGE_MAX_SECONDS=16*3600
EXPECTED_FIELDS=("uid","source","family","image_id","run_index",
    "measurement_source","candidate_path","candidate_format",
    "candidate_sha256","measurement_path","measurement_sha256")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def safe(path):
    p=Path(assert_no_forbidden(str(path)))
    if not p.is_file() or not p.is_absolute() or p.is_symlink():
        raise RuntimeError("STOP_R06_PROTECTED_OR_MISSING_FILE")
    assert_no_forbidden(str(p.resolve()))
    return p

def load(path):
    return json.loads(safe(path).read_text())

def save_x(path,value):
    with Path(path).open("x",encoding="utf-8") as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")

def manifest_rows():
    task_file=CANARY/"prepare/R0_5_GT_FREE_ROOTS.json"
    if sha(safe(task_file))!=CANARY_TASK_SHA:
        raise RuntimeError("STOP_R06_ROOT_LIST_SHA")
    candidates=frozen_roots(load(task_file),CANARY_HEAD)
    if len(candidates)!=720:
        raise RuntimeError("STOP_R06_NOT_720_ROOTS")
    return candidates

def audit(r05_state,old_roots):
    if (r05_state.get("status")!="R0_5_ALL_720_FAST_ROOTS_COMPLETE_4GPU" or
        r05_state.get("head")!=R05_HEAD or
        r05_state.get("completed_roots")!=720 or
        r05_state.get("active_attempts") or
        len(r05_state.get("completed",[]))!=30 or
        r05_state.get("root_sha256")!=CANARY_TASK_SHA or
        r05_state.get("canary_run")!=str(CANARY) or
        r05_state.get("physical_gpus")!=[0,1,2,3]):
        raise RuntimeError("STOP_R06_R05_NOT_PASS")
    canary_state=load(CANARY/"AUTOPILOT_STATUS.json")
    if (canary_state.get("status")!="R0_5_FAST_8_ROOT_CANARY_PASS" or
        canary_state.get("head")!=CANARY_HEAD or
        canary_state.get("frozen_roots_sha256")!=CANARY_TASK_SHA):
        raise RuntimeError("STOP_R06_CANARY_GATE")
    seen=set(range(8))
    chunks=[(0,8,CANARY/"canary_first_8",CANARY_HEAD)]
    for c in r05_state["completed"]:
        start,count=c["start"],c["count"]
        indices=set(range(start,start+count))
        if (start<8 or count<=0 or count>24 or start%4 or count%4 or
            indices & seen):
            raise RuntimeError("STOP_R06_CHUNK_PARTITION_INVALID")
        seen.update(indices)
        chunks.append((start,count,R05/"chunks"/f"batch_{start:04d}_{count:02d}",R05_HEAD))
    if seen!=set(range(720)):
        raise RuntimeError("STOP_R06_MISSING_ROOT_INDICES")
    output={}
    for start,count,folder,head in chunks:
        meta=load(folder/"R0_5_CHUNK_META.json")
        complete=load(folder/"R0_5_CHUNK_COMPLETE.json")
        if (complete.get("status")!="R0_5_CHUNK_PASS" or
            complete.get("start")!=start or complete.get("count")!=count or
            complete.get("meta_sha256")!=sha(folder/"R0_5_CHUNK_META.json") or
            meta.get("head")!=head or
            meta.get("root_sha256")!=CANARY_TASK_SHA or
            meta.get("uids")!=[row["uid"] for row in old_roots[start:start+count]]):
            raise RuntimeError("STOP_R06_CHUNK_RECEIPT "+str(start))
        for idx in range(start,start+count):
            row=old_roots[idx]
            directory=root_folder(folder,row)
            native=validate_complete(directory,row,head,CANARY_TASK_SHA)
            if native.get("rep")!=row["rep"] or native.get("measurement_source")!=row["measurement_source"]:
                raise RuntimeError("STOP_R06_NATIVE_PROVENANCE "+row["uid"])
            png=safe(directory/"reconstruction.png")
            pt=safe(directory/"reconstruction.pt")
            # Confirm native tensor quantization equals stored canonical PNG
            # without accessing GT; catches representation drift across shards.
            import torch
            tensor=torch.load(pt,map_location="cpu",weights_only=True)
            if set(tensor)!={"reconstruction"}:
                raise RuntimeError("STOP_R06_FLOAT_TENSOR_SCHEMA "+row["uid"])
            x=tensor["reconstruction"]
            if (not isinstance(x,torch.Tensor) or x.shape!=(1,3,256,256) or
                not x.is_floating_point() or not bool(torch.isfinite(x).all()) or
                not bool(torch.equal(to_rgb8(x.float()),load_png_rgb8(png)))):
                raise RuntimeError("STOP_R06_CANONICAL_RGB8_MISMATCH "+row["uid"])
            output[idx]={
                "uid":row["uid"],"source":"S5_r0_fast","family":"FAST",
                "image_id":row["image_id"],"run_index":row["rep"],
                "measurement_source":row["measurement_source"],
                "candidate_path":str(png),"candidate_format":"png",
                "candidate_sha256":native["reconstruction_png_sha256"],
                "measurement_path":row["measurement_path"],
                "measurement_sha256":row["measurement_sha256"]}
    if set(output)!=set(range(720)):
        raise RuntimeError("STOP_R06_NOT_EXACTLY_720")
    return [output[i] for i in range(720)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    out=Path(a.out)
    if not str(out).startswith(str(BASE)+"/") or out.exists() or out.is_symlink():
        raise RuntimeError("STOP_R06_OUTPUT_NAMESPACE_OR_USED")
    head=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    if (head!=a.expected_head or subprocess.check_output(
        ["git","-C",str(WT),"status","--porcelain"],text=True)):
        raise RuntimeError("STOP_R06_GIT_DIRTY_OR_HEAD")
    old=manifest_rows()
    state=load(R05/"AUTOPILOT_STATUS.json")
    rows=audit(state,old)
    if any(set(r)!=set(EXPECTED_FIELDS) for r in rows):
        raise RuntimeError("STOP_R06_GT_FREE_WORKER_FIELDS")
    r04=float(state["r04_gpu_seconds"])
    r05=float(state["total_r05_gpu_seconds_including_canary"])
    earlier=float(state["prior_smoke_gpu_seconds"])
    aggregate=float(state["all_r0_known_gpu_seconds"])
    if not 0<aggregate<GPU_STAGE_MAX_SECONDS or abs(r04+r05+earlier-aggregate)>1e-3:
        raise RuntimeError("STOP_R06_AGGREGATE_LEDGER")
    out.mkdir(parents=True,exist_ok=False)
    file=out/"R0_6_GT_FREE_TASKS.json"
    payload={"schema":"prdiff.track_r.r0.6.score-tasks.v1",
        "status":"R0_6_TASKS_FROZEN","head":head,
        "r05_task_sha256":CANARY_TASK_SHA,
        "n_candidates":720,"gt_or_psnr_fields_in_gpu_tasks":False,
        "tasks":rows}
    save_x(file,payload)
    receipt={"schema":"prdiff.track_r.r0.6.cpu-preflight.v1",
        "status":"R0_6_CPU_PREP_PASS","head":head,
        "tasks_sha256":sha(file),"n_candidates":720,"n_images":180,
        "r04_gpu_seconds":r04,"r05_gpu_seconds":r05,
        "prior_smoke_gpu_seconds":earlier,
        "known_total_gpu_seconds":aggregate,
        "gt_access":False,"gpu_seconds_spent_this_stage":0}
    save_x(out/"R0_6_CPU_PREP_RECEIPT.json",receipt)
    print("R06_CPU_PREP_PASS 720/720 ROOTS",
          "TASK_SHA",receipt["tasks_sha256"],
          "KNOWN_GPU_H",round(aggregate/3600,5),flush=True)

if __name__=="__main__":
    sys.exit(main())
