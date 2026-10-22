#!/usr/bin/env python3
"""R0.4 CPU: freeze GT-free scoring tasks from validated 2520-candidate manifest.

The GPU worker sees ONLY canonical candidate paths/hashes, signed-y paths/
hashes and opaque UIDs. Explicitly refuse GT paths, scores and oracle fields.
No S5 candidates are included; R0.5 is a separate future stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from prdiffusion.track_r.manifest import assert_no_forbidden, read_manifest

ROOT=Path("/egr/research-pac/huang248")
WT=ROOT/"pr_diffusion_track_r"
BASE=ROOT/"outputs/pr_diffusion/track_r"
ORIGINAL_SHA="8ab7dc375a39a94d1f5183dd58d82f6d5c1c20a3d7124c7249df6a685c5ef890"
R03B_HEAD="6767f1850235a3df0c94fea9b53135016dbe271f"
R03C_HEAD="9bc30d9a910d267a45bc791cc43dab8597e4ee61"

FIELDS=("uid","source","family","image_id","run_index","candidate_path",
        "candidate_format","candidate_sha256","measurement_path",
        "measurement_sha256")

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def read_safe(path):
    p=Path(assert_no_forbidden(str(path)))
    if not p.is_file() or not p.is_absolute() or p.is_symlink():
        raise RuntimeError("STOP_R04_MISSING_OR_SYMLINK "+str(p))
    assert_no_forbidden(str(p.resolve()))
    return p

def verify_receipts(manifest,g0,g1,r03b,r03c):
    if digest(read_safe(manifest))!=ORIGINAL_SHA:
        raise RuntimeError("STOP_R04_HISTORICAL_MANIFEST_SHA")
    r0=json.loads(read_safe(g0).read_text())
    if (r0.get("status")!="G0_CPU_PRECHECK_PASS_PENDING_DETERMINISM" or
        r0.get("manifest_path")!=str(manifest) or
        r0["gates"]["G0_recorded_psnr_replay"].get("n_match")!=2520):
        raise RuntimeError("STOP_R04_G0_CPU_RECEIPT")
    s=json.loads(read_safe(r03b).read_text())
    if (s.get("status")!="G0_FULL_PASS_R0_3B_NOT_FULL_R0_3" or
        s.get("head")!=R03B_HEAD or
        s.get("comparison",{}).get("status")!="PASS" or
        float(s["comparison"]["max_relative_difference"])>1e-5):
        raise RuntimeError("STOP_R04_G0_GPU_RECEIPT")
    g=json.loads(read_safe(g1).read_text())
    if (g.get("status")!="G1_PASS" or g.get("manifest_sha256")!=ORIGINAL_SHA or
        g.get("prior",{}).get("s_star")!="111"):
        raise RuntimeError("STOP_R04_G1_RECEIPT")
    c=json.loads(read_safe(r03c).read_text())
    if (c.get("status")!="R0_3C_PASS_READY_FOR_R04_IMPLEMENTATION" or
        c.get("source_head")!=R03C_HEAD or
        c.get("budget",{}).get("status")!="PASS" or
        float(c["budget"]["conservative_total_gpu_h"])>16 or
        c.get("replay",{}).get("status")!="PASS"):
        raise RuntimeError("STOP_R04_R03C_BUDGET_OR_REPLAY")
    return c["budget"]

def tasks(rows):
    if len(rows)!=2520 or len({r["uid"] for r in rows})!=2520:
        raise RuntimeError("STOP_R04_MANIFEST_COUNT")
    hashes={}
    result=[]
    for row in rows:
        candidate=read_safe(row["candidate_path"])
        measurement=read_safe(row["measurement_path"])
        if row["source"] not in ("S1_b22","S2_b24_dev80","S3_a12","S4_a15"):
            raise RuntimeError("STOP_R04_UNEXPECTED_SOURCE")
        if str(candidate.suffix).lower() not in (".pt",".png"):
            raise RuntimeError("STOP_R04_BAD_CANDIDATE_SUFFIX")
        key=str(measurement)
        if key not in hashes:
            hashes[key]=digest(measurement)
        if row["measurement_sha256"] and row["measurement_sha256"]!=hashes[key]:
            raise RuntimeError("STOP_R04_MEAS_SHA "+row["uid"])
        result.append({
            "uid":row["uid"],"source":row["source"],"family":row["family"],
            "image_id":row["image_id"],"run_index":row["run_index"],
            "candidate_path":str(candidate),
            "candidate_format":row["candidate_format"],
            "candidate_sha256":digest(candidate),
            "measurement_path":key,"measurement_sha256":hashes[key]
        })
    if any(set(r)!=set(FIELDS) for r in result):
        raise RuntimeError("STOP_R04_GT_FREE_WORKER_SCHEMA")
    return result,hashes

def main():
    p=argparse.ArgumentParser()
    for flag in ("out","expected-head","manifest","g0-receipt","g1-receipt",
                 "r03b-receipt","r03c-receipt"):
        p.add_argument("--"+flag,required=True)
    a=p.parse_args()
    dest=Path(a.out)
    if not str(dest).startswith(str(BASE)+"/") or dest.exists() or dest.is_symlink():
        raise RuntimeError("STOP_R04_OUTPUT_NAMESPACE")
    head=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    status=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],text=True)
    if head!=a.expected_head or status:
        raise RuntimeError("STOP_R04_HEAD_OR_TREE")
    budget=verify_receipts(a.manifest,a.g0_receipt,a.g1_receipt,
                           a.r03b_receipt,a.r03c_receipt)
    source=read_manifest(read_safe(a.manifest))
    clean,measurements=tasks(source)
    dest.mkdir(parents=True,exist_ok=False)
    path=dest/"R0_4_GT_FREE_TASKS.json"
    payload={"schema":"prdiff.track_r.r0.4.gt_free_tasks.v1",
        "status":"R0_4_TASKS_FROZEN","head":head,
        "original_manifest_sha256":ORIGINAL_SHA,
        "gt_or_psnr_fields_in_gpu_tasks":False,
        "n_candidates":len(clean),"n_measurement_files":len(measurements),
        "tasks":clean}
    with path.open("x",encoding="utf-8") as f:
        json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    receipt={"status":"R0_4_TASKS_READY","head":head,
        "n_candidates":len(clean),"n_measurements":len(measurements),
        "tasks_sha256":digest(path),
        "source_manifest_sha256":ORIGINAL_SHA,
        "r03c_budget_projection_gpu_h":budget["conservative_total_gpu_h"],
        "gpu_calls":0,"protected_cohort_access":False,
        "gt_access_after_preparation":False}
    with (dest/"R0_4_PREPARE_RECEIPT.json").open("x") as f:
        json.dump(receipt,f,indent=2,sort_keys=True);f.write("\n")
    print("R04_PREPARED",len(clean),"UIDS",len(measurements),
          "OBSERVATIONS","SHA",receipt["tasks_sha256"],flush=True)

if __name__=="__main__":
    sys.exit(main())
