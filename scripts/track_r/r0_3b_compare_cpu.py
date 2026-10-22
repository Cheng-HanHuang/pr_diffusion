#!/usr/bin/env python3
"""CPU adjudication of independent GPU R0.3b scorer receipts.

The frozen G0 relative threshold is 1e-5 across batch sizes and GPUs.
This is NOT a full R0.3 PASS until FAST-native smoke/timing passes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path("/egr/research-pac/huang248")
BASE=ROOT/"outputs/pr_diffusion/track_r"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda:f.read(4*1024*1024),b""):
            h.update(part)
    return h.hexdigest()


def load_worker(root):
    root=Path(root)
    status=json.loads((root/"R0_3B_WORKER_RECEIPT.json").read_text())
    if status.get("status")!="R0_3B_GPU_WORKER_PASS" or status.get("completed")!=8:
        raise RuntimeError("STOP_R0_3B_NOT_EIGHT_COMPLETE "+str(root))
    file=root/"R0_3B_CANDIDATES.jsonl"
    if sha(file)!=status.get("candidate_jsonl_sha256"):
        raise RuntimeError("STOP_R0_3B_JSONL_HASH "+str(root))
    rows=[json.loads(line) for line in file.read_text().splitlines()]
    if len(rows)!=8 or [r["uid"] for r in rows]!=status["uids"]:
        raise RuntimeError("STOP_R0_3B_UID_ORDER "+str(root))
    return status,rows


def max_relative(a,b):
    if not math.isfinite(a) or not math.isfinite(b):
        raise RuntimeError("STOP_R0_3B_NONFINITE_SCORE")
    return abs(a-b)/max(abs(a),abs(b),1e-12)


def compare(a,b):
    worst=0.0;record=None;delta=[]
    for ra,rb in zip(a,b):
        if (ra["uid"],ra["input_sha256"]) != (rb["uid"],rb["input_sha256"]):
            raise RuntimeError("STOP_R0_3B_DIFFERENT_CANDIDATES")
        pa,pb=ra["prior"],rb["prior"]
        if (pa["unet_forwards"]!=256 or pb["unet_forwards"]!=256 or
                pa["s_star"]!=pb["s_star"]):
            raise RuntimeError("STOP_R0_3B_FORWARD_COUNT_OR_ORBIT_CHOICE")
        for label in ("S","S_mid"):
            scores_a=pa[label];scores_b=pb[label]
            if set(scores_a)!=set(scores_b):
                raise RuntimeError("STOP_R0_3B_SCORE_KEYS")
            for p in scores_a:
                rel=max_relative(scores_a[p],scores_b[p])
                delta.append(rel)
                if rel>worst:
                    worst=rel;record={"uid":ra["uid"],"field":label,
                                      "orbit":p,"relative":rel,
                                      "GPU8":scores_a[p],"GPU64":scores_b[p]}
        for p,v in pa["per_t_means"].items():
            if set(v)!=set(pb["per_t_means"][p]):
                raise RuntimeError("STOP_R0_3B_TIMESTEP_KEYS")
            for t,m in v.items():
                rel=max_relative(m,pb["per_t_means"][p][t])
                delta.append(rel)
                if rel>worst:
                    worst=rel;record={"uid":ra["uid"],"field":"per_t_means",
                                      "orbit":p,"timestep":t,
                                      "relative":rel,
                                      "GPU8":m,
                                      "GPU64":pb["per_t_means"][p][t]}
    return {"max_relative_difference":worst,"worst_case":record,
            "n_compared_values":len(delta),
            "threshold_relative":1e-5,"status":"PASS" if worst<=1e-5 else "FAIL"}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gpu8-root",required=True)
    ap.add_argument("--gpu64-root",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--expected-head",required=True)
    a=ap.parse_args()
    out=Path(a.out)
    if not str(out).startswith(str(BASE)+"/") or out.exists():
        raise RuntimeError("STOP_R0_3B_OUTPUT_NOT_FRESH")
    head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","status","--porcelain"],text=True)
    if head!=a.expected_head or dirty:raise RuntimeError("STOP_R0_3B_GIT")
    s8,r8=load_worker(a.gpu8_root)
    s64,r64=load_worker(a.gpu64_root)
    if (s8["batch_size"]!=8 or s64["batch_size"]!=64 or
            s8["univfd_enabled"]!=True or s64["univfd_enabled"]!=False or
            s8["physical_gpu"]==s64["physical_gpu"] or
            s8["selection_sha256"]!=s64["selection_sha256"] or
            s8["git_head"]!=head or s64["git_head"]!=head):
        raise RuntimeError("STOP_R0_3B_NOT_DIFFERENT_GPU_BATCH")
    comparison=compare(r8,r64)
    if (s8["unet_forwards"]!=2048 or s64["unet_forwards"]!=2048 or
            s8["univfd_image_forwards"]!=72):
        raise RuntimeError("STOP_R0_3B_FORWARD_LEDGER")
    sample_rates=[x["wall_seconds"] for x in r8]
    # Honest projection: all existing + future FAST prior/UnivFD scoring
    # is extrapolated. FAST generation is NOT included until independently
    # measured on the native worker.
    existing_score_gpu_h=(2520/8.0)*sum(sample_rates)/3600.0
    s5_score_gpu_h=(720/8.0)*sum(sample_rates)/3600.0
    receipt={
        "schema":"prdiff.track_r.r0.3b.determinism.v1",
        "status":"G0_FULL_PASS_R0_3B_NOT_FULL_R0_3" if comparison["status"]=="PASS" else "STOP_G0_GPU_DETERMINISM_FAIL",
        "head":head,"comparison":comparison,
        "physical_gpus":[s8["physical_gpu"],s64["physical_gpu"]],
        "batches":[8,64],
        "unet_forwards":s8["unet_forwards"]+s64["unet_forwards"],
        "univfd_images":s8["univfd_image_forwards"],
        "measured_smoke_gpu_hours":s8["gpu_hours_measured"]+s64["gpu_hours_measured"],
        "projected_existing_scoring_gpu_hours":existing_score_gpu_h,
        "projected_future_s5_scoring_gpu_hours":s5_score_gpu_h,
        "FAST_native_generation_gpu_hours":"UNMEASURED_NOT_AUTHORIZED",
        "full_r0_budget_gate":"PENDING_FAST_SAMPLE_TIMING",
        "r0_budget_ceiling_gpu_hours":16,
        "no_protected_source_access":True,
        "gt_free_gpu_workers":True}
    out.mkdir(parents=True,exist_ok=False)
    with (out/"R0_3B_DETERMINISM_RECEIPT.json").open("x") as f:
        json.dump(receipt,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    print(json.dumps({"status":receipt["status"],
                     "max_relative":comparison["max_relative_difference"],
                     "worst_case":comparison["worst_case"],
                     "score_projection_existing_gpu_h":existing_score_gpu_h,
                     "score_projection_s5_gpu_h":s5_score_gpu_h},sort_keys=True),
          flush=True)
    return 0 if comparison["status"]=="PASS" else 7

if __name__=="__main__":
    sys.exit(main())
