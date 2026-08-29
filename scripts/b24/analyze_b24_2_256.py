#!/usr/bin/env python3
"""Aggregate immutable B24.2 prefix-64 plus rows 64--255 into cumulative 256."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_parent64(root: Path):
    rows=[]
    for shard in range(4):
        summary=read_json(root/f"shard{shard}/SHARD_COMPLETE.json")
        if summary.get("status")!="PASS" or int(summary.get("completed",-1))!=16:
            raise RuntimeError(f"bad parent shard {shard}")
        for p in sorted((root/f"shard{shard}").glob("row*/IMAGE_COMPLETE.json")):
            v=read_json(p)
            if v.get("status")!="PASS" or v.get("stage")!="B24.2_64":
                raise RuntimeError(f"bad parent completion {p}")
            rows.append(v)
    if len(rows)!=64 or sorted(int(r["row_index"]) for r in rows)!=list(range(64)):
        raise RuntimeError("parent64 coverage mismatch")
    return rows


def load_extension(root: Path):
    rows=[]; shard_summaries=[]
    for shard in range(4):
        sroot=root/f"shard{shard}"
        summary=read_json(sroot/"SHARD_COMPLETE.json")
        if summary.get("status")!="PASS" or int(summary.get("completed",-1))!=48:
            raise RuntimeError(f"bad extension shard {shard}")
        shard_summaries.append(summary)
        for p in sorted(sroot.glob("row*/IMAGE_COMPLETE.json")):
            v=read_json(p)
            if v.get("status")!="PASS" or v.get("stage")!="B24.2_256_EXTENSION":
                raise RuntimeError(f"bad extension completion {p}")
            rows.append(v)
    if len(rows)!=192 or sorted(int(r["row_index"]) for r in rows)!=list(range(64,256)):
        raise RuntimeError("extension coverage mismatch")
    return rows, shard_summaries


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--parent64",type=Path,required=True)
    ap.add_argument("--extension",type=Path,required=True)
    args=ap.parse_args()
    parent=args.parent64.resolve(); ext=args.extension.resolve()
    p64=load_parent64(parent)
    new, shard_summaries=load_extension(ext)
    rows=p64+new
    if len({r["image_id"] for r in rows})!=256:
        raise RuntimeError("duplicate image across cumulative 256")
    counts=Counter(r["class_label"] for r in rows)
    prevalence={c:counts.get(c,0)/256.0 for c in "ABCD"}
    b=counts.get("B",0); c=counts.get("C",0)
    naive_b=100.0/prevalence["B"] if prevalence["B"] else None
    naive_c=100.0/prevalence["C"] if prevalence["C"] else None
    ext_wall=max(float(x["shard_wall_seconds"]) for x in shard_summaries)
    throughput=192.0/(ext_wall/3600.0)
    waits=sum(float(x.get("gpu_fit_wait_seconds_approx",0.0)) for x in shard_summaries)
    payload={
        "schema_version":"b24.baseline-256-summary.v1",
        "stage":"B24.2_256_CUMULATIVE",
        "status":"PASS",
        "n_images":256,
        "parent64_runroot":str(parent),
        "extension_runroot":str(ext),
        "class_counts":{x:counts.get(x,0) for x in "ABCD"},
        "class_prevalence":prevalence,
        "gating_collection_classes":["B","C"],
        "d_retained_but_not_gating":True,
        "naive_total_images_for_100_B":naive_b,
        "naive_total_images_for_100_C":naive_c,
        "bottleneck_point_estimate_total_images_for_100_BC":max(x for x in (naive_b,naive_c) if x is not None),
        "extension_observed_images_per_hour_four_gpu":throughput,
        "extension_max_shard_wall_seconds":ext_wall,
        "aggregate_gpu_fit_wait_seconds_approx":waits,
        "per_image_wall_seconds":{
            "mean":statistics.mean(float(r["image_wall_seconds"]) for r in new),
            "median":statistics.median(float(r["image_wall_seconds"]) for r in new),
            "max":max(float(r["image_wall_seconds"]) for r in new),
        },
        "best_psnr_raw_rgb_db":{
            "DAPS":{"mean":statistics.mean(float(r["daps_best_psnr_raw_rgb_db"]) for r in rows)},
            "SITCOM":{"mean":statistics.mean(float(r["sitcom_best_psnr_raw_rgb_db"]) for r in rows)},
        },
        "planning_note":"B/C point-prevalence totals remain descriptive; larger-prefix planning should include rare-class uncertainty. D is retained opportunistically but is not an immediate collection gate.",
    }
    out=ext/"B24_2_256_CUMULATIVE_SUMMARY.json"
    out.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","class_counts":payload["class_counts"],"images_per_hour":throughput,"summary":str(out)},sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
