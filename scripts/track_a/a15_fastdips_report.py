#!/usr/bin/env python3
"""A1.5 zero-GPU report: evaluate complete native FAST independent K1/K4 roots.

Raw RGB8 GT PSNR is for retrospective evaluation only, not deployable selection.
Reads original A1.2 oracle data and new A1.5 terminals, never reconstructs.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import sys

from scripts.track_a.a15_fastdips_preflight import CFG, load, sha, check_source, file_ok

def write_json_x(path,value):
    with path.open("x",encoding="utf8") as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-head",required=True)
    ap.add_argument("--k",type=int,choices=[1,2,4],required=True)
    a=ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP report is CPU-only")
    cfg=load(CFG)
    source=check_source(cfg,a.expected_head)
    out=Path(cfg["output_root"])/{1:cfg["report_k1_stage"],2:cfg["report_k2_stage"],4:cfg["report_k4_stage"]}[a.k]
    if out.exists() or out.is_symlink():raise RuntimeError("STOP occupied report output")
    pre=load(Path(cfg["output_root"])/cfg["preflight_stage"]/"A15_PREFLIGHT_PASS.json")
    if pre["status"]!="A15_CPU_PREFLIGHT_PASS" or pre["source"]["a15_commit"]!=a.expected_head:
        raise RuntimeError("STOP mismatched stage0")
    frozen=load(Path(cfg["frozen_input_manifest"]))
    rowsbyid={str(v["image_id"]):v for v in frozen["rows"]}
    reference=load(Path(cfg["a1_oracle_summary"]))["per_image"]
    by_image=[]
    for ident in cfg["ids"]:
        roots=[]
        for rep in range(a.k):
            parent=Path(cfg["output_root"])/cfg["grid_stage"]/ident/f"rep{rep}"
            rec_path=parent/"A15_SMOKE_COMPLETE.json"
            if not rec_path.is_file():raise RuntimeError(f"STOP missing K={a.k} root {ident}/{rep}")
            r=load(rec_path)
            if r.get("status")!="A15_FAST_CANDIDATE_PASS" or \
               r["source_head"]!=a.expected_head or r["fast_head"]!=cfg["fast_head"] or \
               r["image_id"]!=ident or r["rep"]!=rep or \
               r["measurement_tensor_sha256"]!=rowsbyid[ident]["measurement_tensor_sha256"] or \
               r["native_checkpoint_load_validation"]["checkpoint_sha256"]!=cfg["checkpoint_sha256"]:
                raise RuntimeError(f"STOP grid receipt content mismatch {ident}/{rep}")
            file_ok(parent/"reconstruction.png",r["reconstruction_png_sha256"],"FAST PNG")
            file_ok(parent/"reconstruction.pt",r["reconstruction_float_sha256"],"FAST float")
            if not math.isfinite(r["metric"]["raw_rgb8_psnr_db"]):
                raise RuntimeError("STOP nonfinite FAST PSNR")
            roots.append(r)
        best=max(roots,key=lambda z:(z["metric"]["raw_rgb8_psnr_db"],-z["rep"]))
        original=reference[ident]["oracle_by_policy"]
        image=dict(image_id=ident,k=a.k,fast_best_raw_psnr_db=best["metric"]["raw_rgb8_psnr_db"],
            fast_best_rot180_psnr_db=best["metric"]["rot180_rgb8_psnr_db"],
            best_rep=best["rep"],good25=int(best["metric"]["raw_rgb8_psnr_db"]>=25),
            good30=int(best["metric"]["raw_rgb8_psnr_db"]>=30),
            bad20=int(best["metric"]["raw_rgb8_psnr_db"]<20),
            fast_all_roots_wall_s=sum(x["sample_only_wall_seconds"] for x in roots),
            fast_all_roots_event_s=sum(x["sample_cuda_event_seconds"] for x in roots),
            fast_all_roots_native_s=sum(x["native_recorded_step_sum_seconds"] for x in roots),
            **{f"{name}_K{a.k}_raw_oracle_db":original[f"{name}_K{a.k}"]["best_psnr_raw_db"]
                 for name in ("DAPS","SITCOM","NP_LF","NP_S2")})
        by_image.append(image)
    mean=lambda field:statistics.mean(v[field] for v in by_image)
    summary=dict(
        status=f"A15_FAST_K{a.k}_PAIRED_EXPOSED_REPORT_PASS",k=a.k,n_images=len(by_image),
        source_head=a.expected_head,fast_head=cfg["fast_head"],exposed_not_heldout=True,
        fast_mean_raw_oracle_db=mean("fast_best_raw_psnr_db"),
        good25=sum(v["good25"] for v in by_image),good30=sum(v["good30"] for v in by_image),
        bad20=sum(v["bad20"] for v in by_image),
        mean_fast_full_k_sample_wall_s=mean("fast_all_roots_wall_s"),
        mean_fast_full_k_cuda_event_s=mean("fast_all_roots_event_s"),
        mean_fast_full_k_native_step_s=mean("fast_all_roots_native_s"),
        historical_A1_per_method_means={
            n:mean(f"{n}_K{a.k}_raw_oracle_db") for n in ("DAPS","SITCOM","NP_LF","NP_S2")},
        primary_metric="RAW_RGB8_PNG_GT_ORACLE_OFFLINE_FULL_K_CHARGED",
        note="Per-image paired exposed engineering data; FAST times sampling-only, A1 historical GPU candidate subprocess time; compare costs only with differing scope labeled")
    out.mkdir(parents=True,exist_ok=False)
    with (out/"A15_PAIRED_PER_IMAGE.csv").open("x",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(by_image[0]))
        writer.writeheader();writer.writerows(by_image)
    write_json_x(out/"A15_PAIRED_SUMMARY.json",summary)
    print(json.dumps(summary,sort_keys=True))
    return 0

if __name__=="__main__":
    try:sys.exit(main())
    except Exception as e:
        print(f"A15_REPORT_STOP: {type(e).__name__}: {e}",file=sys.stderr)
        sys.exit(9)
