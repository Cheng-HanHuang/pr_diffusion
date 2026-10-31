#!/usr/bin/env python3
"""A16 §2: independent, CPU-only XP-I 180x4 per-particle metric return.

Every candidate is rejoined to frozen CP3 evaluation, native float piq
metric receipts and prior face CLIP decisions. GT-derived scores are
EVALUATION ONLY; no native .pt, paths or source GT are published.
No re-fitting, solver calls, GT-free claims or matched-NFE claims.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

from scripts.track_r.xp_a12_g1a_prepare_cpu import BASE, gate
from scripts.track_r.xp_a15_cp2_d4_cpu import sha, write
from scripts.track_r.xp_a16_cp3_joint_readout_cpu import read_stage

COLUMNS=("image_id","particle","label","PSNR","SSIM","VGG_LPIPS",
         "rho","face_choice","face_margin")
STATUS="A16_XPI_PER_PARTICLE_METRIC_TABLE_PASS"


def rows_from_stage(stage):
    gate(stage["stage"]=="G3"
         and stage["score_rec"]["n_images"]==180
         and stage["score_rec"]["n_candidates"]==720,
         "A18_XPI_FROZEN_G3")
    s=stage["candidates"]
    c=stage["clips"]
    m=stage["metrics"]
    gate(len(s)==len(c)==len(m)==720 and
         set(s)==set(c)==set(m),"A18_XPI_JOIN_KEYS")
    rows=[]
    for (cohort,iid),group in sorted(stage["image_groups"].items()):
        gate(cohort in ("S1_b22","S2_b24_dev80") and
             iid.startswith(cohort+"_") and len(group)==4,
             "A18_XPI_COHORT_KEY")
        measurement_set=set()
        for k in range(4):
            key=(iid,k)
            sc=s[key];cl=c[key];mt=m[key]
            gate(sc["cohort"]==cl["cohort"]==mt["cohort"]
                 and sc["image_id"]==cl["image_id"]==mt["image_id"]==iid
                 and int(sc["run"])==int(cl["run"])==int(mt["run"])==k
                 and sc["uid"]==mt["uid"]
                 and sc["label"]==cl["label"]
                 and cl["face_choice"] in ("000","111"),
                 "A18_XPI_PER_PARTICLE_PROVENANCE")
            measurement_set.add(sc["measurement_sha256"])
            value=dict(image_id=iid,particle=k,label=sc["label"],
                  PSNR=float(sc["psnr_raw"]),
                  SSIM=float(mt["SSIM_raw"]),
                  VGG_LPIPS=float(mt["LPIPS_VGG_raw"]),
                  rho=float(sc["rho_signed"]),
                  face_choice=cl["face_choice"],
                  face_margin=float(cl["face_margin"]))
            gate(all(math.isfinite(value[x]) for x in
                     ("PSNR","SSIM","VGG_LPIPS","rho","face_margin"))
                 and value["rho"]>=0,"A18_XPI_METRIC_FINITE")
            rows.append(value)
        gate(len(measurement_set)==1,"A18_XPI_FROZEN_SIGNED_Y")
    gate(len(rows)==720 and len({(r["image_id"],r["particle"])
                                 for r in rows})==720,
         "A18_XPI_GRID_180X4")
    return rows


def main(a):
    gate(os.environ.get("CUDA_VISIBLE_DEVICES")=="","A18_XPI_CPU_ONLY")
    wt=Path(__file__).resolve().parents[2]
    git=lambda *z:subprocess.check_output(
        ["git","-C",str(wt),*z],text=True,timeout=30).strip()
    gate(git("rev-parse","HEAD")==a.expected_head and
         not git("status","--porcelain"),
         "A18_XPI_PINNED_CLEAN_WORKTREE")
    out=Path(a.out)
    gate(out.parent==BASE and out.name.startswith("xp_a18_xpi_particles_")
         and not out.exists() and not out.is_symlink(),
         "A18_XPI_NEW_OUTPUT_PATH")
    stage=read_stage("G3")
    rows=rows_from_stage(stage)
    if a.preflight_only:
        print("A18_XPI_PARTICLES_PREFLIGHT_PASS",len(rows),
              "IMAGES",len({r["image_id"] for r in rows}),flush=True)
        return
    out.mkdir(mode=0o700)
    p=out/"A16_XPI_PER_PARTICLE_METRICS.csv"
    with p.open("x",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=COLUMNS,lineterminator="\\n")
        writer.writeheader()
        writer.writerows(rows)
    with p.open("r",newline="",encoding="utf-8") as f:
        reread=list(csv.DictReader(f))
    gate(len(reread)==720 and set(reread[0])==set(COLUMNS),
         "A18_XPI_CSV_ROUNDTRIP")
    rec=dict(status=STATUS,expected_head=a.expected_head,
             n_images=180,n_particles=720,per_image_particles=4,
             columns=list(COLUMNS),
             metric_source="CP3 frozen RGB8 raw PSNR; independent original float piq SSIM/LPIPS",
             score_columns_are_gt_evaluation_only=True,
             face_columns_are_gt_free_decisions=True,
             source_candidate_metrics_receipt_sha256=sha(
                 Path(stage["receipt"]["input_stage_scoring_receipt_sha256"])
                 ) if False else stage["receipt"]["input_stage_scoring_receipt_sha256"],
             sha256=sha(p),no_native_tensor_or_path=True,
             unchanged_frozen_data=True,no_gpu_calls=True,
             no_matched_compute_claim=True)
    write(out/"A16_XPI_PER_PARTICLE_METRICS_RECEIPT.json",rec)
    print("A18_XPI_PARTICLES_PASS",p,"SHA256",rec["sha256"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--expected-head",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--preflight-only",action="store_true")
    try:main(p.parse_args())
    except Exception as exc:
        traceback.print_exc()
        print("ACTION NEEDED: A18_XPI_PER_PARTICLE",exc,flush=True)
        sys.exit(8)
