#!/usr/bin/env python3
"""Publish ONLY the approved D1 P1 (FAST ImageNet, runs 0..3) A8-style strings.

This script is CPU-only and does not compute or access new quality scores.
It verifies the already-completed D1 phase-1 offline score receipt against
the immutable native D1_P1 manifest, with no D1_P2/A16/A64 reference read.
Only the specified per-image CSV and a receipt go to theory_inputs/.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,shutil,subprocess,sys,traceback
from pathlib import Path
from collections import Counter
from scripts.track_r.xp_d1p1_score_cpu import BASE,WT,GOOD
from scripts.track_r.xp_night_publish_cpu import GEN_HEAD,EXPECTED_MANIFEST_HASHES
from scripts.track_r.xp_t_fast_prepare_cpu import sha
DEST=WT/"docs/track_r/executors/xp/theory_inputs"
ALLOW={ "image_id","source","family","n_runs","n_GOOD","n_ORBIT_global",
        "n_ORBIT_partial","n_CONSISTENT","n_INCONSISTENT","run_labels",
        "e0_rank_labels","e0_rank_order","rho_signed_by_run",
        "tau_signed_good_q95","orbit_best_member" }
CODE="GOPCI"

def validated(rows,receipt):
    if (receipt.get("status")!="XP_D1_P1_K4_EVAL_PASS"
        or receipt.get("native_generation_head")!=GEN_HEAD
        or receipt.get("stage_manifest_sha256")!=EXPECTED_MANIFEST_HASHES["D1_P1"]
        or receipt.get("n_images")!=100 or receipt.get("n_runs")!=400
        or receipt.get("run_indices")!=[0,1,2,3]
        or receipt.get("primary_good_psnr_db")!=GOOD
        or receipt.get("family")!="FAST" or receipt.get("dataset")!="imagenet"
        or receipt.get("sealed_D1_P2_scored") is not False
        or receipt.get("sealed_A64_scored") is not False
        or receipt.get("sealed_XP_A16_scored") is not False
        or receipt.get("use_for_v2_prediction_freeze") is not True):
        raise RuntimeError("STOP_D1P1_RELEASE_RECEIPT")
    if len(rows)!=100 or any(set(r)!=ALLOW for r in rows):
        raise RuntimeError("STOP_D1P1_RELEASE_SCHEMA")
    ids=set()
    labels=Counter()
    tau=str(rows[0]["tau_signed_good_q95"])
    if not math.isfinite(float(tau)) or float(tau)<=0:
        raise RuntimeError("STOP_D1P1_RELEASE_TAU")
    for row in rows:
        iid=row["image_id"]
        if iid in ids:raise RuntimeError("STOP_D1P1_RELEASE_DUPLICATE_ID")
        ids.add(iid)
        if (row["source"]!="XP_D1_P1" or row["family"]!="FAST"
            or int(row["n_runs"])!=4 or row["tau_signed_good_q95"]!=tau):
            raise RuntimeError("STOP_D1P1_RELEASE_IDENTITY")
        text=row["run_labels"];ranks=row["e0_rank_labels"]
        order=[int(x) for x in row["e0_rank_order"].split(",")]
        res=row["rho_signed_by_run"].split(";")
        if (len(text)!=4 or len(ranks)!=4 or sorted(order)!=[0,1,2,3]
            or any(k not in CODE for k in text)
            or ranks!="".join(text[j] for j in order)
            or len(res)!=4 or any(
                not math.isfinite(float(v)) or float(v)<0 for v in res)
            or len(row["orbit_best_member"])!=4
            or sum(int(row[k]) for k in
                   ("n_GOOD","n_ORBIT_global","n_ORBIT_partial",
                    "n_CONSISTENT","n_INCONSISTENT"))!=4):
            raise RuntimeError("STOP_D1P1_RELEASE_PER_IMAGE")
        labels.update(text)
    # Original evaluator labels use one-letter codes (G/O/P/C/I).
    allowed={k:labels[k] for k in CODE if labels[k]}
    if receipt.get("label_counts")!=allowed:
        raise RuntimeError("STOP_D1P1_RELEASE_GLOBAL_COUNTS")
    return True

def main(a):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("STOP_D1P1_RELEASE_GPU_VISIBLE")
    head=subprocess.check_output(["git","-C",str(WT),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(WT),"status","--porcelain"],text=True).strip()
    if head!=a.expected_head or dirty:
        raise RuntimeError("STOP_D1P1_RELEASE_GIT")
    src=Path(a.eval_dir)
    if (src.parent!=BASE or not src.name.startswith("xp_d1p1_eval_")
        or not src.is_dir() or src.is_symlink()):
        raise RuntimeError("STOP_D1P1_RELEASE_SOURCE")
    csv_path=src/"XP_D1_P1_A8_STYLE_PER_IMAGE.csv"
    receipt_path=src/"XP_D1_P1_K4_SUMMARY.json"
    if csv_path.is_symlink() or receipt_path.is_symlink():
        raise RuntimeError("STOP_D1P1_RELEASE_SYMLINK")
    with csv_path.open(newline="") as f:rows=list(csv.DictReader(f))
    receipt=json.loads(receipt_path.read_text())
    if sha(csv_path)!=receipt.get("rows_sha256"):
        raise RuntimeError("STOP_D1P1_RELEASE_ROWS_HASH")
    validated(rows,receipt)
    # No PSNR-containing scored-candidate tables enter theory_inputs.
    DEST.mkdir(parents=True,exist_ok=True)
    out=DEST/"XP_D1_P1_K4_A8_STYLE_PER_IMAGE.csv"
    meta=DEST/"XP_D1_P1_K4_THEORY_INPUT_RECEIPT.json"
    if out.exists() or meta.exists():
        raise RuntimeError("STOP_D1P1_RELEASE_ALREADY_PRESENT")
    shutil.copyfile(csv_path,out)
    manifest={
      "status":"XP_D1_P1_K4_THEORY_INPUT_PASS",
      "native_generation_head":GEN_HEAD,
      "evaluation_head":receipt["evaluation_head"],
      "publication_head":head,
      "n_images":100,"n_runs":400,"run_indices":[0,1,2,3],
      "family":"FAST","dataset":"imagenet",
      "primary_good_psnr_db":GOOD,
      "tau_in_signed_good_q95":receipt["rho_signed_good_q95_tau"],
      "tau_estimation":receipt["tau_estimation"],
      "per_image_csv_sha256":sha(out),
      "source_evaluation_receipt_sha256":sha(receipt_path),
      "D1_phase2_evaluated":False,
      "FFHQ_K_gt4_evaluated":False,
      "contains_per_run_quality_values":False,
      "CLIP_G1_decisions":"PENDING_XP_C_CLIP_WORKER",
      "theory_v2_clock_starts_at_commit":True,
    }
    with meta.open("x") as f:json.dump(manifest,f,sort_keys=True,indent=2);f.write("\n")
    print("XP_D1P1_THEORY_INPUT_STAGE_PASS images=100 roots=400")
    print("D1P1_THEORY_INPUT_SHA="+sha(out))
    print("CLIP_G1_PENDING=true")
    print("D1_PHASE2_EMBARGO=ACTIVE")
    print("OK")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--eval-dir",required=True)
    p.add_argument("--expected-head",required=True)
    try:main(p.parse_args())
    except Exception as e:
        traceback.print_exc()
        print("ACTION NEEDED: XP_D1P1_THEORY_RELEASE "+str(e),flush=True)
        sys.exit(8)
