#!/usr/bin/env python3
"""A1.5 FAST K1/K2/K4 selection counterfactual using saved CPU M0 CSV only.

GT PSNR is read solely from previously computed metadata; no ground truth,
reconstructions, FFTs, or GPUs are read or recomputed. This is an ORACLE VS
MEASUREMENT-RESIDUAL selector comparison on the *same* existing 32 roots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

A15_SOURCE_HEAD = "d51170f77ee672990c22f1703c5997e949dda9d6"
M0_SOURCE_HEAD = "a9b5dedb0c5586d55991a7f9cb82f13aea847fff"
M0_SHA256 = "b7c2e478982c571bf45cacd3eddc505e6752501047aa1467ead75064115bcb74"
AUDIT_BRANCH = "codex/prdiff-track-a-a15-residual-selector-m1-20261021"
IDS = ("69073", "68539", "67948", "60140", "67293", "67174", "61101", "66631")
ROOT = Path("/egr/research-pac/huang248")
OUTPUT_ROOT = ROOT / "outputs/pr_diffusion/track_a_np_paper/a15_fastdips_exposed_20261020"
STAGE = "a15_8_fast_residual_selector_cpu_v1"


def git(folder: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(folder), *args], text=True).strip()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_read(p: Path) -> dict:
    with p.open(encoding="utf8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("STOP CUDA_VISIBLE_DEVICES must be empty (CPU-only)")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("STOP: bytecode writes disabled")
    if Path.cwd() != ROOT / "pr_diffusion_track_a_a15_residual_selector":
        raise RuntimeError("STOP: use dedicated selector audit checkout")
    if git(Path.cwd(), "branch", "--show-current") != AUDIT_BRANCH:
        raise RuntimeError("STOP: wrong analysis branch")
    if git(Path.cwd(), "rev-parse", "HEAD") != args.expected_head:
        raise RuntimeError("STOP: pinned analysis HEAD mismatch")
    if git(Path.cwd(), "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("STOP: dirty tracked analysis source")
    if git(ROOT/"pr_diffusion_track_a_a15_fastdips", "rev-parse", "HEAD") != A15_SOURCE_HEAD:
        raise RuntimeError("STOP: prior A15 execution commit changed")
    if git(ROOT/"pr_diffusion_track_a_a15_mechanism_audit", "rev-parse", "HEAD") != M0_SOURCE_HEAD:
        raise RuntimeError("STOP: prior M0 audit source commit changed")

    m0path = OUTPUT_ROOT/"a15_6_mechanism_cpu_audit_v1"/"A15_MECHANISM_AUDIT_COMPLETE.json"
    if sha256(m0path) != M0_SHA256:
        raise RuntimeError("STOP: M0 receipt SHA256 mismatch")
    m0 = json_read(m0path)
    if m0.get("status") != "A15_CPU_MECHANISM_AUDIT_PASS" or m0.get("candidate_count") != 32:
        raise RuntimeError("STOP: M0 incomplete")
    csvpath = m0path.with_name("A15_PER_CANDIDATE_MECHANISM.csv")
    with csvpath.open(newline="", encoding="utf8") as f:
        roots = list(csv.DictReader(f))
    if len(roots) != 32:
        raise RuntimeError("STOP: expected 32 archived candidate rows")

    by_id = {i: [] for i in IDS}
    seen = set()
    for rec in roots:
        ident, rep = rec["image_id"], int(rec["rep"])
        if ident not in by_id or rep not in (0,1,2,3) or (ident,rep) in seen:
            raise RuntimeError("STOP: unknown / duplicate candidate")
        seen.add((ident,rep))
        converted = dict(
            image_id=ident, rep=rep,
            psnr=float(rec["raw_rgb8_psnr_db_from_pinned_receipt"]),
            residual=float(rec["actual_output_measurement_l2"]),
            time=float(rec["candidate_sample_only_s"]),
            post_import_wall=float(rec["candidate_program_wall_after_imports_s"]),
        )
        if not all(math.isfinite(converted[k]) for k in ("psnr", "residual", "time", "post_import_wall")):
            raise RuntimeError("STOP: nonfinite candidate diagnostic")
        by_id[ident].append(converted)

    comparisons = []
    results = {}
    m0_by_id = {r["image_id"]:r for r in m0["per_image"]}
    for k in (1,2,4):
        rows = []
        for ident in IDS:
            roots_k = sorted(by_id[ident], key=lambda x:x["rep"])[:k]
            if len(roots_k) != k or [r["rep"] for r in roots_k] != list(range(k)):
                raise RuntimeError("STOP: noncontiguous K roots")
            oracle = max(roots_k, key=lambda x:(x["psnr"], -x["rep"]))
            residual = min(roots_k, key=lambda x:(x["residual"], x["rep"]))
            if k==4 and abs(oracle["psnr"] - float(m0_by_id[ident]["best_of_4_raw_psnr_db"]))>1e-5:
                raise RuntimeError("STOP: saved M0 best4 PSNR differs from root CSV")
            rows.append(dict(
                image_id=ident, k=k,
                oracle_rep=oracle["rep"], residual_rep=residual["rep"],
                oracle_psnr_db=oracle["psnr"],
                residual_chosen_psnr_db=residual["psnr"],
                residual_selector_regret_db=max(0.0, oracle["psnr"]-residual["psnr"]),
                oracle_root_residual_l2=oracle["residual"],
                residual_chosen_residual_l2=residual["residual"],
                picked_oracle=int(oracle["rep"]==residual["rep"]),
                oracle_good25=int(oracle["psnr"]>=25),
                residual_good25=int(residual["psnr"]>=25),
                full_k_sample_seconds=sum(r["time"] for r in roots_k),
                full_k_program_after_import_seconds=sum(r["post_import_wall"] for r in roots_k),
            ))
        mean = lambda field: statistics.mean(r[field] for r in rows)
        results[f"K{k}"] = dict(
            mean_oracle_raw_psnr_db=mean("oracle_psnr_db"),
            mean_residual_selected_raw_psnr_db=mean("residual_chosen_psnr_db"),
            mean_regret_db=mean("residual_selector_regret_db"),
            matched_oracle_count=sum(r["picked_oracle"] for r in rows),
            oracle_good25=sum(r["oracle_good25"] for r in rows),
            residual_good25=sum(r["residual_good25"] for r in rows),
            mean_full_k_sampling_seconds=mean("full_k_sample_seconds"),
            mean_full_k_after_import_seconds=mean("full_k_program_after_import_seconds"),
        )
        comparisons.extend(rows)

    # Cross-check prior frozen FAST K1/K2/K4 report means, not post-hoc retune.
    report_by_k = {1:"a15_3_report_k1_v1",2:"a15_5_report_k2_v1",4:"a15_4_report_k4_v1"}
    for k, stage in report_by_k.items():
        original = json_read(OUTPUT_ROOT/stage/"A15_PAIRED_SUMMARY.json")
        if original.get("status") != f"A15_FAST_K{k}_PAIRED_EXPOSED_REPORT_PASS":
            raise RuntimeError("STOP: FAST K report receipt mismatch")
        err = abs(results[f"K{k}"]["mean_oracle_raw_psnr_db"] - float(original["fast_mean_raw_oracle_db"]))
        if err > 2e-5:
            raise RuntimeError(f"STOP K{k}: original oracle mean PSNR differs by {err}")

    out = OUTPUT_ROOT/STAGE
    if out.exists() or out.is_symlink():
        raise RuntimeError("STOP: nonoverwriting output already exists")
    out.mkdir(parents=True, exist_ok=False)
    with (out/"A15_RESIDUAL_SELECTOR_PER_IMAGE.csv").open("x",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    receipt = dict(
        status="A15_FAST_RESIDUAL_SELECTOR_CPU_PASS",
        audit_head=args.expected_head,
        a15_execution_head=A15_SOURCE_HEAD,
        m0_execution_head=M0_SOURCE_HEAD,
        m0_receipt_sha256=M0_SHA256,
        n_images=8, n_archived_candidates=32,
        n_new_gpu_candidates=0, n_ground_truth_tensors_read=0,
        primary="raw RGB8 per-image oracle vs minimum-forward-measurement-residual selection",
        results=results,
        warning="An oracle uses unavailable GT. Residual-selected policy is a GT-free counterfactual but uses post-computed raw signed measurement residual, not necessarily original FAST native selector.",
    )
    receipt_path=out/"A15_RESIDUAL_SELECTOR_COMPLETE.json"
    with receipt_path.open("x",encoding="utf8") as f:
        json.dump(receipt,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(status=receipt["status"],results=results,receipt_sha256=sha256(receipt_path)),sort_keys=True))
    return 0


if __name__=="__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"A15_RESIDUAL_SELECTOR_STOP: {type(e).__name__}: {e}",file=sys.stderr,flush=True)
        sys.exit(13)
