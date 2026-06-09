#!/usr/bin/env python3
"""Summarize phase_retrieval_20260608 CSV outputs.

This script recursively scans the dated output folder for run_level.csv and
selected_image_level.csv files produced by the 20260608 runners and writes
aggregate CSVs under <root>/summary.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, Iterable, List, Tuple


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def as_float(x, default=math.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def fmean(xs: Iterable[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    return mean(vals) if vals else math.nan


def fmedian(xs: Iterable[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    return median(vals) if vals else math.nan


def fstd(xs: Iterable[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    return stdev(vals) if len(vals) > 1 else 0.0


def summarize_selected(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        key = (
            r.get("selection_method", ""),
            r.get("refine_tag", "none"),
            r.get("alignment_mode", ""),
            r.get("oversample", ""),
            r.get("measurement_noise_std", ""),
        )
        groups[key].append(r)

    out = []
    for key, rs in sorted(groups.items()):
        method, refine, align, oversample, noise = key
        psnr = [as_float(r.get("psnr")) for r in rs]
        ssim = [as_float(r.get("ssim")) for r in rs]
        runt = [as_float(r.get("runtime_s")) for r in rs]
        worst = min(rs, key=lambda r: as_float(r.get("psnr")))
        out.append({
            "selection_method": method,
            "refine_tag": refine,
            "alignment_mode": align,
            "oversample": oversample,
            "measurement_noise_std": noise,
            "n_images": len(rs),
            "psnr_mean": fmean(psnr),
            "psnr_median": fmedian(psnr),
            "psnr_min": min(x for x in psnr if math.isfinite(x)) if any(math.isfinite(x) for x in psnr) else math.nan,
            "psnr_max": max(x for x in psnr if math.isfinite(x)) if any(math.isfinite(x) for x in psnr) else math.nan,
            "psnr_std": fstd(psnr),
            "ssim_mean": fmean(ssim),
            "runtime_s_mean": fmean(runt),
            "below20": sum(1 for x in psnr if x < 20),
            "below25": sum(1 for x in psnr if x < 25),
            "below28": sum(1 for x in psnr if x < 28),
            "worst_image": worst.get("image_basename", ""),
            "worst_psnr": as_float(worst.get("psnr")),
        })
    return out


def summarize_oracle_px(run_rows: List[Dict[str, str]], threshold: float) -> List[Dict[str, object]]:
    # Candidate-generation table: for each image/refine/noise/alignment, how many
    # candidate runs pass threshold?  This estimates p_x under the generated pool.
    groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in run_rows:
        key = (
            r.get("image_basename", ""),
            r.get("refine_tag", "none"),
            r.get("alignment_mode", ""),
            r.get("oversample", ""),
            r.get("measurement_noise_std", ""),
        )
        groups[key].append(r)

    out = []
    for key, rs in sorted(groups.items()):
        image, refine, align, oversample, noise = key
        psnr = [as_float(r.get("psnr")) for r in rs]
        n = len([x for x in psnr if math.isfinite(x)])
        succ = sum(1 for x in psnr if x >= threshold)
        best = max((x for x in psnr if math.isfinite(x)), default=math.nan)
        out.append({
            "image_basename": image,
            "refine_tag": refine,
            "alignment_mode": align,
            "oversample": oversample,
            "measurement_noise_std": noise,
            "threshold": threshold,
            "n_candidate_rows": n,
            "n_success_rows": succ,
            "candidate_success_rate": succ / n if n else math.nan,
            "oracle_best_psnr": best,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260608")
    ap.add_argument("--thresholds", default="20,25,28,30")
    args = ap.parse_args()

    root = Path(args.root)
    selected_files = sorted(root.rglob("selected_image_level.csv"))
    run_files = sorted(root.rglob("run_level.csv"))
    selected_rows: List[Dict[str, str]] = []
    run_rows: List[Dict[str, str]] = []

    for path in selected_files:
        for r in read_csv(path):
            r["source_csv"] = str(path)
            selected_rows.append(r)
    for path in run_files:
        for r in read_csv(path):
            r["source_csv"] = str(path)
            run_rows.append(r)

    summary_dir = root / "summary"
    write_csv(summary_dir / "selected_summary_all.csv", summarize_selected(selected_rows))
    for th in [float(x) for x in args.thresholds.split(",") if x.strip()]:
        write_csv(summary_dir / f"candidate_success_px_threshold_{th:g}.csv", summarize_oracle_px(run_rows, th))

    manifest = [{
        "root": str(root),
        "n_selected_files": len(selected_files),
        "n_run_files": len(run_files),
        "n_selected_rows": len(selected_rows),
        "n_run_rows": len(run_rows),
    }]
    write_csv(summary_dir / "manifest.csv", manifest)
    print(f"[summary] wrote {summary_dir}")
    print(manifest[0])


if __name__ == "__main__":
    main()
