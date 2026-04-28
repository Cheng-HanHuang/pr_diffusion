#!/usr/bin/env python3
from __future__ import annotations

import csv
import glob
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path("/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411")
SCHED_ROOT = ROOT / "np_ffhq_proj_schedule_screen"
STEP_ROOT = ROOT / "np_ffhq_stepA_full_top2"

OUT_SUMMARY = ROOT / "np_ffhq_proj_schedule_screen_bestof2_summary.csv"
OUT_WITH_BASELINES = ROOT / "np_ffhq_proj_schedule_screen_plus_constant_baselines_bestof2.csv"
OUT_IMAGE = ROOT / "np_ffhq_proj_schedule_screen_image_bestof2.csv"
OUT_RUNS = ROOT / "np_ffhq_proj_schedule_screen_merged_run_level.csv"

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"[WARN] no rows for {path}")
        return
    fieldnames, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                fieldnames.append(k); seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def fget(row, key, default=math.nan):
    try:
        v = row.get(key, "")
        if v == "" or v is None:
            return default
        return float(v)
    except Exception:
        return default

def safe_mean(vals):
    vals = [v for v in vals if math.isfinite(v)]
    return mean(vals) if vals else math.nan

def safe_median(vals):
    vals = [v for v in vals if math.isfinite(v)]
    return median(vals) if vals else math.nan

def safe_min(vals):
    vals = [v for v in vals if math.isfinite(v)]
    return min(vals) if vals else math.nan

def safe_max(vals):
    vals = [v for v in vals if math.isfinite(v)]
    return max(vals) if vals else math.nan

def safe_stdev(vals):
    vals = [v for v in vals if math.isfinite(v)]
    return stdev(vals) if len(vals) > 1 else 0.0

def tag_after_root(path, root_name):
    parts = Path(path).parts
    for i, p in enumerate(parts):
        if p == root_name and i + 1 < len(parts):
            return parts[i + 1]
    return ""

def load_schedule_rows():
    files = sorted(glob.glob(str(SCHED_ROOT / "**" / "run_level.csv"), recursive=True))
    print(f"Found schedule run_level.csv files: {len(files)}")
    rows = []
    for path in files:
        tag = tag_after_root(path, "np_ffhq_proj_schedule_screen")
        for r in read_csv(path):
            r["config_tag"] = tag
            r["experiment_family"] = "schedule"
            r["source_file"] = path
            rows.append(r)
    return rows

def load_stepA_baseline_rows(image_set, seed_set):
    files = sorted(glob.glob(str(STEP_ROOT / "**" / "run_level.csv"), recursive=True))
    print(f"Found Step A baseline run_level.csv files: {len(files)}")
    rows = []
    for path in files:
        tag = tag_after_root(path, "np_ffhq_stepA_full_top2")
        for r in read_csv(path):
            if r.get("alignment_mode") != "rot180":
                continue
            if r.get("image_basename") not in image_set:
                continue
            if r.get("seed") not in seed_set:
                continue
            r["config_tag"] = "constant_baseline_" + tag
            r["experiment_family"] = "constant_baseline"
            r["source_file"] = path
            rows.append(r)
    return rows

def summarize_bestof(rows):
    by_cfg_img = defaultdict(list)
    for r in rows:
        if r.get("alignment_mode") != "rot180":
            continue
        by_cfg_img[(r.get("config_tag", ""), r.get("image_basename", ""))].append(r)

    image_best = []
    for (tag, img), gr in sorted(by_cfg_img.items()):
        best = max(gr, key=lambda r: fget(r, "psnr"))
        psnrs = [fget(r, "psnr") for r in gr]
        out = {
            "config_tag": tag,
            "experiment_family": best.get("experiment_family", ""),
            "image_basename": img,
            "n_seeds": len(set(r.get("seed", "") for r in gr)),
            "best_seed_by_psnr": best.get("seed", ""),
            "bestof_psnr": fget(best, "psnr"),
            "bestof_ssim_at_best_psnr": fget(best, "ssim"),
            "runtime_s_at_best_psnr": fget(best, "runtime_s"),
            "psnr_mean_over_seeds": safe_mean(psnrs),
            "psnr_min_over_seeds": safe_min(psnrs),
            "score_radius": best.get("score_radius", ""),
            "proj_radius": best.get("proj_radius", ""),
            "proj_radius_schedule": best.get("proj_radius_schedule", ""),
            "proj_start": best.get("proj_start", ""),
            "num_candidates_soft": best.get("num_candidates_soft", ""),
            "num_candidates_hard": best.get("num_candidates_hard", ""),
        }
        image_best.append(out)

    by_cfg = defaultdict(list)
    for r in image_best:
        by_cfg[r["config_tag"]].append(r)

    summary = []
    for tag, gr in sorted(by_cfg.items()):
        psnr = [fget(r, "bestof_psnr") for r in gr]
        ssim = [fget(r, "bestof_ssim_at_best_psnr") for r in gr]
        runtime = [fget(r, "runtime_s_at_best_psnr") for r in gr]
        worst = min(gr, key=lambda r: fget(r, "bestof_psnr"))
        rep = gr[0]
        summary.append({
            "config_tag": tag,
            "experiment_family": rep.get("experiment_family", ""),
            "n_images": len(gr),
            "bestof_psnr_mean": safe_mean(psnr),
            "bestof_psnr_median": safe_median(psnr),
            "bestof_psnr_min": safe_min(psnr),
            "bestof_psnr_max": safe_max(psnr),
            "bestof_psnr_std": safe_stdev(psnr),
            "bestof_ssim_mean": safe_mean(ssim),
            "runtime_s_mean_at_best_seed": safe_mean(runtime),
            "worst_image": worst.get("image_basename", ""),
            "worst_image_bestof_psnr": fget(worst, "bestof_psnr"),
            "score_radius": rep.get("score_radius", ""),
            "proj_radius": rep.get("proj_radius", ""),
            "proj_radius_schedule": rep.get("proj_radius_schedule", ""),
            "proj_start": rep.get("proj_start", ""),
            "num_candidates_soft": rep.get("num_candidates_soft", ""),
            "num_candidates_hard": rep.get("num_candidates_hard", ""),
        })

    summary.sort(
        key=lambda r: (
            fget(r, "bestof_psnr_mean"),
            fget(r, "bestof_psnr_median"),
            fget(r, "bestof_psnr_min"),
            fget(r, "bestof_ssim_mean"),
        ),
        reverse=True,
    )
    return summary, image_best

def main():
    sched_rows = load_schedule_rows()
    if not sched_rows:
        raise RuntimeError(f"No schedule rows found under {SCHED_ROOT}")

    image_set = set(r.get("image_basename", "") for r in sched_rows)
    seed_set = set(r.get("seed", "") for r in sched_rows)
    baseline_rows = load_stepA_baseline_rows(image_set=image_set, seed_set=seed_set)

    print(f"Schedule rows: {len(sched_rows)}")
    print(f"Schedule images: {len(image_set)}")
    print(f"Schedule seeds: {sorted(seed_set)}")
    print(f"Baseline rows restricted to same image/seed set: {len(baseline_rows)}")

    sched_summary, sched_image = summarize_bestof(sched_rows)
    combined_summary, combined_image = summarize_bestof(sched_rows + baseline_rows)

    write_csv(OUT_SUMMARY, sched_summary)
    write_csv(OUT_WITH_BASELINES, combined_summary)
    write_csv(OUT_IMAGE, combined_image)
    write_csv(OUT_RUNS, sched_rows)

    cols = [
        "config_tag", "experiment_family", "n_images",
        "bestof_psnr_mean", "bestof_psnr_median", "bestof_psnr_min",
        "bestof_ssim_mean", "proj_radius_schedule",
        "worst_image", "worst_image_bestof_psnr",
    ]

    print("\n=== Schedule screen only: best-of-2 summary ===")
    print(",".join(cols))
    for r in sched_summary:
        print(",".join(str(r.get(c, "")) for c in cols))

    print("\n=== Schedule screen + constant baselines on same images/seeds ===")
    print(",".join(cols))
    for r in combined_summary:
        print(",".join(str(r.get(c, "")) for c in cols))

    print("\nSaved:")
    print(" ", OUT_SUMMARY)
    print(" ", OUT_WITH_BASELINES)
    print(" ", OUT_IMAGE)
    print(" ", OUT_RUNS)

if __name__ == "__main__":
    main()
