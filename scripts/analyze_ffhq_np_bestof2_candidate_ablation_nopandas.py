#!/usr/bin/env python3
from __future__ import annotations

import csv
import glob
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path("/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411")
RUN_ROOT = ROOT / "np_ffhq_bestof2_candidate_ablation"

OUT_SUMMARY = ROOT / "np_ffhq_bestof2_candidate_ablation_summary.csv"
OUT_IMAGE = ROOT / "np_ffhq_bestof2_candidate_ablation_image_bestof2.csv"
OUT_RUNS = ROOT / "np_ffhq_bestof2_candidate_ablation_merged_run_level.csv"

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
                fieldnames.append(k)
                seen.add(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def fget(r, k, default=math.nan):
    try:
        v = r.get(k, "")
        if v == "" or v is None:
            return default
        return float(v)
    except Exception:
        return default

def finite(xs):
    return [x for x in xs if math.isfinite(x)]

def safe_mean(xs):
    xs = finite(xs)
    return mean(xs) if xs else math.nan

def safe_median(xs):
    xs = finite(xs)
    return median(xs) if xs else math.nan

def safe_min(xs):
    xs = finite(xs)
    return min(xs) if xs else math.nan

def safe_max(xs):
    xs = finite(xs)
    return max(xs) if xs else math.nan

def safe_std(xs):
    xs = finite(xs)
    return stdev(xs) if len(xs) > 1 else 0.0

def tag_after_root(path, root_name):
    parts = Path(path).parts
    for i, p in enumerate(parts):
        if p == root_name and i + 1 < len(parts):
            return parts[i + 1]
    return ""

def main():
    files = sorted(glob.glob(str(RUN_ROOT / "**" / "run_level.csv"), recursive=True))
    print("run_level files:", len(files))

    rows = []
    for path in files:
        tag = tag_after_root(path, "np_ffhq_bestof2_candidate_ablation")
        for r in read_csv(path):
            r["config_tag"] = tag
            r["source_file"] = path
            rows.append(r)

    if not rows:
        raise RuntimeError(f"No rows found under {RUN_ROOT}")

    group_cols = [
        "config_tag",
        "measurement_noise_std",
        "alignment_mode",
        "image_basename",
    ]

    by_cfg_noise_align_img = defaultdict(list)
    for r in rows:
        key = tuple(r.get(c, "") for c in group_cols)
        by_cfg_noise_align_img[key].append(r)

    image_rows = []
    for key, gr in sorted(by_cfg_noise_align_img.items()):
        config_tag, noise, align, img = key
        best = max(gr, key=lambda r: fget(r, "psnr"))
        psnrs = [fget(r, "psnr") for r in gr]
        image_rows.append({
            "config_tag": config_tag,
            "measurement_noise_std": noise,
            "alignment_mode": align,
            "image_basename": img,
            "n_seeds": len(set(r.get("seed", "") for r in gr)),
            "best_seed_by_psnr": best.get("seed", ""),
            "bestof2_psnr": fget(best, "psnr"),
            "bestof2_ssim_at_best_psnr": fget(best, "ssim"),
            "bestof2_lpips_at_best_psnr": fget(best, "lpips"),
            "runtime_s_at_best_psnr": fget(best, "runtime_s"),
            "psnr_mean_over_seeds": safe_mean(psnrs),
            "psnr_min_over_seeds": safe_min(psnrs),
            "score_radius": best.get("score_radius", ""),
            "proj_radius": best.get("proj_radius", ""),
            "proj_start": best.get("proj_start", ""),
            "num_candidates_soft": best.get("num_candidates_soft", ""),
            "num_candidates_hard": best.get("num_candidates_hard", ""),
        })

    by_cfg_noise_align = defaultdict(list)
    for r in image_rows:
        by_cfg_noise_align[
            (r["config_tag"], r["measurement_noise_std"], r["alignment_mode"])
        ].append(r)

    summary = []
    for (tag, noise, align), gr in sorted(by_cfg_noise_align.items()):
        psnr = [fget(r, "bestof2_psnr") for r in gr]
        ssim = [fget(r, "bestof2_ssim_at_best_psnr") for r in gr]
        lpips = [fget(r, "bestof2_lpips_at_best_psnr") for r in gr]
        runtime = [fget(r, "runtime_s_at_best_psnr") for r in gr]
        worst = min(gr, key=lambda r: fget(r, "bestof2_psnr"))
        rep = gr[0]
        below20 = sum(1 for x in psnr if x < 20)
        below25 = sum(1 for x in psnr if x < 25)
        summary.append({
            "config_tag": tag,
            "measurement_noise_std": noise,
            "alignment_mode": align,
            "n_images": len(gr),
            "bestof2_psnr_mean": safe_mean(psnr),
            "bestof2_psnr_median": safe_median(psnr),
            "bestof2_psnr_min": safe_min(psnr),
            "bestof2_psnr_max": safe_max(psnr),
            "bestof2_psnr_std": safe_std(psnr),
            "bestof2_ssim_mean": safe_mean(ssim),
            "bestof2_lpips_mean": safe_mean(lpips),
            "runtime_s_mean_at_best_seed": safe_mean(runtime),
            "n_images_below20": below20,
            "n_images_below25": below25,
            "worst_image": worst.get("image_basename", ""),
            "worst_image_bestof2_psnr": fget(worst, "bestof2_psnr"),
            "score_radius": rep.get("score_radius", ""),
            "proj_radius": rep.get("proj_radius", ""),
            "proj_start": rep.get("proj_start", ""),
            "num_candidates_soft": rep.get("num_candidates_soft", ""),
            "num_candidates_hard": rep.get("num_candidates_hard", ""),
        })

    summary.sort(
        key=lambda r: (
            float(r["measurement_noise_std"]),
            r["alignment_mode"],
            -fget(r, "bestof2_psnr_mean"),
        )
    )

    write_csv(OUT_SUMMARY, summary)
    write_csv(OUT_IMAGE, image_rows)
    write_csv(OUT_RUNS, rows)

    cols = [
        "config_tag", "measurement_noise_std", "alignment_mode", "n_images",
        "bestof2_psnr_mean", "bestof2_psnr_median", "bestof2_psnr_min",
        "bestof2_ssim_mean", "bestof2_lpips_mean",
        "n_images_below20", "n_images_below25",
        "runtime_s_mean_at_best_seed",
    ]
    print(",".join(cols))
    for r in summary:
        print(",".join(str(r.get(c, "")) for c in cols))

    print("\nSaved:")
    print(" ", OUT_SUMMARY)
    print(" ", OUT_IMAGE)
    print(" ", OUT_RUNS)

if __name__ == "__main__":
    main()
