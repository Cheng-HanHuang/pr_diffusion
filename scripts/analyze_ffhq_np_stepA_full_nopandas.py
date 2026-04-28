#!/usr/bin/env python3
from __future__ import annotations

import csv
import glob
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path("/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411")
STEP_ROOT = ROOT / "np_ffhq_stepA_full_top2"

OUT_SUMMARY = ROOT / "np_ffhq_stepA_full_top2_bestof4_summary.csv"
OUT_IMAGE = ROOT / "np_ffhq_stepA_full_top2_image_bestof4.csv"
OUT_RUNS = ROOT / "np_ffhq_stepA_full_top2_merged_run_level.csv"


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
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


def fget(row: dict, key: str, default: float = math.nan) -> float:
    try:
        v = row.get(key, "")
        if v == "" or v is None:
            return default
        return float(v)
    except Exception:
        return default


def safe_mean(vals: list[float]) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    return mean(vals) if vals else math.nan


def safe_median(vals: list[float]) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    return median(vals) if vals else math.nan


def safe_min(vals: list[float]) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    return min(vals) if vals else math.nan


def safe_max(vals: list[float]) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    return max(vals) if vals else math.nan


def safe_stdev(vals: list[float]) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    return stdev(vals) if len(vals) > 1 else 0.0


def find_config_tag(path: str) -> str:
    # .../np_ffhq_stepA_full_top2/A_s02.../difffpr_np_guided_x/run_level.csv
    parts = Path(path).parts
    for i, p in enumerate(parts):
        if p == "np_ffhq_stepA_full_top2" and i + 1 < len(parts):
            return parts[i + 1]
    return ""


def load_rows() -> list[dict]:
    files = sorted(glob.glob(str(STEP_ROOT / "**" / "run_level.csv"), recursive=True))
    print(f"Found Step A run_level.csv files: {len(files)}")
    rows = []
    for path in files:
        tag = find_config_tag(path)
        for r in read_csv(path):
            r["config_tag"] = tag
            r["source_file"] = path
            rows.append(r)
    return rows


def main() -> None:
    rows = load_rows()
    if not rows:
        raise RuntimeError(f"No run_level.csv files under {STEP_ROOT}")

    # Group by config, alignment, image. Choose best seed by PSNR.
    group_cols = [
        "config_tag",
        "alignment_mode",
        "image_basename",
    ]

    by_cfg_align_img = defaultdict(list)
    for r in rows:
        key = tuple(r.get(c, "") for c in group_cols)
        by_cfg_align_img[key].append(r)

    image_best_rows = []
    for (config_tag, alignment, image), gr in sorted(by_cfg_align_img.items()):
        best = max(gr, key=lambda r: fget(r, "psnr"))
        psnrs = [fget(r, "psnr") for r in gr]
        out = {
            "config_tag": config_tag,
            "alignment_mode": alignment,
            "image_basename": image,
            "n_seeds": len(set(r.get("seed", "") for r in gr)),
            "best_seed_by_psnr": best.get("seed", ""),
            "bestof4_psnr": fget(best, "psnr"),
            "bestof4_ssim_at_best_psnr": fget(best, "ssim"),
            "bestof4_lpips_at_best_psnr": fget(best, "lpips"),
            "clean_mag_l2_at_best_psnr": fget(best, "clean_mag_l2"),
            "noisy_mag_l2_at_best_psnr": fget(best, "noisy_mag_l2"),
            "clean_lowfreq_mag_l2_at_best_psnr": fget(best, "clean_lowfreq_mag_l2"),
            "noisy_lowfreq_mag_l2_at_best_psnr": fget(best, "noisy_lowfreq_mag_l2"),
            "runtime_s_at_best_psnr": fget(best, "runtime_s"),
            "psnr_mean_over_seeds": safe_mean(psnrs),
            "psnr_median_over_seeds": safe_median(psnrs),
            "psnr_min_over_seeds": safe_min(psnrs),
            "psnr_max_over_seeds": safe_max(psnrs),
            "score_radius": best.get("score_radius", ""),
            "proj_radius": best.get("proj_radius", ""),
            "proj_radius_schedule": best.get("proj_radius_schedule", ""),
            "proj_start": best.get("proj_start", ""),
            "num_candidates_soft": best.get("num_candidates_soft", ""),
            "num_candidates_hard": best.get("num_candidates_hard", ""),
        }
        image_best_rows.append(out)

    by_cfg_align = defaultdict(list)
    for r in image_best_rows:
        by_cfg_align[(r["config_tag"], r["alignment_mode"])].append(r)

    summary_rows = []
    for (config_tag, alignment), gr in sorted(by_cfg_align.items()):
        psnr = [fget(r, "bestof4_psnr") for r in gr]
        ssim = [fget(r, "bestof4_ssim_at_best_psnr") for r in gr]
        lpips = [fget(r, "bestof4_lpips_at_best_psnr") for r in gr]
        clean_mag = [fget(r, "clean_mag_l2_at_best_psnr") for r in gr]
        noisy_mag = [fget(r, "noisy_mag_l2_at_best_psnr") for r in gr]
        clean_low = [fget(r, "clean_lowfreq_mag_l2_at_best_psnr") for r in gr]
        noisy_low = [fget(r, "noisy_lowfreq_mag_l2_at_best_psnr") for r in gr]

        worst = min(gr, key=lambda r: fget(r, "bestof4_psnr"))
        best_img = max(gr, key=lambda r: fget(r, "bestof4_psnr"))
        rep = gr[0]

        summary_rows.append({
            "config_tag": config_tag,
            "alignment_mode": alignment,
            "n_images": len(gr),
            "bestof4_psnr_mean": safe_mean(psnr),
            "bestof4_psnr_median": safe_median(psnr),
            "bestof4_psnr_min": safe_min(psnr),
            "bestof4_psnr_max": safe_max(psnr),
            "bestof4_psnr_std": safe_stdev(psnr),
            "bestof4_ssim_mean": safe_mean(ssim),
            "bestof4_ssim_median": safe_median(ssim),
            "bestof4_lpips_mean": safe_mean(lpips),
            "bestof4_lpips_median": safe_median(lpips),
            "clean_mag_l2_mean": safe_mean(clean_mag),
            "noisy_mag_l2_mean": safe_mean(noisy_mag),
            "clean_lowfreq_mag_l2_mean": safe_mean(clean_low),
            "noisy_lowfreq_mag_l2_mean": safe_mean(noisy_low),
            "worst_image": worst["image_basename"],
            "worst_image_bestof4_psnr": fget(worst, "bestof4_psnr"),
            "best_image": best_img["image_basename"],
            "best_image_bestof4_psnr": fget(best_img, "bestof4_psnr"),
            "score_radius": rep.get("score_radius", ""),
            "proj_radius": rep.get("proj_radius", ""),
            "proj_radius_schedule": rep.get("proj_radius_schedule", ""),
            "proj_start": rep.get("proj_start", ""),
            "num_candidates_soft": rep.get("num_candidates_soft", ""),
            "num_candidates_hard": rep.get("num_candidates_hard", ""),
        })

    summary_rows.sort(
        key=lambda r: (
            fget(r, "bestof4_psnr_mean"),
            fget(r, "bestof4_psnr_median"),
            fget(r, "bestof4_psnr_min"),
            fget(r, "bestof4_ssim_mean"),
        ),
        reverse=True,
    )

    write_csv(OUT_SUMMARY, summary_rows)
    write_csv(OUT_IMAGE, image_best_rows)
    write_csv(OUT_RUNS, rows)

    print("\n=== Step A full top-2: best-of-4 summary ===")
    cols = [
        "config_tag", "alignment_mode", "n_images",
        "bestof4_psnr_mean", "bestof4_psnr_median", "bestof4_psnr_min",
        "bestof4_ssim_mean", "bestof4_lpips_mean",
        "clean_mag_l2_mean", "noisy_mag_l2_mean",
        "worst_image", "worst_image_bestof4_psnr",
    ]
    print(",".join(cols))
    for r in summary_rows:
        print(",".join(str(r.get(c, "")) for c in cols))

    print("\nSaved:")
    print(" ", OUT_SUMMARY)
    print(" ", OUT_IMAGE)
    print(" ", OUT_RUNS)


if __name__ == "__main__":
    main()
