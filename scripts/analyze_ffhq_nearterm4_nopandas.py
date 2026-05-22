#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev


def read_csv(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
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


def tag_after_root(path: str, root: Path) -> str:
    try:
        rel = Path(path).resolve().relative_to(root.resolve())
        return rel.parts[0] if rel.parts else ""
    except Exception:
        return Path(path).parts[-3] if len(Path(path).parts) >= 3 else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Near-term output root, e.g. .../np_ffhq_nearterm_4gpu")
    ap.add_argument("--out_prefix", default=None)
    ap.add_argument("--primary_alignment", default="raw")
    ap.add_argument("--promote_mean", type=float, default=28.8)
    ap.add_argument("--promote_min", type=float, default=25.0)
    args = ap.parse_args()

    root = Path(args.root)
    out_prefix = Path(args.out_prefix) if args.out_prefix else root / "nearterm4"

    files = sorted(glob.glob(str(root / "**" / "run_level.csv"), recursive=True))
    print("run_level files:", len(files))
    for p in files:
        print(" ", p)

    rows = []
    for path in files:
        tag = tag_after_root(path, root)
        for r in read_csv(path):
            r["config_tag"] = tag
            r["source_file"] = path
            rows.append(r)

    if not rows:
        raise RuntimeError(f"No run_level.csv files found under {root}")

    by_cfg_align_img = defaultdict(list)
    for r in rows:
        key = (r.get("config_tag", ""), r.get("alignment_mode", ""), r.get("image_basename", ""))
        by_cfg_align_img[key].append(r)

    image_rows = []
    for (tag, align, img), gr in sorted(by_cfg_align_img.items()):
        best = max(gr, key=lambda r: fget(r, "psnr"))
        psnrs = [fget(r, "psnr") for r in gr]
        ssims = [fget(r, "ssim") for r in gr]
        lpips_vals = [fget(r, "lpips") for r in gr]
        image_rows.append({
            "config_tag": tag,
            "alignment_mode": align,
            "image_basename": img,
            "n_seeds": len(set(r.get("seed", "") for r in gr)),
            "best_seed_by_psnr": best.get("seed", ""),
            "bestofk_psnr": fget(best, "psnr"),
            "bestofk_ssim_at_best_psnr": fget(best, "ssim"),
            "bestofk_lpips_at_best_psnr": fget(best, "lpips"),
            "psnr_mean_over_seeds": safe_mean(psnrs),
            "psnr_min_over_seeds": safe_min(psnrs),
            "ssim_mean_over_seeds": safe_mean(ssims),
            "lpips_mean_over_seeds": safe_mean(lpips_vals),
            "score_mode": best.get("score_mode", ""),
            "score_reg_lambda": best.get("score_reg_lambda", ""),
            "score_reg_lambda_schedule": best.get("score_reg_lambda_schedule", ""),
            "noise_memory_k": best.get("noise_memory_k", ""),
            "score_radius": best.get("score_radius", ""),
            "proj_radius": best.get("proj_radius", ""),
            "proj_start": best.get("proj_start", ""),
            "num_candidates_soft": best.get("num_candidates_soft", ""),
            "num_candidates_hard": best.get("num_candidates_hard", ""),
        })

    by_cfg_align = defaultdict(list)
    for r in image_rows:
        by_cfg_align[(r["config_tag"], r["alignment_mode"])].append(r)

    summary = []
    for (tag, align), gr in sorted(by_cfg_align.items()):
        psnr = [fget(r, "bestofk_psnr") for r in gr]
        ssim = [fget(r, "bestofk_ssim_at_best_psnr") for r in gr]
        lpips = [fget(r, "bestofk_lpips_at_best_psnr") for r in gr]
        worst = min(gr, key=lambda r: fget(r, "bestofk_psnr"))
        rep = gr[0]
        below20 = sum(1 for x in psnr if x < 20)
        below25 = sum(1 for x in psnr if x < 25)
        mean_psnr = safe_mean(psnr)
        min_psnr = safe_min(psnr)
        promote = (
            mean_psnr > float(args.promote_mean)
            and min_psnr > float(args.promote_min)
            and below20 == 0
            and below25 == 0
        )
        summary.append({
            "config_tag": tag,
            "alignment_mode": align,
            "n_images": len(gr),
            "bestofk_psnr_mean": mean_psnr,
            "bestofk_psnr_median": safe_median(psnr),
            "bestofk_psnr_min": min_psnr,
            "bestofk_psnr_max": safe_max(psnr),
            "bestofk_psnr_std": safe_std(psnr),
            "bestofk_ssim_mean": safe_mean(ssim),
            "bestofk_lpips_mean": safe_mean(lpips),
            "n_images_below20": below20,
            "n_images_below25": below25,
            "promote_by_rule": int(promote),
            "worst_image": worst.get("image_basename", ""),
            "worst_image_bestofk_psnr": fget(worst, "bestofk_psnr"),
            "score_mode": rep.get("score_mode", ""),
            "score_reg_lambda": rep.get("score_reg_lambda", ""),
            "score_reg_lambda_schedule": rep.get("score_reg_lambda_schedule", ""),
            "noise_memory_k": rep.get("noise_memory_k", ""),
            "score_radius": rep.get("score_radius", ""),
            "proj_radius": rep.get("proj_radius", ""),
            "proj_start": rep.get("proj_start", ""),
            "num_candidates_soft": rep.get("num_candidates_soft", ""),
            "num_candidates_hard": rep.get("num_candidates_hard", ""),
        })

    summary.sort(key=lambda r: (r["alignment_mode"] != args.primary_alignment, -fget(r, "bestofk_psnr_mean")))

    summary_path = Path(str(out_prefix) + "_summary.csv")
    image_path = Path(str(out_prefix) + "_image_bestofk.csv")
    runs_path = Path(str(out_prefix) + "_merged_run_level.csv")
    write_csv(summary_path, summary)
    write_csv(image_path, image_rows)
    write_csv(runs_path, rows)

    cols = [
        "config_tag", "alignment_mode", "n_images", "bestofk_psnr_mean",
        "bestofk_psnr_median", "bestofk_psnr_min", "bestofk_ssim_mean",
        "bestofk_lpips_mean", "n_images_below20", "n_images_below25",
        "promote_by_rule", "worst_image", "worst_image_bestofk_psnr",
        "score_mode", "score_reg_lambda", "score_reg_lambda_schedule", "noise_memory_k",
    ]
    print("\nSummary:")
    print(",".join(cols))
    for r in summary:
        print(",".join(str(r.get(c, "")) for c in cols))

    print("\nPrimary alignment ranking:", args.primary_alignment)
    primary = [r for r in summary if r.get("alignment_mode") == args.primary_alignment]
    for i, r in enumerate(sorted(primary, key=lambda x: -fget(x, "bestofk_psnr_mean")), 1):
        print(
            f"{i}. {r['config_tag']}: mean={fget(r,'bestofk_psnr_mean'):.3f}, "
            f"median={fget(r,'bestofk_psnr_median'):.3f}, min={fget(r,'bestofk_psnr_min'):.3f}, "
            f"below20={r['n_images_below20']}, below25={r['n_images_below25']}, "
            f"promote={r['promote_by_rule']}, worst={r['worst_image']}"
        )

    print("\nSaved:")
    print(" ", summary_path)
    print(" ", image_path)
    print(" ", runs_path)


if __name__ == "__main__":
    main()
