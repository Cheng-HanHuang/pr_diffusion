#!/usr/bin/env python3
"""Apply the LF/S2 seed tie-break selector to a selector run directory.

Input is a run directory produced by pr_external_difffpr_np_guided_lf_s2_selector.py,
containing run_level.csv.  The selector is:

1. Choose config LF vs S2 by mean post-projection winner LF-MSE across seeds.
2. Within the selected config, sort seeds by the same selector statistic.
3. If the best two seeds are within --seed_tie_threshold, choose the seed with
   lower final noisy_lowfreq_mag_l2 instead.
4. Otherwise choose the lower selector-stat seed.

This script writes:
  selected_tiebreak_image_level.csv
  selected_tiebreak_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Tuple


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def fget(row: Dict[str, object], key: str, default: float = math.nan) -> float:
    try:
        val = row.get(key, "")
        if val == "" or val is None:
            return default
        return float(val)
    except Exception:
        return default


def finite(xs: List[float]) -> List[float]:
    return [x for x in xs if math.isfinite(x)]


def fmean(xs: List[float]) -> float:
    xs = finite(xs)
    return mean(xs) if xs else math.nan


def fmedian(xs: List[float]) -> float:
    xs = finite(xs)
    return median(xs) if xs else math.nan


def fmin(xs: List[float]) -> float:
    xs = finite(xs)
    return min(xs) if xs else math.nan


def fmax(xs: List[float]) -> float:
    xs = finite(xs)
    return max(xs) if xs else math.nan


def fstd(xs: List[float]) -> float:
    xs = finite(xs)
    return stdev(xs) if len(xs) > 1 else 0.0


def find_run_dir(root_or_run: Path) -> Path:
    if (root_or_run / "run_level.csv").exists():
        return root_or_run
    candidates = [Path(p) for p in glob.glob(str(root_or_run / "lf_s2_selector_*"))]
    candidates = [p for p in candidates if (p / "run_level.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"No lf_s2_selector_* directory with run_level.csv under {root_or_run}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_method_align: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        by_method_align.setdefault((str(row["selection_method"]), str(row["alignment_mode"])), []).append(row)

    out: List[Dict[str, object]] = []
    for (method, align), group in sorted(by_method_align.items()):
        psnr = [fget(r, "psnr") for r in group]
        ssim = [fget(r, "ssim") for r in group]
        lpips = [fget(r, "lpips") for r in group]
        worst = min(group, key=lambda r: fget(r, "psnr"))
        out.append(
            {
                "selection_method": method,
                "alignment_mode": align,
                "n_images": len(group),
                "psnr_mean": fmean(psnr),
                "psnr_median": fmedian(psnr),
                "psnr_min": fmin(psnr),
                "psnr_max": fmax(psnr),
                "psnr_std": fstd(psnr),
                "ssim_mean": fmean(ssim),
                "lpips_mean": fmean(lpips),
                "n_images_below20": sum(1 for x in psnr if x < 20),
                "n_images_below25": sum(1 for x in psnr if x < 25),
                "worst_image": worst.get("image_basename", ""),
                "worst_image_psnr": fget(worst, "psnr"),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="A selector run directory or parent output root")
    ap.add_argument("--seed_tie_threshold", type=float, default=5e-5)
    ap.add_argument("--out_prefix", default="selected_tiebreak")
    args = ap.parse_args()

    run_dir = find_run_dir(Path(args.run_dir))
    run_path = run_dir / "run_level.csv"
    rows = read_csv(run_path)
    if not rows:
        raise RuntimeError(f"No rows found in {run_path}")

    alignments = sorted({r.get("alignment_mode", "") for r in rows if r.get("alignment_mode", "")})
    raw_rows = [r for r in rows if r.get("alignment_mode") == "raw"]

    by_image: Dict[str, List[Dict[str, str]]] = {}
    for row in raw_rows:
        by_image.setdefault(str(row.get("image_basename", "")), []).append(row)

    selected: List[Dict[str, object]] = []

    for image, image_raw_rows in sorted(by_image.items()):
        by_cfg: Dict[str, List[Dict[str, str]]] = {}
        for row in image_raw_rows:
            by_cfg.setdefault(str(row.get("config_tag", "")), []).append(row)

        cfg_stat = {
            cfg: fmean([fget(r, "selector_post_winner_lf_mse_mean") for r in cfg_rows])
            for cfg, cfg_rows in by_cfg.items()
        }
        selected_cfg = min(cfg_stat, key=lambda cfg: cfg_stat[cfg])
        cfg_rows = sorted(
            by_cfg[selected_cfg],
            key=lambda r: fget(r, "selector_post_winner_lf_mse_mean"),
        )
        best_stat_row = cfg_rows[0]
        second_stat_row = cfg_rows[1] if len(cfg_rows) > 1 else None
        selector_gap = (
            fget(second_stat_row, "selector_post_winner_lf_mse_mean")
            - fget(best_stat_row, "selector_post_winner_lf_mse_mean")
            if second_stat_row is not None
            else math.inf
        )

        if second_stat_row is not None and selector_gap <= float(args.seed_tie_threshold):
            chosen_seed_row = min(cfg_rows, key=lambda r: fget(r, "noisy_lowfreq_mag_l2"))
            seed_rule = "tie_final_noisy_lowfreq_mag_l2"
        else:
            chosen_seed_row = best_stat_row
            seed_rule = "selector_stat"

        selected_seed = str(chosen_seed_row.get("seed", ""))

        for align in alignments:
            match = [
                r for r in rows
                if r.get("image_basename") == image
                and r.get("config_tag") == selected_cfg
                and str(r.get("seed")) == selected_seed
                and r.get("alignment_mode") == align
            ]
            if len(match) != 1:
                raise RuntimeError(
                    f"Expected exactly one row for image={image}, cfg={selected_cfg}, "
                    f"seed={selected_seed}, align={align}; found {len(match)}"
                )
            out = dict(match[0])
            out.update(
                {
                    "selection_method": "selected_config_seed_tiebreak",
                    "selected_config": selected_cfg,
                    "selected_seed": selected_seed,
                    "seed_selection_rule": seed_rule,
                    "seed_tie_threshold": float(args.seed_tie_threshold),
                    "seed_selector_gap": selector_gap,
                    "lf_config_selector_stat_mean": cfg_stat.get("lf", math.nan),
                    "s2_config_selector_stat_mean": cfg_stat.get("s2_preproj_lam001", math.nan),
                    "selector_margin_s2_minus_lf": cfg_stat.get("s2_preproj_lam001", math.nan) - cfg_stat.get("lf", math.nan),
                }
            )
            selected.append(out)

    image_out = run_dir / f"{args.out_prefix}_image_level.csv"
    summary_out = run_dir / f"{args.out_prefix}_summary.csv"
    write_csv(image_out, selected)
    write_csv(summary_out, summarize(selected))

    print(f"Run dir: {run_dir}")
    print(f"Wrote: {image_out}")
    print(f"Wrote: {summary_out}")
    print("\nSummary:")
    for row in summarize(selected):
        print(row)


if __name__ == "__main__":
    main()
