#!/usr/bin/env python3
"""Apply executable multi-config selection to diagnostic trace summaries.

This postprocessor consumes `*_run_trace_summary.csv` produced by
`analyze_ffhq_diagnostic_trace_nopandas.py` and applies a non-ground-truth
selector across multiple candidate configurations.

Primary selector:
  1. For each image and config, average a trajectory statistic across seeds.
  2. Choose the config with the lowest averaged statistic.
  3. Choose a seed inside the chosen config by the same statistic.

It also reports diagnostic oracle views:
  - selected_config_bestofk: choose config by statistic, then choose best seed by PSNR.
  - oracle_all_candidates: choose best PSNR among all configs/seeds.

The selector statistic defaults to `post_winner_lf_mse_mean`, matching the LF/S2
selector experiments.
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
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def find_run_trace(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(glob.glob(str(path / "**" / "*_run_trace_summary.csv"), recursive=True))
    if not candidates:
        candidates = sorted(glob.glob(str(path / "**" / "run_trace_summary.csv"), recursive=True))
    if not candidates:
        raise FileNotFoundError(f"No run trace summary found under {path}")
    # Prefer the newest file in case the root contains multiple old analyses.
    return max((Path(p) for p in candidates), key=lambda p: p.stat().st_mtime)


def metric_key(alignment: str, metric: str) -> str:
    return f"{alignment}_{metric}"


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


def choose_rows_for_method(
    *,
    method: str,
    image: str,
    alignments: List[str],
    chosen_run: Dict[str, str],
    chosen_config: str,
    chosen_seed: str,
    cfg_stat: Dict[str, float],
    rows_for_image: List[Dict[str, str]],
    stat_key: str,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for align in alignments:
        psnr_key = metric_key(align, "psnr")
        ssim_key = metric_key(align, "ssim")
        lpips_key = metric_key(align, "lpips")
        row = dict(chosen_run)
        row.update(
            {
                "selection_method": method,
                "alignment_mode": align,
                "image_basename": image,
                "selected_config": chosen_config,
                "selected_seed": chosen_seed,
                "psnr": fget(chosen_run, psnr_key),
                "ssim": fget(chosen_run, ssim_key),
                "lpips": fget(chosen_run, lpips_key),
                "selector_stat_key": stat_key,
                "selected_run_selector_stat": fget(chosen_run, stat_key),
            }
        )
        for cfg, val in sorted(cfg_stat.items()):
            safe = cfg.replace(".", "p").replace("-", "_")
            row[f"config_stat_{safe}"] = val
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_or_trace", required=True, help="Root containing *_run_trace_summary.csv or the CSV itself")
    ap.add_argument("--out_prefix", default=None)
    ap.add_argument("--selector_stat", default="post_winner_lf_mse_mean")
    ap.add_argument("--alignments", default="raw,rot180,resolve")
    ap.add_argument("--primary_alignment", default="raw")
    args = ap.parse_args()

    trace_path = find_run_trace(Path(args.root_or_trace))
    rows = read_csv(trace_path)
    if not rows:
        raise RuntimeError(f"No rows in {trace_path}")

    stat_key = args.selector_stat
    alignments = [x.strip() for x in args.alignments.split(",") if x.strip()]
    out_prefix = Path(args.out_prefix) if args.out_prefix else trace_path.with_name("multiconfig_selector")

    by_image: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_image.setdefault(str(row.get("image_basename", "")), []).append(row)

    selected_rows: List[Dict[str, object]] = []
    image_diag_rows: List[Dict[str, object]] = []

    for image, image_rows in sorted(by_image.items()):
        by_cfg: Dict[str, List[Dict[str, str]]] = {}
        for row in image_rows:
            by_cfg.setdefault(str(row.get("config_tag", "")), []).append(row)

        cfg_stat = {
            cfg: fmean([fget(r, stat_key) for r in cfg_rows])
            for cfg, cfg_rows in by_cfg.items()
        }
        selected_cfg = min(cfg_stat, key=lambda cfg: cfg_stat[cfg])
        cfg_rows = by_cfg[selected_cfg]
        selected_seed_row = min(cfg_rows, key=lambda r: fget(r, stat_key))
        selected_seed = str(selected_seed_row.get("seed", ""))

        # Diagnostic view: chosen config, oracle seed by PSNR under primary alignment.
        primary_psnr_key = metric_key(args.primary_alignment, "psnr")
        best_seed_in_selected_cfg = max(cfg_rows, key=lambda r: fget(r, primary_psnr_key))
        oracle_all = max(image_rows, key=lambda r: fget(r, primary_psnr_key))
        global_by_stat = min(image_rows, key=lambda r: fget(r, stat_key))

        selected_rows.extend(
            choose_rows_for_method(
                method="selected_config_seed_by_selector",
                image=image,
                alignments=alignments,
                chosen_run=selected_seed_row,
                chosen_config=selected_cfg,
                chosen_seed=selected_seed,
                cfg_stat=cfg_stat,
                rows_for_image=image_rows,
                stat_key=stat_key,
            )
        )
        selected_rows.extend(
            choose_rows_for_method(
                method="selected_config_bestofk",
                image=image,
                alignments=alignments,
                chosen_run=best_seed_in_selected_cfg,
                chosen_config=selected_cfg,
                chosen_seed=str(best_seed_in_selected_cfg.get("seed", "")),
                cfg_stat=cfg_stat,
                rows_for_image=image_rows,
                stat_key=stat_key,
            )
        )
        selected_rows.extend(
            choose_rows_for_method(
                method="global_run_by_selector",
                image=image,
                alignments=alignments,
                chosen_run=global_by_stat,
                chosen_config=str(global_by_stat.get("config_tag", "")),
                chosen_seed=str(global_by_stat.get("seed", "")),
                cfg_stat=cfg_stat,
                rows_for_image=image_rows,
                stat_key=stat_key,
            )
        )
        selected_rows.extend(
            choose_rows_for_method(
                method="oracle_all_candidates",
                image=image,
                alignments=alignments,
                chosen_run=oracle_all,
                chosen_config=str(oracle_all.get("config_tag", "")),
                chosen_seed=str(oracle_all.get("seed", "")),
                cfg_stat=cfg_stat,
                rows_for_image=image_rows,
                stat_key=stat_key,
            )
        )

        diag = {
            "image_basename": image,
            "selected_config": selected_cfg,
            "selected_seed_by_selector": selected_seed,
            "selected_config_best_seed": best_seed_in_selected_cfg.get("seed", ""),
            "oracle_config": oracle_all.get("config_tag", ""),
            "oracle_seed": oracle_all.get("seed", ""),
            "selected_config_seed_by_selector_raw_psnr": fget(selected_seed_row, primary_psnr_key),
            "selected_config_bestofk_raw_psnr": fget(best_seed_in_selected_cfg, primary_psnr_key),
            "oracle_all_candidates_raw_psnr": fget(oracle_all, primary_psnr_key),
            "global_run_by_selector_raw_psnr": fget(global_by_stat, primary_psnr_key),
            "selector_regret_vs_oracle": fget(oracle_all, primary_psnr_key) - fget(selected_seed_row, primary_psnr_key),
            "config_regret_vs_oracle": fget(oracle_all, primary_psnr_key) - fget(best_seed_in_selected_cfg, primary_psnr_key),
        }
        for cfg, val in sorted(cfg_stat.items()):
            safe = cfg.replace(".", "p").replace("-", "_")
            diag[f"config_stat_{safe}"] = val
        image_diag_rows.append(diag)

    selected_path = Path(str(out_prefix) + "_selected_image_level.csv")
    summary_path = Path(str(out_prefix) + "_selected_summary.csv")
    diag_path = Path(str(out_prefix) + "_image_diagnostics.csv")
    write_csv(selected_path, selected_rows)
    write_csv(summary_path, summarize(selected_rows))
    write_csv(diag_path, image_diag_rows)

    print(f"Trace: {trace_path}")
    print(f"Wrote: {selected_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {diag_path}")
    print("\nSummary:")
    for row in summarize(selected_rows):
        print(row)


if __name__ == "__main__":
    main()
