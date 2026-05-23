#!/usr/bin/env python3
"""Post-hoc slice analysis for multi-config selector traces.

This tool reads a run trace summary produced by analyze_ffhq_diagnostic_trace_nopandas.py
and evaluates selector performance on subsets of seeds and/or configuration tags.

Use cases:
  - single-seed simulation: --seed_sets 102 103
  - reduced-lambda pool: --config_sets "LF,S2..." etc.
  - validation of whether reliability comes from seeds or scoring branches.

It does not run new reconstructions.
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
        w = csv.DictWriter(f, fieldnames=fields)
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


def find_trace(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(glob.glob(str(path / "**" / "*_run_trace_summary.csv"), recursive=True))
    if not candidates:
        raise FileNotFoundError(f"No *_run_trace_summary.csv found under {path}")
    return max((Path(p) for p in candidates), key=lambda p: p.stat().st_mtime)


def parse_sets(items: List[str], default_all_name: str = "all") -> List[Tuple[str, List[str]]]:
    out: List[Tuple[str, List[str]]] = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, vals = item.split(":", 1)
            vals_list = [x.strip() for x in vals.split(",") if x.strip()]
            out.append((name.strip(), vals_list))
        else:
            vals_list = [x.strip() for x in item.split(",") if x.strip()]
            name = item.replace(",", "_").replace(".", "p") if len(vals_list) != 1 else vals_list[0]
            out.append((name, vals_list))
    if not out:
        out.append((default_all_name, []))
    return out


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_key: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["analysis_name"]),
            str(row["selection_method"]),
            str(row["alignment_mode"]),
            str(row["seed_set_name"]),
        )
        by_key.setdefault(key, []).append(row)

    out: List[Dict[str, object]] = []
    for (analysis, method, align, seed_name), group in sorted(by_key.items()):
        psnr = [fget(r, "psnr") for r in group]
        ssim = [fget(r, "ssim") for r in group]
        worst = min(group, key=lambda r: fget(r, "psnr"))
        out.append(
            {
                "analysis_name": analysis,
                "seed_set_name": seed_name,
                "selection_method": method,
                "alignment_mode": align,
                "n_images": len(group),
                "psnr_mean": fmean(psnr),
                "psnr_median": fmedian(psnr),
                "psnr_min": fmin(psnr),
                "psnr_max": fmax(psnr),
                "psnr_std": fstd(psnr),
                "ssim_mean": fmean(ssim),
                "n_images_below20": sum(1 for x in psnr if x < 20),
                "n_images_below25": sum(1 for x in psnr if x < 25),
                "worst_image": worst.get("image_basename", ""),
                "worst_image_psnr": fget(worst, "psnr"),
            }
        )
    return out


def metric_key(align: str, metric: str) -> str:
    return f"{align}_{metric}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_or_trace", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--selector_stat", default="post_winner_lf_mse_mean")
    ap.add_argument("--alignments", default="raw,rot180,resolve")
    ap.add_argument("--primary_alignment", default="raw")
    ap.add_argument(
        "--seed_sets",
        nargs="*",
        default=[],
        help="Seed sets, e.g. '102' '103' '102,103' or named 's102:102'. Empty means all seeds.",
    )
    ap.add_argument(
        "--config_sets",
        nargs="*",
        default=[],
        help="Config tag sets. Empty means all configs. Use exact config_tag strings or comma-separated named sets.",
    )
    args = ap.parse_args()

    trace = find_trace(Path(args.root_or_trace))
    rows = read_csv(trace)
    if not rows:
        raise RuntimeError(f"No rows in {trace}")

    all_seeds = sorted({str(r.get("seed", "")) for r in rows if str(r.get("seed", ""))})
    all_configs = sorted({str(r.get("config_tag", "")) for r in rows if str(r.get("config_tag", ""))})
    seed_sets = parse_sets(args.seed_sets, default_all_name="all_seeds")
    config_sets = parse_sets(args.config_sets, default_all_name="all_configs")
    seed_sets = [(name, vals if vals else all_seeds) for name, vals in seed_sets]
    config_sets = [(name, vals if vals else all_configs) for name, vals in config_sets]
    alignments = [x.strip() for x in args.alignments.split(",") if x.strip()]

    selected_rows: List[Dict[str, object]] = []
    image_rows: List[Dict[str, object]] = []

    for seed_name, seeds in seed_sets:
        for cfg_name, configs in config_sets:
            subset = [r for r in rows if str(r.get("seed", "")) in set(seeds) and str(r.get("config_tag", "")) in set(configs)]
            if not subset:
                print(f"[WARN] empty subset seed_set={seed_name} configs={cfg_name}")
                continue
            analysis_name = f"{cfg_name}__{seed_name}"
            by_image: Dict[str, List[Dict[str, str]]] = {}
            for row in subset:
                by_image.setdefault(str(row.get("image_basename", "")), []).append(row)

            for image, image_rows_subset in sorted(by_image.items()):
                by_cfg: Dict[str, List[Dict[str, str]]] = {}
                for row in image_rows_subset:
                    by_cfg.setdefault(str(row.get("config_tag", "")), []).append(row)

                cfg_stat = {cfg: fmean([fget(r, args.selector_stat) for r in rs]) for cfg, rs in by_cfg.items()}
                selected_cfg = min(cfg_stat, key=lambda c: cfg_stat[c])
                cfg_rows = by_cfg[selected_cfg]
                selected_seed_row = min(cfg_rows, key=lambda r: fget(r, args.selector_stat))
                primary_psnr = metric_key(args.primary_alignment, "psnr")
                best_seed_in_cfg = max(cfg_rows, key=lambda r: fget(r, primary_psnr))
                oracle_all = max(image_rows_subset, key=lambda r: fget(r, primary_psnr))
                global_by_stat = min(image_rows_subset, key=lambda r: fget(r, args.selector_stat))

                choices = [
                    ("selected_config_seed_by_selector", selected_seed_row),
                    ("selected_config_bestofk", best_seed_in_cfg),
                    ("global_run_by_selector", global_by_stat),
                    ("oracle_all_candidates", oracle_all),
                ]
                for method, chosen in choices:
                    for align in alignments:
                        row = dict(chosen)
                        row.update(
                            {
                                "analysis_name": analysis_name,
                                "seed_set_name": seed_name,
                                "config_set_name": cfg_name,
                                "allowed_seeds": ",".join(seeds),
                                "allowed_configs": ",".join(configs),
                                "selection_method": method,
                                "alignment_mode": align,
                                "image_basename": image,
                                "selected_config": chosen.get("config_tag", ""),
                                "selected_seed": chosen.get("seed", ""),
                                "psnr": fget(chosen, metric_key(align, "psnr")),
                                "ssim": fget(chosen, metric_key(align, "ssim")),
                                "lpips": fget(chosen, metric_key(align, "lpips")),
                                "selector_stat_key": args.selector_stat,
                                "selected_stat": fget(chosen, args.selector_stat),
                                "oracle_raw_psnr": fget(oracle_all, primary_psnr),
                                "regret_vs_oracle_raw": fget(oracle_all, primary_psnr) - fget(chosen, primary_psnr),
                            }
                        )
                        selected_rows.append(row)

                diag = {
                    "analysis_name": analysis_name,
                    "seed_set_name": seed_name,
                    "config_set_name": cfg_name,
                    "image_basename": image,
                    "selected_config": selected_seed_row.get("config_tag", ""),
                    "selected_seed": selected_seed_row.get("seed", ""),
                    "selected_raw_psnr": fget(selected_seed_row, primary_psnr),
                    "oracle_config": oracle_all.get("config_tag", ""),
                    "oracle_seed": oracle_all.get("seed", ""),
                    "oracle_raw_psnr": fget(oracle_all, primary_psnr),
                    "regret_vs_oracle_raw": fget(oracle_all, primary_psnr) - fget(selected_seed_row, primary_psnr),
                }
                for cfg, val in sorted(cfg_stat.items()):
                    safe = cfg.replace(".", "p").replace("-", "_")
                    diag[f"config_stat_{safe}"] = val
                image_rows.append(diag)

    out_prefix = Path(args.out_prefix)
    write_csv(Path(str(out_prefix) + "_selected_image_level.csv"), selected_rows)
    write_csv(Path(str(out_prefix) + "_selected_summary.csv"), summarize(selected_rows))
    write_csv(Path(str(out_prefix) + "_image_diagnostics.csv"), image_rows)
    print(f"Trace: {trace}")
    print(f"Wrote prefix: {out_prefix}")
    print("Summary:")
    for row in summarize(selected_rows):
        print(row)


if __name__ == "__main__":
    main()
