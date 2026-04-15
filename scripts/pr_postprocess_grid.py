#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import statistics
from typing import Dict, List, Tuple


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def aggregate_image_level(run_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = {}
    for r in run_rows:
        key = (r["mode"], r["setting"], r["image_basename"], r["method"])
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, object]] = []
    for (mode, setting, image_name, method), grp in groups.items():
        psnrs = [float(r["psnr"]) for r in grp]
        mags = [float(r["mag_l2"]) for r in grp]
        low_mags = [float(r["lowfreq_mag_l2"]) for r in grp]
        runtimes = [float(r["runtime_s"]) for r in grp]
        out.append(
            {
                "mode": mode,
                "setting": setting,
                "image_basename": image_name,
                "method": method,
                "num_restarts": len(grp),
                "mean_psnr": statistics.fmean(psnrs),
                "median_psnr": statistics.median(psnrs),
                "max_psnr": max(psnrs),
                "mean_mag_l2": statistics.fmean(mags),
                "mean_lowfreq_mag_l2": statistics.fmean(low_mags),
                "mean_runtime_s": statistics.fmean(runtimes),
                "total_runtime_s": sum(runtimes),
            }
        )
    out.sort(key=lambda x: (x["mode"], x["setting"], x["image_basename"], x["method"]))
    return out


def aggregate_split_level(image_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
    for r in image_rows:
        key = (str(r["mode"]), str(r["setting"]), str(r["method"]))
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, object]] = []
    for (mode, setting, method), grp in groups.items():
        out.append(
            {
                "mode": mode,
                "setting": setting,
                "method": method,
                "num_images": len(grp),
                "avg_image_mean_psnr": statistics.fmean(float(r["mean_psnr"]) for r in grp),
                "avg_image_median_psnr": statistics.fmean(float(r["median_psnr"]) for r in grp),
                "avg_image_max_psnr": statistics.fmean(float(r["max_psnr"]) for r in grp),
                "avg_image_mean_mag_l2": statistics.fmean(float(r["mean_mag_l2"]) for r in grp),
                "avg_image_mean_lowfreq_mag_l2": statistics.fmean(float(r["mean_lowfreq_mag_l2"]) for r in grp),
                "avg_image_mean_runtime_s": statistics.fmean(float(r["mean_runtime_s"]) for r in grp),
                "total_runtime_s": sum(float(r["total_runtime_s"]) for r in grp),
            }
        )
    out.sort(key=lambda x: (x["mode"], x["setting"], x["method"]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Postprocess neurips_grid_experiments run_level CSV into image/split summaries.")
    p.add_argument("--run_dir", required=True, help="Directory produced by neurips_grid_experiments.py containing run_level.csv")
    args = p.parse_args()

    run_level_path = os.path.join(args.run_dir, "run_level.csv")
    run_rows = read_csv(run_level_path)
    if not run_rows:
        raise ValueError(f"No rows found in {run_level_path}")

    image_rows = aggregate_image_level(run_rows)
    split_rows = aggregate_split_level(image_rows)

    write_csv(os.path.join(args.run_dir, "image_level.csv"), image_rows)
    write_csv(os.path.join(args.run_dir, "split_summary.csv"), split_rows)
    print(f"Wrote {len(image_rows)} image rows and {len(split_rows)} split-summary rows under {args.run_dir}")


if __name__ == "__main__":
    main()
