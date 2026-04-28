#!/usr/bin/env python3
from __future__ import annotations

import csv
import glob
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path("/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411")

OUT_SUMMARY = ROOT / "np_ffhq_tuning_triage8w_ranked_condition_summary_nopandas.csv"
OUT_RUNS = ROOT / "np_ffhq_tuning_triage8w_merged_run_level_nopandas.csv"
OUT_BY_IMAGE = ROOT / "np_ffhq_tuning_triage8w_by_image_winners_nopandas.csv"

SETTING_RE = re.compile(
    r"score_(?P<score_label>[^_]+)_"
    r"proj_(?P<proj_label>[^_]+)_"
    r"start_(?P<proj_start>\d+)_"
    r"soft_(?P<soft_k>\d+)_"
    r"hard_(?P<hard_k>\d+)"
)


def read_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"[WARN] no rows for {path}")
        return

    # Preserve first-row order, then add any later keys.
    fieldnames = list(rows[0].keys())
    seen = set(fieldnames)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def find_setting_dir(path: str) -> str:
    for part in Path(path).parts:
        if part.startswith("score_") and "_proj_" in part and "_start_" in part:
            return part
    return ""


def label_to_float(label: str) -> float:
    if label == "full":
        return 0.72
    return float(label)


def parse_setting(path: str) -> dict:
    setting_dir = find_setting_dir(path)
    m = SETTING_RE.search(setting_dir)
    if not m:
        return {
            "setting_dir": setting_dir,
            "score_label": "",
            "proj_label": "",
            "score_radius": "",
            "proj_radius": "",
            "proj_start": "",
            "soft_k": "",
            "hard_k": "",
        }

    d = m.groupdict()
    d["setting_dir"] = setting_dir
    d["score_radius"] = label_to_float(d["score_label"])
    d["proj_radius"] = label_to_float(d["proj_label"])
    d["proj_start"] = int(d["proj_start"])
    d["soft_k"] = int(d["soft_k"])
    d["hard_k"] = int(d["hard_k"])
    return d


def fget(row: dict, key: str, default: float = math.nan) -> float:
    try:
        val = row.get(key, "")
        if val == "" or val is None:
            return default
        return float(val)
    except Exception:
        return default


def safe_stdev(vals: list[float]) -> float:
    vals = [v for v in vals if math.isfinite(v)]
    if len(vals) <= 1:
        return 0.0
    return stdev(vals)


def quantile(vals: list[float], q: float) -> float:
    vals = sorted(v for v in vals if math.isfinite(v))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def load_run_rows() -> list[dict]:
    # Use alias run_level.csv only, not timestamped duplicate files.
    files = sorted(glob.glob(str(ROOT / "np_ffhq_tuning_triage8w_*/**/run_level.csv"), recursive=True))
    print(f"Found run_level.csv files: {len(files)}")

    rows: list[dict] = []
    for path in files:
        meta = parse_setting(path)
        for row in read_csv(path):
            row.update(meta)

            # Prefer fields already saved by the updated runner, otherwise use folder metadata.
            row["score_radius"] = row.get("score_radius") or meta["score_radius"]
            row["proj_radius"] = row.get("proj_radius") or meta["proj_radius"]
            row["num_candidates_soft"] = row.get("num_candidates_soft") or meta["soft_k"]
            row["num_candidates_hard"] = row.get("num_candidates_hard") or meta["hard_k"]
            row["source_file"] = path
            rows.append(row)
    return rows


def main() -> None:
    rows = load_run_rows()
    if not rows:
        raise RuntimeError("No run rows found.")

    # Keep main triage alignment.
    rows_rot = [r for r in rows if r.get("alignment_mode") == "rot180"]
    print(f"Total run rows: {len(rows)}")
    print(f"Rot180 run rows: {len(rows_rot)}")

    group_cols = [
        "setting_dir",
        "score_label",
        "proj_label",
        "score_radius",
        "proj_radius",
        "proj_start",
        "soft_k",
        "hard_k",
        "num_candidates_soft",
        "num_candidates_hard",
        "alignment_mode",
        "oversample",
        "measurement_noise_std",
    ]

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows_rot:
        key = tuple(str(r.get(c, "")) for c in group_cols)
        groups[key].append(r)

    summary_rows: list[dict] = []
    for key, gr in groups.items():
        base = {c: key[i] for i, c in enumerate(group_cols)}
        psnrs = [fget(r, "psnr") for r in gr]
        ssims = [fget(r, "ssim") for r in gr]
        runtimes = [fget(r, "runtime_s") for r in gr]
        nfe = [fget(r, "nfe_calls") for r in gr]
        images = sorted(set(r.get("image_basename", "") for r in gr))

        psnrs_f = [v for v in psnrs if math.isfinite(v)]
        ssims_f = [v for v in ssims if math.isfinite(v)]
        runtimes_f = [v for v in runtimes if math.isfinite(v)]
        nfe_f = [v for v in nfe if math.isfinite(v)]

        if not psnrs_f:
            continue

        base.update(
            {
                "n_runs": len(gr),
                "n_images": len(images),
                "psnr_best": max(psnrs_f),
                "psnr_mean": mean(psnrs_f),
                "psnr_median": median(psnrs_f),
                "psnr_min": min(psnrs_f),
                "psnr_std": safe_stdev(psnrs_f),
                "psnr_q25": quantile(psnrs_f, 0.25),
                "psnr_q75": quantile(psnrs_f, 0.75),
                "ssim_best": max(ssims_f) if ssims_f else math.nan,
                "ssim_mean": mean(ssims_f) if ssims_f else math.nan,
                "ssim_median": median(ssims_f) if ssims_f else math.nan,
                "ssim_std": safe_stdev(ssims_f),
                "runtime_s_mean": mean(runtimes_f) if runtimes_f else math.nan,
                "runtime_s_median": median(runtimes_f) if runtimes_f else math.nan,
                "nfe_calls_mean": mean(nfe_f) if nfe_f else math.nan,
            }
        )
        summary_rows.append(base)

    summary_rows.sort(
        key=lambda r: (
            fget(r, "psnr_mean"),
            fget(r, "psnr_median"),
            fget(r, "psnr_min"),
            fget(r, "ssim_mean"),
        ),
        reverse=True,
    )

    # Best config per image.
    best_by_image = {}
    for r in rows_rot:
        img = r.get("image_basename", "")
        psnr = fget(r, "psnr")
        if img not in best_by_image or psnr > fget(best_by_image[img], "psnr"):
            best_by_image[img] = r
    by_image_rows = sorted(best_by_image.values(), key=lambda r: r.get("image_basename", ""))

    write_csv(OUT_SUMMARY, summary_rows)
    write_csv(OUT_RUNS, rows)
    write_csv(OUT_BY_IMAGE, by_image_rows)

    print("\n=== Expected triage configs: 96 ===")
    print("Completed unique configs:", len(summary_rows))
    print("Unique images:", len(set(r.get("image_basename", "") for r in rows_rot)))

    print("\n=== Top 30 configs by PSNR mean ===")
    cols = [
        "score_label",
        "proj_label",
        "proj_start",
        "soft_k",
        "hard_k",
        "n_runs",
        "n_images",
        "psnr_mean",
        "psnr_median",
        "psnr_best",
        "psnr_min",
        "psnr_std",
        "ssim_mean",
        "runtime_s_mean",
    ]
    print(",".join(cols))
    for r in summary_rows[:30]:
        print(",".join(str(r.get(c, "")) for c in cols))

    print("\nSaved:")
    print(" ", OUT_SUMMARY)
    print(" ", OUT_RUNS)
    print(" ", OUT_BY_IMAGE)


if __name__ == "__main__":
    main()
