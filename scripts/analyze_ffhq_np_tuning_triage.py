#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import pandas as pd


ROOT = Path(
    "/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411"
)

OUT_SUMMARY = ROOT / "np_ffhq_tuning_triage8w_ranked_condition_summary.csv"
OUT_RUNS = ROOT / "np_ffhq_tuning_triage8w_merged_run_level.csv"
OUT_BY_IMAGE = ROOT / "np_ffhq_tuning_triage8w_by_image_winners.csv"


SETTING_RE = re.compile(
    r"score_(?P<score_label>[^_]+)_"
    r"proj_(?P<proj_label>[^_]+)_"
    r"start_(?P<proj_start>\d+)_"
    r"soft_(?P<soft_k>\d+)_"
    r"hard_(?P<hard_k>\d+)"
)


def find_setting_dir(path: str) -> str:
    for part in Path(path).parts:
        if part.startswith("score_") and "_proj_" in part and "_start_" in part:
            return part
    return ""


def parse_setting(path: str) -> dict:
    setting_dir = find_setting_dir(path)
    m = SETTING_RE.search(setting_dir)
    if not m:
        return {
            "setting_dir": setting_dir,
            "score_label": None,
            "proj_label": None,
            "proj_start": None,
            "soft_k": None,
            "hard_k": None,
        }
    d = m.groupdict()
    d["setting_dir"] = setting_dir
    d["proj_start"] = int(d["proj_start"])
    d["soft_k"] = int(d["soft_k"])
    d["hard_k"] = int(d["hard_k"])
    return d


def label_to_float(x):
    if pd.isna(x):
        return float("nan")
    if str(x) == "full":
        return 0.72
    return float(x)


def load_csvs(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / pattern), recursive=True))
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARN] could not read {f}: {e}")
            continue
        meta = parse_setting(f)
        for k, v in meta.items():
            df[k] = v
        df["source_file"] = f
        rows.append(df)

    if not rows:
        raise RuntimeError(f"No files found for pattern: {pattern}")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    # Use alias CSVs only, not timestamped duplicates.
    cond = load_csvs("np_ffhq_tuning_triage8w_*/**/condition_level_summary.csv")
    runs = load_csvs("np_ffhq_tuning_triage8w_*/**/run_level.csv")

    # Prefer CSV fields if present; otherwise use folder labels.
    for df in (cond, runs):
        if "score_radius" not in df.columns:
            df["score_radius"] = df["score_label"].map(label_to_float)
        if "proj_radius" not in df.columns:
            df["proj_radius"] = df["proj_label"].map(label_to_float)
        if "num_candidates_soft" not in df.columns:
            df["num_candidates_soft"] = df["soft_k"]
        if "num_candidates_hard" not in df.columns:
            df["num_candidates_hard"] = df["hard_k"]

    # Keep the main triage alignment.
    cond_rot = cond[cond["alignment_mode"].eq("rot180")].copy()
    runs_rot = runs[runs["alignment_mode"].eq("rot180")].copy()

    # Add stability metrics from run-level data.
    group_cols = [
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

    run_stats = (
        runs_rot.groupby(group_cols, dropna=False)
        .agg(
            n_images=("image_basename", "nunique"),
            n_runs_check=("psnr", "size"),
            psnr_std=("psnr", "std"),
            psnr_min=("psnr", "min"),
            psnr_q25=("psnr", lambda x: x.quantile(0.25)),
            psnr_q75=("psnr", lambda x: x.quantile(0.75)),
            ssim_std=("ssim", "std"),
        )
        .reset_index()
    )

    # The condition CSV already has mean/best/median/runtime.
    # Drop repeated rows if aliases accidentally duplicate.
    cond_cols = [
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
        "n_runs",
        "psnr_best",
        "psnr_mean",
        "psnr_median",
        "ssim_best",
        "ssim_mean",
        "ssim_median",
        "runtime_s_mean",
        "runtime_s_median",
        "nfe_calls_mean",
        "psnr_below_threshold_count",
        "source_file",
    ]
    cond_cols = [c for c in cond_cols if c in cond_rot.columns]
    cond_small = cond_rot[cond_cols].drop_duplicates()

    ranked = cond_small.merge(
        run_stats,
        on=[
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
        ],
        how="left",
    )

    # A simple conservative rank: high mean, high median, high worst-case.
    ranked = ranked.sort_values(
        ["psnr_mean", "psnr_median", "psnr_min", "ssim_mean"],
        ascending=[False, False, False, False],
    )

    ranked.to_csv(OUT_SUMMARY, index=False)
    runs.to_csv(OUT_RUNS, index=False)

    # Best config per image, useful to see if winner is broad or image-specific.
    by_image = (
        runs_rot.sort_values("psnr", ascending=False)
        .groupby("image_basename", as_index=False)
        .first()
    )
    by_image.to_csv(OUT_BY_IMAGE, index=False)

    print("\n=== Completed config count ===")
    print("condition rows:", len(cond_rot))
    print("unique setting dirs:", ranked["setting_dir"].nunique() if "setting_dir" in ranked else "n/a")
    print("run rows:", len(runs_rot))
    print("unique images:", runs_rot["image_basename"].nunique())

    print("\n=== Top 30 configs by PSNR mean ===")
    display_cols = [
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
    display_cols = [c for c in display_cols if c in ranked.columns]
    print(ranked[display_cols].head(30).to_string(index=False))

    print("\nSaved:")
    print(" ", OUT_SUMMARY)
    print(" ", OUT_RUNS)
    print(" ", OUT_BY_IMAGE)


if __name__ == "__main__":
    main()
