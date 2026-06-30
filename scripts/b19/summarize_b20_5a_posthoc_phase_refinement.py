#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import glob
import pandas as pd


BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")
K = int(os.environ.get("K", "16"))
PATTERN = str(BASE / f"B20_5A_posthoc_phase_refinement_K{K}_shard*of*.csv")
GOOD_THRESH = float(os.environ.get("GOOD_THRESH", "25.0"))


def main() -> None:
    paths = sorted(glob.glob(PATTERN))
    print("[paths]", len(paths))
    for p in paths:
        print(" ", p)

    if not paths:
        raise RuntimeError(f"No files matched {PATTERN}")

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    print("[rows]", len(df))

    case_cols = ["image_id", "run_seed", "meas_seed", "K"]

    # Original oracle is repeated on every candidate row.
    orig_case = (
        df.groupby(case_cols)
        .agg(
            orig_oracleK_psnr=("orig_oracleK_psnr", "first"),
            orig_oracleK_run=("orig_oracleK_run", "first"),
        )
        .reset_index()
    )

    refined_case = (
        df.groupby(case_cols + ["config"])
        .agg(
            refined_oracleK_psnr=("refined_psnr", "max"),
            refined_best_run=("refined_psnr", lambda x: int(df.loc[x.idxmax(), "candidate_run"])),
            mean_refined_psnr=("refined_psnr", "mean"),
            max_delta_psnr=("delta_psnr", "max"),
            mean_delta_psnr=("delta_psnr", "mean"),
            min_refined_exact_loss=("refined_exact_loss", "min"),
            candidates=("candidate_run", "nunique"),
        )
        .reset_index()
    )

    merged = refined_case.merge(orig_case, on=case_cols, how="left")
    merged["orig_bad25"] = (merged["orig_oracleK_psnr"] < GOOD_THRESH).astype(int)
    merged["refined_good25"] = (merged["refined_oracleK_psnr"] >= GOOD_THRESH).astype(int)
    merged["refined_bad25"] = (merged["refined_oracleK_psnr"] < GOOD_THRESH).astype(int)
    merged["rescued25"] = (
        (merged["orig_oracleK_psnr"] < GOOD_THRESH)
        & (merged["refined_oracleK_psnr"] >= GOOD_THRESH)
    ).astype(int)
    merged["oracle_improvement"] = merged["refined_oracleK_psnr"] - merged["orig_oracleK_psnr"]

    per_case_path = BASE / f"B20_5A_posthoc_phase_refinement_K{K}_per_case_summary.csv"
    merged.to_csv(per_case_path, index=False)
    print("[write]", per_case_path)

    summary = (
        merged.groupby("config")
        .agg(
            cases=("image_id", "count"),
            images=("image_id", "nunique"),
            mean_orig_oracleK=("orig_oracleK_psnr", "mean"),
            mean_refined_oracleK=("refined_oracleK_psnr", "mean"),
            min_refined_oracleK=("refined_oracleK_psnr", "min"),
            max_refined_oracleK=("refined_oracleK_psnr", "max"),
            orig_bad25=("orig_bad25", "sum"),
            refined_bad25=("refined_bad25", "sum"),
            rescued25=("rescued25", "sum"),
            mean_oracle_improvement=("oracle_improvement", "mean"),
            max_oracle_improvement=("oracle_improvement", "max"),
            mean_max_delta_candidate=("max_delta_psnr", "mean"),
        )
        .reset_index()
        .sort_values(["rescued25", "refined_bad25", "mean_refined_oracleK"], ascending=[False, True, False])
    )

    summary_path = BASE / f"B20_5A_posthoc_phase_refinement_K{K}_summary_by_config.csv"
    summary.to_csv(summary_path, index=False)
    print("[write]", summary_path)

    by_image = (
        merged.groupby(["image_id", "config"])
        .agg(
            cases=("image_id", "count"),
            mean_orig_oracleK=("orig_oracleK_psnr", "mean"),
            mean_refined_oracleK=("refined_oracleK_psnr", "mean"),
            min_refined_oracleK=("refined_oracleK_psnr", "min"),
            max_refined_oracleK=("refined_oracleK_psnr", "max"),
            orig_bad25=("orig_bad25", "sum"),
            refined_bad25=("refined_bad25", "sum"),
            rescued25=("rescued25", "sum"),
            mean_oracle_improvement=("oracle_improvement", "mean"),
            max_oracle_improvement=("oracle_improvement", "max"),
        )
        .reset_index()
        .sort_values(["image_id", "rescued25", "mean_refined_oracleK"], ascending=[True, False, False])
    )

    by_image_path = BASE / f"B20_5A_posthoc_phase_refinement_K{K}_by_image_config.csv"
    by_image.to_csv(by_image_path, index=False)
    print("[write]", by_image_path)

    rescued = merged[merged["rescued25"] == 1].copy()
    rescued_path = BASE / f"B20_5A_posthoc_phase_refinement_K{K}_rescued_cases.csv"
    rescued.to_csv(rescued_path, index=False)
    print("[write]", rescued_path)

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 240)

    print("\n== B20.5A posthoc refinement: summary by config ==")
    print(summary.to_string(index=False))

    print("\n== B20.5A posthoc refinement: by image/config ==")
    print(by_image.to_string(index=False))

    print("\n== B20.5A posthoc refinement: rescued cases ==")
    if len(rescued):
        print(
            rescued[[
                "image_id", "run_seed", "meas_seed", "config",
                "orig_oracleK_psnr", "refined_oracleK_psnr",
                "orig_oracleK_run", "refined_best_run",
                "oracle_improvement",
            ]]
            .sort_values(["image_id", "run_seed", "meas_seed", "config"])
            .to_string(index=False)
        )
    else:
        print("none")


if __name__ == "__main__":
    main()
