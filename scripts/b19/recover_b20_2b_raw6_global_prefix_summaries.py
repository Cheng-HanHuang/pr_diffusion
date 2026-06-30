#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import pandas as pd


BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")
RUN_SEED = int(os.environ.get("RUN_SEED", "4400"))

ALL_ROWS = BASE / f"B20_2B_raw6_global_prefix_anatomy_runseed{RUN_SEED}_all_rows.csv"
FEATURE_AUC = BASE / f"B20_2B_raw6_global_prefix_feature_auc_runseed{RUN_SEED}.csv"


def pick_col(df: pd.DataFrame, names: list[str]) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"None of these columns found: {names}\nAvailable columns:\n{list(df.columns)}")


def main() -> None:
    df = pd.read_csv(ALL_ROWS)
    print("[read]", ALL_ROWS)
    print("[rows]", len(df))
    print("[columns]", len(df.columns))

    # Handle pandas merge suffixes robustly.
    final_psnr_col = pick_col(df, [
        "final_psnr",
        "final_psnr_y",
        "psnr_metrics_json",
        "psnr_metrics_json_y",
        "psnr_recomputed_from_png",
        "psnr_recomputed_from_png_y",
    ])

    run_col = pick_col(df, ["run_index", "run", "sample_index", "candidate_run"])
    image_col = pick_col(df, ["image_id"])
    meas_col = pick_col(df, ["meas_seed"])

    df["_final_psnr"] = pd.to_numeric(df[final_psnr_col], errors="coerce")
    df["_final_good25"] = (df["_final_psnr"] >= 25.0).astype(int)

    final_unique = (
        df[[image_col, meas_col, run_col, "_final_psnr", "_final_good25"]]
        .drop_duplicates()
        .rename(columns={
            image_col: "image_id",
            meas_col: "meas_seed",
            run_col: "run_index",
            "_final_psnr": "final_psnr",
            "_final_good25": "final_good25",
        })
    )

    by_image = (
        final_unique.groupby("image_id")
        .agg(
            cases=("meas_seed", "nunique"),
            candidates=("run_index", "count"),
            mean_final_psnr=("final_psnr", "mean"),
            max_final_psnr=("final_psnr", "max"),
            min_final_psnr=("final_psnr", "min"),
            good25_runs=("final_good25", "sum"),
        )
        .reset_index()
    )
    by_image["good25_rate"] = by_image["good25_runs"] / by_image["candidates"]
    by_image = by_image.sort_values(["good25_rate", "max_final_psnr"], ascending=[True, True])

    by_image_path = BASE / f"B20_2B_raw6_global_prefix_by_image_runseed{RUN_SEED}.csv"
    by_image.to_csv(by_image_path, index=False)
    print("[write]", by_image_path)

    feat = pd.read_csv(FEATURE_AUC)
    print("[read]", FEATURE_AUC)

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 240)

    print("\n== B20.2B raw6 global: top feature AUCs for final_good25 ==")
    if len(feat):
        cols = [
            "feature", "direction_for_good", "rows",
            "auc_good25", "auc_good29", "auc_good30",
            "mean_feature_good25", "mean_feature_bad25",
        ]
        cols = [c for c in cols if c in feat.columns]
        print(feat[cols].head(40).to_string(index=False))
    else:
        print("empty feature AUC file")

    print("\n== B20.2B raw6 global: hardest images by good25_rate in raw6 ==")
    print(
        by_image[[
            "image_id", "cases", "candidates",
            "good25_runs", "good25_rate",
            "max_final_psnr", "mean_final_psnr", "min_final_psnr",
        ]]
        .head(40)
        .to_string(index=False)
    )

    print("\n== Debug: final PSNR column used ==")
    print(final_psnr_col)


if __name__ == "__main__":
    main()
