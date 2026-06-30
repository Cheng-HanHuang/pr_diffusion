#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import math
import re
import numpy as np
import pandas as pd


BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")

PANEL_NAME = "B19_20_ffhq100_seed20260627_from00000to00999_exclude_ffhq25"
IDS_TXT = Path(os.environ.get(
    "IDS_TXT",
    str(BASE / "manifests" / f"{PANEL_NAME}_ids.txt"),
))

RUN_SEED = int(os.environ.get("RUN_SEED", "4400"))
MEAS_SEEDS = [int(x) for x in os.environ.get(
    "MEAS_SEEDS", "5001,5002,5003,5004,5005,5006,5007,5008,5009,5010"
).replace(" ", ",").split(",") if x.strip()]

# Restrict to a few important checkpoints if the raw file has checkpoint/window columns.
FOCUS_CHECKPOINTS = {50, 75, 100, 125}


def exact_path(image_id: str, meas_seed: int) -> Path:
    return BASE / f"B19_16B_daps6S_{image_id}_meas{meas_seed}_runseed{RUN_SEED}_exact_final_loss_selector.csv"


def raw_window_path(image_id: str, meas_seed: int) -> Path:
    return BASE / f"B19_16B_daps_{image_id}_meas{meas_seed}_runseed{RUN_SEED}_6S_rawtraj_window_features.csv"


def psnr_col(df: pd.DataFrame) -> str:
    if "psnr_metrics_json" in df.columns:
        return "psnr_metrics_json"
    if "psnr_recomputed_from_png" in df.columns:
        return "psnr_recomputed_from_png"
    raise KeyError("No PSNR column found")


def find_run_col(df: pd.DataFrame) -> str:
    for c in ["run_index", "run", "sample_index", "candidate_run"]:
        if c in df.columns:
            return c
    raise KeyError(f"No run column found. Columns={list(df.columns)}")


def infer_sort_direction(feature: str) -> str:
    f = feature.lower()
    # For these, smaller should usually be better.
    smaller_terms = [
        "loss", "residual", "norm", "disagreement", "jump", "dist",
        "mse", "error", "correction"
    ]
    if any(t in f for t in smaller_terms):
        return "ascending"
    return "descending"


def auc_rank_score(values: pd.Series, labels: pd.Series) -> float:
    # AUC that larger score predicts positive label.
    x = pd.to_numeric(values, errors="coerce")
    y = pd.to_numeric(labels, errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask].astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = x.rank(method="average")
    sum_pos = float(ranks[y == 1].sum())
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def main() -> None:
    ids = [x.strip() for x in IDS_TXT.read_text().splitlines() if x.strip()]

    rows = []
    missing = []

    for image_id in ids:
        for meas_seed in MEAS_SEEDS:
            ep = exact_path(image_id, meas_seed)
            rp = raw_window_path(image_id, meas_seed)
            if not ep.exists() or not rp.exists():
                missing.append((image_id, meas_seed, ep.exists(), rp.exists()))
                continue

            exact = pd.read_csv(ep).copy()
            run_col_e = find_run_col(exact)
            exact[run_col_e] = exact[run_col_e].astype(int)
            col = psnr_col(exact)
            exact[col] = pd.to_numeric(exact[col])
            exact["exact_operator_loss"] = pd.to_numeric(exact["exact_operator_loss"])

            labels = exact[[run_col_e, col, "exact_operator_loss"]].rename(columns={
                run_col_e: "run_index",
                col: "final_psnr",
            })
            labels["final_good25"] = (labels["final_psnr"] >= 25.0).astype(int)
            labels["final_good29"] = (labels["final_psnr"] >= 29.0).astype(int)
            labels["final_good30"] = (labels["final_psnr"] >= 30.0).astype(int)
            labels["oracle_run6"] = int(labels.loc[labels["final_psnr"].idxmax(), "run_index"])
            labels["exact_selected_run6"] = int(labels.sort_values(["exact_operator_loss", "run_index"]).iloc[0]["run_index"])

            '''raw = pd.read_csv(rp).copy()
            run_col_r = find_run_col(raw)
            raw[run_col_r] = raw[run_col_r].astype(int)
            raw = raw.rename(columns={run_col_r: "run_index"})

            merged = raw.merge(labels, on="run_index", how="left")'''

            raw = pd.read_csv(rp).copy()
            run_col_r = find_run_col(raw)
            raw[run_col_r] = raw[run_col_r].astype(int)
            raw = raw.rename(columns={run_col_r: "run_index"})

            # Avoid pandas suffixes such as final_psnr_x/final_psnr_y if the raw
            # window file already contains old final-label columns.
            stale_label_cols = [
                "final_psnr",
                "final_good25",
                "final_good29",
                "final_good30",
                "exact_operator_loss",
                "oracle_run6",
                "exact_selected_run6",
            ]
            raw = raw.drop(columns=[c for c in stale_label_cols if c in raw.columns], errors="ignore")

            merged = raw.merge(labels, on="run_index", how="left")
            merged["image_id"] = image_id
            merged["meas_seed"] = meas_seed
            merged["run_seed"] = RUN_SEED

            rows.append(merged)

    if not rows:
        raise RuntimeError("No merged rows produced")

    df = pd.concat(rows, ignore_index=True)

    # Optional checkpoint filtering for summary, while still saving all.
    all_path = BASE / f"B20_2B_raw6_global_prefix_anatomy_runseed{RUN_SEED}_all_rows.csv"
    df.to_csv(all_path, index=False)
    print("[write]", all_path)

    work = df.copy()

    # If there is a checkpoint-like column, make a focused version.
    checkpoint_col = None
    for c in ["checkpoint", "step", "timestep", "decision_checkpoint"]:
        if c in work.columns:
            checkpoint_col = c
            break

    if checkpoint_col is not None:
        work_focus = work[work[checkpoint_col].isin(FOCUS_CHECKPOINTS)].copy()
        if len(work_focus):
            work = work_focus

    # Candidate numeric feature columns.
    exclude = {
        "image_id", "meas_seed", "run_seed", "run_index",
        "final_psnr", "final_good25", "final_good29", "final_good30",
        "oracle_run6", "exact_selected_run6",
        "sample_path", "source_file",
    }
    num_cols = []
    for c in work.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(work[c]):
            # Avoid exact final columns as predictors except raw features.
            if c in {"exact_operator_loss"}:
                continue
            if work[c].notna().sum() >= 50:
                num_cols.append(c)

    feature_rows = []
    for feat in num_cols:
        direction = infer_sort_direction(feat)

        # Convert to score where larger means predicted-good.
        vals = pd.to_numeric(work[feat], errors="coerce")
        score = -vals if direction == "ascending" else vals

        auc25 = auc_rank_score(score, work["final_good25"])
        auc29 = auc_rank_score(score, work["final_good29"])
        auc30 = auc_rank_score(score, work["final_good30"])

        feature_rows.append({
            "feature": feat,
            "direction_for_good": direction,
            "rows": int(vals.notna().sum()),
            "auc_good25": auc25,
            "auc_good29": auc29,
            "auc_good30": auc30,
            "mean_feature_good25": float(vals[work["final_good25"] == 1].mean()) if (work["final_good25"] == 1).any() else math.nan,
            "mean_feature_bad25": float(vals[work["final_good25"] == 0].mean()) if (work["final_good25"] == 0).any() else math.nan,
        })

    feat_df = pd.DataFrame(feature_rows)
    if not feat_df.empty:
        feat_df["auc25_abs_from_half"] = (feat_df["auc_good25"] - 0.5).abs()
        feat_df = feat_df.sort_values(["auc25_abs_from_half", "auc_good29"], ascending=[False, False])

    feat_path = BASE / f"B20_2B_raw6_global_prefix_feature_auc_runseed{RUN_SEED}.csv"
    feat_df.to_csv(feat_path, index=False)
    print("[write]", feat_path)

    # Per-image difficulty summary from final labels.
    final_unique = df[["image_id", "meas_seed", "run_index", "final_psnr", "final_good25"]].drop_duplicates()
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

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 240)

    print("\n== B20.2B raw6 global: top feature AUCs for final_good25 ==")
    if len(feat_df):
        print(
            feat_df[[
                "feature", "direction_for_good", "rows",
                "auc_good25", "auc_good29", "auc_good30",
                "mean_feature_good25", "mean_feature_bad25",
            ]]
            .head(40)
            .to_string(index=False)
        )
    else:
        print("no numeric features found")

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

    if missing:
        print(f"\n[missing exact/raw pairs] {len(missing)}")
        print("first missing:", missing[:20])


if __name__ == "__main__":
    main()
