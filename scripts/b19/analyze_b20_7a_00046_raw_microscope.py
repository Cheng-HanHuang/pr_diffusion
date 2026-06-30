#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import pandas as pd


BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")

ID = os.environ.get("ID", "00046").zfill(5)
MEAS_SEED = int(os.environ.get("MEAS_SEED", "5001"))
NUM_RUNS = int(os.environ.get("NUM_RUNS", "16"))
RUN_SEEDS = [
    int(x)
    for x in os.environ.get("RUN_SEEDS", "4400 4500 4600 4700 4900 5500")
    .replace(",", " ")
    .split()
    if x.strip()
]

DESIRED_FEATURES = [
    "correction_rms_rank_mean",
    "exact_loss_x0y_rank_mean",
    "sqrt_loss_x0y_over_y_norm_rank_mean",
    "exact_loss_x0hat_rank_mean",
    "sqrt_loss_x0hat_over_y_norm_rank_mean",
    "x0y_jump_rms_rank_mean",
    "exact_loss_x0y_rank_last",
    "correction_rms_rank_last",
    "exact_loss_x0hat_rank_last",
    "exact_loss_x0y_mean",
    "correction_rms_mean",
    "x0y_jump_rms_mean",
    "exact_loss_x0hat_mean",
    "x0hat_jump_rms_mean",
]


def exact_path(run_seed: int) -> Path:
    return BASE / f"B20_7A_daps{NUM_RUNS}S_{ID}_meas{MEAS_SEED}_runseed{run_seed}_exact_final_loss_selector.csv"


def window_path(run_seed: int) -> Path:
    return BASE / f"B20_7A_daps_{ID}_meas{MEAS_SEED}_runseed{run_seed}_{NUM_RUNS}S_rawtraj_window_features.csv"


def step_path(run_seed: int) -> Path:
    return BASE / f"B20_7A_daps_{ID}_meas{MEAS_SEED}_runseed{run_seed}_{NUM_RUNS}S_rawtraj_step_features.csv"


def pick_run_col(df: pd.DataFrame) -> str:
    for c in ["run_index", "run", "sample_index", "candidate_run", "run_id"]:
        if c in df.columns:
            return c
    raise KeyError(f"No run column found. Available columns:\n{list(df.columns)}")


def pick_psnr_col(df: pd.DataFrame) -> str:
    for c in ["psnr_metrics_json", "psnr_recomputed_from_png", "final_psnr", "psnr"]:
        if c in df.columns:
            return c
    raise KeyError(f"No PSNR column found. Available columns:\n{list(df.columns)}")


def collapse_window(win: pd.DataFrame) -> pd.DataFrame:
    """Collapse window rows to one feature row per candidate."""
    run_col = pick_run_col(win)
    win = win.rename(columns={run_col: "run_index"}).copy()
    win["run_index"] = win["run_index"].astype(int)

    # Drop label-ish columns if they exist, because labels should come from exact CSV.
    labelish = [
        "final_psnr",
        "psnr",
        "psnr_metrics_json",
        "psnr_recomputed_from_png",
        "good25",
        "is_good25",
        "final_good25",
        "final_good29",
        "final_good30",
        "exact_operator_loss",
        "selected",
    ]
    win = win.drop(columns=[c for c in labelish if c in win.columns], errors="ignore")

    numeric_cols = [
        c for c in win.columns
        if c != "run_index" and pd.api.types.is_numeric_dtype(win[c])
    ]

    if not numeric_cols:
        print("[warn] no numeric window feature columns found")
        print("[window columns]", list(win.columns))
        return win[["run_index"]].drop_duplicates()

    return win.groupby("run_index", as_index=False)[numeric_cols].mean()


def main() -> None:
    rows = []
    missing = []

    for run_seed in RUN_SEEDS:
        ep = exact_path(run_seed)
        wp = window_path(run_seed)

        if not ep.exists():
            missing.append(("exact", run_seed, str(ep)))
            continue
        if not wp.exists():
            missing.append(("window", run_seed, str(wp)))
            continue

        exact = pd.read_csv(ep).copy()
        erun = pick_run_col(exact)
        psnr = pick_psnr_col(exact)

        exact = exact.rename(columns={erun: "run_index", psnr: "final_psnr"}).copy()
        exact["run_index"] = exact["run_index"].astype(int)
        exact["final_psnr"] = pd.to_numeric(exact["final_psnr"], errors="coerce")
        exact = exact[exact["run_index"] < NUM_RUNS].copy()

        labels = exact[["run_index", "final_psnr"]].copy()
        if "exact_operator_loss" in exact.columns:
            labels["exact_operator_loss"] = exact["exact_operator_loss"]

        labels["run_seed"] = run_seed
        labels["good25"] = (labels["final_psnr"] >= 25.0).astype(int)

        oracle_idx = labels["final_psnr"].idxmax()
        oracle_run = int(labels.loc[oracle_idx, "run_index"])
        oracle_psnr = float(labels.loc[oracle_idx, "final_psnr"])

        good = labels[labels["good25"] == 1].sort_values("run_index")
        first_good_run = int(good.iloc[0]["run_index"]) if len(good) else -1

        win = pd.read_csv(wp).copy()
        collapsed = collapse_window(win)

        merged = labels.merge(collapsed, on="run_index", how="left")
        merged["oracle_run"] = oracle_run
        merged["oracle_psnr"] = oracle_psnr
        merged["first_good_run"] = first_good_run
        merged["is_oracle"] = (merged["run_index"] == oracle_run).astype(int)
        rows.append(merged)

    if missing:
        print("[missing files]")
        for item in missing:
            print(item)

    if not rows:
        raise RuntimeError("No rows produced. Check exact/window file paths.")

    out = pd.concat(rows, ignore_index=True)

    # Keep only actually available desired features.
    feature_cols = [c for c in DESIRED_FEATURES if c in out.columns]

    # Also include all available rank_mean/rank_last features as backup.
    auto_feature_cols = [
        c for c in out.columns
        if (
            c.endswith("_rank_mean")
            or c.endswith("_rank_last")
            or c.endswith("_mean")
            or c.endswith("_last")
        )
        and c not in {
            "final_psnr",
            "oracle_psnr",
            "run_seed",
            "run_index",
            "oracle_run",
            "first_good_run",
            "good25",
            "is_oracle",
        }
    ]

    # Preserve order: desired first, then extras.
    feature_cols = list(dict.fromkeys(feature_cols + auto_feature_cols))

    out_path = BASE / f"B20_7A_{ID}_meas{MEAS_SEED}_raw_microscope_candidate_features.csv"
    out.to_csv(out_path, index=False)
    print("[write]", out_path)

    by_seed = (
        out.groupby("run_seed")
        .agg(
            candidates=("run_index", "count"),
            oracle_psnr=("oracle_psnr", "first"),
            oracle_run=("oracle_run", "first"),
            first_good_run=("first_good_run", "first"),
            good25_runs=("good25", "sum"),
            mean_final_psnr=("final_psnr", "mean"),
            min_final_psnr=("final_psnr", "min"),
            max_final_psnr=("final_psnr", "max"),
        )
        .reset_index()
        .sort_values("oracle_psnr", ascending=False)
    )

    by_seed_path = BASE / f"B20_7A_{ID}_meas{MEAS_SEED}_raw_microscope_by_seed.csv"
    by_seed.to_csv(by_seed_path, index=False)
    print("[write]", by_seed_path)

    oracle_rows = out[out["is_oracle"] == 1].copy()
    oracle_path = BASE / f"B20_7A_{ID}_meas{MEAS_SEED}_raw_microscope_oracle_rows.csv"
    oracle_rows.to_csv(oracle_path, index=False)
    print("[write]", oracle_path)

    if feature_cols:
        good_vs_bad = (
            out.groupby("good25")[feature_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
    else:
        good_vs_bad = pd.DataFrame()

    good_vs_bad_path = BASE / f"B20_7A_{ID}_meas{MEAS_SEED}_raw_microscope_good_vs_bad_features.csv"
    good_vs_bad.to_csv(good_vs_bad_path, index=False)
    print("[write]", good_vs_bad_path)

    # Simple feature separation table: difference good - bad.
    sep_rows = []
    if feature_cols and out["good25"].nunique() == 2:
        good_df = out[out["good25"] == 1]
        bad_df = out[out["good25"] == 0]
        for c in feature_cols:
            if c not in out.columns:
                continue
            sep_rows.append({
                "feature": c,
                "mean_good25": good_df[c].mean(),
                "mean_bad25": bad_df[c].mean(),
                "good_minus_bad": good_df[c].mean() - bad_df[c].mean(),
                "lower_looks_good": int(good_df[c].mean() < bad_df[c].mean()),
            })

    sep = pd.DataFrame(sep_rows)
    if len(sep):
        sep = sep.sort_values("good_minus_bad")
    sep_path = BASE / f"B20_7A_{ID}_meas{MEAS_SEED}_raw_microscope_feature_separation.csv"
    sep.to_csv(sep_path, index=False)
    print("[write]", sep_path)

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 260)

    print("\n== B20.7A 00046 raw microscope: by seed ==")
    print(by_seed.to_string(index=False))

    print("\n== B20.7A 00046 raw microscope: oracle rows ==")
    show_cols = [
        "run_seed",
        "run_index",
        "final_psnr",
        "good25",
        "oracle_run",
        "first_good_run",
    ] + feature_cols[:16]
    show_cols = [c for c in show_cols if c in oracle_rows.columns]
    print(oracle_rows[show_cols].sort_values("run_seed").to_string(index=False))

    print("\n== B20.7A 00046 raw microscope: good vs bad feature means ==")
    if len(good_vs_bad):
        show_cols = ["good25"] + feature_cols[:24]
        show_cols = [c for c in show_cols if c in good_vs_bad.columns]
        print(good_vs_bad[show_cols].to_string(index=False))
    else:
        print("No feature columns found or no good/bad split.")
        print("Available columns:")
        print(list(out.columns))

    print("\n== B20.7A 00046 raw microscope: feature separation good-bad ==")
    if len(sep):
        print(sep.head(40).to_string(index=False))
    else:
        print("No separation table produced.")


if __name__ == "__main__":
    main()
