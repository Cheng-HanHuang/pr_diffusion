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

Q_GRID = [int(x) for x in os.environ.get("Q_GRID", "2 4 6 8").replace(",", " ").split() if x.strip()]

FEATURES = [
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


def candidate_path() -> Path:
    return BASE / f"B20_7A_{ID}_meas{MEAS_SEED}_raw_microscope_candidate_features.csv"


def main() -> None:
    p = candidate_path()
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p).copy()
    df["run_seed"] = df["run_seed"].astype(int)
    df["run_index"] = df["run_index"].astype(int)
    df["final_psnr"] = pd.to_numeric(df["final_psnr"], errors="coerce")
    df["good25"] = (df["final_psnr"] >= 25.0).astype(int)

    features = [f for f in FEATURES if f in df.columns]
    if not features:
        raise RuntimeError(f"No requested features found. Columns={list(df.columns)}")

    rows = []

    for rs, g in df.groupby("run_seed"):
        g = g.sort_values("run_index").copy()

        full_oracle = float(g["final_psnr"].max())
        full_good = int(full_oracle >= 25.0)
        first_good = int(g[g["good25"] == 1]["run_index"].min()) if full_good else -1
        oracle_run = int(g.loc[g["final_psnr"].idxmax(), "run_index"])

        for q in Q_GRID:
            shallow = g[g["run_index"] < q].copy()
            deep = g[g["run_index"] >= q].copy()

            if len(shallow) == 0:
                continue

            row = {
                "run_seed": rs,
                "q": q,
                "full_oracle_psnr": full_oracle,
                "full_good25": full_good,
                "first_good_run": first_good,
                "oracle_run": oracle_run,
                "shallow_oracle_psnr": float(shallow["final_psnr"].max()),
                "shallow_good25": int((shallow["final_psnr"] >= 25.0).any()),
                "deep_oracle_psnr": float(deep["final_psnr"].max()) if len(deep) else float("nan"),
                "deep_good25": int((deep["final_psnr"] >= 25.0).any()) if len(deep) else 0,
            }

            # Seed-level shallow summaries. Smaller rank/loss features are usually better.
            for f in features:
                vals = pd.to_numeric(shallow[f], errors="coerce")
                row[f"{f}_min_first{q}"] = vals.min()
                row[f"{f}_mean_first{q}"] = vals.mean()
                row[f"{f}_last_first{q}"] = vals.iloc[-1] if len(vals) else float("nan")

            rows.append(row)

    out = pd.DataFrame(rows)
    out_path = BASE / f"B20_7B_{ID}_meas{MEAS_SEED}_shallow_seed_predictor_rows.csv"
    out.to_csv(out_path, index=False)
    print("[write]", out_path)

    # Compare good-seed vs bad-seed shallow summaries.
    sep_rows = []
    for q in Q_GRID:
        sub = out[out["q"] == q].copy()
        if sub["full_good25"].nunique() < 2:
            continue

        good = sub[sub["full_good25"] == 1]
        bad = sub[sub["full_good25"] == 0]

        for c in sub.columns:
            if not (
                c.endswith(f"_min_first{q}")
                or c.endswith(f"_mean_first{q}")
                or c.endswith(f"_last_first{q}")
                or c in ["shallow_oracle_psnr"]
            ):
                continue

            sep_rows.append({
                "q": q,
                "feature": c,
                "mean_good_seed": good[c].mean(),
                "mean_bad_seed": bad[c].mean(),
                "good_minus_bad": good[c].mean() - bad[c].mean(),
                "lower_looks_good": int(good[c].mean() < bad[c].mean()),
            })

    sep = pd.DataFrame(sep_rows)
    if len(sep):
        # Sort by absolute separation, but keep sign.
        sep["abs_sep"] = sep["good_minus_bad"].abs()
        sep = sep.sort_values(["q", "abs_sep"], ascending=[True, False])

    sep_path = BASE / f"B20_7B_{ID}_meas{MEAS_SEED}_shallow_seed_predictor_feature_separation.csv"
    sep.to_csv(sep_path, index=False)
    print("[write]", sep_path)

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 260)

    print("\n== B20.7B shallow seed predictor: seed rows ==")
    show = [
        "run_seed", "q", "full_oracle_psnr", "full_good25",
        "first_good_run", "oracle_run",
        "shallow_oracle_psnr", "shallow_good25",
        "deep_oracle_psnr", "deep_good25",
    ]
    print(out[show].sort_values(["q", "run_seed"]).to_string(index=False))

    print("\n== B20.7B shallow seed predictor: strongest feature separations ==")
    if len(sep):
        print(sep.groupby("q").head(20).to_string(index=False))
    else:
        print("No good-vs-bad seed split available.")


if __name__ == "__main__":
    main()
