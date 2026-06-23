#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


FEATURES = [
    "correction_norm",
    "x0hat_x0y_disagreement",
    "x0y_full_residual_normed",
    "x0y_lowfreq_residual_normed",
]


def image_label(x) -> str:
    try:
        return f"{int(x):05d}"
    except Exception:
        return str(x)


def summarize(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in selected.groupby("selection_method"):
        ps = g["selected_psnr"].astype(float)
        worst = g.loc[ps.idxmin()]
        rows.append({
            "selection_method": method,
            "n_images": len(g),
            "psnr_mean": ps.mean(),
            "psnr_min": ps.min(),
            "psnr_median": ps.median(),
            "bad25": int((ps < 25).sum()),
            "bad20": int((ps < 20).sum()),
            "worst_image": worst["image_id"],
            "worst_psnr": float(worst["selected_psnr"]),
            "worst_run_index": int(worst["selected_run_index"]),
        })
    return pd.DataFrame(rows).sort_values(["bad25", "bad20", "psnr_mean"], ascending=[True, True, False])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory_outdir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--taus", default="0.5,0.6,0.7,0.75,0.8,0.85")
    args = ap.parse_args()

    src = Path(args.trajectory_outdir)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    run = pd.read_csv(src / "run_level_summary.csv")
    step = pd.read_csv(src / "trajectory_step_metrics.csv")

    run["image_id"] = run["image_id"].map(image_label)
    step["image_id"] = step["image_id"].map(image_label)

    max_step = int(step["step"].max())
    n_steps = max_step + 1
    taus = [float(x) for x in args.taus.split(",") if x.strip()]

    rows = []
    for tau in taus:
        target = int(round(tau * n_steps))
        target = max(0, min(max_step, target))
        for (image_id, run_index), g in step.groupby(["image_id", "run_index"]):
            idx = (g["step"].astype(int) - target).abs().idxmin()
            r = g.loc[idx]
            row = {
                "image_id": image_id,
                "run_index": int(run_index),
                "tau": tau,
                "target_step": target,
                "actual_step": int(r["step"]),
            }
            for feat in FEATURES:
                row[feat] = float(r[feat])
            rows.append(row)

    feat = pd.DataFrame(rows)

    # Rank each feature within image and tau. Lower is better.
    rank_cols = []
    for feat_name in FEATURES:
        col = f"{feat_name}_rank"
        feat[col] = feat.groupby(["image_id", "tau"])[feat_name].rank(method="average", ascending=True)
        rank_cols.append(col)

    feat["mean_rank_all_features"] = feat[rank_cols].mean(axis=1)
    feat["mean_rank_correction_only"] = feat["correction_norm_rank"]
    feat["mean_rank_residual_only"] = feat[
        ["x0y_full_residual_normed_rank", "x0y_lowfreq_residual_normed_rank"]
    ].mean(axis=1)
    feat["mean_rank_correction_plus_residual"] = feat[
        ["correction_norm_rank", "x0y_full_residual_normed_rank", "x0y_lowfreq_residual_normed_rank"]
    ].mean(axis=1)

    agg = feat.groupby(["image_id", "run_index"])[
        [
            "mean_rank_all_features",
            "mean_rank_correction_only",
            "mean_rank_residual_only",
            "mean_rank_correction_plus_residual",
        ]
    ].mean().reset_index()

    run_small = run[["image_id", "run_index", "final_psnr"]]
    agg = agg.merge(run_small, on=["image_id", "run_index"], how="left")
    agg.to_csv(out / "tau_window_run_scores.csv", index=False)

    methods = {
        "tau_window_all_features": "mean_rank_all_features",
        "tau_window_correction_only": "mean_rank_correction_only",
        "tau_window_residual_only": "mean_rank_residual_only",
        "tau_window_correction_plus_residual": "mean_rank_correction_plus_residual",
    }

    selected_rows = []
    for image_id, g in agg.groupby("image_id"):
        oracle = g.loc[g["final_psnr"].idxmax()]
        selected_rows.append({
            "image_id": image_id,
            "selection_method": "oracle_best_psnr_diagnostic",
            "selected_run_index": int(oracle["run_index"]),
            "selected_psnr": float(oracle["final_psnr"]),
        })
        for method, score_col in methods.items():
            ch = g.loc[g[score_col].idxmin()]
            selected_rows.append({
                "image_id": image_id,
                "selection_method": method,
                "selected_run_index": int(ch["run_index"]),
                "selected_psnr": float(ch["final_psnr"]),
                "score": float(ch[score_col]),
            })

    selected = pd.DataFrame(selected_rows)
    selected.to_csv(out / "selected_image_level.csv", index=False)
    summary = summarize(selected)
    summary.to_csv(out / "selected_summary.csv", index=False)

    print("taus:", taus)
    print("\nSELECTED SUMMARY")
    print(summary.to_string(index=False))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
