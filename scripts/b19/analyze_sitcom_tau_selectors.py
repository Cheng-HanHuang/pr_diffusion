#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def image_label(x) -> str:
    try:
        return f"{int(x):05d}"
    except Exception:
        return str(x)


def summarize_selected(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby("selection_method"):
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


def choose(g: pd.DataFrame, method: str) -> pd.Series:
    if method == "oracle_best_psnr_diagnostic":
        return g.loc[g["final_psnr"].astype(float).idxmax()]
    if method == "min_correction_tau":
        return g.loc[g["correction_norm_tau"].astype(float).idxmin()]
    if method == "min_x0hat_x0y_disagreement_tau":
        return g.loc[g["x0hat_x0y_disagreement_tau"].astype(float).idxmin()]
    if method == "min_x0y_full_residual_tau":
        return g.loc[g["x0y_full_residual_normed_tau"].astype(float).idxmin()]
    if method == "min_x0y_lowfreq_residual_tau":
        return g.loc[g["x0y_lowfreq_residual_normed_tau"].astype(float).idxmin()]
    if method == "max_x0y_full_residual_tau":
        return g.loc[g["x0y_full_residual_normed_tau"].astype(float).idxmax()]
    raise ValueError(method)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory_outdir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tau", type=float, default=0.8)
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
    target = int(round(args.tau * n_steps))
    target = max(0, min(max_step, target))

    # Pick the row closest to tau for each image/run.
    tau_rows = []
    for (image_id, run_index), g in step.groupby(["image_id", "run_index"]):
        idx = (g["step"].astype(int) - target).abs().idxmin()
        r = g.loc[idx]
        tau_rows.append({
            "image_id": image_id,
            "run_index": int(run_index),
            "tau": args.tau,
            "target_step": target,
            "actual_step": int(r["step"]),
            "sigma_tau": float(r["sigma"]),
            "correction_norm_tau": float(r["correction_norm"]),
            "x0hat_x0y_disagreement_tau": float(r["x0hat_x0y_disagreement"]),
            "x0y_full_residual_normed_tau": float(r["x0y_full_residual_normed"]),
            "x0y_lowfreq_residual_normed_tau": float(r["x0y_lowfreq_residual_normed"]),
            "x0hat_full_residual_normed_tau": float(r["x0hat_full_residual_normed"]),
            "x0hat_lowfreq_residual_normed_tau": float(r["x0hat_lowfreq_residual_normed"]),
        })
    tau_df = pd.DataFrame(tau_rows)

    feat = tau_df.merge(
        run[["image_id", "run_index", "final_psnr"]],
        on=["image_id", "run_index"],
        how="left",
    )
    feat["bad25"] = feat["final_psnr"] < 25
    feat["bad20"] = feat["final_psnr"] < 20
    feat.to_csv(out / "run_features_tau.csv", index=False)

    methods = [
        "oracle_best_psnr_diagnostic",
        "min_correction_tau",
        "min_x0hat_x0y_disagreement_tau",
        "min_x0y_full_residual_tau",
        "min_x0y_lowfreq_residual_tau",
        "max_x0y_full_residual_tau",
    ]

    selected_rows = []
    for image_id, g in feat.groupby("image_id"):
        for method in methods:
            ch = choose(g, method)
            selected_rows.append({
                "image_id": image_id,
                "selection_method": method,
                "selected_run_index": int(ch["run_index"]),
                "selected_psnr": float(ch["final_psnr"]),
                "selected_bad25": bool(ch["final_psnr"] < 25),
                "selected_bad20": bool(ch["final_psnr"] < 20),
                "correction_norm_tau": float(ch["correction_norm_tau"]),
                "x0hat_x0y_disagreement_tau": float(ch["x0hat_x0y_disagreement_tau"]),
                "x0y_full_residual_normed_tau": float(ch["x0y_full_residual_normed_tau"]),
                "x0y_lowfreq_residual_normed_tau": float(ch["x0y_lowfreq_residual_normed_tau"]),
            })

    selected = pd.DataFrame(selected_rows)
    selected.to_csv(out / "selected_image_level.csv", index=False)

    summary = summarize_selected(selected)
    summary.to_csv(out / "selected_summary.csv", index=False)

    print("wrote", out)
    print("\nRUN FEATURES AT TAU")
    print(feat.sort_values(["image_id", "run_index"]).to_string(index=False))
    print("\nSELECTED SUMMARY")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
