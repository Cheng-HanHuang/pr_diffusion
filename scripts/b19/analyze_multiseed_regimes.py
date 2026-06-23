#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def image_label(x):
    try:
        return f"{int(x):05d}"
    except Exception:
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--seeds", default="19050,19051,19052,19053")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--selector", default="tau_window_all_features")
    args = ap.parse_args()

    base = Path(args.base)
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []

    for seed in seeds:
        if seed == "19050":
            out = base / "B19_0_sitcom_special6_noise005_npsitcomroot"
        else:
            out = base / f"B19_0_sitcom_special6_noise005_seed{seed}_npsitcomroot"

        run_path = out / "run_level_summary.csv"
        sel_path = out / "tau_window_selector_analysis" / "selected_image_level.csv"

        run = pd.read_csv(run_path)
        run["image_id"] = run["image_id"].map(image_label)

        sel = pd.read_csv(sel_path)
        sel["image_id"] = sel["image_id"].map(image_label)

        chosen = sel[sel["selection_method"] == args.selector].copy()

        for image_id, g in run.groupby("image_id"):
            oracle = g.loc[g["final_psnr"].idxmax()]
            selected = chosen[chosen["image_id"] == image_id].iloc[0]

            oracle_psnr = float(oracle["final_psnr"])
            selected_psnr = float(selected["selected_psnr"])

            if oracle_psnr < 25:
                regime = "generation_failure_oracle4_bad"
            elif selected_psnr < 25:
                regime = "selector_failure_oracle4_good"
            else:
                regime = "solved_oracle4_good_selector_good"

            rows.append({
                "seed": seed,
                "image_id": image_id,
                "regime": regime,
                "oracle_run": int(oracle["run_index"]),
                "oracle_psnr": oracle_psnr,
                "selected_run": int(selected["selected_run_index"]),
                "selected_psnr": selected_psnr,
                "oracle_bad25": oracle_psnr < 25,
                "selected_bad25": selected_psnr < 25,
                "oracle_gap_selected_minus_oracle": selected_psnr - oracle_psnr,
            })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "multiseed_regime_rows.csv", index=False)

    summary = (
        df.groupby(["regime"])
        .agg(
            count=("regime", "size"),
            min_oracle_psnr=("oracle_psnr", "min"),
            min_selected_psnr=("selected_psnr", "min"),
            mean_selected_psnr=("selected_psnr", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(outdir / "multiseed_regime_summary.csv", index=False)

    print("REGIME ROWS")
    print(df.sort_values(["seed", "image_id"]).to_string(index=False))
    print("\nREGIME SUMMARY")
    print(summary.to_string(index=False))
    print("\nwrote", outdir)


if __name__ == "__main__":
    main()
