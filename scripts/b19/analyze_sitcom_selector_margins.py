#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


FEATURES = [
    "correction_norm_tau",
    "x0hat_x0y_disagreement_tau",
    "x0y_full_residual_normed_tau",
    "x0y_lowfreq_residual_normed_tau",
    "x0hat_full_residual_normed_tau",
    "x0hat_lowfreq_residual_normed_tau",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_features_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.run_features_csv)
    rows = []

    for image_id, g in df.groupby("image_id"):
        oracle = g.loc[g["final_psnr"].idxmax()]
        for feat in FEATURES:
            gg = g.sort_values(feat, ascending=True).reset_index(drop=True)
            chosen = gg.iloc[0]
            second = gg.iloc[1]
            rows.append({
                "image_id": image_id,
                "feature": feat,
                "chosen_run": int(chosen["run_index"]),
                "chosen_psnr": float(chosen["final_psnr"]),
                "chosen_bad25": bool(chosen["final_psnr"] < 25),
                "oracle_run": int(oracle["run_index"]),
                "oracle_psnr": float(oracle["final_psnr"]),
                "oracle_rank_by_feature": int(gg.index[gg["run_index"] == oracle["run_index"]][0]) + 1,
                "chosen_feature": float(chosen[feat]),
                "second_feature": float(second[feat]),
                "margin_second_minus_best": float(second[feat] - chosen[feat]),
                "relative_margin": float((second[feat] - chosen[feat]) / max(abs(chosen[feat]), 1e-12)),
            })

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(out.sort_values(["feature", "image_id"]).to_string(index=False))
    print("wrote", args.out_csv)


if __name__ == "__main__":
    main()
