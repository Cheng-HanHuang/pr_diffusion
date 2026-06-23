#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_features_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--method", default="sitcom_ode_npsitcom")
    ap.add_argument("--noise_std", type=float, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.run_features_csv)
    rows = []

    for _, r in df.iterrows():
        image_id = str(r["image_id"]).zfill(5)
        run_index = int(r["run_index"])

        traj = {
            "tau": float(r["tau"]),
            "target_step": int(r["target_step"]),
            "actual_step": int(r["actual_step"]),
            "sigma_tau": float(r["sigma_tau"]),
            "correction_norm_tau": float(r["correction_norm_tau"]),
            "x0hat_x0y_disagreement_tau": float(r["x0hat_x0y_disagreement_tau"]),
            "x0y_full_residual_normed_tau": float(r["x0y_full_residual_normed_tau"]),
            "x0y_lowfreq_residual_normed_tau": float(r["x0y_lowfreq_residual_normed_tau"]),
            "x0hat_full_residual_normed_tau": float(r["x0hat_full_residual_normed_tau"]),
            "x0hat_lowfreq_residual_normed_tau": float(r["x0hat_lowfreq_residual_normed_tau"]),
        }

        rows.append({
            "candidate_id": f"{args.method}:{image_id}:run{run_index}",
            "method": args.method,
            "method_family": "sitcom",
            "image_id": image_id,
            "noise_std": args.noise_std,
            "seed": "",
            "run_index": run_index,
            "sample_path": "",
            "runtime_sec": math.nan,
            "psnr": float(r["final_psnr"]),
            "bad25": bool(r["final_psnr"] < 25),
            "bad20": bool(r["final_psnr"] < 20),
            "measurement_full_residual_normed": float(r["x0y_full_residual_normed_tau"]),
            "measurement_lowfreq_residual_normed": float(r["x0y_lowfreq_residual_normed_tau"]),
            "correction_norm_tau": float(r["correction_norm_tau"]),
            "x0hat_x0y_disagreement_tau": float(r["x0hat_x0y_disagreement_tau"]),
            "trajectory_json": json.dumps(traj, sort_keys=True),
            "certificate_json": "{}",
        })

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} rows={len(out)}")


if __name__ == "__main__":
    main()
