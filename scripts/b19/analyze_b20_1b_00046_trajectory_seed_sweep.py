#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import pandas as pd


BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")

IMAGE_ID = os.environ.get("IMAGE_ID", "00046").zfill(5)
RUN_SEEDS = [int(x) for x in os.environ.get("RUN_SEEDS", "4400,4500,4600,4700").replace(" ", ",").split(",") if x.strip()]
MEAS_SEEDS = [int(x) for x in os.environ.get("MEAS_SEEDS", "5001,5002,5003").replace(" ", ",").split(",") if x.strip()]
NUM_RUNS = int(os.environ.get("NUM_RUNS", "32"))
K_GRID = [4, 5, 6, 8, 12, 16, 24, 32]


def exact_path(run_seed: int, meas_seed: int) -> Path:
    return BASE / f"B20_1_daps{NUM_RUNS}S_{IMAGE_ID}_meas{meas_seed}_runseed{run_seed}_exact_final_loss_selector.csv"


def psnr_col(df: pd.DataFrame) -> str:
    if "psnr_metrics_json" in df.columns:
        return "psnr_metrics_json"
    if "psnr_recomputed_from_png" in df.columns:
        return "psnr_recomputed_from_png"
    raise KeyError("No PSNR column found")


def main() -> None:
    rows = []
    missing = []

    for run_seed in RUN_SEEDS:
        for meas_seed in MEAS_SEEDS:
            p = exact_path(run_seed, meas_seed)
            if not p.exists():
                missing.append(str(p))
                continue

            df = pd.read_csv(p).copy()
            df["run_index"] = df["run_index"].astype(int)
            col = psnr_col(df)
            df[col] = pd.to_numeric(df[col])
            df["exact_operator_loss"] = pd.to_numeric(df["exact_operator_loss"])

            for K in K_GRID:
                cand = df[df["run_index"] < K].copy()
                if len(cand) < K:
                    continue

                oracle = cand.loc[cand[col].idxmax()]
                selected = cand.sort_values(["exact_operator_loss", "run_index"]).iloc[0]

                oracle_psnr = float(oracle[col])
                selected_psnr = float(selected[col])

                if oracle_psnr < 25:
                    failure_type = "oracle_init_failure_no_good_firstK"
                elif selected_psnr < 25:
                    failure_type = "final_exact_selector_failure"
                else:
                    failure_type = "selected_good"

                rows.append({
                    "image_id": IMAGE_ID,
                    "run_seed": run_seed,
                    "meas_seed": meas_seed,
                    "K": K,
                    "oracleK_psnr": oracle_psnr,
                    "oracleK_run": int(oracle["run_index"]),
                    "selected_exact_psnr": selected_psnr,
                    "selected_exact_run": int(selected["run_index"]),
                    "selected_exact_loss": float(selected["exact_operator_loss"]),
                    "oracle_bad25": int(oracle_psnr < 25),
                    "selected_bad25": int(selected_psnr < 25),
                    "final_exact_failure": int(oracle_psnr >= 25 and selected_psnr < 25),
                    "failure_type": failure_type,
                    "gap_oracle_minus_selected": oracle_psnr - selected_psnr,
                })

    if missing:
        print("[missing files]")
        for x in missing[:50]:
            print(x)
        print("[missing count]", len(missing))

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No rows produced")

    per = BASE / f"B20_1B_{IMAGE_ID}_trajectory_seed_sweep_{NUM_RUNS}S_per_case.csv"
    out.to_csv(per, index=False)
    print("[write]", per)

    by_k = (
        out.groupby("K")
        .agg(
            cases=("image_id", "count"),
            run_seeds=("run_seed", "nunique"),
            meas_seeds=("meas_seed", "nunique"),
            mean_oracleK_psnr=("oracleK_psnr", "mean"),
            min_oracleK_psnr=("oracleK_psnr", "min"),
            max_oracleK_psnr=("oracleK_psnr", "max"),
            oracle_bad25=("oracle_bad25", "sum"),
            selected_bad25=("selected_bad25", "sum"),
            final_exact_failures=("final_exact_failure", "sum"),
            mean_selected_exact_psnr=("selected_exact_psnr", "mean"),
            min_selected_exact_psnr=("selected_exact_psnr", "min"),
            max_selected_exact_psnr=("selected_exact_psnr", "max"),
        )
        .reset_index()
    )

    by_k_path = BASE / f"B20_1B_{IMAGE_ID}_trajectory_seed_sweep_{NUM_RUNS}S_by_K.csv"
    by_k.to_csv(by_k_path, index=False)
    print("[write]", by_k_path)

    by_seed_k = (
        out.groupby(["run_seed", "K"])
        .agg(
            cases=("image_id", "count"),
            mean_oracleK_psnr=("oracleK_psnr", "mean"),
            min_oracleK_psnr=("oracleK_psnr", "min"),
            max_oracleK_psnr=("oracleK_psnr", "max"),
            oracle_bad25=("oracle_bad25", "sum"),
            selected_bad25=("selected_bad25", "sum"),
            final_exact_failures=("final_exact_failure", "sum"),
            best_oracle_run_seen=("oracleK_run", lambda x: ",".join(map(str, sorted(set(x))))),
            selected_runs=("selected_exact_run", lambda x: ",".join(map(str, sorted(set(x))))),
        )
        .reset_index()
        .sort_values(["K", "oracle_bad25", "run_seed"], ascending=[True, False, True])
    )

    by_seed_k_path = BASE / f"B20_1B_{IMAGE_ID}_trajectory_seed_sweep_{NUM_RUNS}S_by_runseed_K.csv"
    by_seed_k.to_csv(by_seed_k_path, index=False)
    print("[write]", by_seed_k_path)

    # Best run-seed/meas-seed cases at K=32.
    kmax = out[out["K"] == NUM_RUNS].copy()
    kmax_sorted = kmax.sort_values(["oracleK_psnr", "selected_exact_psnr"], ascending=[False, False])
    kmax_path = BASE / f"B20_1B_{IMAGE_ID}_trajectory_seed_sweep_{NUM_RUNS}S_K{NUM_RUNS}_ranked_cases.csv"
    kmax_sorted.to_csv(kmax_path, index=False)
    print("[write]", kmax_path)

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 220)

    print("\n== B20.1B 00046 trajectory-seed sweep: summary by K ==")
    print(by_k.to_string(index=False))

    print("\n== B20.1B 00046 trajectory-seed sweep: by run_seed at K=32 ==")
    print(
        by_seed_k[by_seed_k["K"] == NUM_RUNS][[
            "run_seed", "cases", "mean_oracleK_psnr", "min_oracleK_psnr",
            "max_oracleK_psnr", "oracle_bad25", "selected_bad25",
            "final_exact_failures", "best_oracle_run_seen", "selected_runs",
        ]]
        .to_string(index=False)
    )

    print("\n== B20.1B 00046 trajectory-seed sweep: ranked K=32 cases ==")
    print(
        kmax_sorted[[
            "run_seed", "meas_seed", "oracleK_psnr", "oracleK_run",
            "selected_exact_psnr", "selected_exact_run",
            "failure_type",
        ]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
