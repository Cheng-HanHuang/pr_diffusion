#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import pandas as pd


BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")

IDS = [x.strip().zfill(5) for x in os.environ.get(
    "IDS", "00046,00480,00746,00171,00971"
).replace(" ", ",").split(",") if x.strip()]

RUN_SEEDS = [int(x) for x in os.environ.get(
    "RUN_SEEDS", "4400,4800,4900,5000,5100"
).replace(" ", ",").split(",") if x.strip()]

MEAS_SEEDS = [int(x) for x in os.environ.get(
    "MEAS_SEEDS", "5001,5002,5003"
).replace(" ", ",").split(",") if x.strip()]

NUM_RUNS = int(os.environ.get("NUM_RUNS", "16"))
K_GRID = [4, 5, 6, 8, 12, 16]


def exact_path(image_id: str, run_seed: int, meas_seed: int, num_runs: int) -> Path:
    return BASE / f"B20_1_daps{num_runs}S_{image_id}_meas{meas_seed}_runseed{run_seed}_exact_final_loss_selector.csv"


def psnr_col(df: pd.DataFrame) -> str:
    if "psnr_metrics_json" in df.columns:
        return "psnr_metrics_json"
    if "psnr_recomputed_from_png" in df.columns:
        return "psnr_recomputed_from_png"
    raise KeyError("No PSNR column found")


def main() -> None:
    rows = []
    missing = []

    for image_id in IDS:
        for run_seed in RUN_SEEDS:
            for meas_seed in MEAS_SEEDS:
                p = exact_path(image_id, run_seed, meas_seed, NUM_RUNS)

                # Fallback: if a 32S run exists, use it and restrict to K <= 16.
                if not p.exists():
                    p32 = exact_path(image_id, run_seed, meas_seed, 32)
                    if p32.exists():
                        p = p32
                    else:
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
                    first_good = cand[cand[col] >= 25.0].sort_values("run_index")

                    rows.append({
                        "image_id": image_id,
                        "run_seed": run_seed,
                        "meas_seed": meas_seed,
                        "K": K,
                        "oracleK_psnr": oracle_psnr,
                        "oracleK_run": int(oracle["run_index"]),
                        "selected_exact_psnr": selected_psnr,
                        "selected_exact_run": int(selected["run_index"]),
                        "selected_exact_loss": float(selected["exact_operator_loss"]),
                        "oracle_good25": int(oracle_psnr >= 25.0),
                        "oracle_bad25": int(oracle_psnr < 25.0),
                        "selected_good25": int(selected_psnr >= 25.0),
                        "selected_bad25": int(selected_psnr < 25.0),
                        "final_exact_failure": int(oracle_psnr >= 25.0 and selected_psnr < 25.0),
                        "first_good_run": int(first_good.iloc[0]["run_index"]) if len(first_good) else -1,
                        "num_good_runs_in_K": int((cand[col] >= 25.0).sum()),
                        "gap_oracle_minus_selected": oracle_psnr - selected_psnr,
                        "source_file": str(p),
                    })

    if missing:
        print("[missing files]")
        for x in missing[:50]:
            print(x)
        print("[missing count]", len(missing))

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No rows produced")

    per = BASE / f"B20_2A_seed_hit_rate_{NUM_RUNS}S_per_case.csv"
    out.to_csv(per, index=False)
    print("[write]", per)

    by_k = (
        out.groupby("K")
        .agg(
            cases=("image_id", "count"),
            images=("image_id", "nunique"),
            run_seeds=("run_seed", "nunique"),
            meas_seeds=("meas_seed", "nunique"),
            mean_oracleK_psnr=("oracleK_psnr", "mean"),
            min_oracleK_psnr=("oracleK_psnr", "min"),
            max_oracleK_psnr=("oracleK_psnr", "max"),
            oracle_bad25=("oracle_bad25", "sum"),
            selected_bad25=("selected_bad25", "sum"),
            final_exact_failures=("final_exact_failure", "sum"),
            mean_num_good_runs_in_K=("num_good_runs_in_K", "mean"),
            mean_selected_exact_psnr=("selected_exact_psnr", "mean"),
            min_selected_exact_psnr=("selected_exact_psnr", "min"),
        )
        .reset_index()
    )

    by_k_path = BASE / f"B20_2A_seed_hit_rate_{NUM_RUNS}S_by_K.csv"
    by_k.to_csv(by_k_path, index=False)
    print("[write]", by_k_path)

    by_image_k = (
        out.groupby(["image_id", "K"])
        .agg(
            cases=("image_id", "count"),
            run_seeds=("run_seed", "nunique"),
            mean_oracleK_psnr=("oracleK_psnr", "mean"),
            min_oracleK_psnr=("oracleK_psnr", "min"),
            max_oracleK_psnr=("oracleK_psnr", "max"),
            oracle_bad25=("oracle_bad25", "sum"),
            oracle_good25=("oracle_good25", "sum"),
            selected_bad25=("selected_bad25", "sum"),
            final_exact_failures=("final_exact_failure", "sum"),
            mean_num_good_runs_in_K=("num_good_runs_in_K", "mean"),
        )
        .reset_index()
        .sort_values(["K", "oracle_bad25", "image_id"], ascending=[True, False, True])
    )

    by_image_k_path = BASE / f"B20_2A_seed_hit_rate_{NUM_RUNS}S_by_image_K.csv"
    by_image_k.to_csv(by_image_k_path, index=False)
    print("[write]", by_image_k_path)

    # Hit-rate per image/run seed at the largest K.
    kmax = max(K_GRID)
    kmax_df = out[out["K"] == kmax].copy()

    by_image_seed = (
        kmax_df.groupby(["image_id", "run_seed"])
        .agg(
            cases=("meas_seed", "count"),
            mean_oracleK_psnr=("oracleK_psnr", "mean"),
            min_oracleK_psnr=("oracleK_psnr", "min"),
            max_oracleK_psnr=("oracleK_psnr", "max"),
            oracle_bad25=("oracle_bad25", "sum"),
            oracle_good25=("oracle_good25", "sum"),
            selected_bad25=("selected_bad25", "sum"),
            final_exact_failures=("final_exact_failure", "sum"),
            first_good_runs=("first_good_run", lambda x: ",".join(map(str, sorted(set(x))))),
            oracle_runs=("oracleK_run", lambda x: ",".join(map(str, sorted(set(x))))),
            selected_runs=("selected_exact_run", lambda x: ",".join(map(str, sorted(set(x))))),
        )
        .reset_index()
        .sort_values(["image_id", "oracle_bad25", "mean_oracleK_psnr"], ascending=[True, False, False])
    )

    by_image_seed_path = BASE / f"B20_2A_seed_hit_rate_{NUM_RUNS}S_by_image_runseed_K{kmax}.csv"
    by_image_seed.to_csv(by_image_seed_path, index=False)
    print("[write]", by_image_seed_path)

    # First-good run distribution.
    good = kmax_df[kmax_df["first_good_run"] >= 0].copy()
    first_good_summary = (
        good.groupby(["image_id"])
        .agg(
            good_cases=("image_id", "count"),
            mean_first_good_run=("first_good_run", "mean"),
            min_first_good_run=("first_good_run", "min"),
            max_first_good_run=("first_good_run", "max"),
            first_good_runs=("first_good_run", lambda x: ",".join(map(str, sorted(set(x))))),
        )
        .reset_index()
    )

    first_good_path = BASE / f"B20_2A_seed_hit_rate_{NUM_RUNS}S_first_good_run_summary.csv"
    first_good_summary.to_csv(first_good_path, index=False)
    print("[write]", first_good_path)

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 240)

    print("\n== B20.2A summary by K ==")
    print(by_k.to_string(index=False))

    print(f"\n== B20.2A by image at K={kmax} ==")
    print(
        by_image_k[by_image_k["K"] == kmax][[
            "image_id", "cases", "run_seeds",
            "mean_oracleK_psnr", "min_oracleK_psnr", "max_oracleK_psnr",
            "oracle_bad25", "oracle_good25",
            "selected_bad25", "final_exact_failures",
            "mean_num_good_runs_in_K",
        ]]
        .to_string(index=False)
    )

    print(f"\n== B20.2A by image/run_seed at K={kmax} ==")
    print(
        by_image_seed[[
            "image_id", "run_seed", "cases",
            "mean_oracleK_psnr", "min_oracleK_psnr", "max_oracleK_psnr",
            "oracle_bad25", "oracle_good25",
            "selected_bad25", "final_exact_failures",
            "first_good_runs", "oracle_runs", "selected_runs",
        ]]
        .to_string(index=False)
    )

    print("\n== B20.2A first good run summary ==")
    if len(first_good_summary):
        print(first_good_summary.to_string(index=False))
    else:
        print("none")


if __name__ == "__main__":
    main()
