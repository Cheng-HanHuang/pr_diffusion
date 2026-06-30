#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import pandas as pd


BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")

IDS = [x.strip().zfill(5) for x in os.environ.get(
    "IDS", "00046,00480,00746,00171,00971"
).replace(" ", ",").split(",") if x.strip()]

MEAS_SEEDS = [int(x) for x in os.environ.get(
    "MEAS_SEEDS", "5001,5002,5003"
).replace(" ", ",").split(",") if x.strip()]

RUN_SEED = int(os.environ.get("RUN_SEED", "4400"))
NUM_RUNS = int(os.environ.get("NUM_RUNS", "16"))

K_GRID_DEFAULT = [4, 5, 6, 8, 12, 16, 24, 32]
K_GRID = [k for k in K_GRID_DEFAULT if k <= NUM_RUNS]


def exact_path(image_id: str, meas_seed: int) -> Path:
    return BASE / f"B20_1_daps{NUM_RUNS}S_{image_id}_meas{meas_seed}_runseed{RUN_SEED}_exact_final_loss_selector.csv"


def psnr_col(df: pd.DataFrame) -> str:
    if "psnr_metrics_json" in df.columns:
        return "psnr_metrics_json"
    if "psnr_recomputed_from_png" in df.columns:
        return "psnr_recomputed_from_png"
    raise KeyError("no PSNR column found")


def main() -> None:
    rows = []
    missing = []

    for image_id in IDS:
        for meas_seed in MEAS_SEEDS:
            p = exact_path(image_id, meas_seed)
            if not p.exists():
                missing.append((image_id, meas_seed, str(p)))
                continue

            df = pd.read_csv(p).copy()
            df["run_index"] = df["run_index"].astype(int)
            col = psnr_col(df)
            df[col] = pd.to_numeric(df[col])
            df["exact_operator_loss"] = pd.to_numeric(df["exact_operator_loss"])
            df = df.sort_values("run_index")

            max_available = int(df["run_index"].max()) + 1

            for K in K_GRID:
                if K > max_available:
                    continue

                cand = df[df["run_index"] < K].copy()
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
                    "image_id": image_id,
                    "meas_seed": meas_seed,
                    "run_seed": RUN_SEED,
                    "num_runs_available": max_available,
                    "K": K,
                    "oracleK_psnr": oracle_psnr,
                    "oracleK_run": int(oracle["run_index"]),
                    "selected_exact_psnr": selected_psnr,
                    "selected_exact_run": int(selected["run_index"]),
                    "selected_exact_loss": float(selected["exact_operator_loss"]),
                    "bad_oracle25": int(oracle_psnr < 25),
                    "bad_selected25": int(selected_psnr < 25),
                    "final_exact_failure": int(oracle_psnr >= 25 and selected_psnr < 25),
                    "failure_type": failure_type,
                    "gap_oracle_minus_selected": oracle_psnr - selected_psnr,
                })

    if missing:
        print("[missing exact files]")
        for m in missing[:50]:
            print(m)
        print(f"[missing count] {len(missing)}")

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No rows produced. Are the B20.1 exact CSVs available?")

    per_path = BASE / f"B20_1_highK_{NUM_RUNS}S_runseed{RUN_SEED}_oracle_curve_per_case.csv"
    out.to_csv(per_path, index=False)
    print("[write]", per_path)

    summary_K = (
        out.groupby("K")
        .agg(
            cases=("image_id", "count"),
            images=("image_id", "nunique"),
            mean_oracleK_psnr=("oracleK_psnr", "mean"),
            min_oracleK_psnr=("oracleK_psnr", "min"),
            oracle_bad25=("bad_oracle25", "sum"),
            selected_bad25=("bad_selected25", "sum"),
            final_exact_failures=("final_exact_failure", "sum"),
            mean_selected_exact_psnr=("selected_exact_psnr", "mean"),
            min_selected_exact_psnr=("selected_exact_psnr", "min"),
            mean_gap_oracle_minus_selected=("gap_oracle_minus_selected", "mean"),
            max_gap_oracle_minus_selected=("gap_oracle_minus_selected", "max"),
        )
        .reset_index()
    )

    summary_path = BASE / f"B20_1_highK_{NUM_RUNS}S_runseed{RUN_SEED}_summary_by_K.csv"
    summary_K.to_csv(summary_path, index=False)
    print("[write]", summary_path)

    by_image = (
        out.groupby(["image_id", "K"])
        .agg(
            cases=("meas_seed", "count"),
            mean_oracleK_psnr=("oracleK_psnr", "mean"),
            min_oracleK_psnr=("oracleK_psnr", "min"),
            max_oracleK_psnr=("oracleK_psnr", "max"),
            oracle_bad25=("bad_oracle25", "sum"),
            selected_bad25=("bad_selected25", "sum"),
            final_exact_failures=("final_exact_failure", "sum"),
            mean_selected_exact_psnr=("selected_exact_psnr", "mean"),
            min_selected_exact_psnr=("selected_exact_psnr", "min"),
        )
        .reset_index()
        .sort_values(["K", "oracle_bad25", "min_oracleK_psnr"], ascending=[True, False, True])
    )

    by_image_path = BASE / f"B20_1_highK_{NUM_RUNS}S_runseed{RUN_SEED}_by_image_K.csv"
    by_image.to_csv(by_image_path, index=False)
    print("[write]", by_image_path)

    # Rescue table: cases bad at K=6 but good at larger K.
    wide = out.pivot_table(
        index=["image_id", "meas_seed"],
        columns="K",
        values="oracleK_psnr",
        aggfunc="first",
    ).reset_index()

    rescue_rows = []
    for _, r in wide.iterrows():
        base6 = r.get(6, pd.NA)
        if pd.isna(base6):
            continue
        for K in K_GRID:
            if K <= 6 or K not in r or pd.isna(r[K]):
                continue
            if float(base6) < 25 and float(r[K]) >= 25:
                rescue_rows.append({
                    "image_id": r["image_id"],
                    "meas_seed": int(r["meas_seed"]),
                    "oracle6_psnr": float(base6),
                    "rescued_at_K": K,
                    "oracleK_psnr": float(r[K]),
                })
                break

    rescue = pd.DataFrame(rescue_rows)
    rescue_path = BASE / f"B20_1_highK_{NUM_RUNS}S_runseed{RUN_SEED}_rescued_from_K6.csv"
    rescue.to_csv(rescue_path, index=False)
    print("[write]", rescue_path)

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 220)

    print("\n== B20.1 summary by K ==")
    print(summary_K.to_string(index=False))

    print("\n== B20.1 image-level oracle hardness at largest K ==")
    largest = max(K_GRID)
    print(
        by_image[by_image["K"] == largest][[
            "image_id", "cases", "mean_oracleK_psnr", "min_oracleK_psnr",
            "max_oracleK_psnr", "oracle_bad25", "selected_bad25",
            "final_exact_failures",
        ]]
        .sort_values(["oracle_bad25", "min_oracleK_psnr"], ascending=[False, True])
        .to_string(index=False)
    )

    print("\n== B20.1 rescued from K=6 ==")
    if len(rescue):
        print(rescue.to_string(index=False))
    else:
        print("none")


if __name__ == "__main__":
    main()
