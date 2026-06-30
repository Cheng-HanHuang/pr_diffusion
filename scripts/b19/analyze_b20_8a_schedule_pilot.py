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
    int(x) for x in os.environ.get(
        "RUN_SEEDS", "4400 4700 5200 5300 5400 5600 5700 5500"
    ).replace(",", " ").split() if x.strip()
]

TAG = os.environ.get("TAG", "ann400_diff5")
K = int(os.environ.get("K", "16"))


def psnr_col(df: pd.DataFrame) -> str:
    for c in ["psnr_metrics_json", "psnr_recomputed_from_png", "final_psnr", "psnr"]:
        if c in df.columns:
            return c
    raise KeyError(f"No PSNR column. Columns={list(df.columns)}")


def run_col(df: pd.DataFrame) -> str:
    for c in ["run_index", "run", "sample_index", "candidate_run"]:
        if c in df.columns:
            return c
    raise KeyError(f"No run column. Columns={list(df.columns)}")


def baseline_path(rs: int) -> Path | None:
    for n in [32, 16, 6]:
        p = BASE / f"B20_1_daps{n}S_{ID}_meas{MEAS_SEED}_runseed{rs}_exact_final_loss_selector.csv"
        if p.exists():
            return p
    p = BASE / f"B20_7A_daps16S_{ID}_meas{MEAS_SEED}_runseed{rs}_exact_final_loss_selector.csv"
    if p.exists():
        return p
    return None


def schedule_path(rs: int) -> Path:
    return BASE / f"B20_8A_{TAG}_daps{NUM_RUNS}S_{ID}_meas{MEAS_SEED}_runseed{rs}_exact_final_loss_selector.csv"


def summarize_file(path: Path) -> dict:
    df = pd.read_csv(path).copy()
    rc = run_col(df)
    pc = psnr_col(df)
    df[rc] = df[rc].astype(int)
    df[pc] = pd.to_numeric(df[pc], errors="coerce")
    cand = df[df[rc] < K].copy()
    oracle = cand.loc[cand[pc].idxmax()]
    return {
        "oracleK_psnr": float(oracle[pc]),
        "oracleK_run": int(oracle[rc]),
        "good25": int(float(oracle[pc]) >= 25.0),
        "num_good25_runs": int((cand[pc] >= 25.0).sum()),
        "mean_psnr": float(cand[pc].mean()),
        "min_psnr": float(cand[pc].min()),
        "max_psnr": float(cand[pc].max()),
    }


def main() -> None:
    rows = []

    for rs in RUN_SEEDS:
        bp = baseline_path(rs)
        sp = schedule_path(rs)

        if bp is None:
            print("[missing baseline]", rs)
            continue
        if not sp.exists():
            print("[missing schedule]", rs, sp)
            continue

        b = summarize_file(bp)
        s = summarize_file(sp)

        rows.append({
            "image_id": ID,
            "meas_seed": MEAS_SEED,
            "run_seed": rs,
            "baseline_path": str(bp),
            "schedule_path": str(sp),
            "baseline_oracleK_psnr": b["oracleK_psnr"],
            "schedule_oracleK_psnr": s["oracleK_psnr"],
            "delta_oracleK": s["oracleK_psnr"] - b["oracleK_psnr"],
            "baseline_oracleK_run": b["oracleK_run"],
            "schedule_oracleK_run": s["oracleK_run"],
            "baseline_good25": b["good25"],
            "schedule_good25": s["good25"],
            "rescued25": int((b["good25"] == 0) and (s["good25"] == 1)),
            "lost25": int((b["good25"] == 1) and (s["good25"] == 0)),
            "baseline_num_good25_runs": b["num_good25_runs"],
            "schedule_num_good25_runs": s["num_good25_runs"],
            "baseline_mean_psnr": b["mean_psnr"],
            "schedule_mean_psnr": s["mean_psnr"],
        })

    out = pd.DataFrame(rows)
    out_path = BASE / f"B20_8A_{TAG}_{ID}_meas{MEAS_SEED}_schedule_vs_baseline.csv"
    out.to_csv(out_path, index=False)
    print("[write]", out_path)

    summary = pd.DataFrame([{
        "cases": len(out),
        "mean_baseline_oracleK": out["baseline_oracleK_psnr"].mean(),
        "mean_schedule_oracleK": out["schedule_oracleK_psnr"].mean(),
        "mean_delta_oracleK": out["delta_oracleK"].mean(),
        "max_delta_oracleK": out["delta_oracleK"].max(),
        "min_delta_oracleK": out["delta_oracleK"].min(),
        "baseline_good25": int(out["baseline_good25"].sum()),
        "schedule_good25": int(out["schedule_good25"].sum()),
        "rescued25": int(out["rescued25"].sum()),
        "lost25": int(out["lost25"].sum()),
    }])

    summary_path = BASE / f"B20_8A_{TAG}_{ID}_meas{MEAS_SEED}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("[write]", summary_path)

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 220)

    print("\n== B20.8A schedule pilot summary ==")
    print(summary.to_string(index=False))

    print("\n== B20.8A schedule pilot by seed ==")
    print(out.sort_values("delta_oracleK", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
