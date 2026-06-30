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

TAGS = [
    x.strip() for x in os.environ.get(
        "TAGS",
        "ann250_diff5 ann300_diff5 ann350_diff5 ann375_diff5 ann400_diff5 ann450_diff5"
    ).replace(",", " ").split() if x.strip()
]


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


def schedule_path(tag: str, run_seed: int) -> Path:
    return BASE / f"B20_8A_{tag}_daps{NUM_RUNS}S_{ID}_meas{MEAS_SEED}_runseed{run_seed}_exact_final_loss_selector.csv"


def baseline_path(run_seed: int) -> Path | None:
    for n in [32, 16, 6]:
        p = BASE / f"B20_1_daps{n}S_{ID}_meas{MEAS_SEED}_runseed{run_seed}_exact_final_loss_selector.csv"
        if p.exists():
            return p
    p = BASE / f"B20_7A_daps16S_{ID}_meas{MEAS_SEED}_runseed{run_seed}_exact_final_loss_selector.csv"
    if p.exists():
        return p
    return None


def load_psnr_file(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    rc = run_col(df)
    pc = psnr_col(df)
    out = df[[rc, pc]].rename(columns={rc: "run_index", pc: f"{label}_psnr"})
    out["run_index"] = out["run_index"].astype(int)
    out[f"{label}_psnr"] = pd.to_numeric(out[f"{label}_psnr"], errors="coerce")
    out = out[out["run_index"] < NUM_RUNS].copy()
    out[f"{label}_good25"] = (out[f"{label}_psnr"] >= 25.0).astype(int)
    return out


def main() -> None:
    rows = []
    missing = []

    for rs in RUN_SEEDS:
        base_p = baseline_path(rs)
        if base_p is None:
            missing.append(("baseline", rs))
            continue

        merged = load_psnr_file(base_p, "ann200").copy()
        merged["run_seed"] = rs

        for tag in TAGS:
            p = schedule_path(tag, rs)
            if not p.exists():
                missing.append((tag, rs, str(p)))
                continue
            d = load_psnr_file(p, tag)
            merged = merged.merge(d, on="run_index", how="left")

        rows.append(merged)

    if missing:
        print("[missing]", len(missing))
        print("[first missing]", missing[:30])

    if not rows:
        raise RuntimeError("No rows produced.")

    wide = pd.concat(rows, ignore_index=True)

    psnr_cols = [c for c in wide.columns if c.endswith("_psnr")]
    good_cols = [c for c in wide.columns if c.endswith("_good25")]

    wide["best_psnr"] = wide[psnr_cols].max(axis=1)
    wide["best_schedule"] = wide[psnr_cols].idxmax(axis=1).str.replace("_psnr", "", regex=False)
    wide["num_good_schedules"] = wide[good_cols].sum(axis=1)
    wide["any_good_schedule"] = (wide["num_good_schedules"] > 0).astype(int)
    wide["all_good_schedules"] = (wide["num_good_schedules"] == len(good_cols)).astype(int)

    out_path = BASE / f"B20_9A_{ID}_meas{MEAS_SEED}_schedule_response_map.csv"
    wide.to_csv(out_path, index=False)
    print("[write]", out_path)

    by_seed = (
        wide.groupby("run_seed")
        .agg(
            candidates=("run_index", "count"),
            ann200_good=("ann200_good25", "sum"),
            any_good_schedule=("any_good_schedule", "sum"),
            all_good_schedules=("all_good_schedules", "sum"),
            mean_best_psnr=("best_psnr", "mean"),
            max_best_psnr=("best_psnr", "max"),
            mean_num_good_schedules=("num_good_schedules", "mean"),
        )
        .reset_index()
    )

    by_seed_path = BASE / f"B20_9A_{ID}_meas{MEAS_SEED}_schedule_response_by_seed.csv"
    by_seed.to_csv(by_seed_path, index=False)
    print("[write]", by_seed_path)

    best_sched = (
        wide.groupby("best_schedule")
        .agg(
            count=("best_schedule", "count"),
            mean_best_psnr=("best_psnr", "mean"),
            good_candidates=("any_good_schedule", "sum"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )

    best_sched_path = BASE / f"B20_9A_{ID}_meas{MEAS_SEED}_best_schedule_counts.csv"
    best_sched.to_csv(best_sched_path, index=False)
    print("[write]", best_sched_path)

    # Important candidates: those rescued by schedule, and those schedule-sensitive.
    interesting = wide[
        ((wide["ann200_good25"] == 0) & (wide["any_good_schedule"] == 1))
        | (wide["num_good_schedules"].between(1, len(good_cols) - 1))
    ].copy()

    interesting_path = BASE / f"B20_9A_{ID}_meas{MEAS_SEED}_interesting_schedule_sensitive_candidates.csv"
    interesting.to_csv(interesting_path, index=False)
    print("[write]", interesting_path)

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 260)

    print("\n== B20.9A schedule response by seed ==")
    print(by_seed.to_string(index=False))

    print("\n== B20.9A best schedule counts ==")
    print(best_sched.to_string(index=False))

    print("\n== B20.9A interesting schedule-sensitive candidates ==")
    show_cols = ["run_seed", "run_index", "best_schedule", "best_psnr", "num_good_schedules"] + psnr_cols
    print(interesting[show_cols].sort_values(["run_seed", "run_index"]).to_string(index=False))


if __name__ == "__main__":
    main()
