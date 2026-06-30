#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import pandas as pd


BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")

IDS = [x.strip().zfill(5) for x in os.environ.get(
    "IDS", "00046,00480,00746,00171,00971,00116,00272,00599,00136,00154,00253"
).replace(" ", ",").split(",") if x.strip()]

RUN_SEEDS = [int(x) for x in os.environ.get(
    "RUN_SEEDS", "4400,4500,4600,4700,4800,4900,5000,5100"
).replace(" ", ",").split(",") if x.strip()]

MEAS_SEEDS = [int(x) for x in os.environ.get(
    "MEAS_SEEDS", "5001,5002,5003"
).replace(" ", ",").split(",") if x.strip()]

K_GRID = [4, 5, 6, 8, 12, 16, 24, 32]


def exact_path(image_id: str, run_seed: int, meas_seed: int, num_runs: int) -> Path:
    return BASE / f"B20_1_daps{num_runs}S_{image_id}_meas{meas_seed}_runseed{run_seed}_exact_final_loss_selector.csv"


def find_exact_file(image_id: str, run_seed: int, meas_seed: int) -> tuple[Path | None, int]:
    # Prefer 32S if available, otherwise 16S.
    for n in [32, 16]:
        p = exact_path(image_id, run_seed, meas_seed, n)
        if p.exists():
            return p, n
    return None, 0


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
                p, num_runs = find_exact_file(image_id, run_seed, meas_seed)
                if p is None:
                    missing.append((image_id, run_seed, meas_seed))
                    continue

                df = pd.read_csv(p).copy()
                df["run_index"] = df["run_index"].astype(int)
                col = psnr_col(df)
                df[col] = pd.to_numeric(df[col])
                df["exact_operator_loss"] = pd.to_numeric(df["exact_operator_loss"])

                for K in K_GRID:
                    if K > num_runs:
                        continue
                    cand = df[df["run_index"] < K].copy()
                    if len(cand) < K:
                        continue

                    oracle = cand.loc[cand[col].idxmax()]
                    selected = cand.sort_values(["exact_operator_loss", "run_index"]).iloc[0]
                    good = cand[cand[col] >= 25].sort_values("run_index")

                    oracle_psnr = float(oracle[col])
                    selected_psnr = float(selected[col])

                    rows.append({
                        "image_id": image_id,
                        "run_seed": run_seed,
                        "meas_seed": meas_seed,
                        "num_runs_file": num_runs,
                        "K": K,
                        "oracleK_psnr": oracle_psnr,
                        "oracleK_run": int(oracle["run_index"]),
                        "selected_exact_psnr": selected_psnr,
                        "selected_exact_run": int(selected["run_index"]),
                        "oracle_good25": int(oracle_psnr >= 25),
                        "oracle_bad25": int(oracle_psnr < 25),
                        "selected_good25": int(selected_psnr >= 25),
                        "selected_bad25": int(selected_psnr < 25),
                        "final_exact_failure": int(oracle_psnr >= 25 and selected_psnr < 25),
                        "first_good_run": int(good.iloc[0]["run_index"]) if len(good) else -1,
                        "num_good_runs_in_K": int((cand[col] >= 25).sum()),
                        "source_file": str(p),
                    })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No rows found")

    per_path = BASE / "B20_2B_global_final_hit_anatomy_per_case.csv"
    out.to_csv(per_path, index=False)
    print("[write]", per_path)

    by_image_K = (
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

    by_image_K_path = BASE / "B20_2B_global_final_hit_anatomy_by_image_K.csv"
    by_image_K.to_csv(by_image_K_path, index=False)
    print("[write]", by_image_K_path)

    # Largest available K per case.
    idx = out.groupby(["image_id", "run_seed", "meas_seed"])["K"].idxmax()
    largest = out.loc[idx].copy()

    by_image_seed = (
        largest.groupby(["image_id", "run_seed"])
        .agg(
            cases=("meas_seed", "count"),
            max_K=("K", "max"),
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

    by_image_seed_path = BASE / "B20_2B_global_final_hit_anatomy_by_image_runseed.csv"
    by_image_seed.to_csv(by_image_seed_path, index=False)
    print("[write]", by_image_seed_path)

    good = largest[largest["first_good_run"] >= 0].copy()
    first_good = (
        good.groupby("image_id")
        .agg(
            good_cases=("image_id", "count"),
            mean_first_good_run=("first_good_run", "mean"),
            min_first_good_run=("first_good_run", "min"),
            max_first_good_run=("first_good_run", "max"),
            first_good_runs=("first_good_run", lambda x: ",".join(map(str, sorted(set(x))))),
        )
        .reset_index()
        .sort_values(["mean_first_good_run", "image_id"])
    )

    first_good_path = BASE / "B20_2B_global_final_hit_anatomy_first_good_summary.csv"
    first_good.to_csv(first_good_path, index=False)
    print("[write]", first_good_path)

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 240)

    print("\n== B20.2B global final anatomy: by image at K=16 ==")
    print(
        by_image_K[by_image_K["K"] == 16][[
            "image_id", "cases", "run_seeds",
            "mean_oracleK_psnr", "min_oracleK_psnr", "max_oracleK_psnr",
            "oracle_bad25", "oracle_good25",
            "selected_bad25", "final_exact_failures",
            "mean_num_good_runs_in_K",
        ]]
        .to_string(index=False)
    )

    print("\n== B20.2B global final anatomy: by image/run_seed at largest available K ==")
    print(by_image_seed.to_string(index=False))

    print("\n== B20.2B global final anatomy: first good run summary ==")
    if len(first_good):
        print(first_good.to_string(index=False))
    else:
        print("none")

    if missing:
        print(f"\n[missing combinations] {len(missing)}")
        print("first missing:", missing[:20])


if __name__ == "__main__":
    main()
