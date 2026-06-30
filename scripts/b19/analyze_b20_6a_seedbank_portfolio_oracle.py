#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import random
import pandas as pd


BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")

IDS = [x.strip().zfill(5) for x in os.environ.get(
    "IDS", "00046,00154,00171"
).replace(" ", ",").split(",") if x.strip()]

RUN_SEEDS = [int(x) for x in os.environ.get(
    "RUN_SEEDS", "4400,4500,4600,4700,4800,4900,5000,5100,5200,5300,5400,5500,5600,5700"
).replace(" ", ",").split(",") if x.strip()]

MEAS_SEEDS = [int(x) for x in os.environ.get(
    "MEAS_SEEDS", "5001,5002,5003"
).replace(" ", ",").split(",") if x.strip()]

LAYOUTS_RAW = os.environ.get("LAYOUTS", "1x16,2x8,4x4,8x2")
NUM_PORTFOLIOS = int(os.environ.get("NUM_PORTFOLIOS", "300"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "20260630"))
GOOD_THRESH = float(os.environ.get("GOOD_THRESH", "25.0"))


def parse_layouts(raw: str):
    layouts = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        a, b = item.split("x")
        n_seed = int(a)
        n_run = int(b)
        layouts.append((item, n_seed, n_run, n_seed * n_run))
    return layouts


LAYOUTS = parse_layouts(LAYOUTS_RAW)


def exact_path(image_id: str, run_seed: int, meas_seed: int, num_runs: int) -> Path:
    return BASE / f"B20_1_daps{num_runs}S_{image_id}_meas{meas_seed}_runseed{run_seed}_exact_final_loss_selector.csv"


def find_exact_file(image_id: str, run_seed: int, meas_seed: int) -> tuple[Path | None, int]:
    # Prefer 32S if available, but layouts here only require <=16 by default.
    for n in [32, 16, 6]:
        p = exact_path(image_id, run_seed, meas_seed, n)
        if p.exists():
            return p, n
    return None, 0


def psnr_col(df: pd.DataFrame) -> str:
    for c in ["psnr_metrics_json", "psnr_recomputed_from_png"]:
        if c in df.columns:
            return c
    raise KeyError(f"No PSNR column found. Columns={list(df.columns)}")


def load_candidate_table() -> pd.DataFrame:
    rows = []
    missing = []

    for image_id in IDS:
        for meas_seed in MEAS_SEEDS:
            for run_seed in RUN_SEEDS:
                p, n_avail = find_exact_file(image_id, run_seed, meas_seed)
                if p is None:
                    missing.append((image_id, meas_seed, run_seed))
                    continue

                df = pd.read_csv(p).copy()
                df["run_index"] = df["run_index"].astype(int)
                col = psnr_col(df)
                df[col] = pd.to_numeric(df[col])

                for _, r in df.iterrows():
                    rows.append({
                        "image_id": image_id,
                        "meas_seed": meas_seed,
                        "run_seed": run_seed,
                        "num_runs_file": n_avail,
                        "run_index": int(r["run_index"]),
                        "psnr": float(r[col]),
                    })

    cand = pd.DataFrame(rows)
    if cand.empty:
        raise RuntimeError("No candidate rows found")

    print("[candidate rows]", len(cand))
    print("[missing combos]", len(missing))
    if missing:
        print("[first missing]", missing[:20])

    return cand


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    cand = load_candidate_table()

    rows = []

    for image_id in IDS:
        for meas_seed in MEAS_SEEDS:
            sub = cand[(cand["image_id"] == image_id) & (cand["meas_seed"] == meas_seed)].copy()
            available_seeds = sorted(sub["run_seed"].unique().tolist())

            for layout_name, n_seed, n_run, K_total in LAYOUTS:
                eligible_seeds = []
                for rs in available_seeds:
                    g = sub[sub["run_seed"] == rs]
                    if (g["run_index"] < n_run).sum() >= n_run:
                        eligible_seeds.append(rs)

                if len(eligible_seeds) < n_seed:
                    continue

                for rep in range(NUM_PORTFOLIOS):
                    chosen = rng.sample(eligible_seeds, n_seed)
                    pieces = []
                    for rs in chosen:
                        g = sub[
                            (sub["run_seed"] == rs)
                            & (sub["run_index"] < n_run)
                        ].copy()
                        pieces.append(g)

                    pool = pd.concat(pieces, ignore_index=True)
                    if len(pool) != K_total:
                        continue

                    oracle = pool.loc[pool["psnr"].idxmax()]
                    good = pool[pool["psnr"] >= GOOD_THRESH].copy()

                    rows.append({
                        "image_id": image_id,
                        "meas_seed": meas_seed,
                        "layout": layout_name,
                        "n_seed": n_seed,
                        "n_run_per_seed": n_run,
                        "K_total": K_total,
                        "rep": rep,
                        "chosen_seeds": ",".join(map(str, chosen)),
                        "oracleK_psnr": float(oracle["psnr"]),
                        "oracleK_run_seed": int(oracle["run_seed"]),
                        "oracleK_run_index": int(oracle["run_index"]),
                        "oracle_good25": int(float(oracle["psnr"]) >= GOOD_THRESH),
                        "oracle_bad25": int(float(oracle["psnr"]) < GOOD_THRESH),
                        "num_good_runs": int(len(good)),
                        "first_good_global_index": int(good.index.min()) if len(good) else -1,
                    })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No portfolio rows produced")

    per_path = BASE / "B20_6A_seedbank_portfolio_oracle_per_portfolio.csv"
    out.to_csv(per_path, index=False)
    print("[write]", per_path)

    summary = (
        out.groupby(["layout", "n_seed", "n_run_per_seed", "K_total"])
        .agg(
            portfolios=("layout", "count"),
            images=("image_id", "nunique"),
            mean_oracleK_psnr=("oracleK_psnr", "mean"),
            min_oracleK_psnr=("oracleK_psnr", "min"),
            max_oracleK_psnr=("oracleK_psnr", "max"),
            oracle_good25=("oracle_good25", "sum"),
            oracle_bad25=("oracle_bad25", "sum"),
            oracle_good25_rate=("oracle_good25", "mean"),
            mean_num_good_runs=("num_good_runs", "mean"),
        )
        .reset_index()
        .sort_values(["oracle_good25_rate", "mean_oracleK_psnr"], ascending=[False, False])
    )

    summary_path = BASE / "B20_6A_seedbank_portfolio_oracle_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("[write]", summary_path)

    by_image = (
        out.groupby(["image_id", "layout", "n_seed", "n_run_per_seed", "K_total"])
        .agg(
            portfolios=("layout", "count"),
            mean_oracleK_psnr=("oracleK_psnr", "mean"),
            min_oracleK_psnr=("oracleK_psnr", "min"),
            max_oracleK_psnr=("oracleK_psnr", "max"),
            oracle_good25=("oracle_good25", "sum"),
            oracle_bad25=("oracle_bad25", "sum"),
            oracle_good25_rate=("oracle_good25", "mean"),
            mean_num_good_runs=("num_good_runs", "mean"),
        )
        .reset_index()
        .sort_values(["image_id", "oracle_good25_rate", "mean_oracleK_psnr"], ascending=[True, False, False])
    )

    by_image_path = BASE / "B20_6A_seedbank_portfolio_oracle_by_image.csv"
    by_image.to_csv(by_image_path, index=False)
    print("[write]", by_image_path)

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 240)

    print("\n== B20.6A seed-bank portfolio oracle summary ==")
    print(summary.to_string(index=False))

    print("\n== B20.6A seed-bank portfolio oracle by image ==")
    print(by_image.to_string(index=False))


if __name__ == "__main__":
    main()
