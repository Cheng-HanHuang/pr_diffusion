#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import pandas as pd


BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")

ID = os.environ.get("ID", "00046").zfill(5)
MEAS_SEED = int(os.environ.get("MEAS_SEED", "5001"))
RUN_SEEDS = [
    int(x) for x in os.environ.get(
        "RUN_SEEDS", "4400 4700 5200 5300 5400 5600 5700 5500"
    ).replace(",", " ").split() if x.strip()
]

# Portfolio format:
# name=tag:k+tag:k
PORTFOLIOS_RAW = os.environ.get(
    "PORTFOLIOS",
    ",".join([
        "ann250_16=ann250_diff5:16",
        "ann300_16=ann300_diff5:16",
        "ann350_16=ann350_diff5:16",
        "ann375_16=ann375_diff5:16",
        "ann400_16=ann400_diff5:16",
        "ann450_16=ann450_diff5:16",
        "mix375_400_8_8=ann375_diff5:8+ann400_diff5:8",
        "mix375_450_8_8=ann375_diff5:8+ann450_diff5:8",
        "mix400_450_8_8=ann400_diff5:8+ann450_diff5:8",
        "mix350_375_8_8=ann350_diff5:8+ann375_diff5:8",
        "mix300_375_8_8=ann300_diff5:8+ann375_diff5:8",
        "mix375_400_450_4_6_6=ann375_diff5:4+ann400_diff5:6+ann450_diff5:6",
        "mix300_375_400_4_6_6=ann300_diff5:4+ann375_diff5:6+ann400_diff5:6",
        "mix350_375_400_4_6_6=ann350_diff5:4+ann375_diff5:6+ann400_diff5:6",
    ])
)

NUM_RUNS = int(os.environ.get("NUM_RUNS", "16"))


def parse_tag(tag: str) -> tuple[int, int]:
    # tag like ann375_diff5
    ann = None
    diff = None
    for part in tag.split("_"):
        if part.startswith("ann"):
            ann = int(part.replace("ann", ""))
        if part.startswith("diff"):
            diff = int(part.replace("diff", ""))
    if ann is None or diff is None:
        raise ValueError(f"Cannot parse tag: {tag}")
    return ann, diff


def parse_portfolios(raw: str):
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, spec = item.split("=")
        pieces = []
        for p in spec.split("+"):
            tag, k = p.split(":")
            pieces.append((tag.strip(), int(k)))
        out.append((name.strip(), pieces))
    return out


PORTFOLIOS = parse_portfolios(PORTFOLIOS_RAW)


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


def load_candidates(path: Path, tag: str, k: int) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    rc = run_col(df)
    pc = psnr_col(df)

    df[rc] = df[rc].astype(int)
    df[pc] = pd.to_numeric(df[pc], errors="coerce")
    df = df[df[rc] < k].copy()

    ann, diff = parse_tag(tag)

    return pd.DataFrame({
        "tag": tag,
        "ann_steps": ann,
        "diff_steps": diff,
        "run_index": df[rc].astype(int),
        "psnr": df[pc].astype(float),
        "candidate_cost": ann,
    })


def summarize_pool(pool: pd.DataFrame) -> dict:
    oracle = pool.loc[pool["psnr"].idxmax()]
    return {
        "oracle_psnr": float(oracle["psnr"]),
        "oracle_tag": str(oracle["tag"]),
        "oracle_run_index": int(oracle["run_index"]),
        "good25": int(float(oracle["psnr"]) >= 25.0),
        "num_good25_candidates": int((pool["psnr"] >= 25.0).sum()),
        "mean_psnr": float(pool["psnr"].mean()),
        "min_psnr": float(pool["psnr"].min()),
        "max_psnr": float(pool["psnr"].max()),
        "total_candidates": int(len(pool)),
        "total_candidate_step_cost": int(pool["candidate_cost"].sum()),
    }


def summarize_baseline(run_seed: int) -> dict | None:
    p = baseline_path(run_seed)
    if p is None:
        return None
    # Baseline is ann200_diff5 effectively, use first 16 candidates.
    df = pd.read_csv(p).copy()
    rc = run_col(df)
    pc = psnr_col(df)
    df[rc] = df[rc].astype(int)
    df[pc] = pd.to_numeric(df[pc], errors="coerce")
    cand = df[df[rc] < 16].copy()
    oracle = cand.loc[cand[pc].idxmax()]
    return {
        "baseline_oracle_psnr": float(oracle[pc]),
        "baseline_good25": int(float(oracle[pc]) >= 25.0),
        "baseline_oracle_run": int(oracle[rc]),
        "baseline_num_good25_candidates": int((cand[pc] >= 25.0).sum()),
    }


def main() -> None:
    rows = []
    missing = []

    for run_seed in RUN_SEEDS:
        b = summarize_baseline(run_seed)
        if b is None:
            missing.append(("baseline", run_seed))
            continue

        for pname, pieces in PORTFOLIOS:
            pools = []
            ok = True
            for tag, k in pieces:
                p = schedule_path(tag, run_seed)
                if not p.exists():
                    missing.append((pname, tag, run_seed, str(p)))
                    ok = False
                    break
                pools.append(load_candidates(p, tag, k))

            if not ok:
                continue

            pool = pd.concat(pools, ignore_index=True)
            s = summarize_pool(pool)

            rows.append({
                "image_id": ID,
                "meas_seed": MEAS_SEED,
                "run_seed": run_seed,
                "portfolio": pname,
                "pieces": "+".join([f"{tag}:{k}" for tag, k in pieces]),
                **b,
                **s,
                "delta_vs_baseline": s["oracle_psnr"] - b["baseline_oracle_psnr"],
                "rescued25": int((b["baseline_good25"] == 0) and (s["good25"] == 1)),
                "lost25": int((b["baseline_good25"] == 1) and (s["good25"] == 0)),
            })

    if missing:
        print("[missing]", len(missing))
        print("[first missing]", missing[:30])

    if not rows:
        raise RuntimeError("No rows produced.")

    out = pd.DataFrame(rows)
    out_path = BASE / f"B20_8G_{ID}_meas{MEAS_SEED}_schedule_portfolio_per_seed.csv"
    out.to_csv(out_path, index=False)
    print("[write]", out_path)

    summary = (
        out.groupby(["portfolio", "pieces"])
        .agg(
            cases=("run_seed", "count"),
            total_candidate_step_cost=("total_candidate_step_cost", "first"),
            mean_oracle_psnr=("oracle_psnr", "mean"),
            min_oracle_psnr=("oracle_psnr", "min"),
            max_oracle_psnr=("oracle_psnr", "max"),
            good25=("good25", "sum"),
            bad25=("good25", lambda x: int((1 - x).sum())),
            good25_rate=("good25", "mean"),
            rescued25=("rescued25", "sum"),
            lost25=("lost25", "sum"),
            mean_num_good25_candidates=("num_good25_candidates", "mean"),
            mean_delta_vs_baseline=("delta_vs_baseline", "mean"),
        )
        .reset_index()
        .sort_values(
            ["good25", "lost25", "mean_oracle_psnr", "total_candidate_step_cost"],
            ascending=[False, True, False, True],
        )
    )

    summary_path = BASE / f"B20_8G_{ID}_meas{MEAS_SEED}_schedule_portfolio_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("[write]", summary_path)

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 260)

    print("\n== B20.8G schedule portfolio summary ==")
    print(summary.to_string(index=False))

    print("\n== B20.8G schedule portfolio per seed: top portfolios ==")
    top_names = summary.head(8)["portfolio"].tolist()
    print(
        out[out["portfolio"].isin(top_names)]
        .sort_values(["portfolio", "run_seed"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
