#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import math
import pandas as pd


BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")
RUN_SEED = int(os.environ.get("RUN_SEED", "4400"))

ALL_ROWS = BASE / f"B20_2B_raw6_global_prefix_anatomy_runseed{RUN_SEED}_all_rows.csv"

K = int(os.environ.get("K", "6"))
KEEP_LIST = [int(x) for x in os.environ.get("KEEP_LIST", "1,2,3").replace(" ", ",").split(",") if x.strip()]


CANDIDATE_FEATURES = [
    "correction_rms_rank_mean",
    "exact_loss_x0y_rank_mean",
    "sqrt_loss_x0y_over_y_norm_rank_mean",
    "exact_loss_x0hat_rank_mean",
    "sqrt_loss_x0hat_over_y_norm_rank_mean",
    "x0y_jump_rms_rank_mean",
    "exact_loss_x0y_rank_last",
    "correction_rms_rank_last",
    "exact_loss_x0hat_rank_last",
]


COMBO_POLICIES = {
    "combo_loss_corr_rank_mean": [
        "correction_rms_rank_mean",
        "exact_loss_x0y_rank_mean",
        "exact_loss_x0hat_rank_mean",
    ],
    "combo_loss_corr_jump_rank_mean": [
        "correction_rms_rank_mean",
        "exact_loss_x0y_rank_mean",
        "exact_loss_x0hat_rank_mean",
        "x0y_jump_rms_rank_mean",
    ],
    "combo_x0y_x0hat_rank_mean": [
        "exact_loss_x0y_rank_mean",
        "exact_loss_x0hat_rank_mean",
    ],
    "combo_last_loss_corr": [
        "exact_loss_x0y_rank_last",
        "correction_rms_rank_last",
        "exact_loss_x0hat_rank_last",
    ],
}


def pick_col(df: pd.DataFrame, names: list[str]) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"None found: {names}\nAvailable: {list(df.columns)}")


def add_score_columns(df: pd.DataFrame) -> dict[str, str]:
    score_cols: dict[str, str] = {}

    for feat in CANDIDATE_FEATURES:
        if feat in df.columns:
            sc = f"score__{feat}"
            # These are rank/loss/correction features where smaller is better.
            df[sc] = pd.to_numeric(df[feat], errors="coerce")
            score_cols[feat] = sc

    for name, feats in COMBO_POLICIES.items():
        usable = [f for f in feats if f in df.columns]
        if usable:
            sc = f"score__{name}"
            df[sc] = df[usable].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            score_cols[name] = sc

    return score_cols


def main() -> None:
    df = pd.read_csv(ALL_ROWS)
    print("[read]", ALL_ROWS)
    print("[rows]", len(df))

    image_col = pick_col(df, ["image_id"])
    meas_col = pick_col(df, ["meas_seed"])
    run_col = pick_col(df, ["run_index"])

    final_psnr_col = pick_col(df, [
        "final_psnr",
        "final_psnr_y",
        "psnr_metrics_json",
        "psnr_metrics_json_y",
        "psnr_recomputed_from_png",
        "psnr_recomputed_from_png_y",
    ])

    exact_loss_col = pick_col(df, [
        "exact_operator_loss",
        "exact_operator_loss_y",
    ])

    df["image_id_norm"] = df[image_col].astype(str).str.zfill(5)
    df["meas_seed_norm"] = df[meas_col].astype(int)
    df["run_index_norm"] = df[run_col].astype(int)
    df["final_psnr_norm"] = pd.to_numeric(df[final_psnr_col], errors="coerce")
    df["exact_loss_norm"] = pd.to_numeric(df[exact_loss_col], errors="coerce")

    # Raw window files have multiple rows per run. Collapse to one row per candidate
    # by taking the mean of numeric features. Labels/final losses are constant per run.
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in {"meas_seed_norm", "run_index_norm"}
    ]

    agg = {
        c: "mean" for c in numeric_cols
        if c not in {"final_psnr_norm", "exact_loss_norm"}
    }
    agg["final_psnr_norm"] = "first"
    agg["exact_loss_norm"] = "first"

    cand = (
        df.groupby(["image_id_norm", "meas_seed_norm", "run_index_norm"], as_index=False)
        .agg(agg)
    )

    score_cols = add_score_columns(cand)
    if not score_cols:
        raise RuntimeError("No usable score columns found")

    rows = []

    group_cols = ["image_id_norm", "meas_seed_norm"]
    for (image_id, meas_seed), g0 in cand.groupby(group_cols):
        g = g0[g0["run_index_norm"] < K].copy()
        if len(g) < K:
            continue

        oracle = g.loc[g["final_psnr_norm"].idxmax()]
        exact_selected = g.sort_values(["exact_loss_norm", "run_index_norm"]).iloc[0]

        oracle_psnr = float(oracle["final_psnr_norm"])
        exact_psnr = float(exact_selected["final_psnr_norm"])

        # Full exact policy among all K.
        rows.append({
            "policy": f"F{K}_full_exact",
            "feature": "exact_loss_allK",
            "keep_k": K,
            "image_id": image_id,
            "meas_seed": meas_seed,
            "kept_runs": ",".join(map(str, sorted(g["run_index_norm"].tolist()))),
            "selected_run": int(exact_selected["run_index_norm"]),
            "oracleK_run": int(oracle["run_index_norm"]),
            "oracleK_psnr": oracle_psnr,
            "selected_psnr": exact_psnr,
            "selected_bad25": int(exact_psnr < 25),
            "oracle_bad25": int(oracle_psnr < 25),
            "prefix_failure": 0,
            "final_exact_failure": int(oracle_psnr >= 25 and exact_psnr < 25),
            "gap_to_oracleK": oracle_psnr - exact_psnr,
        })

        for feat_name, sc in score_cols.items():
            ranked = g.sort_values([sc, "run_index_norm"], ascending=[True, True])

            for keep_k in KEEP_LIST:
                if keep_k > K:
                    continue

                kept = ranked.head(keep_k).copy()
                selected = kept.sort_values(["exact_loss_norm", "run_index_norm"]).iloc[0]
                selected_psnr = float(selected["final_psnr_norm"])

                kept_oracle = kept.loc[kept["final_psnr_norm"].idxmax()]
                oracle_kept_psnr = float(kept_oracle["final_psnr_norm"])

                rows.append({
                    "policy": f"P{K}_keep{keep_k}_{feat_name}",
                    "feature": feat_name,
                    "keep_k": keep_k,
                    "image_id": image_id,
                    "meas_seed": meas_seed,
                    "kept_runs": ",".join(map(str, sorted(kept["run_index_norm"].tolist()))),
                    "selected_run": int(selected["run_index_norm"]),
                    "oracleK_run": int(oracle["run_index_norm"]),
                    "oracleK_psnr": oracle_psnr,
                    "oracleKept_psnr": oracle_kept_psnr,
                    "selected_psnr": selected_psnr,
                    "selected_bad25": int(selected_psnr < 25),
                    "oracle_bad25": int(oracle_psnr < 25),
                    "prefix_failure": int(oracle_psnr >= 25 and oracle_kept_psnr < 25),
                    "final_exact_failure": int(oracle_kept_psnr >= 25 and selected_psnr < 25),
                    "gap_to_oracleK": oracle_psnr - selected_psnr,
                })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No replay rows produced")

    per_path = BASE / f"B20_3A_raw6_feature_policy_replay_runseed{RUN_SEED}_per_case.csv"
    out.to_csv(per_path, index=False)
    print("[write]", per_path)

    summary = (
        out.groupby(["policy", "feature", "keep_k"])
        .agg(
            cases=("image_id", "count"),
            images=("image_id", "nunique"),
            mean_selected_psnr=("selected_psnr", "mean"),
            min_selected_psnr=("selected_psnr", "min"),
            bad25=("selected_bad25", "sum"),
            oracle_bad25=("oracle_bad25", "sum"),
            prefix_failures=("prefix_failure", "sum"),
            final_exact_failures=("final_exact_failure", "sum"),
            mean_oracleK_psnr=("oracleK_psnr", "mean"),
            min_oracleK_psnr=("oracleK_psnr", "min"),
            mean_gap_to_oracleK=("gap_to_oracleK", "mean"),
            max_gap_to_oracleK=("gap_to_oracleK", "max"),
        )
        .reset_index()
        .sort_values(
            ["bad25", "prefix_failures", "final_exact_failures", "keep_k", "mean_selected_psnr"],
            ascending=[True, True, True, True, False],
        )
    )

    summary_path = BASE / f"B20_3A_raw6_feature_policy_replay_runseed{RUN_SEED}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("[write]", summary_path)

    # Bad-case table for top policies.
    top_policies = summary.head(20)["policy"].tolist()
    bad = out[(out["policy"].isin(top_policies)) & (out["selected_bad25"] == 1)].copy()
    bad_path = BASE / f"B20_3A_raw6_feature_policy_replay_runseed{RUN_SEED}_top_policy_bad_cases.csv"
    bad.to_csv(bad_path, index=False)
    print("[write]", bad_path)

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 260)

    print("\n== B20.3A top raw6 feature policies ==")
    print(
        summary[[
            "policy", "keep_k", "cases", "images",
            "mean_selected_psnr", "min_selected_psnr",
            "bad25", "oracle_bad25",
            "prefix_failures", "final_exact_failures",
            "mean_gap_to_oracleK", "max_gap_to_oracleK",
        ]]
        .head(40)
        .to_string(index=False)
    )

    print("\n== B20.3A baseline/full policies ==")
    base = summary[
        summary["policy"].str.startswith(f"F{K}_")
        | summary["policy"].str.contains("combo")
        | summary["policy"].str.contains("correction_rms_rank_mean")
        | summary["policy"].str.contains("exact_loss_x0y_rank_mean")
    ].copy()
    print(
        base[[
            "policy", "keep_k", "mean_selected_psnr", "min_selected_psnr",
            "bad25", "oracle_bad25",
            "prefix_failures", "final_exact_failures",
            "mean_gap_to_oracleK",
        ]]
        .head(80)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
