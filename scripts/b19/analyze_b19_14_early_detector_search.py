#!/usr/bin/env python3
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import math

import pandas as pd


BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")

CASES = [
    "00000", "00004", "00007", "00008", "00013",
    "00015", "00017", "00018", "00019", "00020",
    "00025", "00027", "00028", "00032", "00034",
]

POLICIES = [
    # name, K, checkpoint, keep_k
    ("P13_c50_keep1_cost4p00", 13, 50, 1),
    ("P10_c50_keep2_cost4p00", 10, 50, 2),
    ("P9_c75_keep1_cost4p00", 9, 75, 1),
    ("P7_c75_keep2_cost3p875", 7, 75, 2),
    ("P6_c100_keep2_cost4p00", 6, 100, 2),
    ("P5_c125_keep2_cost3p875", 5, 125, 2),
]

SCORES = [
    "inst_lc",
    "inst_lhc",
    "histmean_lc",
    "histmean_lhc",
    "slope_lc",
    "loss_only",
    "corr_only",
]

CHECKPOINT_HISTORY = [50, 75, 100, 125]


def prefix_cost(K: int, checkpoint: int, keep_k: int) -> float:
    return K * checkpoint / 200.0 + keep_k * (200 - checkpoint) / 200.0


def window_path(image_id: str) -> Path:
    if image_id == "00007":
        return BASE / "B19_9C_daps_00007_meas3000_16S_rawtraj_window_features.csv"
    return BASE / f"B19_11_daps_{image_id}_meas3000_16S_rawtraj_window_features.csv"


def exact_path(image_id: str) -> Path:
    if image_id == "00007":
        p = BASE / "B19_8B_daps16S_00007_meas3000_exact_final_loss_selector.csv"
        if p.exists():
            return p
    return BASE / f"B19_11_daps16S_{image_id}_meas3000_exact_final_loss_selector.csv"


def load_exact(image_id: str) -> pd.DataFrame:
    df = pd.read_csv(exact_path(image_id)).copy()
    df["run_index"] = df["run_index"].astype(int)
    if "psnr_metrics_json" not in df.columns:
        df["psnr_metrics_json"] = df["psnr_recomputed_from_png"]
    return df


def make_wide(win: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run, g in win.groupby("run_index"):
        out = {"run_index": int(run)}
        for _, r in g.iterrows():
            c = int(r["checkpoint_step"])
            out[f"c{c}_loss"] = float(r["sqrt_loss_x0y_over_y_norm_last"])
            out[f"c{c}_corr"] = float(r["correction_rms_last"])
            if "sqrt_loss_x0hat_over_y_norm_last" in r:
                out[f"c{c}_xhat_loss"] = float(r["sqrt_loss_x0hat_over_y_norm_last"])
            else:
                out[f"c{c}_xhat_loss"] = math.nan
            for col in ["x0y_jump_rms_last", "x0hat_jump_rms_last", "xt_jump_rms_last"]:
                if col in r:
                    out[f"c{c}_{col}"] = float(r[col])
        rows.append(out)
    return pd.DataFrame(rows)


def rank_col(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].rank(method="min", ascending=True)


def score_subbatch(wide: pd.DataFrame, run_tuple: tuple[int, ...], checkpoint: int, score_name: str) -> pd.DataFrame:
    sub = wide[wide["run_index"].isin(run_tuple)].copy()

    loss_c = f"c{checkpoint}_loss"
    corr_c = f"c{checkpoint}_corr"
    xhat_c = f"c{checkpoint}_xhat_loss"

    sub["score_abs_simple"] = sub[loss_c] + sub[corr_c]
    sub["rank_loss_c"] = rank_col(sub, loss_c)
    sub["rank_corr_c"] = rank_col(sub, corr_c)
    sub["rank_xhat_c"] = rank_col(sub, xhat_c) if xhat_c in sub.columns and sub[xhat_c].notna().any() else 0.0

    sub["score_inst_lc"] = sub["rank_loss_c"] + sub["rank_corr_c"]
    sub["score_inst_lhc"] = sub["rank_loss_c"] + sub["rank_corr_c"] + sub["rank_xhat_c"]
    sub["score_loss_only"] = sub["rank_loss_c"]
    sub["score_corr_only"] = sub["rank_corr_c"]

    hist = [c for c in CHECKPOINT_HISTORY if c <= checkpoint and f"c{c}_loss" in sub.columns]
    hist_score_lc = 0.0
    hist_score_lhc = 0.0
    used = 0

    for c in hist:
        lc = f"c{c}_loss"
        cc = f"c{c}_corr"
        xc = f"c{c}_xhat_loss"

        if lc not in sub.columns or cc not in sub.columns:
            continue

        rloss = rank_col(sub, lc)
        rcorr = rank_col(sub, cc)
        if xc in sub.columns and sub[xc].notna().any():
            rxhat = rank_col(sub, xc)
        else:
            rxhat = 0.0

        hist_score_lc = hist_score_lc + rloss + rcorr
        hist_score_lhc = hist_score_lhc + rloss + rcorr + rxhat
        used += 1

    if used:
        sub["score_histmean_lc"] = hist_score_lc / used
        sub["score_histmean_lhc"] = hist_score_lhc / used
    else:
        sub["score_histmean_lc"] = sub["score_inst_lc"]
        sub["score_histmean_lhc"] = sub["score_inst_lhc"]

    prev_hist = [c for c in hist if c < checkpoint]
    if prev_hist:
        prev = max(prev_hist)
        sub["delta_loss"] = sub[f"c{checkpoint}_loss"] - sub[f"c{prev}_loss"]
        sub["delta_corr"] = sub[f"c{checkpoint}_corr"] - sub[f"c{prev}_corr"]
        sub["score_slope_lc"] = (
            sub["score_inst_lc"]
            + rank_col(sub, "delta_loss")
            + rank_col(sub, "delta_corr")
        )
    else:
        sub["score_slope_lc"] = sub["score_inst_lc"]

    score_col = f"score_{score_name}"
    if score_col not in sub.columns:
        raise ValueError(f"unknown score {score_name}")

    sub["score"] = sub[score_col]
    return sub


def exact_select(exact: pd.DataFrame, run_indices: list[int]) -> pd.Series:
    cand = exact[exact["run_index"].isin(run_indices)].copy()
    return cand.sort_values(["exact_operator_loss", "run_index"]).iloc[0]


summary_rows = []
failure_rows = []

for image_id in CASES:
    wp = window_path(image_id)
    ep = exact_path(image_id)
    if not wp.exists() or not ep.exists():
        print("[missing]", image_id, wp, ep)
        continue

    win = pd.read_csv(wp)
    win["run_index"] = win["run_index"].astype(int)
    exact = load_exact(image_id)
    wide = make_wide(win)
    available_runs = sorted(exact["run_index"].astype(int).tolist())

    for policy_name, K, checkpoint, keep_k in POLICIES:
        if len(available_runs) < K:
            continue

        run_tuples = list(combinations(available_runs, K))

        for score_name in SCORES:
            n = 0
            good_batches = 0
            all_bad_batches = 0
            kept_has_good_count = 0
            selected_good_count = 0
            selected_bad25_count = 0
            init_failures = 0
            prefix_failures = 0
            final_failures = 0
            selected_psnrs = []
            gaps_to_batch_oracle = []

            for run_tuple in run_tuples:
                first = exact[exact["run_index"].isin(run_tuple)]
                batch_has_good = bool((first["psnr_metrics_json"] >= 25.0).any())
                batch_oracle_psnr = float(first["psnr_metrics_json"].max())

                scored = score_subbatch(wide, run_tuple, checkpoint, score_name)
                kept = scored.sort_values(["score", "score_abs_simple", "run_index"]).head(keep_k)
                kept_runs = kept["run_index"].astype(int).tolist()

                kept_exact = exact[exact["run_index"].isin(kept_runs)]
                kept_has_good = bool((kept_exact["psnr_metrics_json"] >= 25.0).any())

                selected = exact_select(exact, kept_runs)
                selected_psnr = float(selected["psnr_metrics_json"])
                selected_good = selected_psnr >= 25.0

                n += 1
                good_batches += int(batch_has_good)
                all_bad_batches += int(not batch_has_good)
                kept_has_good_count += int(kept_has_good)
                selected_good_count += int(selected_good)
                selected_bad25_count += int(not selected_good)

                init_failures += int(not batch_has_good)
                prefix_failures += int(batch_has_good and not kept_has_good)
                final_failures += int(kept_has_good and not selected_good)

                selected_psnrs.append(selected_psnr)
                gaps_to_batch_oracle.append(batch_oracle_psnr - selected_psnr)

                if (batch_has_good and not selected_good) or (batch_has_good and not kept_has_good):
                    if len(failure_rows) < 20000:
                        failure_rows.append({
                            "image_id": image_id,
                            "policy": policy_name,
                            "score": score_name,
                            "run_tuple": ",".join(map(str, run_tuple)),
                            "kept_runs": ",".join(map(str, kept_runs)),
                            "batch_oracle_psnr": batch_oracle_psnr,
                            "selected_run": int(selected["run_index"]),
                            "selected_psnr": selected_psnr,
                            "batch_has_good": int(batch_has_good),
                            "kept_has_good": int(kept_has_good),
                            "failure_type": (
                                "prefix_selection_failure" if batch_has_good and not kept_has_good
                                else "final_exact_selection_failure"
                            ),
                        })

            summary_rows.append({
                "image_id": image_id,
                "policy": policy_name,
                "score": score_name,
                "cost_full_equiv": prefix_cost(K, checkpoint, keep_k),
                "K": K,
                "checkpoint_step": checkpoint,
                "keep_k": keep_k,
                "num_batches": n,
                "good_batches": good_batches,
                "all_bad_batches": all_bad_batches,
                "all_bad_rate": all_bad_batches / n if n else float("nan"),
                "kept_has_good_rate_all": kept_has_good_count / n if n else float("nan"),
                "selected_good_rate_all": selected_good_count / n if n else float("nan"),
                "selected_bad25_count": selected_bad25_count,
                "init_failures": init_failures,
                "prefix_selection_failures": prefix_failures,
                "final_exact_selection_failures": final_failures,
                "mean_selected_psnr": sum(selected_psnrs) / len(selected_psnrs) if selected_psnrs else float("nan"),
                "min_selected_psnr": min(selected_psnrs) if selected_psnrs else float("nan"),
                "mean_gap_to_batch_oracle": sum(gaps_to_batch_oracle) / len(gaps_to_batch_oracle) if gaps_to_batch_oracle else float("nan"),
                "max_gap_to_batch_oracle": max(gaps_to_batch_oracle) if gaps_to_batch_oracle else float("nan"),
            })

summary = pd.DataFrame(summary_rows)
per_image_dest = BASE / "B19_14_early_detector_search_by_image.csv"
summary.to_csv(per_image_dest, index=False)
print("[write]", per_image_dest)

fail = pd.DataFrame(failure_rows)
fail_dest = BASE / "B19_14_early_detector_search_failure_examples.csv"
fail.to_csv(fail_dest, index=False)
print("[write]", fail_dest)

overall = (
    summary.groupby(["policy", "score", "cost_full_equiv", "K", "checkpoint_step", "keep_k"])
    .agg(
        images=("image_id", "count"),
        total_batches=("num_batches", "sum"),
        total_good_batches=("good_batches", "sum"),
        total_all_bad_batches=("all_bad_batches", "sum"),
        total_selected_bad25=("selected_bad25_count", "sum"),
        total_init_failures=("init_failures", "sum"),
        total_prefix_failures=("prefix_selection_failures", "sum"),
        total_final_failures=("final_exact_selection_failures", "sum"),
        mean_selected_psnr=("mean_selected_psnr", "mean"),
        min_selected_psnr=("min_selected_psnr", "min"),
        mean_gap=("mean_gap_to_batch_oracle", "mean"),
        max_gap=("max_gap_to_batch_oracle", "max"),
    )
    .reset_index()
)

overall["selected_bad25_rate"] = overall["total_selected_bad25"] / overall["total_batches"]
overall["all_bad_rate"] = overall["total_all_bad_batches"] / overall["total_batches"]
overall["prefix_failure_rate_among_good"] = overall["total_prefix_failures"] / overall["total_good_batches"].clip(lower=1)
overall["final_failure_rate_among_good"] = overall["total_final_failures"] / overall["total_good_batches"].clip(lower=1)

overall = overall.sort_values([
    "cost_full_equiv",
    "selected_bad25_rate",
    "prefix_failure_rate_among_good",
    "mean_selected_psnr",
], ascending=[True, True, True, False])

overall_dest = BASE / "B19_14_early_detector_search_overall.csv"
overall.to_csv(overall_dest, index=False)
print("[write]", overall_dest)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 240)

print("\nTop policies by selected_bad25_rate:")
print(
    overall[[
        "policy", "score", "cost_full_equiv", "checkpoint_step", "K", "keep_k",
        "total_batches", "total_selected_bad25", "selected_bad25_rate",
        "total_init_failures", "total_prefix_failures", "total_final_failures",
        "mean_selected_psnr", "min_selected_psnr", "mean_gap", "max_gap",
    ]]
    .head(80)
    .to_string(index=False)
)

print("\nBest per policy:")
print(
    overall.sort_values(["policy", "selected_bad25_rate", "mean_selected_psnr"], ascending=[True, True, False])
    .groupby("policy")
    .head(3)
    [[
        "policy", "score", "cost_full_equiv", "total_selected_bad25",
        "selected_bad25_rate", "total_init_failures", "total_prefix_failures",
        "total_final_failures", "mean_selected_psnr", "min_selected_psnr",
    ]]
    .to_string(index=False)
)
