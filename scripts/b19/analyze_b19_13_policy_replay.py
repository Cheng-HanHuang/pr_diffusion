#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")

CASES = [
    "00000", "00004", "00007", "00008", "00013",
    "00015", "00017", "00018", "00019", "00020",
    "00025", "00027", "00028", "00032", "00034",
]

FULL_BASELINES = [4, 5, 6, 8, 12, 16]

PREFIX_POLICIES = [
    # name, K, checkpoint, keep_k
    ("P5_c150_keep1_cost4p00", 5, 150, 1),
    ("P5_c125_keep2_cost3p875", 5, 125, 2),
    ("P6_c100_keep2_cost4p00", 6, 100, 2),
    ("P7_c100_keep1_cost4p00", 7, 100, 1),

    ("P8_c100_keep2_cost5p00", 8, 100, 2),
    ("P8_c125_keep2_cost5p75", 8, 125, 2),
    ("P8_c150_keep2_cost6p50", 8, 150, 2),

    ("P12_c125_keep4_cost9p00", 12, 125, 4),
]


def prefix_cost(K: int, checkpoint: int, keep_k: int) -> float:
    return K * checkpoint / 200.0 + keep_k * (200 - checkpoint) / 200.0


def window_path(image_id: str) -> Path:
    if image_id == "00007":
        return BASE / "B19_9C_daps_00007_meas3000_16S_rawtraj_window_features.csv"
    return BASE / f"B19_11_daps_{image_id}_meas3000_16S_rawtraj_window_features.csv"


def exact_path(image_id: str) -> Path:
    candidates = []
    if image_id == "00007":
        candidates.append(BASE / "B19_8B_daps16S_00007_meas3000_exact_final_loss_selector.csv")
    candidates.append(BASE / f"B19_11_daps16S_{image_id}_meas3000_exact_final_loss_selector.csv")
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"missing exact selector csv for {image_id}: {candidates}")


def load_exact(image_id: str) -> pd.DataFrame:
    df = pd.read_csv(exact_path(image_id)).copy()
    df["run_index"] = df["run_index"].astype(int)
    if "psnr_metrics_json" not in df.columns:
        if "psnr_recomputed_from_png" in df.columns:
            df["psnr_metrics_json"] = df["psnr_recomputed_from_png"]
        else:
            raise ValueError(f"no PSNR column in exact selector for {image_id}")
    return df


def load_window(image_id: str) -> pd.DataFrame:
    df = pd.read_csv(window_path(image_id)).copy()
    df["run_index"] = df["run_index"].astype(int)
    return df


def add_prefix_scores(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.copy()
    loss = "sqrt_loss_x0y_over_y_norm_last"
    corr = "correction_rms_last"
    x0hat = "sqrt_loss_x0hat_over_y_norm_last"

    sub["score_abs_simple"] = sub[loss] + sub[corr]
    sub["loss_rank_in_batch"] = sub[loss].rank(method="min", ascending=True)
    sub["corr_rank_in_batch"] = sub[corr].rank(method="min", ascending=True)
    sub["score_rel_loss_corr"] = sub["loss_rank_in_batch"] + sub["corr_rank_in_batch"]

    if x0hat in sub.columns:
        sub["x0hat_rank_in_batch"] = sub[x0hat].rank(method="min", ascending=True)
        sub["score_rel_loss_hat_corr"] = (
            sub["loss_rank_in_batch"]
            + sub["x0hat_rank_in_batch"]
            + sub["corr_rank_in_batch"]
        )
    else:
        sub["score_rel_loss_hat_corr"] = sub["score_rel_loss_corr"]

    return sub


def exact_select(final_df: pd.DataFrame, run_indices: list[int]) -> pd.Series:
    cand = final_df[final_df["run_index"].isin(run_indices)].copy()
    if cand.empty:
        raise ValueError(f"empty candidate set for runs {run_indices}")
    return cand.sort_values(["exact_operator_loss", "run_index"]).iloc[0]


rows = []

for image_id in CASES:
    try:
        final_df = load_exact(image_id)
        win_df = load_window(image_id)
    except Exception as e:
        print("[skip]", image_id, e)
        continue

    final_df = final_df.sort_values("run_index")
    available_runs = sorted(final_df["run_index"].astype(int).tolist())
    oracle16_psnr = float(final_df["psnr_metrics_json"].max())
    oracle16_run = int(final_df.loc[final_df["psnr_metrics_json"].idxmax(), "run_index"])

    # Full DAPS baselines: first M completed trajectories, final exact-loss selection.
    for M in FULL_BASELINES:
        runs = [r for r in available_runs if r < M]
        if len(runs) < M:
            continue

        selected = exact_select(final_df, runs)
        oracleM = float(final_df[final_df["run_index"].isin(runs)]["psnr_metrics_json"].max())

        rows.append({
            "image_id": image_id,
            "policy": f"F{M}_full_exact",
            "policy_type": "full",
            "cost_full_equiv": float(M),
            "K": M,
            "checkpoint_step": 200,
            "keep_k": M,
            "candidate_runs": ",".join(map(str, runs)),
            "kept_runs": ",".join(map(str, runs)),
            "selected_run": int(selected["run_index"]),
            "selected_psnr": float(selected["psnr_metrics_json"]),
            "selected_exact_loss": float(selected["exact_operator_loss"]),
            "oracle_policy_psnr": oracleM,
            "oracle16_psnr": oracle16_psnr,
            "oracle16_run": oracle16_run,
            "gap_to_policy_oracle": oracleM - float(selected["psnr_metrics_json"]),
            "gap_to_oracle16": oracle16_psnr - float(selected["psnr_metrics_json"]),
            "firstK_has_good25": int((final_df[final_df["run_index"].isin(runs)]["psnr_metrics_json"] >= 25.0).any()),
            "kept_has_good25": int((final_df[final_df["run_index"].isin(runs)]["psnr_metrics_json"] >= 25.0).any()),
            "selected_bad25": int(float(selected["psnr_metrics_json"]) < 25.0),
            "selected_bad20": int(float(selected["psnr_metrics_json"]) < 20.0),
            "selected_29": int(float(selected["psnr_metrics_json"]) >= 29.0),
            "selected_30": int(float(selected["psnr_metrics_json"]) >= 30.0),
            "init_failure_no_good_firstK": int(not (final_df[final_df["run_index"].isin(runs)]["psnr_metrics_json"] >= 25.0).any()),
            "prefix_selection_failure": 0,
            "final_exact_selection_failure": int(
                (oracleM >= 25.0) and (float(selected["psnr_metrics_json"]) < 25.0)
            ),
        })

    # Prefix policies: first K prefixes to checkpoint, relative keep, final exact-loss selection among kept.
    for policy_name, K, checkpoint, keep_k in PREFIX_POLICIES:
        g = win_df[win_df["checkpoint_step"] == checkpoint].copy()
        runs = [r for r in available_runs if r < K]
        sub = g[g["run_index"].isin(runs)].copy()

        if len(runs) < K or len(sub) < K:
            print("[missing prefix runs]", image_id, policy_name, "runs", runs, "sub", len(sub))
            continue

        sub = add_prefix_scores(sub)
        kept = sub.sort_values(["score_rel_loss_corr", "score_abs_simple", "run_index"]).head(keep_k)
        kept_runs = kept["run_index"].astype(int).tolist()

        selected = exact_select(final_df, kept_runs)

        firstK_final = final_df[final_df["run_index"].isin(runs)].copy()
        kept_final = final_df[final_df["run_index"].isin(kept_runs)].copy()

        oracleK = float(firstK_final["psnr_metrics_json"].max())
        oracleKept = float(kept_final["psnr_metrics_json"].max())

        firstK_has_good = int((firstK_final["psnr_metrics_json"] >= 25.0).any())
        kept_has_good = int((kept_final["psnr_metrics_json"] >= 25.0).any())

        rows.append({
            "image_id": image_id,
            "policy": policy_name,
            "policy_type": "prefix",
            "cost_full_equiv": prefix_cost(K, checkpoint, keep_k),
            "K": K,
            "checkpoint_step": checkpoint,
            "keep_k": keep_k,
            "candidate_runs": ",".join(map(str, runs)),
            "kept_runs": ",".join(map(str, kept_runs)),
            "selected_run": int(selected["run_index"]),
            "selected_psnr": float(selected["psnr_metrics_json"]),
            "selected_exact_loss": float(selected["exact_operator_loss"]),
            "oracle_policy_psnr": oracleK,
            "oracle_kept_psnr": oracleKept,
            "oracle16_psnr": oracle16_psnr,
            "oracle16_run": oracle16_run,
            "gap_to_policy_oracle": oracleK - float(selected["psnr_metrics_json"]),
            "gap_to_kept_oracle": oracleKept - float(selected["psnr_metrics_json"]),
            "gap_to_oracle16": oracle16_psnr - float(selected["psnr_metrics_json"]),
            "firstK_has_good25": firstK_has_good,
            "kept_has_good25": kept_has_good,
            "selected_bad25": int(float(selected["psnr_metrics_json"]) < 25.0),
            "selected_bad20": int(float(selected["psnr_metrics_json"]) < 20.0),
            "selected_29": int(float(selected["psnr_metrics_json"]) >= 29.0),
            "selected_30": int(float(selected["psnr_metrics_json"]) >= 30.0),
            "init_failure_no_good_firstK": int(firstK_has_good == 0),
            "prefix_selection_failure": int((firstK_has_good == 1) and (kept_has_good == 0)),
            "final_exact_selection_failure": int(
                (kept_has_good == 1) and (float(selected["psnr_metrics_json"]) < 25.0)
            ),
            "best_prefix_score_abs": float(sub["score_abs_simple"].min()),
        })

out = pd.DataFrame(rows).sort_values(["cost_full_equiv", "policy", "image_id"])

per_image_dest = BASE / "B19_13_policy_replay_per_image.csv"
out.to_csv(per_image_dest, index=False)
print("[write]", per_image_dest)
print("rows:", len(out))


summary_rows = []
for policy, g in out.groupby("policy"):
    g = g.copy()
    summary_rows.append({
        "policy": policy,
        "policy_type": g["policy_type"].iloc[0],
        "cost_full_equiv": float(g["cost_full_equiv"].iloc[0]),
        "K": int(g["K"].iloc[0]),
        "checkpoint_step": int(g["checkpoint_step"].iloc[0]),
        "keep_k": int(g["keep_k"].iloc[0]),
        "images": len(g),
        "mean_selected_psnr": float(g["selected_psnr"].mean()),
        "min_selected_psnr": float(g["selected_psnr"].min()),
        "bad25": int(g["selected_bad25"].sum()),
        "bad20": int(g["selected_bad20"].sum()),
        "selected_29": int(g["selected_29"].sum()),
        "selected_30": int(g["selected_30"].sum()),
        "mean_gap_to_policy_oracle": float(g["gap_to_policy_oracle"].mean()),
        "max_gap_to_policy_oracle": float(g["gap_to_policy_oracle"].max()),
        "mean_gap_to_oracle16": float(g["gap_to_oracle16"].mean()),
        "max_gap_to_oracle16": float(g["gap_to_oracle16"].max()),
        "init_failures_no_good_firstK": int(g["init_failure_no_good_firstK"].sum()),
        "prefix_selection_failures": int(g["prefix_selection_failure"].sum()),
        "final_exact_selection_failures": int(g["final_exact_selection_failure"].sum()),
    })

summary = pd.DataFrame(summary_rows).sort_values([
    "cost_full_equiv", "bad25", "mean_selected_psnr"
], ascending=[True, True, False])

summary_dest = BASE / "B19_13_policy_replay_summary.csv"
summary.to_csv(summary_dest, index=False)
print("[write]", summary_dest)

print("\nSummary:")
print(summary.to_string(index=False))

print("\nBad25 cases by policy:")
bad = out[out["selected_bad25"] == 1].copy()
if len(bad):
    print(
        bad[[
            "policy", "image_id", "cost_full_equiv", "candidate_runs", "kept_runs",
            "selected_run", "selected_psnr", "oracle_policy_psnr", "oracle16_psnr",
            "init_failure_no_good_firstK", "prefix_selection_failure",
            "final_exact_selection_failure",
        ]]
        .sort_values(["policy", "image_id"])
        .to_string(index=False)
    )
else:
    print("none")

print("\nPrefix selection failures:")
pf = out[out["prefix_selection_failure"] == 1].copy()
if len(pf):
    print(
        pf[[
            "policy", "image_id", "cost_full_equiv", "candidate_runs", "kept_runs",
            "selected_run", "selected_psnr", "oracle_policy_psnr", "oracle16_psnr",
        ]]
        .sort_values(["policy", "image_id"])
        .to_string(index=False)
    )
else:
    print("none")
