#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd


BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")

ALL25 = [
    "00000", "00004", "00005", "00007", "00008",
    "00009", "00010", "00011", "00012", "00013",
    "00014", "00015", "00016", "00017", "00018",
    "00019", "00020", "00025", "00027", "00028",
    "00029", "00032", "00034", "00037", "00039",
]

HARD15 = {
    "00000", "00004", "00007", "00008", "00013",
    "00015", "00017", "00018", "00019", "00020",
    "00025", "00027", "00028", "00032", "00034",
}

POLICIES = [
    ("F4_full_exact", "full", 4, 200, 4, "exact"),
    ("F6_full_exact", "full", 6, 200, 6, "exact"),

    ("P5_c125_keep2_inst_lc", "prefix", 5, 125, 2, "inst_lc"),
    ("P5_c125_keep2_loss_only", "prefix", 5, 125, 2, "loss_only"),

    ("P6_c100_keep2_inst_lhc", "prefix", 6, 100, 2, "inst_lhc"),
    ("P6_c100_keep2_inst_lc", "prefix", 6, 100, 2, "inst_lc"),
    ("P6_c100_keep2_loss_only", "prefix", 6, 100, 2, "loss_only"),
]


def prefix_cost(K: int, checkpoint: int, keep_k: int) -> float:
    if checkpoint == 200:
        return float(K)
    return K * checkpoint / 200.0 + keep_k * (200 - checkpoint) / 200.0


def exact_path(image_id: str, seed: int = 3000) -> Path:
    if image_id == "00007":
        p = BASE / f"B19_8B_daps16S_00007_meas{seed}_exact_final_loss_selector.csv"
        if p.exists():
            return p

    if image_id in HARD15:
        p = BASE / f"B19_11_daps16S_{image_id}_meas{seed}_exact_final_loss_selector.csv"
        if p.exists():
            return p

    return BASE / f"B19_16_daps6S_{image_id}_meas{seed}_exact_final_loss_selector.csv"


def window_path(image_id: str, seed: int = 3000) -> Path:
    if image_id == "00007":
        p = BASE / f"B19_9C_daps_00007_meas{seed}_16S_rawtraj_window_features.csv"
        if p.exists():
            return p

    if image_id in HARD15:
        p = BASE / f"B19_11_daps_{image_id}_meas{seed}_16S_rawtraj_window_features.csv"
        if p.exists():
            return p

    return BASE / f"B19_16_daps_{image_id}_meas{seed}_6S_rawtraj_window_features.csv"


def load_exact(image_id: str) -> pd.DataFrame:
    p = exact_path(image_id)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p).copy()
    df["run_index"] = df["run_index"].astype(int)
    if "psnr_metrics_json" not in df.columns:
        df["psnr_metrics_json"] = df["psnr_recomputed_from_png"]
    return df


def load_window(image_id: str) -> pd.DataFrame:
    p = window_path(image_id)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p).copy()
    df["run_index"] = df["run_index"].astype(int)
    return df


def add_scores(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.copy()

    sub["score_abs_simple"] = (
        sub["sqrt_loss_x0y_over_y_norm_last"]
        + sub["correction_rms_last"]
    )

    sub["rank_loss"] = sub["sqrt_loss_x0y_over_y_norm_last"].rank(method="min", ascending=True)
    sub["rank_corr"] = sub["correction_rms_last"].rank(method="min", ascending=True)
    sub["score_inst_lc"] = sub["rank_loss"] + sub["rank_corr"]
    sub["score_loss_only"] = sub["rank_loss"]

    if "sqrt_loss_x0hat_over_y_norm_last" in sub.columns:
        sub["rank_xhat"] = sub["sqrt_loss_x0hat_over_y_norm_last"].rank(method="min", ascending=True)
    else:
        sub["rank_xhat"] = 0.0

    sub["score_inst_lhc"] = sub["rank_loss"] + sub["rank_corr"] + sub["rank_xhat"]
    return sub


def exact_select(exact: pd.DataFrame, run_indices: list[int]) -> pd.Series:
    cand = exact[exact["run_index"].isin(run_indices)].copy()
    if cand.empty:
        raise RuntimeError(f"empty exact candidate set for runs {run_indices}")
    return cand.sort_values(["exact_operator_loss", "run_index"]).iloc[0]


rows = []

for image_id in ALL25:
    exact = load_exact(image_id)
    win = load_window(image_id)

    for policy, policy_type, K, checkpoint, keep_k, score_name in POLICIES:
        runs = list(range(K))

        if policy_type == "full":
            kept_runs = runs
        else:
            sub = win[(win["checkpoint_step"] == checkpoint) & (win["run_index"].isin(runs))].copy()
            if len(sub) < K:
                raise RuntimeError(f"{image_id} {policy}: expected {K} prefix rows, got {len(sub)}")
            sub = add_scores(sub)

            score_col = f"score_{score_name}"
            kept = sub.sort_values([score_col, "score_abs_simple", "run_index"]).head(keep_k)
            kept_runs = kept["run_index"].astype(int).tolist()

        selected = exact_select(exact, kept_runs)
        selected_run = int(selected["run_index"])
        selected_psnr = float(selected["psnr_metrics_json"])

        cand_final = exact[exact["run_index"].isin(runs)].copy()
        kept_final = exact[exact["run_index"].isin(kept_runs)].copy()

        firstK_has_good = bool((cand_final["psnr_metrics_json"] >= 25.0).any())
        kept_has_good = bool((kept_final["psnr_metrics_json"] >= 25.0).any())

        rows.append({
            "image_id": image_id,
            "policy": policy,
            "policy_type": policy_type,
            "cost_full_equiv": prefix_cost(K, checkpoint, keep_k),
            "K": K,
            "checkpoint_step": checkpoint,
            "keep_k": keep_k,
            "score": score_name,
            "candidate_runs": ",".join(map(str, runs)),
            "kept_runs": ",".join(map(str, kept_runs)),
            "selected_run": selected_run,
            "selected_psnr": selected_psnr,
            "selected_exact_loss": float(selected["exact_operator_loss"]),
            "oracleK_psnr": float(cand_final["psnr_metrics_json"].max()),
            "oracleKept_psnr": float(kept_final["psnr_metrics_json"].max()),
            "gap_to_oracleK": float(cand_final["psnr_metrics_json"].max()) - selected_psnr,
            "firstK_has_good25": int(firstK_has_good),
            "kept_has_good25": int(kept_has_good),
            "selected_bad25": int(selected_psnr < 25.0),
            "selected_bad20": int(selected_psnr < 20.0),
            "selected_29": int(selected_psnr >= 29.0),
            "selected_30": int(selected_psnr >= 30.0),
            "init_failure_no_good_firstK": int(not firstK_has_good),
            "prefix_selection_failure": int(firstK_has_good and not kept_has_good),
            "final_exact_selection_failure": int(kept_has_good and selected_psnr < 25.0),
        })

out = pd.DataFrame(rows).sort_values(["cost_full_equiv", "policy", "image_id"])
per_image_dest = BASE / "B19_16A_full25_seed3000_frozen_policy_replay_per_image.csv"
out.to_csv(per_image_dest, index=False)
print("[write]", per_image_dest)

summary = (
    out.groupby(["policy", "policy_type", "score", "cost_full_equiv", "K", "checkpoint_step", "keep_k"])
    .agg(
        images=("image_id", "count"),
        mean_psnr=("selected_psnr", "mean"),
        min_psnr=("selected_psnr", "min"),
        bad25=("selected_bad25", "sum"),
        bad20=("selected_bad20", "sum"),
        selected_29=("selected_29", "sum"),
        selected_30=("selected_30", "sum"),
        init_failures=("init_failure_no_good_firstK", "sum"),
        prefix_failures=("prefix_selection_failure", "sum"),
        final_failures=("final_exact_selection_failure", "sum"),
        mean_gap_to_oracleK=("gap_to_oracleK", "mean"),
        max_gap_to_oracleK=("gap_to_oracleK", "max"),
    )
    .reset_index()
    .sort_values(["bad25", "cost_full_equiv", "mean_psnr"], ascending=[True, True, False])
)

summary_dest = BASE / "B19_16A_full25_seed3000_frozen_policy_replay_summary.csv"
summary.to_csv(summary_dest, index=False)
print("[write]", summary_dest)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 240)
print(summary.to_string(index=False))

bad = out[out["selected_bad25"] == 1].copy()
print("\nBad25 cases:")
if len(bad):
    print(
        bad[[
            "policy", "image_id", "candidate_runs", "kept_runs",
            "selected_run", "selected_psnr", "oracleK_psnr",
            "init_failure_no_good_firstK", "prefix_selection_failure",
            "final_exact_selection_failure",
        ]]
        .sort_values(["policy", "image_id"])
        .to_string(index=False)
    )
else:
    print("none")
