#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")
IMAGE_ID = "00034"

EXACT_CSV = BASE / "B19_11_daps16S_00034_meas3000_exact_final_loss_selector.csv"
WINDOW_CSV = BASE / "B19_11_daps_00034_meas3000_16S_rawtraj_window_features.csv"

CHECKPOINTS = [75, 100, 125, 150, 199]

FULL_BASELINES = [4, 5, 6, 8, 12, 16]

PREFIX_POLICIES = [
    ("P5_c125_keep2_cost3p875", 5, 125, 2),
    ("P5_c150_keep1_cost4p00", 5, 150, 1),
    ("P6_c100_keep2_cost4p00", 6, 100, 2),
    ("P7_c100_keep1_cost4p00", 7, 100, 1),
    ("P8_c125_keep2_cost5p75", 8, 125, 2),
    ("P8_c150_keep2_cost6p50", 8, 150, 2),
    ("P12_c125_keep4_cost9p00", 12, 125, 4),
]


def prefix_cost(K: int, checkpoint: int, keep_k: int) -> float:
    return K * checkpoint / 200.0 + keep_k * (200 - checkpoint) / 200.0


def add_scores(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.copy()

    loss = "sqrt_loss_x0y_over_y_norm_last"
    corr = "correction_rms_last"
    x0hat = "sqrt_loss_x0hat_over_y_norm_last"

    sub["score_abs_simple"] = sub[loss] + sub[corr]

    sub["loss_rank"] = sub[loss].rank(method="min", ascending=True)
    sub["corr_rank"] = sub[corr].rank(method="min", ascending=True)
    sub["score_rel_loss_corr"] = sub["loss_rank"] + sub["corr_rank"]

    if x0hat in sub.columns:
        sub["x0hat_rank"] = sub[x0hat].rank(method="min", ascending=True)
        sub["score_rel_loss_hat_corr"] = (
            sub["loss_rank"] + sub["x0hat_rank"] + sub["corr_rank"]
        )
    else:
        sub["x0hat_rank"] = float("nan")
        sub["score_rel_loss_hat_corr"] = sub["score_rel_loss_corr"]

    return sub


def exact_select(exact: pd.DataFrame, run_indices: list[int]) -> pd.Series:
    cand = exact[exact["run_index"].isin(run_indices)].copy()
    return cand.sort_values(["exact_operator_loss", "run_index"]).iloc[0]


def oracle_select(exact: pd.DataFrame, run_indices: list[int]) -> pd.Series:
    cand = exact[exact["run_index"].isin(run_indices)].copy()
    return cand.sort_values(["psnr_metrics_json", "run_index"], ascending=[False, True]).iloc[0]


def main() -> None:
    exact = pd.read_csv(EXACT_CSV).copy()
    win = pd.read_csv(WINDOW_CSV).copy()

    exact["run_index"] = exact["run_index"].astype(int)
    win["run_index"] = win["run_index"].astype(int)

    if "psnr_metrics_json" not in exact.columns:
        exact["psnr_metrics_json"] = exact["psnr_recomputed_from_png"]

    exact = exact.sort_values("run_index").reset_index(drop=True)

    # -------------------------
    # Candidate-level audit table
    # -------------------------
    rows = []

    for _, erow in exact.iterrows():
        r = int(erow["run_index"])
        row = {
            "run_index": r,
            "final_psnr": float(erow["psnr_metrics_json"]),
            "is_good25": int(float(erow["psnr_metrics_json"]) >= 25.0),
            "is_bad25": int(float(erow["psnr_metrics_json"]) < 25.0),
            "final_exact_operator_loss": float(erow["exact_operator_loss"]),
            "final_exact_loss_rank_all16": int(
                exact["exact_operator_loss"].rank(method="min", ascending=True).loc[erow.name]
            ),
            "final_psnr_rank_all16": int(
                exact["psnr_metrics_json"].rank(method="min", ascending=False).loc[erow.name]
            ),
        }

        for M in FULL_BASELINES:
            runs = list(range(M))
            if r in runs:
                sel = exact_select(exact, runs)
                ora = oracle_select(exact, runs)
                row[f"in_F{M}"] = 1
                row[f"is_F{M}_exact_selected"] = int(r == int(sel["run_index"]))
                row[f"is_F{M}_oracle"] = int(r == int(ora["run_index"]))
            else:
                row[f"in_F{M}"] = 0
                row[f"is_F{M}_exact_selected"] = 0
                row[f"is_F{M}_oracle"] = 0

        for ckpt in CHECKPOINTS:
            g = win[win["checkpoint_step"] == ckpt].copy()
            if g.empty or r not in set(g["run_index"].astype(int)):
                continue

            g_full = add_scores(g)
            grow = g_full[g_full["run_index"] == r].iloc[0]

            row[f"c{ckpt}_sqrt_loss_x0y"] = float(grow["sqrt_loss_x0y_over_y_norm_last"])
            row[f"c{ckpt}_correction_rms"] = float(grow["correction_rms_last"])
            row[f"c{ckpt}_score_abs_simple"] = float(grow["score_abs_simple"])
            row[f"c{ckpt}_rank_loss_full16"] = int(grow["loss_rank"])
            row[f"c{ckpt}_rank_corr_full16"] = int(grow["corr_rank"])
            row[f"c{ckpt}_score_rel_full16"] = float(grow["score_rel_loss_corr"])

            # Also compute rank within the first 5 and first 8, because F5/F8 are where
            # the full exact selector fails.
            for K in [5, 8]:
                gk = g[g["run_index"] < K].copy()
                if r in set(gk["run_index"].astype(int)):
                    gk = add_scores(gk)
                    gkrow = gk[gk["run_index"] == r].iloc[0]
                    row[f"c{ckpt}_rank_loss_first{K}"] = int(gkrow["loss_rank"])
                    row[f"c{ckpt}_rank_corr_first{K}"] = int(gkrow["corr_rank"])
                    row[f"c{ckpt}_score_rel_first{K}"] = float(gkrow["score_rel_loss_corr"])

        rows.append(row)

    audit = pd.DataFrame(rows).sort_values("run_index")
    audit_dest = BASE / "B19_13C_00034_candidate_audit.csv"
    audit.to_csv(audit_dest, index=False)
    print("[write]", audit_dest)

    # -------------------------
    # Policy audit table
    # -------------------------
    policy_rows = []

    for M in FULL_BASELINES:
        runs = list(range(M))
        sel = exact_select(exact, runs)
        ora = oracle_select(exact, runs)

        policy_rows.append({
            "policy": f"F{M}_full_exact",
            "policy_type": "full",
            "cost_full_equiv": float(M),
            "K": M,
            "checkpoint_step": 200,
            "keep_k": M,
            "candidate_runs": ",".join(map(str, runs)),
            "kept_runs": ",".join(map(str, runs)),
            "selected_run": int(sel["run_index"]),
            "selected_psnr": float(sel["psnr_metrics_json"]),
            "selected_exact_loss": float(sel["exact_operator_loss"]),
            "oracle_run": int(ora["run_index"]),
            "oracle_psnr": float(ora["psnr_metrics_json"]),
            "gap_to_oracle": float(ora["psnr_metrics_json"]) - float(sel["psnr_metrics_json"]),
            "selected_bad25": int(float(sel["psnr_metrics_json"]) < 25.0),
            "failure_type": (
                "final_exact_selection_failure"
                if float(ora["psnr_metrics_json"]) >= 25.0 and float(sel["psnr_metrics_json"]) < 25.0
                else "ok"
            ),
        })

    for name, K, ckpt, keep_k in PREFIX_POLICIES:
        g = win[win["checkpoint_step"] == ckpt].copy()
        runs = list(range(K))
        sub = g[g["run_index"].isin(runs)].copy()
        sub = add_scores(sub)

        kept = sub.sort_values(["score_rel_loss_corr", "score_abs_simple", "run_index"]).head(keep_k)
        kept_runs = kept["run_index"].astype(int).tolist()

        sel = exact_select(exact, kept_runs)
        oraK = oracle_select(exact, runs)
        oraKept = oracle_select(exact, kept_runs)

        firstK_has_good25 = int((exact[exact["run_index"].isin(runs)]["psnr_metrics_json"] >= 25.0).any())
        kept_has_good25 = int((exact[exact["run_index"].isin(kept_runs)]["psnr_metrics_json"] >= 25.0).any())

        if firstK_has_good25 == 0:
            failure_type = "init_failure"
        elif kept_has_good25 == 0:
            failure_type = "prefix_selection_failure"
        elif float(sel["psnr_metrics_json"]) < 25.0:
            failure_type = "final_exact_selection_failure"
        else:
            failure_type = "ok"

        policy_rows.append({
            "policy": name,
            "policy_type": "prefix",
            "cost_full_equiv": prefix_cost(K, ckpt, keep_k),
            "K": K,
            "checkpoint_step": ckpt,
            "keep_k": keep_k,
            "candidate_runs": ",".join(map(str, runs)),
            "kept_runs": ",".join(map(str, kept_runs)),
            "selected_run": int(sel["run_index"]),
            "selected_psnr": float(sel["psnr_metrics_json"]),
            "selected_exact_loss": float(sel["exact_operator_loss"]),
            "oracle_run": int(oraK["run_index"]),
            "oracle_psnr": float(oraK["psnr_metrics_json"]),
            "oracle_kept_run": int(oraKept["run_index"]),
            "oracle_kept_psnr": float(oraKept["psnr_metrics_json"]),
            "gap_to_oracle": float(oraK["psnr_metrics_json"]) - float(sel["psnr_metrics_json"]),
            "gap_to_kept_oracle": float(oraKept["psnr_metrics_json"]) - float(sel["psnr_metrics_json"]),
            "selected_bad25": int(float(sel["psnr_metrics_json"]) < 25.0),
            "firstK_has_good25": firstK_has_good25,
            "kept_has_good25": kept_has_good25,
            "failure_type": failure_type,
        })

    policy = pd.DataFrame(policy_rows).sort_values(["cost_full_equiv", "policy"])
    policy_dest = BASE / "B19_13C_00034_policy_audit.csv"
    policy.to_csv(policy_dest, index=False)
    print("[write]", policy_dest)

    # -------------------------
    # Human-readable prints
    # -------------------------
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 260)

    print("\n=== Final candidates sorted by final exact operator loss ===")
    print(
        exact[["run_index", "psnr_metrics_json", "exact_operator_loss", "sqrt_loss_over_y_norm"]]
        .sort_values(["exact_operator_loss", "run_index"])
        .to_string(index=False)
    )

    print("\n=== Final candidates sorted by PSNR ===")
    print(
        exact[["run_index", "psnr_metrics_json", "exact_operator_loss", "sqrt_loss_over_y_norm"]]
        .sort_values(["psnr_metrics_json", "run_index"], ascending=[False, True])
        .to_string(index=False)
    )

    for ckpt in [100, 125, 150]:
        print(f"\n=== Prefix score at checkpoint {ckpt}, first 5 runs ===")
        g = win[(win["checkpoint_step"] == ckpt) & (win["run_index"] < 5)].copy()
        g = add_scores(g)
        view_cols = [
            "run_index", "final_psnr",
            "sqrt_loss_x0y_over_y_norm_last", "correction_rms_last",
            "score_abs_simple", "loss_rank", "corr_rank", "score_rel_loss_corr",
        ]
        print(g[view_cols].sort_values(["score_rel_loss_corr", "score_abs_simple", "run_index"]).to_string(index=False))

    print("\n=== Policy audit ===")
    print(policy.to_string(index=False))

    print("\n=== Key run-4 audit row ===")
    run4_cols = [
        "run_index", "final_psnr", "final_exact_operator_loss",
        "final_exact_loss_rank_all16", "final_psnr_rank_all16",
        "is_F5_exact_selected", "is_F5_oracle",
        "c125_sqrt_loss_x0y", "c125_correction_rms",
        "c125_score_abs_simple", "c125_score_rel_first5",
        "c150_score_abs_simple", "c150_score_rel_first5",
    ]
    existing = [c for c in run4_cols if c in audit.columns]
    print(audit[audit["run_index"] == 4][existing].to_string(index=False))


if __name__ == "__main__":
    main()
