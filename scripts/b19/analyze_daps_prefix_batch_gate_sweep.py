#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd


LOSS_COL = "sqrt_loss_x0y_over_y_norm_last"
CORR_COL = "correction_rms_last"
X0HAT_LOSS_COL = "sqrt_loss_x0hat_over_y_norm_last"


def add_scores_for_subbatch(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.copy()

    # Absolute score: lower is better.
    sub["score_abs_simple"] = sub[LOSS_COL] + sub[CORR_COL]

    # Relative rank score recomputed within this subbatch.
    sub["loss_rank_in_batch"] = sub[LOSS_COL].rank(method="min", ascending=True)
    sub["corr_rank_in_batch"] = sub[CORR_COL].rank(method="min", ascending=True)
    sub["score_rel_loss_corr"] = sub["loss_rank_in_batch"] + sub["corr_rank_in_batch"]

    if X0HAT_LOSS_COL in sub.columns:
        sub["x0hat_loss_rank_in_batch"] = sub[X0HAT_LOSS_COL].rank(method="min", ascending=True)
        sub["score_rel_loss_hat_corr"] = (
            sub["loss_rank_in_batch"]
            + sub["x0hat_loss_rank_in_batch"]
            + sub["corr_rank_in_batch"]
        )
    else:
        sub["score_rel_loss_hat_corr"] = sub["score_rel_loss_corr"]

    return sub


def summarize_one_subbatch(
    image_id: str,
    measurement_seed: int,
    source_group: str,
    checkpoint_step: int,
    full_g: pd.DataFrame,
    run_tuple: tuple[int, ...],
    keep_k: int,
) -> dict:
    sub = full_g[full_g["run_index"].astype(int).isin(run_tuple)].copy()
    sub = add_scores_for_subbatch(sub)

    batch_has_good25 = int((sub["final_psnr"] >= 25.0).any())
    batch_all_bad25 = int((sub["final_psnr"] < 25.0).all())
    batch_best_psnr = float(sub["final_psnr"].max())
    oracle_run_in_batch = int(sub.loc[sub["final_psnr"].idxmax(), "run_index"])

    kept_rel = sub.sort_values(["score_rel_loss_corr", "score_abs_simple", "run_index"]).head(keep_k)
    kept_rel_hat = sub.sort_values(["score_rel_loss_hat_corr", "score_abs_simple", "run_index"]).head(keep_k)

    top1_abs = sub.sort_values(["score_abs_simple", "run_index"]).head(1)
    top2_abs = sub.sort_values(["score_abs_simple", "run_index"]).head(min(2, len(sub)))

    return {
        "image_id": image_id,
        "measurement_seed": measurement_seed,
        "source_group": source_group,
        "checkpoint_step": checkpoint_step,
        "checkpoint_sigma": float(sub["checkpoint_sigma"].iloc[0]),
        "num_runs_in_batch": len(sub),
        "run_tuple": ",".join(map(str, run_tuple)),
        "batch_has_good25": batch_has_good25,
        "batch_all_bad25": batch_all_bad25,
        "batch_best_psnr": batch_best_psnr,
        "oracle_run_in_batch": oracle_run_in_batch,

        # Absolute gate candidates. Lower means better batch.
        "gate_min_abs_score": float(sub["score_abs_simple"].min()),
        "gate_top2_mean_abs_score": float(top2_abs["score_abs_simple"].mean()),
        "gate_min_sqrt_loss": float(sub[LOSS_COL].min()),
        "gate_top2_mean_sqrt_loss": float(sub.nsmallest(min(2, len(sub)), LOSS_COL)[LOSS_COL].mean()),
        "gate_min_correction": float(sub[CORR_COL].min()),
        "gate_top2_mean_correction": float(sub.nsmallest(min(2, len(sub)), CORR_COL)[CORR_COL].mean()),

        # Relative keep-k result.
        "keep_k": keep_k,
        "rel_keep_runs": ",".join(map(str, kept_rel["run_index"].astype(int).tolist())),
        "rel_keep_psnrs": ",".join(f"{x:.3f}" for x in kept_rel["final_psnr"].tolist()),
        "rel_best_kept_psnr": float(kept_rel["final_psnr"].max()),
        "rel_gap_to_batch_oracle": batch_best_psnr - float(kept_rel["final_psnr"].max()),
        "rel_kept_has_good25": int((kept_rel["final_psnr"] >= 25.0).any()),
        "rel_num_bad25_kept": int((kept_rel["final_psnr"] < 25.0).sum()),

        # Three-feature relative keep-k result.
        "relhat_keep_runs": ",".join(map(str, kept_rel_hat["run_index"].astype(int).tolist())),
        "relhat_keep_psnrs": ",".join(f"{x:.3f}" for x in kept_rel_hat["final_psnr"].tolist()),
        "relhat_best_kept_psnr": float(kept_rel_hat["final_psnr"].max()),
        "relhat_gap_to_batch_oracle": batch_best_psnr - float(kept_rel_hat["final_psnr"].max()),
        "relhat_kept_has_good25": int((kept_rel_hat["final_psnr"] >= 25.0).any()),
        "relhat_num_bad25_kept": int((kept_rel_hat["final_psnr"] < 25.0).sum()),
    }


def threshold_sweep(batch_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gate_cols = [
        "gate_min_abs_score",
        "gate_top2_mean_abs_score",
        "gate_min_sqrt_loss",
        "gate_top2_mean_sqrt_loss",
        "gate_min_correction",
        "gate_top2_mean_correction",
    ]

    for (ckpt, keep_k), g0 in batch_df.groupby(["checkpoint_step", "keep_k"]):
        g0 = g0.copy()

        for gate_col in gate_cols:
            values = sorted(g0[gate_col].unique())
            if not values:
                continue

            candidates = []
            candidates.append(min(values) - 1e-9)
            for a, b in zip(values[:-1], values[1:]):
                candidates.append((a + b) / 2.0)
            candidates.append(max(values) + 1e-9)

            for tau in candidates:
                g = g0.copy()

                # Lower gate score means better. Reject if score is too high.
                g["gate_reject"] = (g[gate_col] > tau).astype(int)

                pos = g[g["batch_has_good25"] == 1]
                neg = g[g["batch_has_good25"] == 0]

                accepted = g[g["gate_reject"] == 0]
                accepted_pos = accepted[accepted["batch_has_good25"] == 1]
                accepted_neg = accepted[accepted["batch_has_good25"] == 0]

                rows.append({
                    "checkpoint_step": int(ckpt),
                    "keep_k": int(keep_k),
                    "gate_col": gate_col,
                    "tau": float(tau),
                    "num_batches": len(g),
                    "num_good_batches": len(pos),
                    "num_all_bad_batches": len(neg),

                    "good_batches_rejected": int((pos["gate_reject"] == 1).sum()),
                    "all_bad_batches_accepted": int((neg["gate_reject"] == 0).sum()),
                    "all_bad_batches_rejected": int((neg["gate_reject"] == 1).sum()),

                    "accepted_batches": len(accepted),
                    "accepted_good_batches": len(accepted_pos),
                    "accepted_all_bad_batches": len(accepted_neg),

                    "accepted_good_rel_keep_success": int(accepted_pos["rel_kept_has_good25"].sum()) if len(accepted_pos) else 0,
                    "accepted_good_relhat_keep_success": int(accepted_pos["relhat_kept_has_good25"].sum()) if len(accepted_pos) else 0,

                    "mean_rel_gap_accepted_good": float(accepted_pos["rel_gap_to_batch_oracle"].mean()) if len(accepted_pos) else float("nan"),
                    "max_rel_gap_accepted_good": float(accepted_pos["rel_gap_to_batch_oracle"].max()) if len(accepted_pos) else float("nan"),
                    "mean_relhat_gap_accepted_good": float(accepted_pos["relhat_gap_to_batch_oracle"].mean()) if len(accepted_pos) else float("nan"),
                    "max_relhat_gap_accepted_good": float(accepted_pos["relhat_gap_to_batch_oracle"].max()) if len(accepted_pos) else float("nan"),
                })

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")
    ap.add_argument("--out_batch_csv", required=True)
    ap.add_argument("--out_sweep_csv", required=True)
    ap.add_argument("--subset_size", type=int, default=4)
    ap.add_argument("--keep_k_values", default="1,2")
    args = ap.parse_args()

    base = Path(args.base)
    keep_k_values = [int(x) for x in args.keep_k_values.split(",") if x.strip()]

    cases = [
        ("00028", 3000, "panel4S", base / "B19_9_daps_00028_meas3000_4S_rawtraj_window_features.csv"),
        ("00013", 3002, "panel4S", base / "B19_9B_daps_00013_meas3002_4S_rawtraj_window_features.csv"),
        ("00017", 3000, "panel4S", base / "B19_9B_daps_00017_meas3000_4S_rawtraj_window_features.csv"),
        ("00018", 3000, "panel4S", base / "B19_9B_daps_00018_meas3000_4S_rawtraj_window_features.csv"),
        ("00027", 3000, "panel4S", base / "B19_9B_daps_00027_meas3000_4S_rawtraj_window_features.csv"),
        ("00034", 3000, "panel4S", base / "B19_9B_daps_00034_meas3000_4S_rawtraj_window_features.csv"),
        ("00007", 3000, "failure16S", base / "B19_9C_daps_00007_meas3000_16S_rawtraj_window_features.csv"),
    ]

    batch_rows = []
    checkpoints = [75, 100, 125, 150]

    for image_id, seed, group, path in cases:
        if not path.exists():
            print("[missing]", image_id, seed, path)
            continue

        df = pd.read_csv(path)

        for ckpt in checkpoints:
            g = df[df["checkpoint_step"] == ckpt].copy()
            if g.empty:
                continue

            runs = sorted(g["run_index"].astype(int).unique().tolist())

            # For 16S, enumerate every 4-subset. For 4S panels, this is just the one observed batch.
            if len(runs) >= args.subset_size:
                run_tuples = list(combinations(runs, args.subset_size))
            else:
                continue

            for run_tuple in run_tuples:
                for keep_k in keep_k_values:
                    batch_rows.append(
                        summarize_one_subbatch(
                            image_id=image_id,
                            measurement_seed=seed,
                            source_group=group,
                            checkpoint_step=ckpt,
                            full_g=g,
                            run_tuple=run_tuple,
                            keep_k=keep_k,
                        )
                    )

    batch_df = pd.DataFrame(batch_rows).sort_values(
        ["checkpoint_step", "source_group", "image_id", "run_tuple", "keep_k"]
    )
    Path(args.out_batch_csv).parent.mkdir(parents=True, exist_ok=True)
    batch_df.to_csv(args.out_batch_csv, index=False)
    print("[write]", args.out_batch_csv)
    print("batch rows:", len(batch_df))

    sweep = threshold_sweep(batch_df)
    sweep.to_csv(args.out_sweep_csv, index=False)
    print("[write]", args.out_sweep_csv)
    print("sweep rows:", len(sweep))

    print("\nSubbatch label counts by checkpoint/source:")
    print(
        batch_df.drop_duplicates(["checkpoint_step", "source_group", "image_id", "run_tuple"])
        .groupby(["checkpoint_step", "source_group"])
        .agg(
            batches=("run_tuple", "count"),
            good_batches=("batch_has_good25", "sum"),
            all_bad_batches=("batch_all_bad25", "sum"),
        )
        .to_string()
    )

    print("\nBest no-good-rejection thresholds, sorted by fewest all-bad accepted:")
    cand = sweep[sweep["good_batches_rejected"] == 0].copy()
    if len(cand):
        print(
            cand.sort_values(
                ["checkpoint_step", "keep_k", "all_bad_batches_accepted", "gate_col"]
            )
            .groupby(["checkpoint_step", "keep_k"])
            .head(5)
            [[
                "checkpoint_step", "keep_k", "gate_col", "tau",
                "num_good_batches", "num_all_bad_batches",
                "good_batches_rejected", "all_bad_batches_accepted",
                "all_bad_batches_rejected",
                "accepted_good_rel_keep_success",
                "accepted_good_relhat_keep_success",
                "max_rel_gap_accepted_good",
            ]]
            .to_string(index=False)
        )

    print("\nBest no-all-bad-accepted thresholds, sorted by fewest good rejected:")
    cand = sweep[sweep["all_bad_batches_accepted"] == 0].copy()
    if len(cand):
        print(
            cand.sort_values(
                ["checkpoint_step", "keep_k", "good_batches_rejected", "gate_col"]
            )
            .groupby(["checkpoint_step", "keep_k"])
            .head(5)
            [[
                "checkpoint_step", "keep_k", "gate_col", "tau",
                "num_good_batches", "num_all_bad_batches",
                "good_batches_rejected", "all_bad_batches_accepted",
                "all_bad_batches_rejected",
                "accepted_good_rel_keep_success",
                "accepted_good_relhat_keep_success",
                "max_rel_gap_accepted_good",
            ]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
