#!/usr/bin/env python3
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd


BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")
CHECKPOINTS = [75, 100, 125, 150]
SUBSET_SIZES = [4, 6, 8, 12, 16]
KEEP_K_VALUES = [1, 2, 4]

CASES = [
    "00000", "00004", "00007", "00008", "00013",
    "00015", "00017", "00018", "00019", "00020",
    "00025", "00027", "00028", "00032", "00034",
]


def window_path(image_id: str) -> Path:
    if image_id == "00007":
        return BASE / "B19_9C_daps_00007_meas3000_16S_rawtraj_window_features.csv"
    return BASE / f"B19_11_daps_{image_id}_meas3000_16S_rawtraj_window_features.csv"


def add_scores(sub: pd.DataFrame) -> pd.DataFrame:
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


rows = []

for image_id in CASES:
    path = window_path(image_id)
    if not path.exists():
        print("[missing]", image_id, path)
        continue

    df = pd.read_csv(path)

    for ckpt in CHECKPOINTS:
        g = df[df["checkpoint_step"] == ckpt].copy()
        if g.empty:
            print("[missing checkpoint]", image_id, ckpt)
            continue

        runs = sorted(g["run_index"].astype(int).unique().tolist())
        full_good_count = int((g["final_psnr"] >= 25.0).sum())
        full_bad_count = int((g["final_psnr"] < 25.0).sum())
        full_oracle = float(g["final_psnr"].max())

        for subset_size in SUBSET_SIZES:
            if len(runs) < subset_size:
                continue

            for keep_k in KEEP_K_VALUES:
                if keep_k > subset_size:
                    continue

                num_batches = 0
                good_batches = 0
                all_bad_batches = 0

                rel_success = 0
                relhat_success = 0

                rel_gaps = []
                relhat_gaps = []

                rel_bad_kept = 0
                relhat_bad_kept = 0

                for run_tuple in combinations(runs, subset_size):
                    sub = g[g["run_index"].astype(int).isin(run_tuple)].copy()
                    sub = add_scores(sub)

                    batch_has_good = int((sub["final_psnr"] >= 25.0).any())
                    batch_all_bad = int((sub["final_psnr"] < 25.0).all())
                    batch_oracle = float(sub["final_psnr"].max())

                    num_batches += 1
                    good_batches += batch_has_good
                    all_bad_batches += batch_all_bad

                    if batch_has_good:
                        kept = sub.sort_values(
                            ["score_rel_loss_corr", "score_abs_simple", "run_index"]
                        ).head(keep_k)
                        kept_hat = sub.sort_values(
                            ["score_rel_loss_hat_corr", "score_abs_simple", "run_index"]
                        ).head(keep_k)

                        best_kept = float(kept["final_psnr"].max())
                        best_kept_hat = float(kept_hat["final_psnr"].max())

                        rel_success += int((kept["final_psnr"] >= 25.0).any())
                        relhat_success += int((kept_hat["final_psnr"] >= 25.0).any())

                        rel_gaps.append(batch_oracle - best_kept)
                        relhat_gaps.append(batch_oracle - best_kept_hat)

                        rel_bad_kept += int((kept["final_psnr"] < 25.0).sum())
                        relhat_bad_kept += int((kept_hat["final_psnr"] < 25.0).sum())

                rows.append({
                    "image_id": image_id,
                    "checkpoint_step": ckpt,
                    "subset_size": subset_size,
                    "keep_k": keep_k,
                    "full_good_count_16": full_good_count,
                    "full_bad_count_16": full_bad_count,
                    "full_oracle_psnr": full_oracle,
                    "num_batches": num_batches,
                    "good_batches": good_batches,
                    "all_bad_batches": all_bad_batches,
                    "all_bad_rate": all_bad_batches / num_batches if num_batches else float("nan"),
                    "rel_success": rel_success,
                    "rel_success_rate_among_good_batches": rel_success / good_batches if good_batches else float("nan"),
                    "relhat_success": relhat_success,
                    "relhat_success_rate_among_good_batches": relhat_success / good_batches if good_batches else float("nan"),
                    "rel_mean_gap_good_batches": sum(rel_gaps) / len(rel_gaps) if rel_gaps else float("nan"),
                    "rel_max_gap_good_batches": max(rel_gaps) if rel_gaps else float("nan"),
                    "relhat_mean_gap_good_batches": sum(relhat_gaps) / len(relhat_gaps) if relhat_gaps else float("nan"),
                    "relhat_max_gap_good_batches": max(relhat_gaps) if relhat_gaps else float("nan"),
                    "rel_bad_kept_total_good_batches": rel_bad_kept,
                    "relhat_bad_kept_total_good_batches": relhat_bad_kept,
                })

out = pd.DataFrame(rows).sort_values(["checkpoint_step", "subset_size", "keep_k", "image_id"])

dest = BASE / "B19_12_prefix_population_size_summary.csv"
out.to_csv(dest, index=False)
print("[write]", dest)

print("\nBy image / checkpoint / K / keep_k:")
print(out.to_string(index=False))

overall = (
    out.groupby(["checkpoint_step", "subset_size", "keep_k"])
    .agg(
        images=("image_id", "count"),
        total_batches=("num_batches", "sum"),
        total_good_batches=("good_batches", "sum"),
        total_all_bad_batches=("all_bad_batches", "sum"),
        mean_all_bad_rate=("all_bad_rate", "mean"),
        total_rel_success=("rel_success", "sum"),
        total_relhat_success=("relhat_success", "sum"),
        mean_rel_success_rate=("rel_success_rate_among_good_batches", "mean"),
        mean_relhat_success_rate=("relhat_success_rate_among_good_batches", "mean"),
        max_rel_gap=("rel_max_gap_good_batches", "max"),
        max_relhat_gap=("relhat_max_gap_good_batches", "max"),
        mean_rel_gap=("rel_mean_gap_good_batches", "mean"),
        mean_relhat_gap=("relhat_mean_gap_good_batches", "mean"),
    )
    .reset_index()
)

overall_dest = BASE / "B19_12_prefix_population_size_overall.csv"
overall.to_csv(overall_dest, index=False)
print("\n[write]", overall_dest)

print("\nOverall:")
print(overall.to_string(index=False))

print("\nBest rows by low all-bad and high relative success:")
view = overall.sort_values([
    "checkpoint_step",
    "subset_size",
    "keep_k",
])
print(view.to_string(index=False))
