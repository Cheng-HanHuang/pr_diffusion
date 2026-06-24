#!/usr/bin/env python3
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd


BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")
CKPT = 125
TAU = 0.339962
SUBSET_SIZE = 4
KEEP_K = 2

CASES = [
    "00000", "00007", "00008", "00013", "00015", "00017", "00018",
    "00019", "00020", "00027", "00028", "00032", "00034",
    "00004", "00025",
]


def window_path(image_id: str) -> Path:
    if image_id == "00007":
        return BASE / "B19_9C_daps_00007_meas3000_16S_rawtraj_window_features.csv"
    return BASE / f"B19_11_daps_{image_id}_meas3000_16S_rawtraj_window_features.csv"


def add_scores(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.copy()
    sub["score_abs_simple"] = (
        sub["sqrt_loss_x0y_over_y_norm_last"]
        + sub["correction_rms_last"]
    )
    sub["loss_rank_in_batch"] = sub["sqrt_loss_x0y_over_y_norm_last"].rank(method="min", ascending=True)
    sub["corr_rank_in_batch"] = sub["correction_rms_last"].rank(method="min", ascending=True)
    sub["score_rel_loss_corr"] = sub["loss_rank_in_batch"] + sub["corr_rank_in_batch"]
    return sub


rows = []

for image_id in CASES:
    path = window_path(image_id)
    if not path.exists():
        print("[missing]", image_id, path)
        continue

    df = pd.read_csv(path)
    g = df[df["checkpoint_step"] == CKPT].copy()
    if g.empty:
        print("[missing ckpt]", image_id, CKPT)
        continue

    runs = sorted(g["run_index"].astype(int).unique().tolist())
    if len(runs) < SUBSET_SIZE:
        print("[too few runs]", image_id, runs)
        continue

    for run_tuple in combinations(runs, SUBSET_SIZE):
        sub = g[g["run_index"].astype(int).isin(run_tuple)].copy()
        sub = add_scores(sub)

        batch_has_good25 = int((sub["final_psnr"] >= 25.0).any())
        batch_all_bad25 = int((sub["final_psnr"] < 25.0).all())
        batch_best_psnr = float(sub["final_psnr"].max())

        gate_min_abs_score = float(sub["score_abs_simple"].min())
        gate_accept = int(gate_min_abs_score <= TAU)

        kept = sub.sort_values(["score_rel_loss_corr", "score_abs_simple", "run_index"]).head(KEEP_K)
        kept_has_good25 = int((kept["final_psnr"] >= 25.0).any())
        best_kept_psnr = float(kept["final_psnr"].max())

        rows.append({
            "image_id": image_id,
            "checkpoint_step": CKPT,
            "tau": TAU,
            "run_tuple": ",".join(map(str, run_tuple)),
            "batch_has_good25": batch_has_good25,
            "batch_all_bad25": batch_all_bad25,
            "batch_best_psnr": batch_best_psnr,
            "gate_min_abs_score": gate_min_abs_score,
            "gate_accept": gate_accept,
            "gate_reject": int(not gate_accept),
            "kept_runs": ",".join(map(str, kept["run_index"].astype(int).tolist())),
            "kept_psnrs": ",".join(f"{x:.3f}" for x in kept["final_psnr"].tolist()),
            "kept_has_good25": kept_has_good25,
            "best_kept_psnr": best_kept_psnr,
            "gap_best_kept_to_batch_oracle": batch_best_psnr - best_kept_psnr,
            "num_bad25_in_batch": int((sub["final_psnr"] < 25.0).sum()),
            "num_bad25_kept": int((kept["final_psnr"] < 25.0).sum()),
        })

out = pd.DataFrame(rows)
dest = BASE / "B19_11_hard_mixed_gate_validation_subbatch_table.csv"
out.to_csv(dest, index=False)
print("[write]", dest)
print("rows:", len(out))

summary = (
    out.groupby("image_id")
    .agg(
        subbatches=("run_tuple", "count"),
        good_batches=("batch_has_good25", "sum"),
        all_bad_batches=("batch_all_bad25", "sum"),
        good_batches_rejected=("gate_reject", lambda s: int(((s == 1) & (out.loc[s.index, "batch_has_good25"] == 1)).sum())),
        all_bad_batches_accepted=("gate_accept", lambda s: int(((s == 1) & (out.loc[s.index, "batch_all_bad25"] == 1)).sum())),
        accepted_good_batches=("gate_accept", lambda s: int(((s == 1) & (out.loc[s.index, "batch_has_good25"] == 1)).sum())),
        accepted_good_keep_success=("kept_has_good25", lambda s: int(((s == 1) & (out.loc[s.index, "gate_accept"] == 1) & (out.loc[s.index, "batch_has_good25"] == 1)).sum())),
        max_gap_accepted_good=("gap_best_kept_to_batch_oracle", lambda s: float(out.loc[s.index][(out.loc[s.index, "gate_accept"] == 1) & (out.loc[s.index, "batch_has_good25"] == 1)]["gap_best_kept_to_batch_oracle"].max()) if len(out.loc[s.index][(out.loc[s.index, "gate_accept"] == 1) & (out.loc[s.index, "batch_has_good25"] == 1)]) else float("nan")),
        mean_gap_accepted_good=("gap_best_kept_to_batch_oracle", lambda s: float(out.loc[s.index][(out.loc[s.index, "gate_accept"] == 1) & (out.loc[s.index, "batch_has_good25"] == 1)]["gap_best_kept_to_batch_oracle"].mean()) if len(out.loc[s.index][(out.loc[s.index, "gate_accept"] == 1) & (out.loc[s.index, "batch_has_good25"] == 1)]) else float("nan")),
    )
    .reset_index()
)

summary_dest = BASE / "B19_11_hard_mixed_gate_validation_by_image.csv"
summary.to_csv(summary_dest, index=False)
print("[write]", summary_dest)
print("\nBy image:")
print(summary.to_string(index=False))

print("\nOverall:")
print("images:", summary["image_id"].nunique())
print("subbatches:", len(out))
print("good batches:", int(out["batch_has_good25"].sum()))
print("all-bad batches:", int(out["batch_all_bad25"].sum()))
print("good batches rejected:", int(((out["gate_reject"] == 1) & (out["batch_has_good25"] == 1)).sum()))
print("all-bad batches accepted:", int(((out["gate_accept"] == 1) & (out["batch_all_bad25"] == 1)).sum()))

accepted_good = out[(out["gate_accept"] == 1) & (out["batch_has_good25"] == 1)]
print("accepted good batches:", len(accepted_good))
print("accepted good keep success:", int(accepted_good["kept_has_good25"].sum()))
print("max gap accepted good:", float(accepted_good["gap_best_kept_to_batch_oracle"].max()) if len(accepted_good) else float("nan"))
print("mean gap accepted good:", float(accepted_good["gap_best_kept_to_batch_oracle"].mean()) if len(accepted_good) else float("nan"))

print("\nFailures / suspicious accepted all-bad:")
bad_acc = out[(out["gate_accept"] == 1) & (out["batch_all_bad25"] == 1)]
print(bad_acc.sort_values(["image_id", "gate_min_abs_score"]).head(50).to_string(index=False))

print("\nRejected good batches:")
good_rej = out[(out["gate_reject"] == 1) & (out["batch_has_good25"] == 1)]
print(good_rej.sort_values(["image_id", "gate_min_abs_score"]).head(50).to_string(index=False))
