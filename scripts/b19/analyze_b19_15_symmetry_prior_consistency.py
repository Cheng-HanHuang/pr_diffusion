#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image


REPO = Path("/egr/research-pac/huang248/pr_diffusion_b19_solver")
BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")

sys.path.insert(0, str(REPO / "external/daps"))
from forward_operator import get_operator  # noqa: E402


CASES = [
    "00000", "00004", "00007", "00008", "00013",
    "00015", "00017", "00018", "00019", "00020",
    "00025", "00027", "00028", "00032", "00034",
]

CHECKPOINTS = [50, 75, 100, 125, 150, 199]


def zfill_id(x) -> str:
    return str(x).zfill(5)


def window_path(image_id: str) -> Path:
    if image_id == "00007":
        return BASE / "B19_9C_daps_00007_meas3000_16S_rawtraj_window_features.csv"
    return BASE / f"B19_11_daps_{image_id}_meas3000_16S_rawtraj_window_features.csv"


def samples_root(image_id: str) -> Path:
    if image_id == "00007":
        return REPO / "external/daps/results/b19_early_diag_failure/b19_daps_00007_meas3000_16S_rawtraj/samples"
    return REPO / f"external/daps/results/b19_11_hard_mixed_raw16S/b19_daps_{image_id}_meas3000_16S_rawtraj/samples"


def measurement_path(image_id: str) -> Path:
    return BASE / f"measurements/ffhq{image_id}_phase_noise005_meas3000.pt"


def load_model_range(path: Path, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img).astype("float32") / 255.0
    x01 = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x01 * 2.0 - 1.0


def candidate_class(row: pd.Series) -> str:
    if float(row["psnr_identity_recomputed"]) >= 25.0:
        return "unaligned_good"
    if int(row["symmetry_rescuable25"]) == 1:
        return "symmetry_rescuable"
    return "true_bad"


sym = pd.read_csv(BASE / "B19_13D_flip_symmetry_candidate_audit.csv")
sym["image_id"] = sym["image_id"].map(zfill_id)
sym["run_index"] = sym["run_index"].astype(int)
sym["candidate_class"] = sym.apply(candidate_class, axis=1)

rows = []

for image_id in CASES:
    wp = window_path(image_id)
    if not wp.exists():
        print("[missing window]", image_id, wp)
        continue

    win = pd.read_csv(wp)
    win["run_index"] = win["run_index"].astype(int)

    img_sym = sym[sym["image_id"] == image_id].copy()
    if img_sym.empty:
        print("[missing sym]", image_id)
        continue

    for ckpt in CHECKPOINTS:
        g = win[win["checkpoint_step"] == ckpt].copy()
        if g.empty:
            continue

        g["score_abs_simple"] = (
            g["sqrt_loss_x0y_over_y_norm_last"]
            + g["correction_rms_last"]
        )
        g["rank_loss"] = g["sqrt_loss_x0y_over_y_norm_last"].rank(method="min", ascending=True)
        g["rank_corr"] = g["correction_rms_last"].rank(method="min", ascending=True)
        g["score_rel_lc"] = g["rank_loss"] + g["rank_corr"]

        if "sqrt_loss_x0hat_over_y_norm_last" in g.columns:
            g["rank_xhat_loss"] = g["sqrt_loss_x0hat_over_y_norm_last"].rank(method="min", ascending=True)
            g["score_rel_lhc"] = g["rank_loss"] + g["rank_corr"] + g["rank_xhat_loss"]
        else:
            g["rank_xhat_loss"] = np.nan
            g["score_rel_lhc"] = g["score_rel_lc"]

        merged = g.merge(
            img_sym[[
                "image_id", "run_index", "candidate_class",
                "psnr_identity_recomputed", "best_aligned_psnr",
                "best_alignment", "aligned_gain",
                "symmetry_rescuable25", "rot180_rescuable25",
                "exact_operator_loss",
            ]],
            on="run_index",
            how="left",
        )
        merged["image_id"] = image_id
        merged["checkpoint_step"] = ckpt
        merged["top1_lc"] = (merged["score_rel_lc"].rank(method="min") <= 1).astype(int)
        merged["top2_lc"] = (merged["score_rel_lc"].rank(method="min") <= 2).astype(int)
        merged["top4_lc"] = (merged["score_rel_lc"].rank(method="min") <= 4).astype(int)
        merged["top2_lhc"] = (merged["score_rel_lhc"].rank(method="min") <= 2).astype(int)
        rows.append(merged)

feat = pd.concat(rows, ignore_index=True)

feat_dest = BASE / "B19_15_symmetry_prefix_feature_audit.csv"
feat.to_csv(feat_dest, index=False)
print("[write]", feat_dest)

summary = (
    feat.groupby(["checkpoint_step", "candidate_class"])
    .agg(
        candidates=("run_index", "count"),
        mean_score_rel_lc=("score_rel_lc", "mean"),
        median_score_rel_lc=("score_rel_lc", "median"),
        mean_rank_loss=("rank_loss", "mean"),
        mean_rank_corr=("rank_corr", "mean"),
        mean_score_abs=("score_abs_simple", "mean"),
        top1_lc_rate=("top1_lc", "mean"),
        top2_lc_rate=("top2_lc", "mean"),
        top4_lc_rate=("top4_lc", "mean"),
        top2_lhc_rate=("top2_lhc", "mean"),
        mean_identity_psnr=("psnr_identity_recomputed", "mean"),
        mean_best_aligned_psnr=("best_aligned_psnr", "mean"),
    )
    .reset_index()
)

summary_dest = BASE / "B19_15_symmetry_prefix_feature_summary.csv"
summary.to_csv(summary_dest, index=False)
print("[write]", summary_dest)

print("\nPrefix feature summary:")
print(summary.to_string(index=False))

# Measurement invariance check for symmetry-rescuable candidates.
op = get_operator(name="phase_retrieval", sigma=0.05, oversample=2.0)

loss_rows = []
rescue = sym[sym["symmetry_rescuable25"] == 1].copy()

for _, r in rescue.iterrows():
    image_id = str(r["image_id"]).zfill(5)
    run = int(r["run_index"])
    sr = samples_root(image_id)
    mp = measurement_path(image_id)
    sp = sr / f"00000_run{run:04d}.png"

    if not sp.exists() or not mp.exists():
        print("[missing sample/measurement]", image_id, run, sp, mp)
        continue

    payload = torch.load(mp, map_location="cpu")
    y = payload["measurement"]

    x = load_model_range(sp)
    xrot = torch.flip(x, dims=[2, 3])

    with torch.no_grad():
        loss_id = float(op.loss(x, y).detach().flatten()[0])
        loss_rot = float(op.loss(xrot, y).detach().flatten()[0])

    loss_rows.append({
        "image_id": image_id,
        "run_index": run,
        "identity_operator_loss": loss_id,
        "rot180_operator_loss": loss_rot,
        "loss_ratio_rot_over_identity": loss_rot / loss_id if loss_id != 0 else np.nan,
        "identity_psnr": float(r["psnr_identity_recomputed"]),
        "rot180_or_best_psnr": float(r["best_aligned_psnr"]),
        "best_alignment": r["best_alignment"],
    })

loss_df = pd.DataFrame(loss_rows)
loss_dest = BASE / "B19_15_rot180_measurement_invariance.csv"
loss_df.to_csv(loss_dest, index=False)
print("[write]", loss_dest)

print("\nRot180 measurement invariance check:")
print(loss_df.to_string(index=False))
