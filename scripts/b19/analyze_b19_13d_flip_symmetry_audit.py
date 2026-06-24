#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image


REPO = Path("/egr/research-pac/huang248/pr_diffusion_b19_solver")
BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")
DATA_ROOT = Path("/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024")

CASES = [
    "00000", "00004", "00007", "00008", "00013",
    "00015", "00017", "00018", "00019", "00020",
    "00025", "00027", "00028", "00032", "00034",
]


def load_model_range(path: Path, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img).astype("float32") / 255.0
    x01 = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x01 * 2.0 - 1.0


def psnr_model_range(x: torch.Tensor, y: torch.Tensor) -> float:
    x01 = (x.clamp(-1, 1) + 1.0) / 2.0
    y01 = (y.clamp(-1, 1) + 1.0) / 2.0
    mse = (x01 - y01).pow(2).flatten(1).mean().clamp_min(1e-12)
    return float(-10.0 * torch.log10(mse))


def exact_path(image_id: str) -> Path:
    if image_id == "00007":
        p = BASE / "B19_8B_daps16S_00007_meas3000_exact_final_loss_selector.csv"
        if p.exists():
            return p
    return BASE / f"B19_11_daps16S_{image_id}_meas3000_exact_final_loss_selector.csv"


def samples_root(image_id: str) -> Path:
    if image_id == "00007":
        return REPO / "external/daps/results/b19_early_diag_failure/b19_daps_00007_meas3000_16S_rawtraj/samples"
    return REPO / f"external/daps/results/b19_11_hard_mixed_raw16S/b19_daps_{image_id}_meas3000_16S_rawtraj/samples"


def source_path(image_id: str) -> Path:
    return DATA_ROOT / "00000" / f"{image_id}.png"


rows = []

for image_id in CASES:
    ep = exact_path(image_id)
    sr = samples_root(image_id)
    sp = source_path(image_id)

    if not ep.exists():
        print("[missing exact]", image_id, ep)
        continue
    if not sr.exists():
        print("[missing samples]", image_id, sr)
        continue
    if not sp.exists():
        print("[missing source]", image_id, sp)
        continue

    exact = pd.read_csv(ep)
    gt = load_model_range(sp, size=256)

    for _, row in exact.iterrows():
        run = int(row["run_index"])
        sample_path = sr / f"00000_run{run:04d}.png"
        if not sample_path.exists():
            print("[missing sample]", image_id, run, sample_path)
            continue

        x = load_model_range(sample_path, size=256)

        variants = {
            "identity": x,
            "hflip": torch.flip(x, dims=[3]),
            "vflip": torch.flip(x, dims=[2]),
            "rot180": torch.flip(x, dims=[2, 3]),
        }

        ps = {name: psnr_model_range(v, gt) for name, v in variants.items()}
        best_align = max(ps, key=ps.get)
        best_psnr = ps[best_align]

        unaligned = float(row.get("psnr_metrics_json", row.get("psnr_recomputed_from_png", ps["identity"])))

        rows.append({
            "image_id": image_id,
            "run_index": run,
            "exact_operator_loss": float(row["exact_operator_loss"]),
            "psnr_unaligned_metrics": unaligned,
            "psnr_identity_recomputed": ps["identity"],
            "psnr_hflip": ps["hflip"],
            "psnr_vflip": ps["vflip"],
            "psnr_rot180": ps["rot180"],
            "best_aligned_psnr": best_psnr,
            "best_alignment": best_align,
            "aligned_gain": best_psnr - ps["identity"],
            "bad25_unaligned": int(ps["identity"] < 25.0),
            "good25_after_alignment": int(best_psnr >= 25.0),
            "symmetry_rescuable25": int((ps["identity"] < 25.0) and (best_psnr >= 25.0)),
            "rot180_rescuable25": int((ps["identity"] < 25.0) and (ps["rot180"] >= 25.0)),
        })

out = pd.DataFrame(rows).sort_values(["image_id", "run_index"])
dest = BASE / "B19_13D_flip_symmetry_candidate_audit.csv"
out.to_csv(dest, index=False)
print("[write]", dest)

summary = (
    out.groupby("image_id")
    .agg(
        candidates=("run_index", "count"),
        bad25_unaligned=("bad25_unaligned", "sum"),
        symmetry_rescuable25=("symmetry_rescuable25", "sum"),
        rot180_rescuable25=("rot180_rescuable25", "sum"),
        max_aligned_gain=("aligned_gain", "max"),
        mean_aligned_gain=("aligned_gain", "mean"),
    )
    .reset_index()
)

summary_dest = BASE / "B19_13D_flip_symmetry_by_image.csv"
summary.to_csv(summary_dest, index=False)
print("[write]", summary_dest)

print("\nBy image:")
print(summary.to_string(index=False))

print("\nLikely symmetry-rescuable candidates:")
view = out[out["symmetry_rescuable25"] == 1].copy()
if len(view):
    print(
        view[[
            "image_id", "run_index", "psnr_identity_recomputed",
            "best_aligned_psnr", "best_alignment", "aligned_gain",
            "exact_operator_loss",
        ]]
        .sort_values(["image_id", "run_index"])
        .to_string(index=False)
    )
else:
    print("none")
