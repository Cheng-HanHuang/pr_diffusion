#!/usr/bin/env python3
"""Run NP variants with a guided-diffusion FFHQ checkpoint.

This is the guided-diffusion backend counterpart to
``scripts/pr_external_difffpr_np_benchmark.py``.  It reuses that script's
DiffFPR-style oversampled Fourier measurement, scoring, projection, alignment,
and CSV summary code, but swaps the prior backend from Hugging Face diffusers to
OpenAI guided-diffusion checkpoint format.

Primary intended use:

    ffhq_10m.pt + guided_diffusion package from DiffFPR/DiffPIR/DPS/OpenAI.

Example:

    CUDA_VISIBLE_DEVICES=0 python scripts/pr_external_difffpr_np_guided_benchmark.py \
      --guided_diffusion_dir /egr/research-pac/huang248/external/DiffFPR \
      --guided_model_path /egr/research-pac/huang248/models/ffhq_10m.pt \
      --data_root /egr/research-pac/huang248/data/ffhq/ffhq-dataset \
      --image_list_file /egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/splits/ffhq_available25.txt \
      --outdir /egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/external_ffhq25_np_canonical_ffhq10m_sigma005 \
      --variants np_canonical --seeds 100,101
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

# Import helper functions from the existing diffusers benchmark script by path.
# This keeps the measurement operator and reporting exactly aligned.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE_SCRIPT = _REPO_ROOT / "scripts" / "pr_external_difffpr_np_benchmark.py"
_spec = importlib.util.spec_from_file_location("pr_external_difffpr_np_benchmark", _BASE_SCRIPT)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load helper script at {_BASE_SCRIPT}")
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)

from prdiffusion.guided_backend import load_guided_diffusion_model
from prdiffusion.io import load_image


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run np_canonical / np_fixedk_lateproj with guided-diffusion ffhq_10m.pt."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--image_list_file", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--guided_model_path", required=True)
    parser.add_argument(
        "--guided_diffusion_dir",
        default=None,
        help="Repo root containing guided_diffusion/. Can be DiffFPR, DPS, or OpenAI guided-diffusion.",
    )
    parser.add_argument("--guided_preset", default="difffpr_ffhq_10m")
    parser.add_argument("--guided_strict", action="store_true", help="Use strict checkpoint loading.")
    parser.add_argument("--variants", default="np_canonical,np_fixedk_lateproj")
    parser.add_argument("--seeds", default="100,101,102,103,104,105,106,107,108,109")
    parser.add_argument("--np_steps", type=int, default=1000)
    parser.add_argument("--late_start", type=int, default=400)
    parser.add_argument("--fixed_k", type=int, default=5)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--oversample", type=float, default=4.0)
    parser.add_argument("--measurement_noise_std", type=float, default=0.05)
    parser.add_argument("--measurement_noise_seed", type=int, default=20260423)
    parser.add_argument("--clip_noisy_magnitude", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    image_names = base.collect_images(args.data_root, args.image_list_file)
    if not image_names:
        raise ValueError(f"No images found in split {args.image_list_file}")
    seeds = base.parse_int_list(args.seeds)
    variant_names = [tok.strip() for tok in args.variants.split(",") if tok.strip()]
    variants = base.make_variants(variant_names, late_start=args.late_start, fixed_k=args.fixed_k)

    os.makedirs(args.outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.outdir, f"difffpr_np_guided_{stamp}")
    os.makedirs(run_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_guided_diffusion_model(
        model_path=args.guided_model_path,
        device=device,
        preset=args.guided_preset,
        guided_diffusion_dir=args.guided_diffusion_dir,
        strict=bool(args.guided_strict),
    )
    image_size = int(bundle.unet.config.sample_size)
    pad = base.oversample_pad(image_size, args.oversample)

    config_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []

    for variant in variants:
        config_rows.append(
            {
                "variant": variant.name,
                "backend": "guided_diffusion",
                "guided_preset": args.guided_preset,
                "guided_model_path": str(Path(args.guided_model_path).expanduser()),
                "guided_diffusion_dir": args.guided_diffusion_dir or "PYTHONPATH/default",
                "num_steps": args.np_steps,
                "score_radius": args.radius,
                "proj_radius": args.radius,
                "proj_start": variant.proj_start,
                "num_candidates_soft": variant.soft,
                "num_candidates_hard": variant.hard,
                "use_lowfreq_score": variant.use_lowfreq_score,
                "use_lowfreq_projection": variant.use_lowfreq_projection,
                "measurement_operator": "DiffFPR-style centered FFT magnitude after symmetric zero padding",
                "oversample_arg": args.oversample,
                "pad_pixels_each_side": pad,
                "measurement_noise_std": args.measurement_noise_std,
                "clip_noisy_magnitude": bool(args.clip_noisy_magnitude),
                "seeds": ",".join(map(str, seeds)),
                "guided_model_config": repr(bundle.model_config),
            }
        )

    for image_index, image_name in enumerate(image_names):
        img_path = base.resolve_image_path(args.data_root, image_name)
        x_gt = load_image(img_path, size=image_size, device=device)
        mag_clean = base.oversampled_magnitude(x_gt, pad)

        mag_target = mag_clean
        if args.measurement_noise_std > 0:
            gen = torch.Generator(device=device).manual_seed(args.measurement_noise_seed + image_index)
            noise = torch.randn(mag_clean.shape, device=device, dtype=mag_clean.dtype, generator=gen)
            mag_target = mag_clean + float(args.measurement_noise_std) * noise
            if args.clip_noisy_magnitude:
                mag_target = mag_target.clamp_min(0.0)

        for variant in variants:
            for seed in seeds:
                t0 = time.perf_counter()
                x_rec = base.noise_picking_reconstruct_oversampled(
                    mag_target,
                    pad=pad,
                    seed=seed,
                    unet=bundle.unet,
                    scheduler=bundle.scheduler,
                    device=device,
                    variant=variant,
                    num_steps=args.np_steps,
                    radius=args.radius,
                    log_every=args.log_every,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                runtime_s = time.perf_counter() - t0

                x_aligned = base.best_rot180_channel_alignment(x_rec, x_gt)
                row = {
                    "timestamp": stamp,
                    "variant": variant.name,
                    "backend": "guided_diffusion",
                    "guided_preset": args.guided_preset,
                    "image_basename": image_name,
                    "seed": seed,
                    "raw_psnr": base.psnr01_from_model_range(x_rec, x_gt),
                    "aligned_psnr": base.psnr01_from_model_range(x_aligned, x_gt),
                    "clean_mag_l2": float(base.oversampled_mag_l2(x_aligned, mag_clean, pad).cpu().item()),
                    "noisy_mag_l2": float(base.oversampled_mag_l2(x_aligned, mag_target, pad).cpu().item()),
                    "clean_lowfreq_mag_l2": float(
                        base.oversampled_lowfreq_mag_l2(x_aligned, mag_clean, pad, args.radius).cpu().item()
                    ),
                    "noisy_lowfreq_mag_l2": float(
                        base.oversampled_lowfreq_mag_l2(x_aligned, mag_target, pad, args.radius).cpu().item()
                    ),
                    "runtime_s": runtime_s,
                    "num_steps": args.np_steps,
                    "proj_start": variant.proj_start,
                    "num_candidates_soft": variant.soft,
                    "num_candidates_hard": variant.hard,
                    "radius": args.radius,
                    "oversample": args.oversample,
                    "pad_pixels_each_side": pad,
                    "measurement_noise_std": args.measurement_noise_std,
                }
                run_rows.append(row)
                print(
                    f"[Guided-FFHQ-NP] {variant.name} | {image_name} seed={seed} "
                    f"aligned_psnr={row['aligned_psnr']:.2f} raw_psnr={row['raw_psnr']:.2f} "
                    f"({runtime_s:.1f}s)"
                )

    write_csv(os.path.join(run_root, "configs.csv"), config_rows)
    write_csv(os.path.join(run_root, "run_level.csv"), run_rows)
    write_csv(os.path.join(run_root, "image_level_summary.csv"), base.summarize_image_level(run_rows))
    print(f"Saved: {run_root}")


if __name__ == "__main__":
    main()
