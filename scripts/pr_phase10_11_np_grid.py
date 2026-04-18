#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import torch

from prdiffusion.algorithms.noise_picking import NoisePickingConfig, noise_picking_reconstruct
from prdiffusion.diffusion import load_model
from prdiffusion.fft_ops import magnitude
from prdiffusion.io import find_image_by_basename, load_image
from prdiffusion.metrics import lowfreq_mag_l2, mag_l2, psnr


def parse_int_list(text: str) -> List[int]:
    return [int(tok.strip()) for tok in text.split(",") if tok.strip()]


def read_image_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class NPVariant:
    name: str
    soft: int
    hard: int
    proj_start: int
    use_score: bool
    use_projection: bool


def variants_for_phase(phase: str, late_start: int, fixed_k: int, canonical_soft: int, canonical_hard: int) -> List[NPVariant]:
    if phase == "phase10":
        return [
            NPVariant("np_canonical", canonical_soft, canonical_hard, late_start, True, True),
            NPVariant("np_fixedk_lateproj", fixed_k, fixed_k, late_start, True, True),
            NPVariant("np_fixedk_alwaysproj", fixed_k, fixed_k, 0, True, True),
            NPVariant("np_fixedk_noproj", fixed_k, fixed_k, late_start, True, False),
            NPVariant("np_candidate_switch_only", canonical_soft, canonical_hard, late_start, True, False),
            NPVariant("np_projection_only_switch", fixed_k, fixed_k, late_start, True, True),
        ]

    return [
        NPVariant("hard_from_start", fixed_k, fixed_k, 0, True, True),
        NPVariant("hard_late", fixed_k, fixed_k, late_start, True, True),
        NPVariant("hard_never", fixed_k, fixed_k, late_start, True, False),
        NPVariant("soft_only", fixed_k, fixed_k, late_start, True, False),
        NPVariant("soft_then_hard", fixed_k, fixed_k, late_start, True, True),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 10/11 NP mechanism suites.")
    parser.add_argument("--phase", choices=["phase10", "phase11"], required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--image_list_file", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model_id", default="google/ddpm-celebahq-256")
    parser.add_argument("--seeds", default="100,101,102,103,104")
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--np_steps", type=int, default=1000)
    parser.add_argument("--late_start", type=int, default=400)
    parser.add_argument("--fixed_k", type=int, default=5)
    parser.add_argument("--canonical_soft", type=int, default=5)
    parser.add_argument("--canonical_hard", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    images = read_image_list(args.image_list_file)
    seeds = parse_int_list(args.seeds)
    variants = variants_for_phase(
        phase=args.phase,
        late_start=args.late_start,
        fixed_k=args.fixed_k,
        canonical_soft=args.canonical_soft,
        canonical_hard=args.canonical_hard,
    )

    os.makedirs(args.outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.outdir, f"{args.phase}_{stamp}")
    os.makedirs(run_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_model(args.model_id, device=device)

    config_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []

    for variant in variants:
        config_rows.append(
            {
                "phase": args.phase,
                "variant": variant.name,
                "num_steps": args.np_steps,
                "score_radius": args.radius,
                "proj_radius": args.radius,
                "proj_start": variant.proj_start,
                "num_candidates_soft": variant.soft,
                "num_candidates_hard": variant.hard,
                "use_lowfreq_score": variant.use_score,
                "use_lowfreq_projection": variant.use_projection,
            }
        )

        cfg = NoisePickingConfig(
            num_steps=args.np_steps,
            score_radius=args.radius,
            proj_radius=args.radius,
            proj_start=variant.proj_start,
            num_candidates_soft=variant.soft,
            num_candidates_hard=variant.hard,
            log_every=args.log_every,
            use_lowfreq_score=variant.use_score,
            use_lowfreq_projection=variant.use_projection,
        )

        for image_name in images:
            img_path = find_image_by_basename(args.data_root, image_name)
            if img_path is None:
                raise FileNotFoundError(f"Could not find {image_name} under {args.data_root}")

            x_gt = load_image(img_path, size=256, device=device)
            mag_target = magnitude(x_gt)

            for seed in seeds:
                t0 = time.perf_counter()
                x_np = noise_picking_reconstruct(
                    mag_target,
                    seed=seed,
                    unet=bundle.unet,
                    scheduler=bundle.scheduler,
                    device=device,
                    cfg=cfg,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                runtime_s = time.perf_counter() - t0

                with torch.no_grad():
                    run_rows.append(
                        {
                            "timestamp": stamp,
                            "phase": args.phase,
                            "variant": variant.name,
                            "image_basename": image_name,
                            "seed": seed,
                            "radius": args.radius,
                            "psnr": float(psnr(x_np, x_gt).cpu().item()),
                            "mag_l2": float(mag_l2(x_np, mag_target).cpu().item()),
                            "lowfreq_mag_l2": float(lowfreq_mag_l2(x_np, mag_target, args.radius).cpu().item()),
                            "runtime_s": runtime_s,
                            "num_steps": args.np_steps,
                            "proj_start": variant.proj_start,
                            "num_candidates_soft": variant.soft,
                            "num_candidates_hard": variant.hard,
                            "use_lowfreq_score": variant.use_score,
                            "use_lowfreq_projection": variant.use_projection,
                        }
                    )

                print(
                    f"[{args.phase}] {variant.name} | {image_name} seed={seed} "
                    f"psnr={run_rows[-1]['psnr']:.2f} dB ({runtime_s:.1f}s)"
                )

    write_csv(os.path.join(run_root, "configs.csv"), config_rows)
    write_csv(os.path.join(run_root, "run_level.csv"), run_rows)
    print(f"Saved: {run_root}")


if __name__ == "__main__":
    main()
