#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
import time
from dataclasses import asdict
from typing import Dict, List

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prdiffusion.algorithms.hybrid_np_sitcom import HybridNPSitcomConfig, np_to_sitcom_hybrid_reconstruct
from prdiffusion.diffusion import load_model
from prdiffusion.fft_ops import magnitude
from prdiffusion.io import find_image_by_basename, load_image
from prdiffusion.metrics import lowfreq_mag_l2, mag_l2, psnr


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def read_image_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)


def build_cfg(
    method: str,
    radius: float,
    num_steps: int,
    sitcom_outer_steps: int,
    sitcom_k: int,
    sitcom_lr: float,
    sitcom_lam: float,
) -> HybridNPSitcomConfig:
    masked = "masked" in method
    switch = 400 if "400" in method else 600
    return HybridNPSitcomConfig(
        num_steps=num_steps,
        switch_timestep=switch,
        np_score_radius=radius,
        np_proj_radius=radius,
        np_proj_start=400,
        np_num_candidates_soft=5,
        np_num_candidates_hard=1,
        np_use_lowfreq_score=True,
        np_use_lowfreq_projection=True,
        sitcom_outer_steps=sitcom_outer_steps,
        sitcom_K=sitcom_k,
        sitcom_lr_inner=sitcom_lr,
        sitcom_lam=sitcom_lam,
        sitcom_meas_radius=radius if masked else None,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 12 NP->SITCOM hybrid ladder runner.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--model_id", type=str, default="google/ddpm-celebahq-256")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", type=str)
    group.add_argument("--image_list_file", type=str)

    p.add_argument("--seeds", type=str, default="100,101,102,103,104")
    p.add_argument("--methods", type=str, default="np_to_sitcom_400,np_to_sitcom_600,np_to_sitcom_masked_400,np_to_sitcom_masked_600")

    p.add_argument("--radius", type=float, default=0.5)
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--sitcom_outer_steps", type=int, default=20)
    p.add_argument("--sitcom_inner_steps", type=int, default=20)
    p.add_argument("--sitcom_lr", type=float, default=0.02)
    p.add_argument("--sitcom_lam", type=float, default=0.1)
    args = p.parse_args()

    images = parse_csv_list(args.images) if args.images else read_image_list(args.image_list_file)
    seeds = [int(s) for s in parse_csv_list(args.seeds)]
    methods = parse_csv_list(args.methods)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.outdir, f"phase12_hybrid_ladder_{stamp}")
    os.makedirs(run_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_model(args.model_id, device=device)

    run_rows: List[Dict[str, object]] = []
    config_rows: List[Dict[str, object]] = []

    for method in methods:
        cfg = build_cfg(
            method=method,
            radius=args.radius,
            num_steps=args.num_steps,
            sitcom_outer_steps=args.sitcom_outer_steps,
            sitcom_k=args.sitcom_inner_steps,
            sitcom_lr=args.sitcom_lr,
            sitcom_lam=args.sitcom_lam,
        )
        config_rows.append({"method": method, **asdict(cfg)})

        for image_name in images:
            img_path = find_image_by_basename(args.data_root, image_name)
            if img_path is None:
                raise FileNotFoundError(f"Could not find {image_name} under {args.data_root}")

            x_gt = load_image(img_path, size=256, device=device)
            mag_target = magnitude(x_gt)

            for seed in seeds:
                t0 = time.perf_counter()
                x_rec = np_to_sitcom_hybrid_reconstruct(
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
                            "method": method,
                            "image_basename": image_name,
                            "seed": seed,
                            "radius": args.radius,
                            "switch_timestep": cfg.switch_timestep,
                            "sitcom_meas_radius": cfg.sitcom_meas_radius,
                            "psnr": float(psnr(x_rec, x_gt).cpu().item()),
                            "full_mag_l2": float(mag_l2(x_rec, mag_target).cpu().item()),
                            "lowfreq_mag_l2": float(lowfreq_mag_l2(x_rec, mag_target, args.radius).cpu().item()),
                            "runtime_s": runtime_s,
                        }
                    )

                print(
                    f"[Phase12-Hybrid] {method} | {image_name} seed={seed} "
                    f"psnr={run_rows[-1]['psnr']:.2f} dB ({runtime_s:.1f}s)"
                )

    write_csv(os.path.join(run_root, "configs.csv"), config_rows)
    write_csv(os.path.join(run_root, "run_level.csv"), run_rows)
    print(f"Saved: {run_root}")


if __name__ == "__main__":
    main()
