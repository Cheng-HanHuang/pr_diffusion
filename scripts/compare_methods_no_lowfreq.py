#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import asdict
from typing import List

import torch
from torchvision.utils import save_image

from prdiffusion.algorithms.noise_picking import NoisePickingConfig, noise_picking_reconstruct
from prdiffusion.algorithms.sitcom import SitcomConfig, sitcom_reconstruct
from prdiffusion.diffusion import load_model
from prdiffusion.fft_ops import magnitude
from prdiffusion.io import find_image_by_basename, load_image
from prdiffusion.metrics import mag_l2, psnr


def parse_images(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _save_tensor_png(x: torch.Tensor, path: str) -> None:
    save_image((x.clamp(-1, 1) + 1) / 2, path)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Compare SITCOM vs Noise Picking with low-frequency masking disabled for both methods."
        )
    )
    p.add_argument("--images", type=str, default="09375.jpg,09671.jpg", help="Comma-separated image basenames")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--outdir", type=str, default="out_compare_no_lowfreq")
    p.add_argument("--model_id", type=str, default="google/ddpm-celebahq-256")

    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--base_seed", type=int, default=100)

    p.add_argument("--sitcom_outer_steps", type=int, default=20)
    p.add_argument("--sitcom_inner_steps", type=int, default=20)
    p.add_argument("--sitcom_lr", type=float, default=0.05)
    p.add_argument("--sitcom_lam", type=float, default=0.1)
    p.add_argument("--sitcom_eta_scale", type=float, default=1.0)
    p.add_argument("--sitcom_stop_meas_l2", type=float, default=None)

    p.add_argument("--noise_picking_steps", type=int, default=1000)
    p.add_argument("--np_num_candidates_soft", type=int, default=5)
    p.add_argument("--np_num_candidates_hard", type=int, default=2)
    p.add_argument("--np_proj_start", type=int, default=400)
    p.add_argument("--np_proj_radius", type=float, default=0.3)
    p.add_argument("--np_score_radius", type=float, default=0.02)

    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_model(args.model_id, device=device)
    unet, scheduler = bundle.unet, bundle.scheduler

    images = parse_images(args.images)
    seeds = [args.base_seed + i for i in range(args.n_runs)]

    summary_rows = []

    for image_name in images:
        img_path = find_image_by_basename(args.data_root, image_name)
        if img_path is None:
            raise FileNotFoundError(f"Could not find {image_name} under {args.data_root}")

        image_tag = os.path.splitext(os.path.basename(image_name))[0]
        image_outdir = os.path.join(args.outdir, image_tag)
        os.makedirs(image_outdir, exist_ok=True)

        x_gt = load_image(img_path, size=256, device=device)
        mag_target = magnitude(x_gt)

        gt_path = os.path.join(image_outdir, "gt.png")
        _save_tensor_png(x_gt, gt_path)

        sitcom_cfg = SitcomConfig(
            num_steps=args.sitcom_outer_steps,
            K=args.sitcom_inner_steps,
            lr_inner=args.sitcom_lr,
            lam=args.sitcom_lam,
            eta_scale=args.sitcom_eta_scale,
            meas_radius=None,
            stop_meas_l2=args.sitcom_stop_meas_l2,
            backprop_unet=True,
            inner_optim="adam",
        )

        np_cfg = NoisePickingConfig(
            num_steps=args.noise_picking_steps,
            score_radius=args.np_score_radius,
            proj_radius=args.np_proj_radius,
            proj_start=args.np_proj_start,
            num_candidates_soft=args.np_num_candidates_soft,
            num_candidates_hard=args.np_num_candidates_hard,
            use_lowfreq_score=False,
            use_lowfreq_projection=False,
        )

        cfg_rows = []
        for method_name, cfg in (("sitcom", sitcom_cfg), ("noise_picking", np_cfg)):
            row = {"image": image_name, "method": method_name, **asdict(cfg)}
            cfg_rows.append(row)

        cfg_fieldnames = sorted({k for r in cfg_rows for k in r.keys()})
        with open(os.path.join(image_outdir, "configs.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cfg_fieldnames)
            w.writeheader()
            w.writerows(cfg_rows)

        for run_idx, seed in enumerate(seeds):
            t0 = time.perf_counter()
            x_sit = sitcom_reconstruct(
                mag_target,
                seed=seed,
                unet=unet,
                scheduler=scheduler,
                device=device,
                cfg=sitcom_cfg,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_sit = time.perf_counter() - t0

            t0 = time.perf_counter()
            x_np = noise_picking_reconstruct(
                mag_target,
                seed=seed,
                unet=unet,
                scheduler=scheduler,
                device=device,
                cfg=np_cfg,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_np = time.perf_counter() - t0

            with torch.no_grad():
                psnr_sit = float(psnr(x_sit, x_gt).cpu().item())
                psnr_np = float(psnr(x_np, x_gt).cpu().item())
                magerr_sit = float(mag_l2(x_sit, mag_target).cpu().item())
                magerr_np = float(mag_l2(x_np, mag_target).cpu().item())

            sit_path = os.path.join(image_outdir, f"sitcom_seed{seed}.png")
            np_path = os.path.join(image_outdir, f"noise_picking_seed{seed}.png")
            _save_tensor_png(x_sit, sit_path)
            _save_tensor_png(x_np, np_path)

            summary_rows.append(
                {
                    "image_basename": image_name,
                    "image_path": img_path,
                    "gt_path": gt_path,
                    "run_idx": run_idx,
                    "seed": seed,
                    "sitcom_outer_steps": args.sitcom_outer_steps,
                    "sitcom_inner_steps": args.sitcom_inner_steps,
                    "noise_picking_steps": args.noise_picking_steps,
                    "sitcom_psnr": psnr_sit,
                    "noise_picking_psnr": psnr_np,
                    "sitcom_magerr_l2": magerr_sit,
                    "noise_picking_magerr_l2": magerr_np,
                    "sitcom_time_s": t_sit,
                    "noise_picking_time_s": t_np,
                    "sitcom_recon_path": sit_path,
                    "noise_picking_recon_path": np_path,
                }
            )

            print(
                f"[{image_name} run={run_idx} seed={seed}] "
                f"SITCOM PSNR={psnr_sit:.2f} dB | NoisePicking PSNR={psnr_np:.2f} dB"
            )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.outdir, f"compare_no_lowfreq_{stamp}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
