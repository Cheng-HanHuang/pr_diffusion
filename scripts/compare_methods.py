#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, os, time
from typing import List

import torch

from prdiffusion.io import find_image_by_basename, load_image
from prdiffusion.fft_ops import magnitude
from prdiffusion.metrics import psnr, mag_l2
from prdiffusion.diffusion import load_model
from prdiffusion.algorithms.sitcom import SitcomConfig, sitcom_reconstruct
from prdiffusion.algorithms.noise_picking import NoisePickingConfig, noise_picking_reconstruct


def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--outdir", type=str, default="out_compare")
    p.add_argument("--model_id", type=str, default="google/ddpm-celebahq-256")

    p.add_argument("--score_radii", type=str, default="0.005,0.01,0.02,0.03,0.04,0.05")
    p.add_argument("--n_runs", type=int, default=4)
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--noise_std", type=float, default=0.0)

    # noise picking knobs
    p.add_argument("--proj_radius", type=float, default=0.3)
    p.add_argument("--proj_start", type=int, default=400)
    p.add_argument("--num_candidates_soft", type=int, default=5)
    p.add_argument("--num_candidates_hard", type=int, default=2)

    # sitcom knobs
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lam", type=float, default=0.1)

    p.add_argument("--save_png", action="store_true")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)


    img_path = find_image_by_basename(args.data_root, args.image)
    if img_path is None:
        raise FileNotFoundError(f"Could not find {args.image} under {args.data_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_model(args.model_id, device=device)
    unet, scheduler = bundle.unet, bundle.scheduler

    x_gt = load_image(img_path, size=256, device=device)
    mag_gt = magnitude(x_gt)
    mag_target = mag_gt + args.noise_std * torch.randn_like(mag_gt)

    score_radii = parse_float_list(args.score_radii)

    # deterministic seeds: 10,11,... for fairness across radii
    seeds = [10 + i for i in range(args.n_runs)]

    rows = []
    for r in score_radii:
        for run_idx, seed in enumerate(seeds):
            cfg_np = NoisePickingConfig(
                num_steps=args.num_steps,
                score_radius=float(r),
                proj_radius=args.proj_radius,
                proj_start=args.proj_start,
                num_candidates_soft=args.num_candidates_soft,
                num_candidates_hard=args.num_candidates_hard,
            )
            cfg_sit = SitcomConfig(
                num_steps=args.num_steps,
                K=args.K,
                lr_inner=args.lr,
                lam=args.lam,
                backprop_unet=True,
                inner_optim="adam",
            )

            t0 = time.perf_counter()
            xA = noise_picking_reconstruct(mag_target, seed=seed, unet=unet, scheduler=scheduler, device=device, cfg=cfg_np)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            tA = time.perf_counter() - t0

            t0 = time.perf_counter()
            xB = sitcom_reconstruct(mag_target, seed=seed, unet=unet, scheduler=scheduler, device=device, cfg=cfg_sit)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            tB = time.perf_counter() - t0

            with torch.no_grad():
                psA = float(psnr(xA, x_gt).cpu().item())
                psB = float(psnr(xB, x_gt).cpu().item())
                meA = float(mag_l2(xA, mag_target).cpu().item())
                meB = float(mag_l2(xB, mag_target).cpu().item())

            rows.append({
                "image_basename": args.image,
                "image_path": img_path,
                "score_radius": float(r),
                "run_idx": int(run_idx),
                "seed": int(seed),
                "num_steps": int(args.num_steps),
                "noise_std": float(args.noise_std),

                "psnr_noise_picking": psA,
                "psnr_sitcom": psB,
                "magerr_l2_noise_picking": meA,
                "magerr_l2_sitcom": meB,
                "time_s_noise_picking": float(tA),
                "time_s_sitcom": float(tB),
            })

            print(f"[r={r:g} seed={seed}] PSNR A={psA:.2f} | PSNR B={psB:.2f}")

            if args.save_png:
                from torchvision.utils import save_image
                save_image((x_gt.clamp(-1,1) + 1) / 2, os.path.join(args.outdir, "x_gt.png"))
                save_image((xA.clamp(-1,1) + 1) / 2, os.path.join(args.outdir, f"xA_r{r:g}_seed{seed}.png"))
                save_image((xB.clamp(-1,1) + 1) / 2, os.path.join(args.outdir, f"xB_r{r:g}_seed{seed}.png"))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.outdir, f"compare_methods_{args.image}_{stamp}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
