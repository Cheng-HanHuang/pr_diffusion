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


def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True, help="image basename, e.g. 09375.jpg")
    p.add_argument("--data_root", type=str, required=True, help="root folder containing images")
    p.add_argument("--outdir", type=str, default="out_sitcom_lr_sweep")
    p.add_argument("--model_id", type=str, default="google/ddpm-celebahq-256")

    p.add_argument("--lr_list", type=str, default="0.01,0.02,0.05,0.1")
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--lam", type=float, default=0.1)

    p.add_argument("--eta_scale", type=float, default=1.0)
    p.add_argument("--init_scale", type=float, default=1.0)

    p.add_argument("--noise_std", type=float, default=0.0, help="std of noise added to |FFT(x)| measurement")

    p.add_argument("--seeds", type=str, default="0,1,2,3", help="comma-separated base seeds")
    p.add_argument("--save_png", action="store_true")

    # switches
    p.add_argument("--plugin_denoiser", action="store_true",
                   help="If set, disables backprop through UNet in the inner loop (not paper SITCOM).")

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

    lr_list = parse_float_list(args.lr_list)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    rows = []
    for lr in lr_list:
        for seed in seeds:
            cfg = SitcomConfig(
                num_steps=args.num_steps,
                K=args.K,
                lr_inner=lr,
                lam=args.lam,
                eta_scale=args.eta_scale,
                init_scale=args.init_scale,
                backprop_unet=(not args.plugin_denoiser),
                inner_optim="adam",
            )

            t0 = time.perf_counter()
            x_hat = sitcom_reconstruct(mag_target, seed=seed, unet=unet, scheduler=scheduler, device=device, cfg=cfg)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            with torch.no_grad():
                ps = float(psnr(x_hat, x_gt).cpu().item())
                me = float(mag_l2(x_hat, mag_target).cpu().item())

            rows.append({
                "image_basename": args.image,
                "image_path": img_path,
                "seed": seed,
                "lr_inner": float(lr),
                "K": int(args.K),
                "lam": float(args.lam),
                "num_steps": int(args.num_steps),
                "eta_scale": float(args.eta_scale),
                "init_scale": float(args.init_scale),
                "noise_std": float(args.noise_std),
                "backprop_unet": int(not args.plugin_denoiser),
                "psnr": ps,
                "magerr_l2": me,
                "time_s": float(dt),
            })

            print(f"[lr={lr:g} seed={seed}] PSNR={ps:.2f} dB | magerr_l2={me:.4g} | time={dt:.2f}s")

            if args.save_png:
                from torchvision.utils import save_image
                save_image((x_hat.clamp(-1,1) + 1) / 2, os.path.join(args.outdir, f"recon_lr{lr:g}_seed{seed}.png"))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.outdir, f"sitcom_lr_sweep_{args.image}_{stamp}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
