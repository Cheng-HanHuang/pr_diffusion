#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, os, time
from typing import List, Optional

import torch

from prdiffusion.io import find_image_by_basename, load_image, maybe_download_celeba_hq_256
from prdiffusion.fft_ops import magnitude
from prdiffusion.metrics import psnr, mag_l2
from prdiffusion.diffusion import load_model
from prdiffusion.algorithms.sitcom import SitcomConfig, sitcom_reconstruct


def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def maybe_select(values: list, idxs: Optional[str]) -> list:
    if idxs is None:
        return values
    I = parse_int_list(idxs)
    return [values[i] for i in I]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--outdir", type=str, default="out_sitcom_noise_ablation")
    p.add_argument("--model_id", type=str, default="google/ddpm-celebahq-256")

    # fixed LR for this script
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--lam", type=float, default=0.1)

    p.add_argument("--eta_list", type=str, default="0.0,0.25,0.5,1.0")
    p.add_argument("--init_list", type=str, default="0.75,1.0,1.25")
    p.add_argument("--eta_idxs", type=str, default=None)
    p.add_argument("--init_idxs", type=str, default=None)

    p.add_argument("--noise_std", type=float, default=0.0)

    p.add_argument("--seeds", type=str, default="0,1,2,3")
    p.add_argument("--seed_idxs", type=str, default=None)

    p.add_argument("--save_png", action="store_true")
    p.add_argument("--plugin_denoiser", action="store_true")

    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.data_root is None:
        args.data_root = maybe_download_celeba_hq_256()

    img_path = find_image_by_basename(args.data_root, args.image)
    if img_path is None:
        raise FileNotFoundError(f"Could not find {args.image} under {args.data_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_model(args.model_id, device=device)
    unet, scheduler = bundle.unet, bundle.scheduler

    x_gt = load_image(img_path, size=256, device=device)
    mag_gt = magnitude(x_gt)
    mag_target = mag_gt + args.noise_std * torch.randn_like(mag_gt)

    eta_list = parse_float_list(args.eta_list)
    init_list = parse_float_list(args.init_list)
    eta_list = maybe_select(eta_list, args.eta_idxs)
    init_list = maybe_select(init_list, args.init_idxs)

    seeds = parse_int_list(args.seeds)
    seeds = maybe_select(seeds, args.seed_idxs)

    rows = []
    for eta in eta_list:
        for init in init_list:
            for seed in seeds:
                cfg = SitcomConfig(
                    num_steps=args.num_steps,
                    K=args.K,
                    lr_inner=args.lr,
                    lam=args.lam,
                    eta_scale=float(eta),
                    init_scale=float(init),
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
                    "eta_scale": float(eta),
                    "init_scale": float(init),
                    "lr_inner": float(args.lr),
                    "K": int(args.K),
                    "lam": float(args.lam),
                    "num_steps": int(args.num_steps),
                    "noise_std": float(args.noise_std),
                    "backprop_unet": int(not args.plugin_denoiser),
                    "psnr": ps,
                    "magerr_l2": me,
                    "time_s": float(dt),
                })

                print(f"[eta={eta:g} init={init:g} seed={seed}] PSNR={ps:.2f} | magerr_l2={me:.4g} | {dt:.2f}s")

                if args.save_png:
                    from torchvision.utils import save_image
                    save_image((x_hat.clamp(-1,1) + 1) / 2,
                               os.path.join(args.outdir, f"recon_eta{eta:g}_init{init:g}_seed{seed}.png"))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.outdir, f"sitcom_noise_ablation_{args.image}_{stamp}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
