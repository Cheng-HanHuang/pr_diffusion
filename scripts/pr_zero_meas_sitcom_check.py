#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from typing import Dict, List

import torch
from torch.nn import Parameter

from prdiffusion.diffusion import load_model, resample_x_t, tweedie_x0_from_v
from prdiffusion.fft_ops import magnitude
from prdiffusion.io import find_image_by_basename, load_image
from prdiffusion.metrics import lowfreq_mag_l2, mag_l2, psnr
from prdiffusion.seed import seed_everything

DEFAULT_SEEDS = [100, 101, 102, 103, 104]


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def read_image_list(path: str) -> List[str]:
    images: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                images.append(line)
    if not images:
        raise ValueError(f"No images found in list file: {path}")
    return images


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_by_image(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row["image_basename"]), str(row["method"]))
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, object]] = []
    for (image_name, method), grp in groups.items():
        psnrs = [float(r["psnr"]) for r in grp]
        times = [float(r["runtime_s"]) for r in grp]
        full_mag = [float(r["full_mag_l2"]) for r in grp]
        low_mag = [float(r["lowfreq_mag_l2"]) for r in grp]
        out.append(
            {
                "image_basename": image_name,
                "method": method,
                "num_restarts": len(grp),
                "mean_psnr": statistics.fmean(psnrs),
                "median_psnr": statistics.median(psnrs),
                "max_psnr": max(psnrs),
                "mean_runtime_s": statistics.fmean(times),
                "mean_full_mag_l2": statistics.fmean(full_mag),
                "mean_lowfreq_mag_l2": statistics.fmean(low_mag),
            }
        )
    return out


@torch.no_grad()
def plain_diffusion_from_seed(seed: int, unet, scheduler, device: torch.device) -> torch.Tensor:
    seed_everything(seed)
    scheduler.set_timesteps(20, device=device)
    timesteps = scheduler.timesteps
    x_t = torch.randn((1, 3, unet.config.sample_size, unet.config.sample_size), device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 12345)
    for i in range(len(timesteps) - 1):
        t_int = int(timesteps[i])
        t_next = int(timesteps[i + 1])
        x0_hat = tweedie_x0_from_v(x_t, t_int, unet=unet, scheduler=scheduler, backprop_unet=False).detach()
        x_t = resample_x_t(x0_hat, t_next, scheduler=scheduler, eta_scale=1.0, generator=gen)
    t_last = int(timesteps[-1])
    return tweedie_x0_from_v(x_t, t_last, unet=unet, scheduler=scheduler, backprop_unet=False).detach()


def zero_meas_sitcom(seed: int, unet, scheduler, device: torch.device, K: int, lr_inner: float, lam: float) -> torch.Tensor:
    seed_everything(seed)
    scheduler.set_timesteps(20, device=device)
    timesteps = scheduler.timesteps
    x_t = torch.randn((1, 3, unet.config.sample_size, unet.config.sample_size), device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 12345)

    for i in range(len(timesteps) - 1):
        t_int = int(timesteps[i])
        t_next = int(timesteps[i + 1])
        v = Parameter(x_t.detach().clone())
        opt = torch.optim.Adam([v], lr=lr_inner)
        for _ in range(K):
            opt.zero_grad(set_to_none=True)
            # zero measurement term: optimization only keeps v close to x_t
            loss_reg = lam * torch.mean((v - x_t) ** 2)
            loss_reg.backward()
            opt.step()
        x0_hat = tweedie_x0_from_v(v.detach(), t_int, unet=unet, scheduler=scheduler, backprop_unet=False).detach()
        x_t = resample_x_t(x0_hat, t_next, scheduler=scheduler, eta_scale=1.0, generator=gen)

    t_last = int(timesteps[-1])
    return tweedie_x0_from_v(x_t, t_last, unet=unet, scheduler=scheduler, backprop_unet=False).detach()


def main() -> None:
    p = argparse.ArgumentParser(description="Zero-measurement SITCOM sanity check.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", type=str)
    group.add_argument("--image_list_file", type=str)
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--radius", type=float, default=0.5)
    p.add_argument("--model_id", type=str, default="google/ddpm-celebahq-256")
    p.add_argument("--sitcom_steps", type=int, default=20)
    p.add_argument("--sitcom_inner_steps", type=int, default=20)
    p.add_argument("--sitcom_lr", type=float, default=0.02)
    p.add_argument("--sitcom_lam", type=float, default=0.1)
    args = p.parse_args()

    images = parse_csv_list(args.images) if args.images else read_image_list(args.image_list_file)
    seeds = [int(x) for x in parse_csv_list(args.seeds)]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.outdir, f"zero_meas_sitcom_{stamp}")
    os.makedirs(run_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_model(args.model_id, device=device)

    run_rows: List[Dict[str, object]] = []
    for image_name in images:
        img_path = find_image_by_basename(args.data_root, image_name)
        if img_path is None:
            raise FileNotFoundError(f"Could not find {image_name} under {args.data_root}")
        x_gt = load_image(img_path, size=256, device=device)
        mag_target = magnitude(x_gt)

        for seed in seeds:
            for method in ["plain_diffusion_20step", "zero_meas_sitcom_20x20"]:
                t0 = time.perf_counter()
                if method == "plain_diffusion_20step":
                    x_rec = plain_diffusion_from_seed(seed, bundle.unet, bundle.scheduler, device)
                else:
                    x_rec = zero_meas_sitcom(seed, bundle.unet, bundle.scheduler, device, args.sitcom_inner_steps, args.sitcom_lr, args.sitcom_lam)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                runtime_s = time.perf_counter() - t0
                with torch.no_grad():
                    run_rows.append(
                        {
                            "timestamp": stamp,
                            "image_basename": image_name,
                            "seed": seed,
                            "method": method,
                            "psnr": float(psnr(x_rec, x_gt).cpu().item()),
                            "full_mag_l2": float(mag_l2(x_rec, mag_target).cpu().item()),
                            "lowfreq_mag_l2": float(lowfreq_mag_l2(x_rec, mag_target, args.radius).cpu().item()),
                            "runtime_s": runtime_s,
                        }
                    )
                print(f"[{method} | {image_name} seed={seed}] {run_rows[-1]['psnr']:.2f} dB ({runtime_s:.1f}s)")

    write_csv(os.path.join(run_root, "run_level.csv"), run_rows)
    write_csv(os.path.join(run_root, "image_level.csv"), aggregate_by_image(run_rows))
    print(f"Saved under {run_root}")


if __name__ == "__main__":
    main()
