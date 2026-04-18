#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import statistics
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prdiffusion.algorithms.sitcom import SitcomConfig
from prdiffusion.diffusion import load_model, tweedie_x0_from_v, resample_x_t
from prdiffusion.fft_ops import magnitude
from prdiffusion.io import find_image_by_basename, load_image
from prdiffusion.metrics import lowfreq_mag_l2, mag_l2, mag_mse, psnr
from prdiffusion.seed import seed_everything


DEFAULT_SEEDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]


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
    if not rows:
        raise ValueError(f"No rows to write to {path}")
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
        full_mag = [float(r["full_mag_l2"]) for r in grp]
        low_mag = [float(r["lowfreq_mag_l2"]) for r in grp]
        times = [float(r["runtime_s"]) for r in grp]
        out.append(
            {
                "image_basename": image_name,
                "method": method,
                "num_restarts": len(grp),
                "mean_psnr": statistics.fmean(psnrs),
                "median_psnr": statistics.median(psnrs),
                "max_psnr": max(psnrs),
                "mean_full_mag_l2": statistics.fmean(full_mag),
                "mean_lowfreq_mag_l2": statistics.fmean(low_mag),
                "mean_runtime_s": statistics.fmean(times),
                "total_runtime_s": sum(times),
            }
        )
    out.sort(key=lambda x: (x["image_basename"], x["method"]))
    return out


def effective_meas_radius(mask_mode: str, timestep: int, mask_radius: float, mask_start: int) -> Optional[float]:
    if mask_mode == "unmasked":
        return None
    if mask_mode == "masked":
        return mask_radius
    if mask_mode == "weighted":
        return None
    # late mode: activate low-frequency masking for late diffusion timesteps.
    return mask_radius if timestep <= mask_start else None


def sitcom_reconstruct_scheduled(
    mag_target: torch.Tensor,
    *,
    seed: int,
    unet,
    scheduler,
    device: torch.device,
    cfg: SitcomConfig,
    mask_mode: str,
    mask_radius: float,
    mask_start: int,
    early_meas_weight: float,
    late_meas_weight: float,
) -> torch.Tensor:
    seed_everything(seed)
    scheduler.set_timesteps(cfg.num_steps, device=device)
    timesteps = scheduler.timesteps

    x_t = cfg.init_scale * torch.randn((1, 3, unet.config.sample_size, unet.config.sample_size), device=device)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 12345)

    for i in range(len(timesteps) - 1):
        t_int = int(timesteps[i])
        t_next_int = int(timesteps[i + 1])

        meas_radius = effective_meas_radius(mask_mode, t_int, mask_radius, mask_start)
        meas_weight = late_meas_weight if t_int <= mask_start else early_meas_weight

        v = torch.nn.Parameter(x_t.detach().clone())
        if cfg.inner_optim.lower() == "adam":
            opt = torch.optim.Adam([v], lr=cfg.lr_inner)
        elif cfg.inner_optim.lower() == "sgd":
            opt = torch.optim.SGD([v], lr=cfg.lr_inner)
        else:
            raise ValueError(f"Unknown inner_optim={cfg.inner_optim}")

        for _ in range(cfg.K):
            opt.zero_grad(set_to_none=True)
            x0_v = tweedie_x0_from_v(v, t_int, unet=unet, scheduler=scheduler, backprop_unet=cfg.backprop_unet)

            if meas_radius is None:
                loss_meas = mag_mse(x0_v, mag_target)
                meas_l2 = mag_l2(x0_v, mag_target)
            else:
                meas_l2 = lowfreq_mag_l2(x0_v, mag_target, meas_radius)
                loss_meas = meas_l2.pow(2)

            loss_reg = cfg.lam * torch.mean((v - x_t) ** 2)
            loss = meas_weight * loss_meas + loss_reg
            loss.backward()
            opt.step()

            if cfg.stop_meas_l2 is not None and float(meas_l2.detach().cpu()) <= float(cfg.stop_meas_l2):
                break
            if cfg.stop_meas_mse is not None and float(loss_meas.detach().cpu()) <= float(cfg.stop_meas_mse):
                break

        v_hat = v.detach()
        x0_hat = tweedie_x0_from_v(v_hat, t_int, unet=unet, scheduler=scheduler, backprop_unet=False).detach()
        x_t = resample_x_t(x0_hat, t_next_int, scheduler=scheduler, eta_scale=cfg.eta_scale, generator=gen)

        if cfg.log_every > 0 and (i + 1) % cfg.log_every == 0:
            print(
                f"[SITCOM-Scheduled] step {i+1}/{len(timesteps)-1} | t={t_int} "
                f"| mode={mask_mode} | meas_weight={meas_weight:.3g}"
            )

    t_last = int(timesteps[-1])
    return tweedie_x0_from_v(x_t, t_last, unet=unet, scheduler=scheduler, backprop_unet=False).detach()


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 8/9 SITCOM schedule runner (unmasked/masked/late).")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--model_id", type=str, default="google/ddpm-celebahq-256")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", type=str, help="Comma-separated image basenames.")
    group.add_argument("--image_list_file", type=str, help="Path to newline-separated list of image basenames.")

    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))

    p.add_argument("--sitcom_steps", type=int, default=20)
    p.add_argument("--sitcom_inner_steps", type=int, default=20)
    p.add_argument("--sitcom_lr", type=float, default=0.02)
    p.add_argument("--sitcom_lam", type=float, default=0.1)
    p.add_argument("--sitcom_eta_scale", type=float, default=1.0)
    p.add_argument("--sitcom_init_scale", type=float, default=1.0)

    p.add_argument("--mask_mode", choices=["unmasked", "masked", "late", "weighted"], default="late")
    p.add_argument("--mask_radius", type=float, default=0.5)
    p.add_argument("--mask_start", type=int, default=400, help="Late-mask activation threshold in diffusion timestep index.")
    p.add_argument("--metrics_radius", type=float, default=None, help="Radius for low-frequency metric reporting. Defaults to mask_radius.")
    p.add_argument(
        "--early_meas_weight",
        type=float,
        default=1.0,
        help="Measurement loss weight before mask_start (for weighted schedule mode).",
    )
    p.add_argument(
        "--late_meas_weight",
        type=float,
        default=1.0,
        help="Measurement loss weight at/after mask_start (for weighted schedule mode).",
    )

    args = p.parse_args()

    if args.mask_mode in {"masked", "late"} and not (0.0 < args.mask_radius <= 1.0):
        raise ValueError("--mask_radius must be in (0, 1] for masked/late modes.")
    if args.early_meas_weight <= 0 or args.late_meas_weight <= 0:
        raise ValueError("--early_meas_weight and --late_meas_weight must be positive.")

    images = parse_csv_list(args.images) if args.images else read_image_list(args.image_list_file)
    seeds = [int(x) for x in parse_csv_list(args.seeds)]
    metrics_radius = args.metrics_radius if args.metrics_radius is not None else args.mask_radius

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.outdir, f"phase89_schedule_{stamp}")
    os.makedirs(run_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_model(args.model_id, device=device)

    cfg = SitcomConfig(
        num_steps=args.sitcom_steps,
        K=args.sitcom_inner_steps,
        lr_inner=args.sitcom_lr,
        lam=args.sitcom_lam,
        eta_scale=args.sitcom_eta_scale,
        init_scale=args.sitcom_init_scale,
        meas_radius=None,
        backprop_unet=True,
        inner_optim="adam",
    )

    run_rows: List[Dict[str, object]] = []
    config_rows = [
        {
            "method": "sitcom",
            "mask_mode": args.mask_mode,
            "mask_radius": args.mask_radius,
            "mask_start": args.mask_start,
            "metrics_radius": metrics_radius,
            "early_meas_weight": args.early_meas_weight,
            "late_meas_weight": args.late_meas_weight,
            **asdict(cfg),
        }
    ]
    write_csv(os.path.join(run_root, "configs.csv"), config_rows)

    for image_name in images:
        img_path = find_image_by_basename(args.data_root, image_name)
        if img_path is None:
            raise FileNotFoundError(f"Could not find {image_name} under {args.data_root}")

        x_gt = load_image(img_path, size=256, device=device)
        mag_target = magnitude(x_gt)

        for seed in seeds:
            t0 = time.perf_counter()
            x_sit = sitcom_reconstruct_scheduled(
                mag_target,
                seed=seed,
                unet=bundle.unet,
                scheduler=bundle.scheduler,
                device=device,
                cfg=cfg,
                mask_mode=args.mask_mode,
                mask_radius=args.mask_radius,
                mask_start=args.mask_start,
                early_meas_weight=args.early_meas_weight,
                late_meas_weight=args.late_meas_weight,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            runtime_s = time.perf_counter() - t0

            with torch.no_grad():
                val_psnr = float(psnr(x_sit, x_gt).cpu().item())
                val_full = float(mag_l2(x_sit, mag_target).cpu().item())
                val_low = float(lowfreq_mag_l2(x_sit, mag_target, metrics_radius).cpu().item())

            run_rows.append(
                {
                    "timestamp": stamp,
                    "image_basename": image_name,
                    "seed": seed,
                    "method": f"sitcom_{args.mask_mode}",
                    "mask_mode": args.mask_mode,
                    "mask_radius": args.mask_radius,
                    "mask_start": args.mask_start,
                    "metrics_radius": metrics_radius,
                    "early_meas_weight": args.early_meas_weight,
                    "late_meas_weight": args.late_meas_weight,
                    "psnr": val_psnr,
                    "full_mag_l2": val_full,
                    "lowfreq_mag_l2": val_low,
                    "runtime_s": runtime_s,
                }
            )
            print(
                f"[{image_name} seed={seed}] SITCOM({args.mask_mode}, r={args.mask_radius:g}, start={args.mask_start}, "
                f"w={args.early_meas_weight:g}->{args.late_meas_weight:g}) "
                f"= {val_psnr:.2f} dB ({runtime_s:.1f}s)"
            )

    run_rows.sort(key=lambda x: (x["image_basename"], x["seed"]))
    image_rows = aggregate_by_image(run_rows)

    write_csv(os.path.join(run_root, "run_level.csv"), run_rows)
    write_csv(os.path.join(run_root, "image_level.csv"), image_rows)

    print(f"Saved under {run_root}")


if __name__ == "__main__":
    main()
