#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import statistics
import time
from dataclasses import asdict
from typing import Dict, List

import torch

from prdiffusion.algorithms.noise_picking import (
    NoisePickingConfig,
    enforce_magnitude_lowfreq,
    pick_noise,
)
from prdiffusion.diffusion import load_model
from prdiffusion.fft_ops import magnitude
from prdiffusion.io import find_image_by_basename, load_image
from prdiffusion.metrics import lowfreq_mag_l2, mag_l2, psnr

DEFAULT_SEEDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


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
    groups: Dict[tuple[str, str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (str(row["image_basename"]), str(row["method"]), str(row["radius"]))
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, object]] = []
    for (image_name, method, radius), grp in groups.items():
        psnrs = [float(r["psnr"]) for r in grp]
        full_mag = [float(r["full_mag_l2"]) for r in grp]
        low_mag = [float(r["lowfreq_mag_l2"]) for r in grp]
        times = [float(r["runtime_s"]) for r in grp]
        out.append(
            {
                "image_basename": image_name,
                "method": method,
                "radius": radius,
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
    out.sort(key=lambda x: (x["image_basename"], x["method"], x["radius"]))
    return out


def get_schedule(scheduler, num_steps: int, device: torch.device) -> List[int]:
    scheduler.set_timesteps(num_steps, device=device)
    return [int(t) for t in scheduler.timesteps.tolist()]


def build_piecewise_timesteps(
    full_ts: List[int],
    reduced_ts: List[int],
    proj_timestep: int,
    mode: str,
) -> List[int]:
    if mode == "full_early_reduced_late":
        prefix = [t for t in full_ts if t > proj_timestep]
        suffix = [t for t in reduced_ts if t <= proj_timestep]
    elif mode == "reduced_early_full_late":
        prefix = [t for t in reduced_ts if t > proj_timestep]
        suffix = [t for t in full_ts if t <= proj_timestep]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    merged = prefix + suffix
    if len(merged) < 2:
        raise ValueError("Merged timestep list must contain at least 2 timesteps.")
    if any(merged[i] <= merged[i + 1] for i in range(len(merged) - 1)):
        raise ValueError(f"Merged timesteps must be strictly descending: {merged[:20]} ...")
    return merged


@torch.no_grad()
def noise_picking_reconstruct_custom_timesteps(
    mag_target: torch.Tensor,
    *,
    seed: int,
    unet,
    scheduler,
    device: torch.device,
    cfg: NoisePickingConfig,
    timesteps: List[int],
    proj_timestep: int,
) -> torch.Tensor:
    # Same NP logic as repo, except the schedule is a custom explicit timestep list,
    # and projection/candidate switch are triggered by diffusion-timestep threshold.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x_t = torch.randn((1, 3, unet.config.sample_size, unet.config.sample_size), device=device)
    x_prev = None
    eps_prev = None

    for i in range(len(timesteps) - 1):
        t_int = int(timesteps[i])
        t_next_int = int(timesteps[i + 1])

        if i == 0:
            t_tensor = torch.tensor([t_int], device=device, dtype=torch.long)
            eps = unet(x_t, t_tensor).sample
            alpha_bar_t = scheduler.alphas_cumprod[t_int].to(device=device, dtype=x_t.dtype)
            sqrt_ab = torch.sqrt(alpha_bar_t)
            sqrt_1mab = torch.sqrt(1.0 - alpha_bar_t)
            x0_hat = (x_t - sqrt_1mab * eps) / sqrt_ab
        else:
            assert x_prev is not None
            x0_hat = x_prev

        # Turn on hard projection only once we are in the late regime.
        if cfg.use_lowfreq_projection and t_int <= proj_timestep:
            x0_hat = enforce_magnitude_lowfreq(x0_hat, mag_target, cfg.proj_radius)

        # Use soft candidates before proj_timestep, hard candidates after.
        k = cfg.num_candidates_soft if t_int > proj_timestep else cfg.num_candidates_hard

        x_prev, eps_prev = pick_noise(
            x0_target=x0_hat,
            t_int=t_next_int,
            eps_prev=eps_prev,
            mag_target=mag_target,
            num_candidates=k,
            score_radius=cfg.score_radius if cfg.use_lowfreq_score else None,
            unet=unet,
            scheduler=scheduler,
        )

    assert x_prev is not None
    return x_prev.detach()


def main() -> None:
    p = argparse.ArgumentParser(description="Piecewise-schedule NP comparison runner.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--model_id", type=str, default="google/ddpm-celebahq-256")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", type=str)
    group.add_argument("--image_list_file", type=str)
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--radii", type=str, default="0.5")

    p.add_argument("--mode", type=str, choices=["full_early_reduced_late", "reduced_early_full_late"], required=True)
    p.add_argument("--reduced_steps_list", type=str, default="100,400,500")
    p.add_argument("--proj_timestep", type=int, default=400, help="Absolute diffusion timestep where late hard projection begins.")

    p.add_argument("--np_num_candidates_soft", type=int, default=5)
    p.add_argument("--np_num_candidates_hard", type=int, default=1)
    p.add_argument("--log_schedule_preview", action="store_true")
    args = p.parse_args()

    images = parse_csv_list(args.images) if args.images else read_image_list(args.image_list_file)
    seeds = [int(x) for x in parse_csv_list(args.seeds)]
    radii = parse_float_list(args.radii)
    reduced_steps_list = parse_int_list(args.reduced_steps_list)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.outdir, f"piecewise_np_{args.mode}_{stamp}")
    os.makedirs(run_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_model(args.model_id, device=device)

    full_ts = get_schedule(bundle.scheduler, 1000, device)
    run_rows: List[Dict[str, object]] = []
    config_rows: List[Dict[str, object]] = []

    for reduced_steps in reduced_steps_list:
        reduced_ts = get_schedule(bundle.scheduler, reduced_steps, device)
        piecewise_ts = build_piecewise_timesteps(full_ts, reduced_ts, args.proj_timestep, args.mode)
        method_name = f"np_{args.mode}_red{reduced_steps}_pt{args.proj_timestep}"

        if args.log_schedule_preview:
            print(f"\n[{method_name}] len={len(piecewise_ts)}")
            print(f"  head: {piecewise_ts[:20]}")
            print(f"  tail: {piecewise_ts[-20:]}")

        for radius in radii:
            np_cfg = NoisePickingConfig(
                num_steps=1000,  # informational only; actual schedule is piecewise_ts
                score_radius=radius,
                proj_radius=radius,
                proj_start=0,  # unused here
                num_candidates_soft=args.np_num_candidates_soft,
                num_candidates_hard=args.np_num_candidates_hard,
                use_lowfreq_score=True,
                use_lowfreq_projection=True,
            )
            config_rows.append(
                {
                    "method": method_name,
                    "mode": args.mode,
                    "reduced_steps": reduced_steps,
                    "proj_timestep": args.proj_timestep,
                    "num_piecewise_timesteps": len(piecewise_ts),
                    **asdict(np_cfg),
                }
            )

            for image_name in images:
                img_path = find_image_by_basename(args.data_root, image_name)
                if img_path is None:
                    raise FileNotFoundError(f"Could not find {image_name} under {args.data_root}")
                x_gt = load_image(img_path, size=256, device=device)
                mag_target = magnitude(x_gt)

                for seed in seeds:
                    t0 = time.perf_counter()
                    x_np = noise_picking_reconstruct_custom_timesteps(
                        mag_target,
                        seed=seed,
                        unet=bundle.unet,
                        scheduler=bundle.scheduler,
                        device=device,
                        cfg=np_cfg,
                        timesteps=piecewise_ts,
                        proj_timestep=args.proj_timestep,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    np_t = time.perf_counter() - t0

                    with torch.no_grad():
                        np_psnr = float(psnr(x_np, x_gt).cpu().item())
                        np_full = float(mag_l2(x_np, mag_target).cpu().item())
                        np_low = float(lowfreq_mag_l2(x_np, mag_target, radius).cpu().item())

                    run_rows.append(
                        {
                            "timestamp": stamp,
                            "image_basename": image_name,
                            "seed": seed,
                            "radius": radius,
                            "method": method_name,
                            "mode": args.mode,
                            "reduced_steps": reduced_steps,
                            "proj_timestep": args.proj_timestep,
                            "num_piecewise_timesteps": len(piecewise_ts),
                            "psnr": np_psnr,
                            "full_mag_l2": np_full,
                            "lowfreq_mag_l2": np_low,
                            "runtime_s": np_t,
                        }
                    )
                    print(f"[{method_name} | {image_name} seed={seed}] {np_psnr:.2f} dB ({np_t:.1f}s)")

    image_rows = aggregate_by_image(run_rows)
    write_csv(os.path.join(run_root, "configs.csv"), config_rows)
    write_csv(os.path.join(run_root, "run_level.csv"), run_rows)
    write_csv(os.path.join(run_root, "image_level.csv"), image_rows)
    print(f"Saved under {run_root}")


if __name__ == "__main__":
    main()
