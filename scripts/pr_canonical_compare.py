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

from prdiffusion.algorithms.noise_picking import NoisePickingConfig, noise_picking_reconstruct
from prdiffusion.algorithms.sitcom import SitcomConfig, sitcom_reconstruct
from prdiffusion.diffusion import load_model
from prdiffusion.fft_ops import magnitude
from prdiffusion.io import find_image_by_basename, load_image
from prdiffusion.metrics import lowfreq_mag_l2, mag_l2, psnr


DEFAULT_SEEDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


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

def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Canonical NeurIPS comparison runner: Noise Picking (masked) vs SITCOM (unmasked/masked)."
    )
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--model_id", type=str, default="google/ddpm-celebahq-256")

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", type=str, help="Comma-separated image basenames.")
    group.add_argument("--image_list_file", type=str, help="Path to newline-separated list of image basenames.")

    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    p.add_argument("--radii", type=str, default="0.2", help="Comma-separated radii (used by Noise Picking; optional for SITCOM masked).")
    p.add_argument("--sitcom_variant", choices=["unmasked", "masked"], default="unmasked")

    p.add_argument("--sitcom_steps", type=int, default=20)
    p.add_argument("--sitcom_inner_steps", type=int, default=20)
    p.add_argument("--sitcom_lr", type=float, default=0.05)
    p.add_argument("--sitcom_lam", type=float, default=0.1)
    p.add_argument("--sitcom_eta_scale", type=float, default=1.0)
    p.add_argument("--sitcom_init_scale", type=float, default=1.0)

    p.add_argument("--np_steps", type=int, default=1000)
    p.add_argument("--np_num_candidates_soft", type=int, default=5)
    p.add_argument("--np_num_candidates_hard", type=int, default=2)
    p.add_argument("--np_proj_start", type=int, default=400)

    args = p.parse_args()

    images = parse_csv_list(args.images) if args.images else read_image_list(args.image_list_file)
    seeds = [int(x) for x in parse_csv_list(args.seeds)]
    radii = parse_float_list(args.radii)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.outdir, f"canonical_{stamp}")
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

        sitcom_cache: Dict[int, Dict[str, object]] = {}
        if args.sitcom_variant == "unmasked":
            shared_sitcom_cfg = SitcomConfig(
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
            for seed in seeds:
                t0 = time.perf_counter()
                x_sit = sitcom_reconstruct(mag_target, seed=seed, unet=bundle.unet, scheduler=bundle.scheduler, device=device, cfg=shared_sitcom_cfg)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                sit_t = time.perf_counter() - t0
                with torch.no_grad():
                    sit_psnr = float(psnr(x_sit, x_gt).cpu().item())
                    sit_full = float(mag_l2(x_sit, mag_target).cpu().item())
                sitcom_cache[seed] = {
                    "recon": x_sit,
                    "runtime_s": sit_t,
                    "psnr": sit_psnr,
                    "full_mag_l2": sit_full,
                }

        for radius in radii:
            sitcom_cfg = SitcomConfig(
                num_steps=args.sitcom_steps,
                K=args.sitcom_inner_steps,
                lr_inner=args.sitcom_lr,
                lam=args.sitcom_lam,
                eta_scale=args.sitcom_eta_scale,
                init_scale=args.sitcom_init_scale,
                meas_radius=radius if args.sitcom_variant == "masked" else None,
                backprop_unet=True,
                inner_optim="adam",
            )
            np_cfg = NoisePickingConfig(
                num_steps=args.np_steps,
                score_radius=radius,
                proj_radius=radius,
                proj_start=args.np_proj_start,
                num_candidates_soft=args.np_num_candidates_soft,
                num_candidates_hard=args.np_num_candidates_hard,
                use_lowfreq_score=True,
                use_lowfreq_projection=True,
            )

            cfg_rows = [
                {"method": "sitcom", "radius": radius, **asdict(sitcom_cfg)},
                {"method": "noise_picking", "radius": radius, **asdict(np_cfg)},
            ]
            write_csv(os.path.join(run_root, f"configs_{os.path.splitext(image_name)[0]}_r{radius:g}.csv"), cfg_rows)

            for seed in seeds:
                if args.sitcom_variant == "masked":
                    t0 = time.perf_counter()
                    x_sit = sitcom_reconstruct(mag_target, seed=seed, unet=bundle.unet, scheduler=bundle.scheduler, device=device, cfg=sitcom_cfg)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    sit_t = time.perf_counter() - t0
                    with torch.no_grad():
                        sit_psnr = float(psnr(x_sit, x_gt).cpu().item())
                        sit_full = float(mag_l2(x_sit, mag_target).cpu().item())
                else:
                    cached = sitcom_cache[seed]
                    x_sit = cached["recon"]
                    sit_t = float(cached["runtime_s"])
                    sit_psnr = float(cached["psnr"])
                    sit_full = float(cached["full_mag_l2"])

                t0 = time.perf_counter()
                x_np = noise_picking_reconstruct(mag_target, seed=seed, unet=bundle.unet, scheduler=bundle.scheduler, device=device, cfg=np_cfg)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                np_t = time.perf_counter() - t0

                with torch.no_grad():
                    np_psnr = float(psnr(x_np, x_gt).cpu().item())
                    np_full = float(mag_l2(x_np, mag_target).cpu().item())
                    sit_low = float(lowfreq_mag_l2(x_sit, mag_target, radius).cpu().item())
                    np_low = float(lowfreq_mag_l2(x_np, mag_target, radius).cpu().item())

                run_rows.append(
                    {
                        "timestamp": stamp,
                        "image_basename": image_name,
                        "seed": seed,
                        "radius": radius,
                        "method": "sitcom",
                        "sitcom_variant": args.sitcom_variant,
                        "psnr": sit_psnr,
                        "full_mag_l2": sit_full,
                        "lowfreq_mag_l2": sit_low,
                        "runtime_s": sit_t,
                    }
                )
                run_rows.append(
                    {
                        "timestamp": stamp,
                        "image_basename": image_name,
                        "seed": seed,
                        "radius": radius,
                        "method": "noise_picking",
                        "sitcom_variant": args.sitcom_variant,
                        "psnr": np_psnr,
                        "full_mag_l2": np_full,
                        "lowfreq_mag_l2": np_low,
                        "runtime_s": np_t,
                    }
                )

                print(
                    f"[{image_name} r={radius:g} seed={seed}] "
                    f"SITCOM({args.sitcom_variant})={sit_psnr:.2f}dB ({sit_t:.1f}s) | "
                    f"NP(masked)={np_psnr:.2f}dB ({np_t:.1f}s)"
                )

    run_rows.sort(key=lambda x: (x["image_basename"], x["radius"], x["seed"], x["method"]))
    image_rows = aggregate_by_image(run_rows)

    write_csv(os.path.join(run_root, "run_level.csv"), run_rows)
    write_csv(os.path.join(run_root, "image_level.csv"), image_rows)

    est_noise_pick_h = (len(images) * len(radii) * len(seeds) * args.np_steps / 1000.0) * (800.0 / 3600.0)
    print(f"Saved under {run_root}")
    print(f"Estimated Noise Picking walltime (serial): ~{est_noise_pick_h:.2f} hours")


if __name__ == "__main__":
    main()
