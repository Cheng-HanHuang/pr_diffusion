#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import asdict
from typing import List, Optional

import torch
from torchvision.utils import save_image

from prdiffusion.algorithms.noise_picking import NoisePickingConfig, noise_picking_reconstruct
from prdiffusion.diffusion import load_model
from prdiffusion.fft_ops import magnitude
from prdiffusion.io import find_image_by_basename, load_image
from prdiffusion.metrics import lowfreq_mag_l2, mag_l2, psnr


ALLOWED_DEFAULT_RADII = "0.1,0.2,0.3,0.4,0.5"
DEFAULT_PROJ_START_LIST = "none,0,200,400,600,800"


def parse_images(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if any(r < 0.1 for r in vals):
        raise ValueError(f"All score/projection radii must be >= 0.1. Got: {vals}")
    return vals


def parse_proj_start_list(s: str) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for tok in [x.strip().lower() for x in s.split(",") if x.strip()]:
        if tok in {"none", "off", "no", "disable", "disabled"}:
            out.append(None)
        else:
            out.append(int(tok))
    return out


def _save_tensor_png(x: torch.Tensor, path: str) -> None:
    save_image((x.clamp(-1, 1) + 1) / 2, path)


def _radius_tag(radius: float) -> str:
    return f"{radius:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _proj_tag(proj_start: Optional[int]) -> str:
    return "none" if proj_start is None else f"t{proj_start}"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Ablate Noise Picking projection start and score/projection radius (radius >= 0.1 only)."
    )
    p.add_argument("--images", type=str, default="09375.jpg,09671.jpg", help="Comma-separated image basenames")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--outdir", type=str, default="out_noise_picking_projstart_ablation")
    p.add_argument("--model_id", type=str, default="google/ddpm-celebahq-256")

    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--base_seed", type=int, default=100)
    p.add_argument("--radius_list", type=str, default=ALLOWED_DEFAULT_RADII)
    p.add_argument("--proj_start_list", type=str, default=DEFAULT_PROJ_START_LIST)

    p.add_argument("--noise_picking_steps", type=int, default=1000)
    p.add_argument("--np_num_candidates_soft", type=int, default=5)
    p.add_argument("--np_num_candidates_hard", type=int, default=2)

    p.add_argument("--toy_case", action="store_true", help="Use a synthetic random target instead of dataset images")
    p.add_argument("--toy_size", type=int, default=256)

    args = p.parse_args()

    if not args.toy_case and not args.data_root:
        raise ValueError("--data_root is required unless --toy_case is provided")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.outdir, f"noise_picking_projstart_ablation_{stamp}")
    os.makedirs(run_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_model(args.model_id, device=device)
    unet, scheduler = bundle.unet, bundle.scheduler

    images = parse_images(args.images)
    radii = parse_float_list(args.radius_list)
    proj_starts = parse_proj_start_list(args.proj_start_list)
    seeds = [args.base_seed + i for i in range(args.n_runs)]

    summary_rows = []

    for image_name in images:
        if args.toy_case:
            x_gt = torch.randn(1, 3, args.toy_size, args.toy_size, device=device).clamp(-1, 1)
            img_path = "toy_case_synthetic"
            image_tag = image_name
        else:
            img_path = find_image_by_basename(args.data_root, image_name)
            if img_path is None:
                raise FileNotFoundError(f"Could not find {image_name} under {args.data_root}")
            image_tag = os.path.splitext(os.path.basename(image_name))[0]
            x_gt = load_image(img_path, size=256, device=device)

        image_outdir = os.path.join(run_root, f"{image_tag}_{stamp}")
        os.makedirs(image_outdir, exist_ok=True)

        mag_target = magnitude(x_gt)

        gt_path = os.path.join(image_outdir, f"gt_{image_tag}_{stamp}.png")
        _save_tensor_png(x_gt, gt_path)

        for radius in radii:
            radius_tag = _radius_tag(radius)
            radius_dir = os.path.join(image_outdir, f"radius_{radius_tag}")
            os.makedirs(radius_dir, exist_ok=True)

            for proj_start in proj_starts:
                proj_tag = _proj_tag(proj_start)
                setting_dir = os.path.join(radius_dir, f"proj_{proj_tag}")
                os.makedirs(setting_dir, exist_ok=True)

                np_cfg = NoisePickingConfig(
                    num_steps=args.noise_picking_steps,
                    score_radius=radius,
                    proj_radius=radius,
                    proj_start=proj_start if proj_start is not None else 0,
                    num_candidates_soft=args.np_num_candidates_soft,
                    num_candidates_hard=args.np_num_candidates_hard,
                    use_lowfreq_score=True,
                    use_lowfreq_projection=(proj_start is not None),
                )

                cfg_row = {
                    "image": image_name,
                    "timestamp": stamp,
                    "radius": radius,
                    "proj_start_setting": proj_start,
                    "projection_enabled": proj_start is not None,
                    **asdict(np_cfg),
                }
                cfg_fieldnames = sorted(cfg_row.keys())
                with open(
                    os.path.join(setting_dir, f"config_np_{image_tag}_r{radius_tag}_proj{proj_tag}_{stamp}.csv"),
                    "w",
                    newline="",
                ) as f:
                    w = csv.DictWriter(f, fieldnames=cfg_fieldnames)
                    w.writeheader()
                    w.writerow(cfg_row)

                for run_idx, seed in enumerate(seeds):
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
                        psnr_np = float(psnr(x_np, x_gt).cpu().item())
                        magerr_np = float(mag_l2(x_np, mag_target).cpu().item())
                        lowfreq_magerr_np = float(lowfreq_mag_l2(x_np, mag_target, radius).cpu().item())

                    np_path = os.path.join(
                        setting_dir,
                        f"noise_picking_{image_tag}_r{radius_tag}_proj{proj_tag}_seed{seed}_{stamp}.png",
                    )
                    _save_tensor_png(x_np, np_path)

                    summary_rows.append(
                        {
                            "timestamp": stamp,
                            "image_basename": image_name,
                            "image_id": image_tag,
                            "image_path": img_path,
                            "gt_path": gt_path,
                            "radius": radius,
                            "projection_enabled": proj_start is not None,
                            "proj_start_setting": proj_start,
                            "run_idx": run_idx,
                            "seed": seed,
                            "noise_picking_steps": args.noise_picking_steps,
                            "np_num_candidates_soft": args.np_num_candidates_soft,
                            "np_num_candidates_hard": args.np_num_candidates_hard,
                            "noise_picking_psnr": psnr_np,
                            "noise_picking_magerr_l2": magerr_np,
                            "noise_picking_lowfreq_magerr_l2": lowfreq_magerr_np,
                            "noise_picking_time_s": t_np,
                            "noise_picking_recon_path": np_path,
                        }
                    )

                    print(
                        f"[{image_name} radius={radius:g} proj={proj_tag} run={run_idx} seed={seed}] "
                        f"NoisePicking PSNR={psnr_np:.2f} dB time={t_np:.2f}s"
                    )

    out_csv = os.path.join(run_root, f"noise_picking_projstart_ablation_{stamp}.csv")
    if summary_rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print("Saved:", out_csv)
    else:
        print("No runs executed.")


if __name__ == "__main__":
    main()
