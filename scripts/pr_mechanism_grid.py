#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import os
import time
from typing import Dict, Iterable, List, Tuple

import torch

from prdiffusion.algorithms.noise_picking import NoisePickingConfig, noise_picking_reconstruct
from prdiffusion.algorithms.sitcom import SitcomConfig, sitcom_reconstruct
from prdiffusion.diffusion import load_model
from prdiffusion.fft_ops import magnitude
from prdiffusion.io import find_image_by_basename, load_image
from prdiffusion.metrics import lowfreq_mag_l2, mag_l2, psnr


def parse_list(s: str, cast):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def grid(params: Dict[str, Iterable[object]]) -> List[Dict[str, object]]:
    keys = list(params.keys())
    vals = [list(params[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def default_methods_for_mode(mode: str) -> Tuple[bool, bool]:
    if mode in {"sitcom_lr", "sitcom_noise"}:
        return True, False
    if mode in {"np_schedule", "mechanism"}:
        return False, True
    return True, True


def main() -> None:
    p = argparse.ArgumentParser(description="Grid runner for NeurIPS phase 2-5 sweeps.")
    p.add_argument("--mode", choices=["sitcom_lr", "sitcom_noise", "np_schedule", "budget", "mechanism"], required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--image_list_file", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--model_id", default="google/ddpm-celebahq-256")
    p.add_argument("--seeds", default="100,101,102,103,104")
    p.add_argument("--radius", type=float, default=0.2)
    p.add_argument(
        "--methods",
        choices=["auto", "both", "sitcom", "noise_picking"],
        default="auto",
        help="Which methods to run. auto picks compute-efficient defaults per mode.",
    )

    p.add_argument("--sitcom_steps", type=int, default=20)
    p.add_argument("--sitcom_inner_steps", type=int, default=20)
    p.add_argument("--sitcom_lr_values", default="0.02,0.05,0.1")
    p.add_argument("--sitcom_eta_values", default="0.5,1.0")
    p.add_argument("--sitcom_init_values", default="0.75,1.0,1.25")

    p.add_argument("--np_steps", type=int, default=1000)
    p.add_argument("--np_soft_values", default="3,5,7")
    p.add_argument("--np_hard_values", default="1,2,3")
    p.add_argument("--np_proj_start_values", default="200,400,600")
    p.add_argument("--budget_np_steps", default="250,500,750,1000")
    p.add_argument("--budget_sitcom_pairs", default="20:20,50:10,100:5")
    args = p.parse_args()

    images = read_list(args.image_list_file)
    seeds = parse_list(args.seeds, int)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.outdir, f"{args.mode}_{stamp}")
    os.makedirs(run_root, exist_ok=True)

    if args.methods == "auto":
        run_sitcom, run_np = default_methods_for_mode(args.mode)
    elif args.methods == "both":
        run_sitcom, run_np = True, True
    elif args.methods == "sitcom":
        run_sitcom, run_np = True, False
    else:
        run_sitcom, run_np = False, True

    if not (run_sitcom or run_np):
        raise ValueError("At least one method must be selected.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_model(args.model_id, device=device)

    if args.mode == "sitcom_lr":
        settings = grid({"sitcom_lr": parse_list(args.sitcom_lr_values, float)})
    elif args.mode == "sitcom_noise":
        settings = grid(
            {
                "sitcom_eta": parse_list(args.sitcom_eta_values, float),
                "sitcom_init": parse_list(args.sitcom_init_values, float),
            }
        )
    elif args.mode == "np_schedule":
        settings = (
            grid({"np_soft": parse_list(args.np_soft_values, int)})
            + grid({"np_hard": parse_list(args.np_hard_values, int)})
            + grid({"np_proj_start": parse_list(args.np_proj_start_values, int)})
        )
    elif args.mode == "budget":
        sitcom_pairs = []
        for tok in parse_list(args.budget_sitcom_pairs, str):
            o, k = tok.split(":")
            sitcom_pairs.append((int(o), int(k)))
        settings = grid({"np_steps": parse_list(args.budget_np_steps, int)}) + [
            {"sitcom_steps": o, "sitcom_inner": k} for o, k in sitcom_pairs
        ]
    else:
        settings = [
            {"np_mask_score": True, "np_mask_proj": True, "np_soft": 5, "np_hard": 1, "np_proj_start": 400, "name": "full"},
            {"np_mask_score": True, "np_mask_proj": False, "np_soft": 5, "np_hard": 1, "np_proj_start": 400, "name": "score_only"},
            {"np_mask_score": False, "np_mask_proj": True, "np_soft": 5, "np_hard": 1, "np_proj_start": 400, "name": "projection_only"},
            {"np_mask_score": False, "np_mask_proj": False, "np_soft": 5, "np_hard": 1, "np_proj_start": 400, "name": "no_masking"},
        ]

    rows: List[Dict[str, object]] = []
    for s in settings:
        for image_name in images:
            img_path = find_image_by_basename(args.data_root, image_name)
            if img_path is None:
                raise FileNotFoundError(f"Missing {image_name}")
            x_gt = load_image(img_path, size=256, device=device)
            mag_target = magnitude(x_gt)

            for seed in seeds:
                if run_sitcom:
                    sitcom_cfg = SitcomConfig(
                        num_steps=int(s.get("sitcom_steps", args.sitcom_steps)),
                        K=int(s.get("sitcom_inner", args.sitcom_inner_steps)),
                        lr_inner=float(s.get("sitcom_lr", 0.05)),
                        lam=0.1,
                        eta_scale=float(s.get("sitcom_eta", 1.0)),
                        init_scale=float(s.get("sitcom_init", 1.0)),
                        meas_radius=None,
                    )
                    t0 = time.perf_counter()
                    x_sit = sitcom_reconstruct(mag_target, seed=seed, unet=bundle.unet, scheduler=bundle.scheduler, device=device, cfg=sitcom_cfg)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    sit_t = time.perf_counter() - t0

                    with torch.no_grad():
                        rows.append(
                            {
                                "timestamp": stamp,
                                "mode": args.mode,
                                "setting": str(s),
                                "image_basename": image_name,
                                "seed": seed,
                                "radius": args.radius,
                                "method": "sitcom",
                                "psnr": float(psnr(x_sit, x_gt).cpu().item()),
                                "mag_l2": float(mag_l2(x_sit, mag_target).cpu().item()),
                                "lowfreq_mag_l2": float(lowfreq_mag_l2(x_sit, mag_target, args.radius).cpu().item()),
                                "runtime_s": sit_t,
                            }
                        )

                if run_np:
                    np_cfg = NoisePickingConfig(
                        num_steps=int(s.get("np_steps", args.np_steps)),
                        score_radius=args.radius,
                        proj_radius=args.radius,
                        proj_start=int(s.get("np_proj_start", 400)),
                        num_candidates_soft=int(s.get("np_soft", 5)),
                        num_candidates_hard=int(s.get("np_hard", 2)),
                        use_lowfreq_score=bool(s.get("np_mask_score", True)),
                        use_lowfreq_projection=bool(s.get("np_mask_proj", True)),
                    )
                    t0 = time.perf_counter()
                    x_np = noise_picking_reconstruct(mag_target, seed=seed, unet=bundle.unet, scheduler=bundle.scheduler, device=device, cfg=np_cfg)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    np_t = time.perf_counter() - t0

                    with torch.no_grad():
                        rows.append(
                            {
                                "timestamp": stamp,
                                "mode": args.mode,
                                "setting": str(s),
                                "image_basename": image_name,
                                "seed": seed,
                                "radius": args.radius,
                                "method": "noise_picking",
                                "psnr": float(psnr(x_np, x_gt).cpu().item()),
                                "mag_l2": float(mag_l2(x_np, mag_target).cpu().item()),
                                "lowfreq_mag_l2": float(lowfreq_mag_l2(x_np, mag_target, args.radius).cpu().item()),
                                "runtime_s": np_t,
                            }
                        )

                print(
                    f"[{args.mode}] {image_name} seed={seed} setting={s} "
                    f"methods={'+'.join(m for m, enabled in [('sitcom', run_sitcom), ('noise_picking', run_np)] if enabled)}"
                )

    write_csv(os.path.join(run_root, "run_level.csv"), rows)
    print(f"Saved: {run_root}")


if __name__ == "__main__":
    main()
