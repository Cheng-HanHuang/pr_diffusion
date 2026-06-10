#!/usr/bin/env python3
"""Export NP handoff states with the exact NP noisy measurement included.

This is a drop-in alternative to scripts/npsitcom/export_np_handoff_states.py.
It saves x_sigma, x0_np, measurement, sigma, image index, and selector stats.
"""
from __future__ import annotations

import argparse, csv, importlib.util, os, sys, time
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
SEL = ROOT / "scripts" / "pr_external_difffpr_np_guided_lf_s2_selector.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


selector = load_module("npsitcom_selector", SEL)
base = selector.base
from prdiffusion.guided_backend import load_guided_diffusion_model
from prdiffusion.io import load_image


def write_csv(path: str, rows: List[Dict[str, object]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise ValueError("no rows")
    keys, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def ints(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def floats(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--image_list_file", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--guided_model_path", required=True)
    p.add_argument("--guided_diffusion_dir", default=None)
    p.add_argument("--guided_preset", default="difffpr_ffhq_10m")
    p.add_argument("--seeds", default="100,101,102,103")
    p.add_argument("--noise_values", default="0.05")
    p.add_argument("--max_images", type=int, default=5)
    p.add_argument("--np_steps", type=int, default=1000)
    p.add_argument("--late_start", type=int, default=300)
    p.add_argument("--soft_candidates", type=int, default=5)
    p.add_argument("--hard_candidates", type=int, default=1)
    p.add_argument("--score_radius", type=float, default=0.6)
    p.add_argument("--proj_radius", type=float, default=0.2)
    p.add_argument("--s2_lambda", type=float, default=0.01)
    p.add_argument("--s2_lambda_schedule", default="pre_projection_only")
    p.add_argument("--score_huber_delta", type=float, default=0.05)
    p.add_argument("--oversample", type=float, default=2.0)
    p.add_argument("--handoff_sigmas", default="50,20,10,5,2,1,0.5")
    p.add_argument("--measurement_noise_seed", type=int, default=20260423)
    p.add_argument("--clip_noisy_magnitude", action="store_true")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    state_dir = os.path.join(args.outdir, "states")
    os.makedirs(state_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_guided_diffusion_model(
        model_path=args.guided_model_path,
        device=device,
        preset=args.guided_preset,
        guided_diffusion_dir=args.guided_diffusion_dir,
    )
    image_size = int(bundle.unet.config.sample_size)
    pad = base.oversample_pad(image_size, args.oversample)
    images = base.collect_images(args.data_root, args.image_list_file)[: args.max_images]
    seeds, noises, sigmas = ints(args.seeds), floats(args.noise_values), floats(args.handoff_sigmas)
    variant = base.NPVariant(
        name=f"np_soft{args.soft_candidates}_hard{args.hard_candidates}",
        soft=args.soft_candidates,
        hard=args.hard_candidates,
        proj_start=args.late_start,
        use_lowfreq_score=True,
        use_lowfreq_projection=True,
    )
    configs = [("lf", "lf", 0.0, "constant"), ("s2_preproj", "prev_l2", args.s2_lambda, args.s2_lambda_schedule)]
    rows = []

    for noise_std in noises:
        for image_i, name in enumerate(images):
            x_gt = load_image(base.resolve_image_path(args.data_root, name), size=image_size, device=device)
            mag_clean = base.oversampled_magnitude(x_gt, pad)
            mag_target = mag_clean
            if noise_std > 0:
                gen = torch.Generator(device=device).manual_seed(args.measurement_noise_seed + image_i)
                mag_target = mag_clean + noise_std * torch.randn(
                    mag_clean.shape, device=device, dtype=mag_clean.dtype, generator=gen
                )
                if args.clip_noisy_magnitude:
                    mag_target = mag_target.clamp_min(0.0)
            for cfg_tag, score_mode, lam, sched in configs:
                for seed in seeds:
                    x_np, stats = selector.reconstruct_with_selector_stat(
                        mag_target,
                        pad=pad,
                        seed=seed,
                        unet=bundle.unet,
                        scheduler=bundle.scheduler,
                        device=device,
                        variant=variant,
                        num_steps=args.np_steps,
                        score_radius=args.score_radius,
                        proj_radius=args.proj_radius,
                        proj_radius_schedule=None,
                        score_mode=score_mode,
                        score_reg_lambda=lam,
                        score_reg_lambda_schedule=sched,
                        score_huber_delta=args.score_huber_delta,
                        log_every=0,
                    )
                    for sigma in sigmas:
                        gen = torch.Generator(device=device).manual_seed(
                            900000 + seed * 1000 + int(round(100 * sigma)) + image_i
                        )
                        x_sigma = x_np + float(sigma) * torch.randn(
                            x_np.shape, device=device, dtype=x_np.dtype, generator=gen
                        )
                        fn = f"{stamp}_{name}_noise{noise_std:g}_{cfg_tag}_seed{seed}_sigma{sigma:g}.pt".replace("/", "_")
                        path = os.path.join(state_dir, fn)
                        torch.save(
                            {
                                "x_sigma": x_sigma.cpu(),
                                "x0_np": x_np.cpu(),
                                "measurement": mag_target.cpu(),
                                "sigma": float(sigma),
                                "image_basename": name,
                                "image_index_in_split": int(image_i),
                                "noise_std": float(noise_std),
                                "seed": int(seed),
                                "config_tag": cfg_tag,
                                "selector_stats": stats,
                            },
                            path,
                        )
                        row = dict(
                            state_path=path,
                            image_basename=name,
                            image_index_in_split=image_i,
                            measurement_noise_std=noise_std,
                            seed=seed,
                            config_tag=cfg_tag,
                            handoff_sigma=sigma,
                            np_steps=args.np_steps,
                            score_radius=args.score_radius,
                            proj_radius=args.proj_radius,
                            **stats,
                        )
                        rows.append(row)
                        print("[handoff]", name, noise_std, cfg_tag, seed, "sigma", sigma, flush=True)
    write_csv(os.path.join(args.outdir, "handoff_manifest.csv"), rows)


if __name__ == "__main__":
    main()
