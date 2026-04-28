#!/usr/bin/env python3
"""Run NP variants with a guided-diffusion FFHQ checkpoint.

This guided backend benchmark is designed for FFHQ pilot/full runs that compare
NP settings in a DiffFPR-style oversampled Fourier magnitude setting.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

# Reuse measurement/reconstruction utilities from the diffusers benchmark script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE_SCRIPT = _REPO_ROOT / "scripts" / "pr_external_difffpr_np_benchmark.py"
_spec = importlib.util.spec_from_file_location("pr_external_difffpr_np_benchmark", _BASE_SCRIPT)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load helper script at {_BASE_SCRIPT}")
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)

from prdiffusion.guided_backend import load_guided_diffusion_model
from prdiffusion.io import load_image


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ssim01(x: torch.Tensor, y: torch.Tensor) -> float:
    """Differentiation-free SSIM estimate for images in model range [-1,1]."""
    x01 = base.to01(x)
    y01 = base.to01(y)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x01, kernel_size=11, stride=1, padding=5)
    mu_y = F.avg_pool2d(y01, kernel_size=11, stride=1, padding=5)
    sigma_x = F.avg_pool2d(x01 * x01, kernel_size=11, stride=1, padding=5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y01 * y01, kernel_size=11, stride=1, padding=5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x01 * y01, kernel_size=11, stride=1, padding=5) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return float((num / den.clamp_min(1e-12)).mean().cpu().item())


def _estimate_roll_by_phase_correlation(x: torch.Tensor, ref: torch.Tensor) -> Tuple[int, int]:
    """Estimate integer spatial shift (dy, dx) for x -> ref using phase correlation."""
    xg = base.to01(x).mean(dim=1, keepdim=False)
    rg = base.to01(ref).mean(dim=1, keepdim=False)
    X = torch.fft.fftn(xg, dim=(-2, -1))
    R = torch.fft.fftn(rg, dim=(-2, -1))
    cps = X * torch.conj(R)
    cps_mag = base.complex_abs_safe(cps).clamp_min(1e-8)
    cps = cps / cps_mag
    corr = torch.fft.ifftn(cps, dim=(-2, -1)).real
    h, w = corr.shape[-2:]
    idx = int(torch.argmax(corr.reshape(-1)).item())
    dy = idx // w
    dx = idx % w
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return int(dy), int(dx)


def _best_channel_rot180(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return base.best_rot180_channel_alignment(x, ref)


def _resolve_shift_conjflip(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Resolve common PR ambiguities: per-channel 180°, axis flips, and cyclic shifts."""
    rot = _best_channel_rot180(x, ref)
    spatial_variants = [
        rot,
        torch.flip(rot, dims=(-2,)),
        torch.flip(rot, dims=(-1,)),
        torch.flip(rot, dims=(-2, -1)),
    ]
    best = None
    best_mse = None
    for cand in spatial_variants:
        dy, dx = _estimate_roll_by_phase_correlation(cand, ref)
        aligned = torch.roll(cand, shifts=(dy, dx), dims=(-2, -1))
        mse = torch.mean((base.to01(aligned) - base.to01(ref)) ** 2)
        if best is None or float(mse) < float(best_mse):
            best = aligned
            best_mse = mse
    assert best is not None
    return best


def maybe_lpips_metric(x: torch.Tensor, y: torch.Tensor, lpips_model) -> float:
    if lpips_model is None:
        return math.nan
    with torch.no_grad():
        return float(lpips_model(x, y).mean().cpu().item())


def make_alignment(alignment: str, x_rec: torch.Tensor, x_gt: torch.Tensor) -> torch.Tensor:
    if alignment == "raw":
        return x_rec
    if alignment == "rot180":
        return _best_channel_rot180(x_rec, x_gt)
    if alignment == "resolve":
        return _resolve_shift_conjflip(x_rec, x_gt)
    raise ValueError(f"Unknown alignment: {alignment}")




def summarize_image_level(run_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    for row in run_rows:
        key = (
            str(row["variant"]),
            str(row["alignment_mode"]),
            str(row["image_basename"]),
            float(row["score_radius"]),
            float(row["proj_radius"]),
            int(row["proj_start"]),
            int(row["num_candidates_soft"]),
            int(row["num_candidates_hard"]),
            float(row["oversample"]),
            float(row["measurement_noise_std"]),
        )
        grouped.setdefault(key, []).append(row)

    out: List[Dict[str, object]] = []
    for (
        variant,
        alignment,
        image,
        score_radius,
        proj_radius,
        proj_start,
        soft_k,
        hard_k,
        oversample,
        noise_std,
    ), rows in sorted(grouped.items()):
        psnr = torch.tensor([float(r["psnr"]) for r in rows], dtype=torch.float32)
        ssim = torch.tensor([float(r["ssim"]) for r in rows], dtype=torch.float32)
        runtime = torch.tensor([float(r["runtime_s"]) for r in rows], dtype=torch.float32)
        out.append(
            {
                "variant": variant,
                "alignment_mode": alignment,
                "image_basename": image,
                "score_radius": score_radius,
                "proj_radius": proj_radius,
                "proj_start": proj_start,
                "num_candidates_soft": soft_k,
                "num_candidates_hard": hard_k,
                "oversample": oversample,
                "measurement_noise_std": noise_std,
                "n_runs": len(rows),
                "psnr_mean": float(psnr.mean().item()),
                "psnr_median": float(psnr.median().item()),
                "psnr_best": float(psnr.max().item()),
                "ssim_mean": float(ssim.mean().item()),
                "ssim_median": float(ssim.median().item()),
                "runtime_s_mean": float(runtime.mean().item()),
            }
        )
    return out


def summarize_condition_level(run_rows: List[Dict[str, object]], psnr_threshold: float) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = {}
    for row in run_rows:
        key = (
            str(row["variant"]),
            str(row["alignment_mode"]),
            float(row["score_radius"]),
            float(row["proj_radius"]),
            int(row["proj_start"]),
            int(row["num_candidates_soft"]),
            int(row["num_candidates_hard"]),
            float(row["oversample"]),
            float(row["measurement_noise_std"]),
        )
        grouped.setdefault(key, []).append(row)

    out: List[Dict[str, object]] = []
    for (
        variant,
        alignment,
        score_radius,
        proj_radius,
        proj_start,
        soft_k,
        hard_k,
        oversample,
        noise_std,
    ), rows in sorted(grouped.items()):
        psnr = torch.tensor([float(r["psnr"]) for r in rows], dtype=torch.float32)
        ssim = torch.tensor([float(r["ssim"]) for r in rows], dtype=torch.float32)
        lpips_vals = torch.tensor([float(r["lpips"]) for r in rows], dtype=torch.float32)
        runtime = torch.tensor([float(r["runtime_s"]) for r in rows], dtype=torch.float32)
        nfe = torch.tensor([float(r["nfe_calls"]) for r in rows], dtype=torch.float32)
        out.append(
            {
                "variant": variant,
                "alignment_mode": alignment,
                "score_radius": score_radius,
                "proj_radius": proj_radius,
                "proj_start": proj_start,
                "num_candidates_soft": soft_k,
                "num_candidates_hard": hard_k,
                "oversample": oversample,
                "measurement_noise_std": noise_std,
                "n_runs": len(rows),
                "psnr_best": float(psnr.max().item()),
                "psnr_mean": float(psnr.mean().item()),
                "psnr_median": float(psnr.median().item()),
                "psnr_below_threshold_count": int((psnr < psnr_threshold).sum().item()),
                "psnr_threshold": float(psnr_threshold),
                "ssim_best": float(ssim.max().item()),
                "ssim_mean": float(ssim.mean().item()),
                "ssim_median": float(ssim.median().item()),
                "lpips_best": float(lpips_vals.min().item()) if not torch.isnan(lpips_vals).all() else math.nan,
                "lpips_mean": float(lpips_vals.nanmean().item()) if not torch.isnan(lpips_vals).all() else math.nan,
                "lpips_median": float(lpips_vals.nanmedian().item()) if not torch.isnan(lpips_vals).all() else math.nan,
                "runtime_s_mean": float(runtime.mean().item()),
                "runtime_s_median": float(runtime.median().item()),
                "nfe_calls_mean": float(nfe.mean().item()),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Guided-diffusion FFHQ NP benchmark/pilot runner.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--image_list_file", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--guided_model_path", required=True)
    parser.add_argument("--guided_diffusion_dir", default=None)
    parser.add_argument("--guided_preset", default="difffpr_ffhq_10m")
    parser.add_argument("--guided_strict", action="store_true")
    parser.add_argument("--variants", default="np_canonical")
    parser.add_argument("--seeds", default="100,101,102,103")
    parser.add_argument("--np_steps", type=int, default=1000)
    parser.add_argument("--late_start", type=int, default=400)
    parser.add_argument("--fixed_k", type=int, default=5)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--score_radius", type=float, default=None)
    parser.add_argument("--proj_radius", type=float, default=None)
    parser.add_argument("--oversample_values", default="4")
    parser.add_argument("--measurement_noise_values", default="0,0.01,0.05")
    parser.add_argument("--measurement_noise_seed", type=int, default=20260423)
    parser.add_argument("--clip_noisy_magnitude", action="store_true")
    parser.add_argument("--alignments", default="raw,rot180,resolve")
    parser.add_argument("--psnr_threshold", type=float, default=20.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--soft_candidates", type=int, default=None)
    parser.add_argument("--hard_candidates", type=int, default=None)
    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument("--fast_eval", action="store_true")
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()
    score_radius = float(args.score_radius) if args.score_radius is not None else float(args.radius)
    proj_radius = float(args.proj_radius) if args.proj_radius is not None else float(args.radius)

    image_names = base.collect_images(args.data_root, args.image_list_file)
    if not image_names:
        raise ValueError(f"No images found in split {args.image_list_file}")
    if args.max_images is not None:
        image_names = image_names[: args.max_images]
    seeds = base.parse_int_list(args.seeds)
    oversample_values = [float(x.strip()) for x in args.oversample_values.split(",") if x.strip()]
    noise_values = [float(x.strip()) for x in args.measurement_noise_values.split(",") if x.strip()]
    alignment_modes = [x.strip() for x in args.alignments.split(",") if x.strip()]

    variant_names = [tok.strip() for tok in args.variants.split(",") if tok.strip()]
    variants = base.make_variants(variant_names, late_start=args.late_start, fixed_k=args.fixed_k)

    if args.soft_candidates is not None or args.hard_candidates is not None:
        new_variants = []
        for v in variants:
            soft = args.soft_candidates if args.soft_candidates is not None else v.soft
            hard = args.hard_candidates if args.hard_candidates is not None else v.hard
            name = f"{v.name}_soft{soft}_hard{hard}"
            new_variants.append(
                base.NPVariant(
                    name=name,
                    soft=int(soft),
                    hard=int(hard),
                    proj_start=v.proj_start,
                    use_lowfreq_score=v.use_lowfreq_score,
                    use_lowfreq_projection=v.use_lowfreq_projection,
                )
            )
        variants = new_variants

    os.makedirs(args.outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.outdir, f"difffpr_np_guided_{stamp}")
    os.makedirs(run_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_guided_diffusion_model(
        model_path=args.guided_model_path,
        device=device,
        preset=args.guided_preset,
        guided_diffusion_dir=args.guided_diffusion_dir,
        strict=bool(args.guided_strict),
    )
    image_size = int(bundle.unet.config.sample_size)

    lpips_model = None
    lpips_status = "skipped" if args.skip_lpips else "not_available"
    if not args.skip_lpips:
        try:
            import lpips  # type: ignore
            lpips_model = lpips.LPIPS(net="alex").to(device)
            lpips_model.eval()
            lpips_status = "lpips_alex"
        except Exception:
            lpips_model = None
            lpips_status = "not_available"

    config_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []

    for variant in variants:
        for oversample in oversample_values:
            pad = base.oversample_pad(image_size, oversample)
            for noise_std in noise_values:
                config_rows.append(
                    {
                        "variant": variant.name,
                        "backend": "guided_diffusion",
                        "guided_preset": args.guided_preset,
                        "guided_model_path": str(Path(args.guided_model_path).expanduser()),
                        "guided_diffusion_dir": args.guided_diffusion_dir or "PYTHONPATH/default",
                        "num_steps": args.np_steps,
                        "score_radius": score_radius,
                        "proj_radius": proj_radius,
                        "proj_start": variant.proj_start,
                        "num_candidates_soft": variant.soft,
                        "num_candidates_hard": variant.hard,
                        "measurement_operator": "DiffFPR-style centered FFT magnitude after symmetric zero padding",
                        "oversample_arg": oversample,
                        "pad_pixels_each_side": pad,
                        "measurement_noise_std": noise_std,
                        "clip_noisy_magnitude": bool(args.clip_noisy_magnitude),
                        "seeds": ",".join(map(str, seeds)),
                        "alignments": ",".join(alignment_modes),
                        "psnr_threshold": args.psnr_threshold,
                        "lpips_backend": lpips_status,
                        "guided_model_config": repr(bundle.model_config),
                    }
                )

                for image_index, image_name in enumerate(image_names):
                    img_path = base.resolve_image_path(args.data_root, image_name)
                    x_gt = load_image(img_path, size=image_size, device=device)
                    mag_clean = base.oversampled_magnitude(x_gt, pad)

                    mag_target = mag_clean
                    if noise_std > 0:
                        gen = torch.Generator(device=device).manual_seed(args.measurement_noise_seed + image_index)
                        noise = torch.randn(mag_clean.shape, device=device, dtype=mag_clean.dtype, generator=gen)
                        mag_target = mag_clean + float(noise_std) * noise
                        if args.clip_noisy_magnitude:
                            mag_target = mag_target.clamp_min(0.0)

                    for seed in seeds:
                        t0 = time.perf_counter()
                        x_rec = base.noise_picking_reconstruct_oversampled(
                            mag_target,
                            pad=pad,
                            seed=seed,
                            unet=bundle.unet,
                            scheduler=bundle.scheduler,
                            device=device,
                            variant=variant,
                            num_steps=args.np_steps,
                            score_radius=score_radius,
                            proj_radius=proj_radius,
                            log_every=args.log_every,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        runtime_s = time.perf_counter() - t0

                        for alignment in alignment_modes:
                            x_eval = make_alignment(alignment, x_rec, x_gt)

                            if args.fast_eval:
                                clean_mag_l2 = math.nan
                                noisy_mag_l2 = math.nan
                                clean_lowfreq_mag_l2 = math.nan
                                noisy_lowfreq_mag_l2 = math.nan
                            else:
                                clean_mag_l2 = float(base.oversampled_mag_l2(x_eval, mag_clean, pad).cpu().item())
                                noisy_mag_l2 = float(base.oversampled_mag_l2(x_eval, mag_target, pad).cpu().item())
                                clean_lowfreq_mag_l2 = float(
                                    base.oversampled_lowfreq_mag_l2(
                                        x_eval, mag_clean, pad, score_radius
                                    ).cpu().item()
                                )
                                noisy_lowfreq_mag_l2 = float(
                                    base.oversampled_lowfreq_mag_l2(
                                        x_eval, mag_target, pad, score_radius
                                    ).cpu().item()
                                )
                            row = {
                                "timestamp": stamp,
                                "variant": variant.name,
                                "backend": "guided_diffusion",
                                "guided_preset": args.guided_preset,
                                "alignment_mode": alignment,
                                "image_basename": image_name,
                                "seed": seed,
                                "psnr": base.psnr01_from_model_range(x_eval, x_gt),
                                "ssim": ssim01(x_eval, x_gt),
                                "lpips": maybe_lpips_metric(x_eval, x_gt, lpips_model),
                                "clean_mag_l2": clean_mag_l2,
                                "noisy_mag_l2": noisy_mag_l2,
                                "clean_lowfreq_mag_l2": clean_lowfreq_mag_l2,
                                "noisy_lowfreq_mag_l2": noisy_lowfreq_mag_l2,
                                "runtime_s": runtime_s,
                                "nfe_calls": args.np_steps - 1,
                                "num_steps": args.np_steps,
                                "proj_start": variant.proj_start,
                                "num_candidates_soft": variant.soft,
                                "num_candidates_hard": variant.hard,
                                "score_radius": score_radius,
                                "proj_radius": proj_radius,
                                "radius": score_radius,  # legacy/backward-compatible field
                                "oversample": oversample,
                                "pad_pixels_each_side": pad,
                                "measurement_noise_std": noise_std,
                            }
                            run_rows.append(row)

                        last_rows = run_rows[-len(alignment_modes):]
                        best_seen = max(float(r["psnr"]) for r in last_rows)
                        print(
                            f"[Guided-FFHQ-NP] {variant.name} | {image_name} seed={seed} "
                            f"oversample={oversample} sigma={noise_std:.3f} "
                            f"best_psnr_this_run={best_seen:.2f} ({runtime_s:.1f}s)"
                        )

    file_prefix = f"difffpr_np_guided_{stamp}"
    outputs = {
        "configs": os.path.join(run_root, f"{file_prefix}__configs.csv"),
        "run_level": os.path.join(run_root, f"{file_prefix}__run_level.csv"),
        "image_level_summary": os.path.join(run_root, f"{file_prefix}__image_level_summary.csv"),
        "condition_level_summary": os.path.join(run_root, f"{file_prefix}__condition_level_summary.csv"),
    }

    write_csv(outputs["configs"], config_rows)
    write_csv(outputs["run_level"], run_rows)
    write_csv(outputs["image_level_summary"], summarize_image_level(run_rows))
    write_csv(
        outputs["condition_level_summary"],
        summarize_condition_level(run_rows, psnr_threshold=args.psnr_threshold),
    )

    # Backward-compatible aliases for existing tooling.
    write_csv(os.path.join(run_root, "configs.csv"), config_rows)
    write_csv(os.path.join(run_root, "run_level.csv"), run_rows)
    write_csv(os.path.join(run_root, "image_level_summary.csv"), summarize_image_level(run_rows))
    write_csv(
        os.path.join(run_root, "condition_level_summary.csv"),
        summarize_condition_level(run_rows, psnr_threshold=args.psnr_threshold),
    )

    print(f"Saved: {run_root}")
    for key, path in outputs.items():
        print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()
