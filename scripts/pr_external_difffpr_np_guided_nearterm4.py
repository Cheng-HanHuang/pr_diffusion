#!/usr/bin/env python3
"""Run near-term FFHQ NP experiments with scheduled S2 and noise memory.

This script is intentionally separate from pr_external_difffpr_np_guided_benchmark.py
so the older benchmark runner remains backward-compatible.  It reuses the same
measurement, metric, and guided-diffusion loading utilities, but adds:

- score_reg_lambda_schedule:
    constant
    linear_decay_to_proj_start
    pre_projection_only
- noise_memory_k:
    keep the winning noise directions from previous steps as explicit candidate
    slots.  With noise_memory_k=1 and hard_candidates=1, the hard stage reuses
    the previous winning noise direction instead of sampling a fresh singleton.
- adaptive_prev_l2 score mode:
    use the previous-state regularizer only when the normalized low-frequency
    score margin between the best and second-best candidates is small.
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
from typing import Dict, List, Optional, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE_SCRIPT = _REPO_ROOT / "scripts" / "pr_external_difffpr_np_benchmark.py"
_GUIDED_SCRIPT = _REPO_ROOT / "scripts" / "pr_external_difffpr_np_guided_benchmark.py"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


base = _load_module("pr_external_difffpr_np_benchmark", _BASE_SCRIPT)
guided = _load_module("pr_external_difffpr_np_guided_benchmark", _GUIDED_SCRIPT)

from prdiffusion.guided_backend import load_guided_diffusion_model
from prdiffusion.io import load_image
from prdiffusion.seed import seed_everything


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_lambda_at_step(
    lambda0: float,
    schedule: str,
    step_index: int,
    proj_start: int,
) -> float:
    """Return the effective S2 regularization weight at NP step index i."""
    mode = str(schedule).lower().strip()
    if mode in {"constant", "static", "none"}:
        return float(lambda0)
    if mode in {"linear_decay_to_proj_start", "linear_decay", "decay"}:
        if proj_start <= 0:
            return 0.0
        return float(lambda0) * max(0.0, 1.0 - float(step_index) / float(proj_start))
    if mode in {"pre_projection_only", "preproj", "pre_proj"}:
        return float(lambda0) if int(step_index) < int(proj_start) else 0.0
    raise ValueError(
        f"Unknown score_reg_lambda_schedule={schedule!r}; expected constant, "
        "linear_decay_to_proj_start, or pre_projection_only."
    )


@torch.no_grad()
def pick_noise_oversampled_nearterm(
    *,
    x0_target: torch.Tensor,
    t_int: int,
    eps_prev: Optional[torch.Tensor],
    eps_memory: Optional[List[torch.Tensor]],
    mag_target: torch.Tensor,
    pad: int,
    num_candidates: int,
    score_radius: Optional[float],
    unet,
    scheduler,
    score_mode: str = "lf",
    score_reg_lambda: float = 0.25,
    score_huber_delta: float = 0.05,
    noise_memory_k: int = 0,
    adaptive_s2_margin: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick a noise candidate with optional explicit memory-bank slots.

    For noise_memory_k=0 this preserves the legacy behavior: if there are
    multiple candidates, the first candidate reuses eps_prev.  For
    noise_memory_k>0, the first up-to-k slots are filled from the explicit memory
    queue even when num_candidates=1.

    For score_mode=adaptive_prev_l2, the LF scores are median-normalized and the
    previous-state regularizer is applied only when the gap between the best and
    second-best normalized LF score is at most adaptive_s2_margin.
    """
    if num_candidates <= 0:
        raise ValueError(f"num_candidates must be positive, got {num_candidates}")

    alpha_bar_t = scheduler.alphas_cumprod[t_int].to(
        device=x0_target.device, dtype=x0_target.dtype
    )
    sqrt_at = torch.sqrt(alpha_bar_t)
    sqrt_1mat = torch.sqrt(1.0 - alpha_bar_t)

    memory_slots: List[torch.Tensor] = []
    if noise_memory_k > 0:
        for eps in eps_memory or []:
            if eps is not None:
                memory_slots.append(eps)
            if len(memory_slots) >= min(int(noise_memory_k), int(num_candidates)):
                break
    elif eps_prev is not None and num_candidates > 1:
        memory_slots.append(eps_prev)

    candidate_x0s: List[torch.Tensor] = []
    candidate_eps: List[torch.Tensor] = []
    lf_scores: List[torch.Tensor] = []
    huber_scores: List[torch.Tensor] = []
    prev_regs: List[torch.Tensor] = []

    for j in range(num_candidates):
        if j < len(memory_slots):
            eps = memory_slots[j]
        else:
            eps = torch.randn_like(x0_target)

        x_t = sqrt_at * x0_target + sqrt_1mat * eps
        t_tensor = torch.tensor([t_int], device=x0_target.device, dtype=torch.long)
        eps_pred = unet(x_t, t_tensor).sample
        x0_hat = (x_t - sqrt_1mat * eps_pred) / sqrt_at

        if score_radius is None:
            lf_score = base.oversampled_mag_l2(x0_hat, mag_target, pad)
            huber_score = lf_score
        else:
            lf_score = base.oversampled_lowfreq_mag_l2(
                x0_hat, mag_target, pad, score_radius
            )
            huber_score = base.oversampled_lowfreq_mag_huber(
                x0_hat,
                mag_target,
                pad,
                score_radius,
                delta=score_huber_delta,
            )

        prev_reg = torch.sqrt(
            torch.mean((base.to01(x0_hat) - base.to01(x0_target)) ** 2).clamp_min(1e-12)
        )

        candidate_x0s.append(x0_hat)
        candidate_eps.append(eps.detach().clone())
        lf_scores.append(lf_score)
        huber_scores.append(huber_score)
        prev_regs.append(prev_reg)

    mode = str(score_mode).lower()
    if mode in {"lf", "s1"}:
        final_scores = torch.stack([x.float().reshape(()) for x in lf_scores])
    elif mode in {"prev_l2", "s2"}:
        final_scores = (
            base._normalized_candidate_scores(lf_scores)
            + float(score_reg_lambda) * base._normalized_candidate_scores(prev_regs)
        )
    elif mode in {"adaptive_prev_l2", "adaptive_s2", "s2_adaptive"}:
        lf_norm = base._normalized_candidate_scores(lf_scores)
        prev_norm = base._normalized_candidate_scores(prev_regs)
        if int(num_candidates) >= 2:
            lf_sorted, _ = torch.sort(lf_norm)
            lf_margin = lf_sorted[1] - lf_sorted[0]
        else:
            lf_margin = torch.tensor(float("inf"), device=lf_norm.device, dtype=lf_norm.dtype)
        if float(lf_margin.detach().cpu().item()) <= float(adaptive_s2_margin):
            final_scores = lf_norm + float(score_reg_lambda) * prev_norm
        else:
            final_scores = lf_norm
    elif mode in {"consensus_l2", "s3"}:
        mean_x0 = torch.stack(candidate_x0s, dim=0).mean(dim=0)
        consensus_regs = [
            torch.sqrt(
                torch.mean((base.to01(x0_hat) - base.to01(mean_x0)) ** 2).clamp_min(1e-12)
            )
            for x0_hat in candidate_x0s
        ]
        final_scores = (
            base._normalized_candidate_scores(lf_scores)
            + float(score_reg_lambda) * base._normalized_candidate_scores(consensus_regs)
        )
    elif mode in {"huber_lf", "s4"}:
        final_scores = torch.stack([x.float().reshape(()) for x in huber_scores])
    else:
        raise ValueError(
            f"Unknown score_mode={score_mode!r}; expected lf, prev_l2, adaptive_prev_l2, "
            "consensus_l2, or huber_lf."
        )

    idx = int(torch.argmin(final_scores).item())
    return candidate_x0s[idx], candidate_eps[idx]


@torch.no_grad()
def noise_picking_reconstruct_oversampled_nearterm(
    mag_target: torch.Tensor,
    *,
    pad: int,
    seed: int,
    unet,
    scheduler,
    device: torch.device,
    variant,
    num_steps: int,
    score_radius: float,
    proj_radius: float,
    log_every: int,
    proj_radius_schedule: str | None = None,
    score_mode: str = "lf",
    score_reg_lambda: float = 0.25,
    score_reg_lambda_schedule: str = "constant",
    score_huber_delta: float = 0.05,
    noise_memory_k: int = 0,
    adaptive_s2_margin: float = 0.05,
) -> torch.Tensor:
    seed_everything(seed)
    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps
    proj_schedule = base.parse_radius_schedule(proj_radius_schedule, proj_radius)

    x_t = torch.randn((1, 3, unet.config.sample_size, unet.config.sample_size), device=device)
    x_prev = None
    eps_prev = None
    eps_memory: List[torch.Tensor] = []

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

        if variant.use_lowfreq_projection and i >= variant.proj_start:
            current_proj_radius = base.radius_at_step(proj_schedule, i)
            x0_hat = base.enforce_oversampled_lowfreq(
                x0_hat, mag_target, pad, current_proj_radius
            )

        k = variant.soft if i < variant.proj_start else variant.hard
        lambda_eff = score_lambda_at_step(
            score_reg_lambda,
            score_reg_lambda_schedule,
            step_index=i,
            proj_start=variant.proj_start,
        )

        x_prev, eps_prev = pick_noise_oversampled_nearterm(
            x0_target=x0_hat,
            t_int=t_next_int,
            eps_prev=eps_prev,
            eps_memory=eps_memory,
            mag_target=mag_target,
            pad=pad,
            num_candidates=k,
            score_radius=score_radius if variant.use_lowfreq_score else None,
            unet=unet,
            scheduler=scheduler,
            score_mode=score_mode,
            score_reg_lambda=lambda_eff,
            score_huber_delta=score_huber_delta,
            noise_memory_k=noise_memory_k,
            adaptive_s2_margin=adaptive_s2_margin,
        )

        if noise_memory_k > 0:
            eps_memory = [eps_prev.detach().clone()] + [
                eps.detach().clone() for eps in eps_memory[: max(0, int(noise_memory_k) - 1)]
            ]

        if log_every > 0 and (i + 1) % log_every == 0:
            print(
                f"[{variant.name}] step {i+1}/{len(timesteps)-1} "
                f"lambda_eff={lambda_eff:.6g} memory_k={int(noise_memory_k)} "
                f"adaptive_margin={adaptive_s2_margin:g}"
            )

    assert x_prev is not None
    return x_prev.detach()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Guided FFHQ NP near-term runner with scheduled S2 and memory bank."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--image_list_file", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--guided_model_path", required=True)
    parser.add_argument("--guided_diffusion_dir", default=None)
    parser.add_argument("--guided_preset", default="difffpr_ffhq_10m")
    parser.add_argument("--guided_strict", action="store_true")
    parser.add_argument("--variants", default="np_canonical")
    parser.add_argument("--seeds", default="100,101")
    parser.add_argument("--np_steps", type=int, default=1000)
    parser.add_argument("--late_start", type=int, default=300)
    parser.add_argument("--fixed_k", type=int, default=5)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--score_radius", type=float, default=None)
    parser.add_argument("--proj_radius", type=float, default=None)
    parser.add_argument("--proj_radius_schedule", default=None)
    parser.add_argument("--oversample_values", default="2")
    parser.add_argument("--measurement_noise_values", default="0.05")
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
    parser.add_argument(
        "--score_mode",
        default="lf",
        choices=[
            "lf", "prev_l2", "adaptive_prev_l2", "consensus_l2", "huber_lf",
            "s1", "s2", "adaptive_s2", "s2_adaptive", "s3", "s4",
        ],
    )
    parser.add_argument("--score_reg_lambda", type=float, default=0.25)
    parser.add_argument(
        "--score_reg_lambda_schedule",
        default="constant",
        choices=["constant", "linear_decay_to_proj_start", "pre_projection_only"],
    )
    parser.add_argument("--score_huber_delta", type=float, default=0.05)
    parser.add_argument("--noise_memory_k", type=int, default=0)
    parser.add_argument("--adaptive_s2_margin", type=float, default=0.05)
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
    run_root = os.path.join(args.outdir, f"difffpr_np_guided_nearterm_{stamp}")
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
                        "backend": "guided_diffusion_nearterm",
                        "guided_preset": args.guided_preset,
                        "guided_model_path": str(Path(args.guided_model_path).expanduser()),
                        "guided_diffusion_dir": args.guided_diffusion_dir or "PYTHONPATH/default",
                        "num_steps": args.np_steps,
                        "score_radius": score_radius,
                        "proj_radius": proj_radius,
                        "proj_radius_schedule": args.proj_radius_schedule or f"{variant.proj_start}:{proj_radius}",
                        "score_mode": args.score_mode,
                        "score_reg_lambda": args.score_reg_lambda,
                        "score_reg_lambda_schedule": args.score_reg_lambda_schedule,
                        "adaptive_s2_margin": args.adaptive_s2_margin,
                        "score_huber_delta": args.score_huber_delta,
                        "noise_memory_k": int(args.noise_memory_k),
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
                        gen = torch.Generator(device=device).manual_seed(
                            args.measurement_noise_seed + image_index
                        )
                        noise = torch.randn(
                            mag_clean.shape,
                            device=device,
                            dtype=mag_clean.dtype,
                            generator=gen,
                        )
                        mag_target = mag_clean + float(noise_std) * noise
                        if args.clip_noisy_magnitude:
                            mag_target = mag_target.clamp_min(0.0)

                    for seed in seeds:
                        t0 = time.perf_counter()
                        x_rec = noise_picking_reconstruct_oversampled_nearterm(
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
                            proj_radius_schedule=args.proj_radius_schedule,
                            score_mode=args.score_mode,
                            score_reg_lambda=args.score_reg_lambda,
                            score_reg_lambda_schedule=args.score_reg_lambda_schedule,
                            score_huber_delta=args.score_huber_delta,
                            noise_memory_k=int(args.noise_memory_k),
                            adaptive_s2_margin=args.adaptive_s2_margin,
                            log_every=args.log_every,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        runtime_s = time.perf_counter() - t0

                        for alignment in alignment_modes:
                            x_eval = guided.make_alignment(alignment, x_rec, x_gt)
                            if args.fast_eval:
                                clean_mag_l2 = math.nan
                                noisy_mag_l2 = math.nan
                                clean_lowfreq_mag_l2 = math.nan
                                noisy_lowfreq_mag_l2 = math.nan
                            else:
                                clean_mag_l2 = float(
                                    base.oversampled_mag_l2(x_eval, mag_clean, pad).cpu().item()
                                )
                                noisy_mag_l2 = float(
                                    base.oversampled_mag_l2(x_eval, mag_target, pad).cpu().item()
                                )
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
                                "backend": "guided_diffusion_nearterm",
                                "guided_preset": args.guided_preset,
                                "alignment_mode": alignment,
                                "image_basename": image_name,
                                "seed": seed,
                                "psnr": base.psnr01_from_model_range(x_eval, x_gt),
                                "ssim": guided.ssim01(x_eval, x_gt),
                                "lpips": guided.maybe_lpips_metric(x_eval, x_gt, lpips_model),
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
                                "proj_radius_schedule": args.proj_radius_schedule or f"{variant.proj_start}:{proj_radius}",
                                "score_mode": args.score_mode,
                                "score_reg_lambda": args.score_reg_lambda,
                                "score_reg_lambda_schedule": args.score_reg_lambda_schedule,
                                "adaptive_s2_margin": args.adaptive_s2_margin,
                                "score_huber_delta": args.score_huber_delta,
                                "noise_memory_k": int(args.noise_memory_k),
                                "radius": score_radius,
                                "oversample": oversample,
                                "pad_pixels_each_side": pad,
                                "measurement_noise_std": noise_std,
                            }
                            run_rows.append(row)

                        last_rows = run_rows[-len(alignment_modes):]
                        best_seen = max(float(r["psnr"]) for r in last_rows)
                        print(
                            f"[Guided-FFHQ-NP-nearterm] {variant.name} | {image_name} "
                            f"seed={seed} oversample={oversample} sigma={noise_std:.3f} "
                            f"score={args.score_mode} lambda={args.score_reg_lambda} "
                            f"lambda_sched={args.score_reg_lambda_schedule} margin={args.adaptive_s2_margin:g} "
                            f"memory_k={int(args.noise_memory_k)} "
                            f"best_psnr_this_run={best_seen:.2f} ({runtime_s:.1f}s)"
                        )

    file_prefix = f"difffpr_np_guided_nearterm_{stamp}"
    outputs = {
        "configs": os.path.join(run_root, f"{file_prefix}__configs.csv"),
        "run_level": os.path.join(run_root, f"{file_prefix}__run_level.csv"),
        "image_level_summary": os.path.join(run_root, f"{file_prefix}__image_level_summary.csv"),
        "condition_level_summary": os.path.join(run_root, f"{file_prefix}__condition_level_summary.csv"),
    }

    write_csv(outputs["configs"], config_rows)
    write_csv(outputs["run_level"], run_rows)
    write_csv(outputs["image_level_summary"], guided.summarize_image_level(run_rows))
    write_csv(
        outputs["condition_level_summary"],
        guided.summarize_condition_level(run_rows, psnr_threshold=args.psnr_threshold),
    )

    write_csv(os.path.join(run_root, "configs.csv"), config_rows)
    write_csv(os.path.join(run_root, "run_level.csv"), run_rows)
    write_csv(os.path.join(run_root, "image_level_summary.csv"), guided.summarize_image_level(run_rows))
    write_csv(
        os.path.join(run_root, "condition_level_summary.csv"),
        guided.summarize_condition_level(run_rows, psnr_threshold=args.psnr_threshold),
    )

    print(f"Saved: {run_root}")
    for key, path in outputs.items():
        print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()
