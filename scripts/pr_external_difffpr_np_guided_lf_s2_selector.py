#!/usr/bin/env python3
"""Lightweight LF/S2 selector experiment for guided FFHQ phase retrieval.

This runner tests the trajectory-level selector suggested by the full-25
selector diagnostic:

  run LF and pre-projection S2 for the same image/seed budget;
  compute post-projection mean winner low-frequency MSE vs the observation;
  choose the config with lower mean selector statistic.

Outputs include both:

- selected_config_bestofk: choose LF vs S2 by selector statistic, then evaluate
  best-of-k among seeds inside the chosen config.  This matches the diagnostic
  question: can the statistic choose the correct config?
- selected_run_by_selector: choose LF vs S2 by selector statistic, then choose
  the seed/run inside the chosen config by the same statistic.  This is the
  fully non-ground-truth executable selector.
- global_run_by_selector: choose the single run among LF/S2 and all seeds with
  minimum selector statistic.
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
from statistics import mean, median, stdev
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
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmean(xs: List[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return mean(xs) if xs else math.nan


def fmedian(xs: List[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return median(xs) if xs else math.nan


def fmin(xs: List[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return min(xs) if xs else math.nan


def fmax(xs: List[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return max(xs) if xs else math.nan


def fstd(xs: List[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return stdev(xs) if len(xs) > 1 else 0.0


def score_lambda_at_step(lambda0: float, schedule: str, step_index: int, proj_start: int) -> float:
    mode = str(schedule).lower().strip()
    if mode in {"constant", "static", "none"}:
        return float(lambda0)
    if mode in {"linear_decay_to_proj_start", "linear_decay", "decay"}:
        if proj_start <= 0:
            return 0.0
        return float(lambda0) * max(0.0, 1.0 - float(step_index) / float(proj_start))
    if mode in {"pre_projection_only", "preproj", "pre_proj"}:
        return float(lambda0) if int(step_index) < int(proj_start) else 0.0
    raise ValueError(f"Unknown score_reg_lambda_schedule={schedule!r}")


@torch.no_grad()
def lowfreq_mse_vs_observation(x: torch.Tensor, mag_target: torch.Tensor, pad: int, radius: float) -> float:
    mag = base.oversampled_magnitude(x, pad)
    resid = mag - mag_target
    _, _, h, w = mag.shape
    mask_hw = base.centered_lowfreq_mask(h, w, radius, mag.device)
    mask = mask_hw[None, None, :, :].expand_as(mag)
    return float(torch.mean(resid[mask].square()).detach().cpu().item())


@torch.no_grad()
def full_mse_vs_observation(x: torch.Tensor, mag_target: torch.Tensor, pad: int) -> float:
    mag = base.oversampled_magnitude(x, pad)
    resid = mag - mag_target
    return float(torch.mean(resid.square()).detach().cpu().item())


@torch.no_grad()
def reconstruct_with_selector_stat(
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
    proj_radius_schedule: Optional[str],
    score_mode: str,
    score_reg_lambda: float,
    score_reg_lambda_schedule: str,
    score_huber_delta: float,
    log_every: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Reconstruct and accumulate trajectory selector features.

    The key selector statistic is the mean low-frequency MSE of the selected
    winner after projection starts.  This is the lightweight version of the
    diagnostic field `post_winner_lf_mse_mean`.
    """
    seed_everything(seed)
    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps
    proj_schedule = base.parse_radius_schedule(proj_radius_schedule, proj_radius)

    x_t = torch.randn((1, 3, unet.config.sample_size, unet.config.sample_size), device=device)
    x_prev = None
    eps_prev = None

    selector_post_lf_mse: List[float] = []
    selector_post_full_mse: List[float] = []
    selector_pre_lf_mse: List[float] = []
    pre_lf_margins: List[float] = []
    post_lf_margins: List[float] = []
    winner_is_lf_best_pre: List[float] = []
    winner_is_lf_best_post: List[float] = []

    for i in range(len(timesteps) - 1):
        t_int = int(timesteps[i])
        t_next_int = int(timesteps[i + 1])

        if i == 0:
            t_tensor = torch.tensor([t_int], device=device, dtype=torch.long)
            eps = unet(x_t, t_tensor).sample
            alpha_bar_t0 = scheduler.alphas_cumprod[t_int].to(device=device, dtype=x_t.dtype)
            sqrt_ab = torch.sqrt(alpha_bar_t0)
            sqrt_1mab = torch.sqrt(1.0 - alpha_bar_t0)
            x0_hat = (x_t - sqrt_1mab * eps) / sqrt_ab
        else:
            assert x_prev is not None
            x0_hat = x_prev

        projection_applied = bool(variant.use_lowfreq_projection and i >= variant.proj_start)
        if projection_applied:
            current_proj_radius = base.radius_at_step(proj_schedule, i)
            x0_hat = base.enforce_oversampled_lowfreq(
                x0_hat, mag_target, pad, current_proj_radius
            )

        k = int(variant.soft if i < variant.proj_start else variant.hard)
        lambda_eff = score_lambda_at_step(
            score_reg_lambda, score_reg_lambda_schedule, step_index=i, proj_start=variant.proj_start
        )

        alpha_bar_t = scheduler.alphas_cumprod[t_next_int].to(device=device, dtype=x0_hat.dtype)
        sqrt_at = torch.sqrt(alpha_bar_t)
        sqrt_1mat = torch.sqrt(1.0 - alpha_bar_t)

        cand_x0s: List[torch.Tensor] = []
        cand_eps: List[torch.Tensor] = []
        lf_l2_scores: List[torch.Tensor] = []
        prev_regs: List[torch.Tensor] = []
        huber_scores: List[torch.Tensor] = []
        cand_lf_mse: List[float] = []
        cand_full_mse: List[float] = []

        for j in range(k):
            if j == 0 and eps_prev is not None and k > 1:
                eps_cand = eps_prev
            else:
                eps_cand = torch.randn_like(x0_hat)

            x_t_cand = sqrt_at * x0_hat + sqrt_1mat * eps_cand
            t_tensor = torch.tensor([t_next_int], device=device, dtype=torch.long)
            eps_pred = unet(x_t_cand, t_tensor).sample
            x0_cand = (x_t_cand - sqrt_1mat * eps_pred) / sqrt_at

            lf_l2 = base.oversampled_lowfreq_mag_l2(x0_cand, mag_target, pad, score_radius)
            prev_reg = torch.sqrt(
                torch.mean((base.to01(x0_cand) - base.to01(x0_hat)) ** 2).clamp_min(1e-12)
            )
            huber_score = base.oversampled_lowfreq_mag_huber(
                x0_cand, mag_target, pad, score_radius, delta=score_huber_delta
            )

            cand_x0s.append(x0_cand)
            cand_eps.append(eps_cand.detach().clone())
            lf_l2_scores.append(lf_l2)
            prev_regs.append(prev_reg)
            huber_scores.append(huber_score)
            cand_lf_mse.append(lowfreq_mse_vs_observation(x0_cand, mag_target, pad, score_radius))
            cand_full_mse.append(full_mse_vs_observation(x0_cand, mag_target, pad))

        mode = str(score_mode).lower()
        if mode in {"lf", "s1"}:
            final_scores = torch.stack([x.float().reshape(()) for x in lf_l2_scores])
        elif mode in {"prev_l2", "s2"}:
            final_scores = (
                base._normalized_candidate_scores(lf_l2_scores)
                + float(lambda_eff) * base._normalized_candidate_scores(prev_regs)
            )
        elif mode in {"huber_lf", "s4"}:
            final_scores = torch.stack([x.float().reshape(()) for x in huber_scores])
        else:
            raise ValueError(f"Selector runner supports score_mode lf, prev_l2, huber_lf; got {score_mode!r}")

        winner_idx = int(torch.argmin(final_scores).item())
        lf_order = sorted(range(k), key=lambda j: float(lf_l2_scores[j].detach().cpu().item()))
        lf_best_idx = int(lf_order[0])
        lf_second_idx = int(lf_order[1]) if k >= 2 else -1
        lf_margin = (
            cand_lf_mse[lf_second_idx] - cand_lf_mse[lf_best_idx] if k >= 2 else math.nan
        )
        winner_is_lf = 1.0 if winner_idx == lf_best_idx else 0.0

        if i < variant.proj_start:
            selector_pre_lf_mse.append(cand_lf_mse[winner_idx])
            if math.isfinite(lf_margin):
                pre_lf_margins.append(lf_margin)
            winner_is_lf_best_pre.append(winner_is_lf)
        else:
            selector_post_lf_mse.append(cand_lf_mse[winner_idx])
            selector_post_full_mse.append(cand_full_mse[winner_idx])
            if math.isfinite(lf_margin):
                post_lf_margins.append(lf_margin)
            winner_is_lf_best_post.append(winner_is_lf)

        x_prev = cand_x0s[winner_idx]
        eps_prev = cand_eps[winner_idx]

        if log_every > 0 and (i + 1) % log_every == 0:
            print(
                f"[{score_mode}] seed={seed} step {i+1}/{len(timesteps)-1} "
                f"winner_lf_mse={cand_lf_mse[winner_idx]:.4e} post_mean={fmean(selector_post_lf_mse):.4e}"
            )

    assert x_prev is not None
    stats = {
        "selector_post_winner_lf_mse_mean": fmean(selector_post_lf_mse),
        "selector_post_winner_lf_mse_median": fmedian(selector_post_lf_mse),
        "selector_post_winner_lf_mse_min": fmin(selector_post_lf_mse),
        "selector_post_winner_lf_mse_max": fmax(selector_post_lf_mse),
        "selector_post_winner_full_mse_mean": fmean(selector_post_full_mse),
        "selector_pre_winner_lf_mse_mean": fmean(selector_pre_lf_mse),
        "selector_pre_lf_mse_margin_mean": fmean(pre_lf_margins),
        "selector_post_lf_mse_margin_mean": fmean(post_lf_margins),
        "selector_winner_is_lf_best_frac_pre": fmean(winner_is_lf_best_pre),
        "selector_winner_is_lf_best_frac_post": fmean(winner_is_lf_best_post),
        "n_pre_selector_steps": len(selector_pre_lf_mse),
        "n_post_selector_steps": len(selector_post_lf_mse),
    }
    return x_prev.detach(), stats


def summarize_selected(selected_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_method_align: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for r in selected_rows:
        by_method_align.setdefault((str(r["selection_method"]), str(r["alignment_mode"])), []).append(r)

    out: List[Dict[str, object]] = []
    for (method, align), rows in sorted(by_method_align.items()):
        psnr = [float(r["psnr"]) for r in rows]
        ssim = [float(r["ssim"]) for r in rows]
        lpips = [float(r["lpips"]) for r in rows if str(r.get("lpips", "nan")) != "nan"]
        worst = min(rows, key=lambda r: float(r["psnr"]))
        out.append(
            {
                "selection_method": method,
                "alignment_mode": align,
                "n_images": len(rows),
                "psnr_mean": fmean(psnr),
                "psnr_median": fmedian(psnr),
                "psnr_min": fmin(psnr),
                "psnr_max": fmax(psnr),
                "psnr_std": fstd(psnr),
                "ssim_mean": fmean(ssim),
                "lpips_mean": fmean(lpips) if lpips else math.nan,
                "n_images_below20": sum(1 for x in psnr if x < 20),
                "n_images_below25": sum(1 for x in psnr if x < 25),
                "worst_image": worst.get("image_basename", ""),
                "worst_image_psnr": float(worst["psnr"]),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Run LF/S2 trajectory-stat selector experiment.")
    p.add_argument("--data_root", required=True)
    p.add_argument("--image_list_file", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--guided_model_path", required=True)
    p.add_argument("--guided_diffusion_dir", default=None)
    p.add_argument("--guided_preset", default="difffpr_ffhq_10m")
    p.add_argument("--guided_strict", action="store_true")
    p.add_argument("--seeds", default="100,101")
    p.add_argument("--np_steps", type=int, default=1000)
    p.add_argument("--late_start", type=int, default=300)
    p.add_argument("--soft_candidates", type=int, default=5)
    p.add_argument("--hard_candidates", type=int, default=1)
    p.add_argument("--score_radius", type=float, default=0.6)
    p.add_argument("--proj_radius", type=float, default=0.2)
    p.add_argument("--proj_radius_schedule", default=None)
    p.add_argument("--s2_lambda", type=float, default=0.01)
    p.add_argument("--s2_lambda_schedule", default="pre_projection_only", choices=["constant", "linear_decay_to_proj_start", "pre_projection_only"])
    p.add_argument("--score_huber_delta", type=float, default=0.05)
    p.add_argument("--oversample_values", default="2")
    p.add_argument("--measurement_noise_values", default="0.05")
    p.add_argument("--measurement_noise_seed", type=int, default=20260423)
    p.add_argument("--clip_noisy_magnitude", action="store_true")
    p.add_argument("--alignments", default="raw,rot180,resolve")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--skip_lpips", action="store_true")
    p.add_argument("--log_every", type=int, default=100)
    args = p.parse_args()

    all_image_names = base.collect_images(args.data_root, args.image_list_file)
    if not all_image_names:
        raise ValueError(f"No images found in split {args.image_list_file}")
    image_names = all_image_names[: args.max_images] if args.max_images is not None else all_image_names
    original_index = {name: i for i, name in enumerate(all_image_names)}

    seeds = base.parse_int_list(args.seeds)
    oversample_values = [float(x.strip()) for x in args.oversample_values.split(",") if x.strip()]
    noise_values = [float(x.strip()) for x in args.measurement_noise_values.split(",") if x.strip()]
    alignment_modes = [x.strip() for x in args.alignments.split(",") if x.strip()]

    os.makedirs(args.outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.outdir, f"lf_s2_selector_{stamp}")
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

    variant = base.NPVariant(
        name=f"np_canonical_soft{args.soft_candidates}_hard{args.hard_candidates}",
        soft=int(args.soft_candidates),
        hard=int(args.hard_candidates),
        proj_start=int(args.late_start),
        use_lowfreq_score=True,
        use_lowfreq_projection=True,
    )

    configs = [
        {
            "config_tag": "lf",
            "score_mode": "lf",
            "score_reg_lambda": 0.0,
            "score_reg_lambda_schedule": "constant",
        },
        {
            "config_tag": "s2_preproj_lam001",
            "score_mode": "prev_l2",
            "score_reg_lambda": float(args.s2_lambda),
            "score_reg_lambda_schedule": args.s2_lambda_schedule,
        },
    ]

    config_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []

    for oversample in oversample_values:
        pad = base.oversample_pad(image_size, oversample)
        for noise_std in noise_values:
            config_rows.append(
                {
                    "backend": "guided_diffusion_lf_s2_selector",
                    "guided_model_path": str(Path(args.guided_model_path).expanduser()),
                    "guided_diffusion_dir": args.guided_diffusion_dir or "PYTHONPATH/default",
                    "guided_preset": args.guided_preset,
                    "num_steps": args.np_steps,
                    "score_radius": args.score_radius,
                    "proj_radius": args.proj_radius,
                    "proj_radius_schedule": args.proj_radius_schedule or f"{args.late_start}:{args.proj_radius}",
                    "s2_lambda": args.s2_lambda,
                    "s2_lambda_schedule": args.s2_lambda_schedule,
                    "proj_start": args.late_start,
                    "num_candidates_soft": args.soft_candidates,
                    "num_candidates_hard": args.hard_candidates,
                    "oversample_arg": oversample,
                    "pad_pixels_each_side": pad,
                    "measurement_noise_std": noise_std,
                    "clip_noisy_magnitude": bool(args.clip_noisy_magnitude),
                    "seeds": ",".join(map(str, seeds)),
                    "alignments": ",".join(alignment_modes),
                    "lpips_backend": lpips_status,
                    "selector_stat": "mean post-projection winner LF MSE vs noisy observation",
                }
            )

            for image_name in image_names:
                img_path = base.resolve_image_path(args.data_root, image_name)
                x_gt = load_image(img_path, size=image_size, device=device)
                mag_clean = base.oversampled_magnitude(x_gt, pad)
                mag_target = mag_clean
                image_index = original_index[image_name]
                if noise_std > 0:
                    gen = torch.Generator(device=device).manual_seed(args.measurement_noise_seed + image_index)
                    noise = torch.randn(mag_clean.shape, device=device, dtype=mag_clean.dtype, generator=gen)
                    mag_target = mag_clean + float(noise_std) * noise
                    if args.clip_noisy_magnitude:
                        mag_target = mag_target.clamp_min(0.0)

                image_run_rows: List[Dict[str, object]] = []
                for cfg in configs:
                    for seed in seeds:
                        t0 = time.perf_counter()
                        x_rec, selector_stats = reconstruct_with_selector_stat(
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
                            proj_radius_schedule=args.proj_radius_schedule,
                            score_mode=str(cfg["score_mode"]),
                            score_reg_lambda=float(cfg["score_reg_lambda"]),
                            score_reg_lambda_schedule=str(cfg["score_reg_lambda_schedule"]),
                            score_huber_delta=args.score_huber_delta,
                            log_every=args.log_every,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        runtime_s = time.perf_counter() - t0

                        for alignment in alignment_modes:
                            x_eval = guided.make_alignment(alignment, x_rec, x_gt)
                            row = {
                                "timestamp": stamp,
                                "config_tag": cfg["config_tag"],
                                "variant": variant.name,
                                "alignment_mode": alignment,
                                "image_basename": image_name,
                                "image_index_in_split": image_index,
                                "seed": seed,
                                "psnr": base.psnr01_from_model_range(x_eval, x_gt),
                                "ssim": guided.ssim01(x_eval, x_gt),
                                "lpips": guided.maybe_lpips_metric(x_eval, x_gt, lpips_model),
                                "clean_mag_l2": float(base.oversampled_mag_l2(x_eval, mag_clean, pad).cpu().item()),
                                "noisy_mag_l2": float(base.oversampled_mag_l2(x_eval, mag_target, pad).cpu().item()),
                                "clean_lowfreq_mag_l2": float(base.oversampled_lowfreq_mag_l2(x_eval, mag_clean, pad, args.score_radius).cpu().item()),
                                "noisy_lowfreq_mag_l2": float(base.oversampled_lowfreq_mag_l2(x_eval, mag_target, pad, args.score_radius).cpu().item()),
                                "runtime_s": runtime_s,
                                "num_steps": args.np_steps,
                                "proj_start": args.late_start,
                                "num_candidates_soft": args.soft_candidates,
                                "num_candidates_hard": args.hard_candidates,
                                "score_radius": args.score_radius,
                                "proj_radius": args.proj_radius,
                                "score_mode": cfg["score_mode"],
                                "score_reg_lambda": cfg["score_reg_lambda"],
                                "score_reg_lambda_schedule": cfg["score_reg_lambda_schedule"],
                                "oversample": oversample,
                                "measurement_noise_std": noise_std,
                                **selector_stats,
                            }
                            run_rows.append(row)
                            image_run_rows.append(row)
                        print(
                            f"[selector] {cfg['config_tag']} {image_name} seed={seed} "
                            f"selector_post_lf_mse={selector_stats['selector_post_winner_lf_mse_mean']:.4e} "
                            f"runtime={runtime_s:.1f}s"
                        )

                # Select only on raw-alignment rows; metrics for the selected run/config are then
                # copied for every requested alignment.
                raw_rows = [r for r in image_run_rows if r["alignment_mode"] == "raw"]
                by_cfg: Dict[str, List[Dict[str, object]]] = {}
                for r in raw_rows:
                    by_cfg.setdefault(str(r["config_tag"]), []).append(r)
                cfg_stat = {
                    cfg: fmean([float(r["selector_post_winner_lf_mse_mean"]) for r in rows])
                    for cfg, rows in by_cfg.items()
                }
                selected_cfg = min(cfg_stat, key=lambda c: cfg_stat[c])

                # View 1: config selected by statistic, seed evaluated as best-of-k inside config.
                for alignment in alignment_modes:
                    rows = [r for r in image_run_rows if r["config_tag"] == selected_cfg and r["alignment_mode"] == alignment]
                    best = max(rows, key=lambda r: float(r["psnr"]))
                    out = dict(best)
                    out.update(
                        {
                            "selection_method": "selected_config_bestofk",
                            "selected_config": selected_cfg,
                            "selected_seed": best["seed"],
                            "lf_config_selector_stat_mean": cfg_stat.get("lf", math.nan),
                            "s2_config_selector_stat_mean": cfg_stat.get("s2_preproj_lam001", math.nan),
                            "selector_margin_s2_minus_lf": cfg_stat.get("s2_preproj_lam001", math.nan) - cfg_stat.get("lf", math.nan),
                        }
                    )
                    selected_rows.append(out)

                # View 2: config selected by statistic, seed selected by statistic inside config.
                raw_cfg_rows = by_cfg[selected_cfg]
                selected_seed_row = min(raw_cfg_rows, key=lambda r: float(r["selector_post_winner_lf_mse_mean"]))
                selected_seed = selected_seed_row["seed"]
                for alignment in alignment_modes:
                    row = next(
                        r for r in image_run_rows
                        if r["config_tag"] == selected_cfg and r["seed"] == selected_seed and r["alignment_mode"] == alignment
                    )
                    out = dict(row)
                    out.update(
                        {
                            "selection_method": "selected_config_seed_by_selector",
                            "selected_config": selected_cfg,
                            "selected_seed": selected_seed,
                            "lf_config_selector_stat_mean": cfg_stat.get("lf", math.nan),
                            "s2_config_selector_stat_mean": cfg_stat.get("s2_preproj_lam001", math.nan),
                            "selector_margin_s2_minus_lf": cfg_stat.get("s2_preproj_lam001", math.nan) - cfg_stat.get("lf", math.nan),
                        }
                    )
                    selected_rows.append(out)

                # View 3: choose the single run globally by selector statistic.
                selected_global_raw = min(raw_rows, key=lambda r: float(r["selector_post_winner_lf_mse_mean"]))
                global_cfg = selected_global_raw["config_tag"]
                global_seed = selected_global_raw["seed"]
                for alignment in alignment_modes:
                    row = next(
                        r for r in image_run_rows
                        if r["config_tag"] == global_cfg and r["seed"] == global_seed and r["alignment_mode"] == alignment
                    )
                    out = dict(row)
                    out.update(
                        {
                            "selection_method": "global_run_by_selector",
                            "selected_config": global_cfg,
                            "selected_seed": global_seed,
                            "lf_config_selector_stat_mean": cfg_stat.get("lf", math.nan),
                            "s2_config_selector_stat_mean": cfg_stat.get("s2_preproj_lam001", math.nan),
                            "selector_margin_s2_minus_lf": cfg_stat.get("s2_preproj_lam001", math.nan) - cfg_stat.get("lf", math.nan),
                        }
                    )
                    selected_rows.append(out)

    write_csv(os.path.join(run_root, "configs.csv"), config_rows)
    write_csv(os.path.join(run_root, "run_level.csv"), run_rows)
    write_csv(os.path.join(run_root, "selected_image_level.csv"), selected_rows)
    summary = summarize_selected(selected_rows)
    write_csv(os.path.join(run_root, "selected_summary.csv"), summary)

    print(f"Saved: {run_root}")
    print("Selected summary:")
    for r in summary:
        print(r)


if __name__ == "__main__":
    main()
