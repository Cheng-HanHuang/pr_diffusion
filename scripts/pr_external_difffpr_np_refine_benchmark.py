#!/usr/bin/env python3
"""NP + anchored magnitude-refinement benchmark for DiffFPR-style phase retrieval.

This runner is intentionally conservative: it reuses the existing guided NP
candidate generation and LF/S2 trajectory selector, then applies a small local
measurement-consistency refinement to the selected/each candidate.  The goal is
not to reproduce SITCOM/DAPS/DiffFPR, but to test the hypothesis:

    NP is a good mode/trajectory selector; local optimization should improve a
    candidate once NP has placed it in a good basin.

Outputs:
  - configs.csv
  - run_level.csv                 one row per candidate/config/seed/alignment/refine setting
  - selected_image_level.csv      executable selector views per image/alignment/refine setting
  - selected_summary.csv          aggregate selected performance

The default is FFHQ-compatible and matches the project convention of
oversample=2 and best-of-4 via seeds="100,101,102,103".
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
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE_SCRIPT = _REPO_ROOT / "scripts" / "pr_external_difffpr_np_benchmark.py"
_GUIDED_SCRIPT = _REPO_ROOT / "scripts" / "pr_external_difffpr_np_guided_benchmark.py"
_SELECTOR_SCRIPT = _REPO_ROOT / "scripts" / "pr_external_difffpr_np_guided_lf_s2_selector.py"


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
selector = _load_module("pr_external_difffpr_np_guided_lf_s2_selector", _SELECTOR_SCRIPT)

from prdiffusion.guided_backend import load_guided_diffusion_model
from prdiffusion.io import load_image


def parse_csv_floats(text: str) -> List[float]:
    return [float(tok.strip()) for tok in str(text).split(",") if tok.strip()]


def parse_csv_ints(text: str) -> List[int]:
    return [int(tok.strip()) for tok in str(text).split(",") if tok.strip()]


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return mean(vals) if vals else math.nan


def fmedian(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return median(vals) if vals else math.nan


def fstd(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return stdev(vals) if len(vals) > 1 else 0.0


def fmin(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return min(vals) if vals else math.nan


def fmax(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return max(vals) if vals else math.nan


@torch.no_grad()
def gs_magnitude_refine(
    x: torch.Tensor,
    mag_target: torch.Tensor,
    *,
    pad: int,
    radius: float,
    steps: int,
) -> torch.Tensor:
    """Repeated magnitude replacement in the padded Fourier domain.

    radius >= 1.0 covers the whole centered frequency square for normal FFT
    frequency coordinates.  Smaller radii give low-frequency-only correction.
    """
    out = x.detach()
    for _ in range(max(0, int(steps))):
        out = base.enforce_oversampled_lowfreq(out, mag_target, pad, float(radius))
        out = out.clamp(-1.0, 1.0)
    return out.detach()


def _masked_mag_mse(
    x: torch.Tensor,
    mag_target: torch.Tensor,
    *,
    pad: int,
    radius: Optional[float],
    huber_delta: Optional[float] = None,
) -> torch.Tensor:
    mag = base.oversampled_magnitude(x, pad)
    resid = mag - mag_target
    if radius is not None and float(radius) < 1.0:
        _, _, h, w = mag.shape
        mask_hw = base.centered_lowfreq_mask(h, w, float(radius), mag.device)
        resid = resid[mask_hw[None, None, :, :].expand_as(resid)]
    if huber_delta is not None and float(huber_delta) > 0:
        d = torch.as_tensor(float(huber_delta), dtype=resid.dtype, device=resid.device)
        abs_r = resid.abs()
        quad = torch.minimum(abs_r, d)
        lin = abs_r - quad
        return (0.5 * quad.square() + d * lin).mean()
    return resid.square().mean()


def tv_loss01(x: torch.Tensor) -> torch.Tensor:
    x01 = base.to01(x)
    dy = x01[..., 1:, :] - x01[..., :-1, :]
    dx = x01[..., :, 1:] - x01[..., :, :-1]
    return dy.abs().mean() + dx.abs().mean()


def adam_anchor_refine(
    x_init: torch.Tensor,
    mag_target: torch.Tensor,
    *,
    pad: int,
    steps: int,
    lr: float,
    mag_radius: Optional[float],
    anchor_weight: float,
    tv_weight: float,
    huber_delta: Optional[float],
    clamp_each_step: bool,
    log_every: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Local measurement refinement anchored to the NP candidate.

    Optimizes directly over the image tensor in model range [-1, 1]:

        L = magnitude_loss(x, y) + anchor_weight ||to01(x)-to01(x_np)||^2
            + tv_weight TV(to01(x)).

    This is intentionally small and solver-agnostic: it tests whether NP picked a
    good basin where ordinary measurement optimization can improve details.
    """
    if int(steps) <= 0:
        return x_init.detach(), {
            "refine_initial_loss": math.nan,
            "refine_final_loss": math.nan,
            "refine_initial_mag_loss": math.nan,
            "refine_final_mag_loss": math.nan,
            "refine_initial_anchor_loss": math.nan,
            "refine_final_anchor_loss": math.nan,
        }

    x_anchor = x_init.detach()
    z = x_init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=float(lr))

    last: Dict[str, float] = {}
    first: Optional[Dict[str, float]] = None
    for it in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        mag_loss = _masked_mag_mse(
            z, mag_target, pad=pad, radius=mag_radius, huber_delta=huber_delta
        )
        anchor_loss = torch.mean((base.to01(z) - base.to01(x_anchor)) ** 2)
        tv = tv_loss01(z) if float(tv_weight) > 0 else z.new_tensor(0.0)
        total = mag_loss + float(anchor_weight) * anchor_loss + float(tv_weight) * tv
        total.backward()
        opt.step()
        if clamp_each_step:
            with torch.no_grad():
                z.clamp_(-1.0, 1.0)

        last = {
            "loss": float(total.detach().cpu().item()),
            "mag_loss": float(mag_loss.detach().cpu().item()),
            "anchor_loss": float(anchor_loss.detach().cpu().item()),
            "tv_loss": float(tv.detach().cpu().item()),
        }
        if first is None:
            first = dict(last)
        if log_every > 0 and ((it + 1) % log_every == 0 or it == int(steps) - 1):
            print(
                f"[refine] iter={it+1}/{steps} loss={last['loss']:.4e} "
                f"mag={last['mag_loss']:.4e} anchor={last['anchor_loss']:.4e} tv={last['tv_loss']:.4e}",
                flush=True,
            )

    assert first is not None
    return z.detach().clamp(-1.0, 1.0), {
        "refine_initial_loss": first["loss"],
        "refine_final_loss": last["loss"],
        "refine_initial_mag_loss": first["mag_loss"],
        "refine_final_mag_loss": last["mag_loss"],
        "refine_initial_anchor_loss": first["anchor_loss"],
        "refine_final_anchor_loss": last["anchor_loss"],
        "refine_initial_tv_loss": first["tv_loss"],
        "refine_final_tv_loss": last["tv_loss"],
    }


@torch.no_grad()
def eval_alignment_rows(
    *,
    x_rec: torch.Tensor,
    x_gt: torch.Tensor,
    mag_clean: torch.Tensor,
    mag_target: torch.Tensor,
    pad: int,
    score_radius: float,
    lpips_model,
    alignment_modes: List[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for alignment in alignment_modes:
        x_eval = guided.make_alignment(alignment, x_rec, x_gt)
        rows.append(
            {
                "alignment_mode": alignment,
                "psnr": base.psnr01_from_model_range(x_eval, x_gt),
                "ssim": guided.ssim01(x_eval, x_gt),
                "lpips": guided.maybe_lpips_metric(x_eval, x_gt, lpips_model),
                "clean_mag_l2": float(base.oversampled_mag_l2(x_eval, mag_clean, pad).cpu().item()),
                "noisy_mag_l2": float(base.oversampled_mag_l2(x_eval, mag_target, pad).cpu().item()),
                "clean_lowfreq_mag_l2": float(
                    base.oversampled_lowfreq_mag_l2(x_eval, mag_clean, pad, score_radius).cpu().item()
                ),
                "noisy_lowfreq_mag_l2": float(
                    base.oversampled_lowfreq_mag_l2(x_eval, mag_target, pad, score_radius).cpu().item()
                ),
            }
        )
    return rows


def summarize_selected(selected_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
    for r in selected_rows:
        key = (str(r["selection_method"]), str(r["refine_tag"]), str(r["alignment_mode"]))
        grouped.setdefault(key, []).append(r)

    out: List[Dict[str, object]] = []
    for (method, refine_tag, align), rows in sorted(grouped.items()):
        psnr = [float(r["psnr"]) for r in rows]
        ssim = [float(r["ssim"]) for r in rows]
        lpips = [float(r["lpips"]) for r in rows if math.isfinite(float(r.get("lpips", math.nan)))]
        worst = min(rows, key=lambda r: float(r["psnr"]))
        out.append(
            {
                "selection_method": method,
                "refine_tag": refine_tag,
                "alignment_mode": align,
                "n_images": len(rows),
                "psnr_mean": fmean(psnr),
                "psnr_median": fmedian(psnr),
                "psnr_min": fmin(psnr),
                "psnr_max": fmax(psnr),
                "psnr_std": fstd(psnr),
                "ssim_mean": fmean(ssim),
                "lpips_mean": fmean(lpips) if lpips else math.nan,
                "n_images_below20": sum(1 for x in psnr if x < 20.0),
                "n_images_below25": sum(1 for x in psnr if x < 25.0),
                "n_images_below28": sum(1 for x in psnr if x < 28.0),
                "worst_image": worst.get("image_basename", ""),
                "worst_image_psnr": float(worst["psnr"]),
            }
        )
    return out


def add_selected_views(
    *,
    image_rows: List[Dict[str, object]],
    alignment_modes: List[str],
    selected_rows: List[Dict[str, object]],
) -> None:
    """Add executable selector and oracle diagnostic views for one image/noise/refine_tag."""
    raw_rows = [r for r in image_rows if r["alignment_mode"] == "raw"]
    if not raw_rows:
        return

    by_cfg: Dict[str, List[Dict[str, object]]] = {}
    for r in raw_rows:
        by_cfg.setdefault(str(r["config_tag"]), []).append(r)

    cfg_stat = {
        cfg: fmean(float(r["selector_post_winner_lf_mse_mean"]) for r in rows)
        for cfg, rows in by_cfg.items()
    }
    selected_cfg = min(cfg_stat, key=lambda c: cfg_stat[c])

    selected_global_raw = min(raw_rows, key=lambda r: float(r["selector_post_winner_lf_mse_mean"]))
    global_cfg = str(selected_global_raw["config_tag"])
    global_seed = int(selected_global_raw["seed"])

    raw_cfg_rows = by_cfg[selected_cfg]
    selected_seed_row = min(raw_cfg_rows, key=lambda r: float(r["selector_post_winner_lf_mse_mean"]))
    selected_seed = int(selected_seed_row["seed"])

    views = [
        ("selected_config_seed_by_selector", selected_cfg, selected_seed),
        ("global_run_by_selector", global_cfg, global_seed),
    ]

    # This is explicitly diagnostic/oracle; it is useful for candidate generation p_x.
    oracle_raw = max(raw_rows, key=lambda r: float(r["psnr"]))
    views.append(("oracle_best_run_psnr_diagnostic", str(oracle_raw["config_tag"]), int(oracle_raw["seed"])))

    for method, cfg, seed in views:
        for alignment in alignment_modes:
            row = next(
                r
                for r in image_rows
                if str(r["config_tag"]) == cfg
                and int(r["seed"]) == int(seed)
                and str(r["alignment_mode"]) == alignment
            )
            out = dict(row)
            out.update(
                {
                    "selection_method": method,
                    "selected_config": cfg,
                    "selected_seed": seed,
                    "lf_config_selector_stat_mean": cfg_stat.get("lf", math.nan),
                    "s2_config_selector_stat_mean": cfg_stat.get("s2_preproj", math.nan),
                    "selector_margin_s2_minus_lf": cfg_stat.get("s2_preproj", math.nan)
                    - cfg_stat.get("lf", math.nan),
                }
            )
            selected_rows.append(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Run NP + anchored refinement experiments.")
    p.add_argument("--data_root", required=True)
    p.add_argument("--image_list_file", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--guided_model_path", required=True)
    p.add_argument("--guided_diffusion_dir", default=None)
    p.add_argument("--guided_preset", default="difffpr_ffhq_10m")
    p.add_argument("--guided_strict", action="store_true")
    p.add_argument("--seeds", default="100,101,102,103")
    p.add_argument("--np_steps", type=int, default=1000)
    p.add_argument("--late_start", type=int, default=300)
    p.add_argument("--soft_candidates", type=int, default=5)
    p.add_argument("--hard_candidates", type=int, default=1)
    p.add_argument("--score_radius", type=float, default=0.6)
    p.add_argument("--proj_radius", type=float, default=0.2)
    p.add_argument("--proj_radius_schedule", default=None)
    p.add_argument("--s2_lambda", type=float, default=0.01)
    p.add_argument(
        "--s2_lambda_schedule",
        default="pre_projection_only",
        choices=["constant", "linear_decay_to_proj_start", "pre_projection_only"],
    )
    p.add_argument("--score_huber_delta", type=float, default=0.05)
    p.add_argument("--oversample_values", default="2")
    p.add_argument("--measurement_noise_values", default="0,0.01,0.05,0.08,0.10")
    p.add_argument("--measurement_noise_seed", type=int, default=20260423)
    p.add_argument("--clip_noisy_magnitude", action="store_true")
    p.add_argument("--alignments", default="raw,rot180,resolve")
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--skip_lpips", action="store_true")
    p.add_argument("--log_every", type=int, default=100)

    # Refinement controls.
    p.add_argument("--refine_steps", type=int, default=100)
    p.add_argument("--refine_lr", type=float, default=0.01)
    p.add_argument("--refine_anchor_weights", default="0,0.01,0.05")
    p.add_argument("--refine_tv_weight", type=float, default=0.0)
    p.add_argument("--refine_mag_radius", default="full", help="full or a radius such as 0.6")
    p.add_argument("--refine_huber_delta", type=float, default=0.0)
    p.add_argument("--no_clamp_each_refine_step", action="store_true")
    p.add_argument("--gs_pre_steps", type=int, default=0)
    p.add_argument("--gs_post_steps", type=int, default=0)
    p.add_argument("--gs_radius", type=float, default=1.0)
    args = p.parse_args()

    all_image_names = base.collect_images(args.data_root, args.image_list_file)
    if not all_image_names:
        raise ValueError(f"No images found in split {args.image_list_file}")
    image_names = all_image_names[: args.max_images] if args.max_images is not None else all_image_names
    original_index = {name: i for i, name in enumerate(all_image_names)}

    seeds = parse_csv_ints(args.seeds)
    oversample_values = parse_csv_floats(args.oversample_values)
    noise_values = parse_csv_floats(args.measurement_noise_values)
    alignment_modes = [x.strip() for x in args.alignments.split(",") if x.strip()]
    anchor_weights = parse_csv_floats(args.refine_anchor_weights)
    refine_mag_radius: Optional[float]
    if str(args.refine_mag_radius).lower() in {"full", "all", "none"}:
        refine_mag_radius = None
    else:
        refine_mag_radius = float(args.refine_mag_radius)
    huber_delta = float(args.refine_huber_delta) if float(args.refine_huber_delta) > 0 else None

    os.makedirs(args.outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.outdir, f"np_refine_{stamp}")
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
        except Exception as exc:
            print(f"[warn] LPIPS unavailable: {exc}", flush=True)
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
            "config_tag": "s2_preproj",
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
                    "backend": "guided_np_plus_anchor_refine",
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
                    "refine_steps": args.refine_steps,
                    "refine_lr": args.refine_lr,
                    "refine_anchor_weights": args.refine_anchor_weights,
                    "refine_tv_weight": args.refine_tv_weight,
                    "refine_mag_radius": args.refine_mag_radius,
                    "refine_huber_delta": args.refine_huber_delta,
                    "gs_pre_steps": args.gs_pre_steps,
                    "gs_post_steps": args.gs_post_steps,
                    "gs_radius": args.gs_radius,
                }
            )

            for image_name in image_names:
                img_path = base.resolve_image_path(args.data_root, image_name)
                x_gt = load_image(img_path, size=image_size, device=device)
                mag_clean = base.oversampled_magnitude(x_gt, pad)
                mag_target = mag_clean
                image_index = original_index[image_name]
                if float(noise_std) > 0:
                    gen = torch.Generator(device=device).manual_seed(args.measurement_noise_seed + image_index)
                    noise = torch.randn(mag_clean.shape, device=device, dtype=mag_clean.dtype, generator=gen)
                    mag_target = mag_clean + float(noise_std) * noise
                    if args.clip_noisy_magnitude:
                        mag_target = mag_target.clamp_min(0.0)

                image_rows_this_condition: List[Dict[str, object]] = []
                for cfg in configs:
                    for seed in seeds:
                        t0 = time.perf_counter()
                        x_np, selector_stats = selector.reconstruct_with_selector_stat(
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
                        np_runtime_s = time.perf_counter() - t0

                        candidate_versions: List[Tuple[str, torch.Tensor, Dict[str, object], float]] = []
                        candidate_versions.append(("none", x_np, {}, 0.0))

                        x_ref_start = x_np
                        if int(args.gs_pre_steps) > 0:
                            t_gs0 = time.perf_counter()
                            x_ref_start = gs_magnitude_refine(
                                x_ref_start,
                                mag_target,
                                pad=pad,
                                radius=float(args.gs_radius),
                                steps=int(args.gs_pre_steps),
                            )
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            candidate_versions.append(
                                (f"gs_pre{args.gs_pre_steps}_r{args.gs_radius:g}", x_ref_start, {}, time.perf_counter() - t_gs0)
                            )

                        for aw in anchor_weights:
                            if int(args.refine_steps) <= 0:
                                continue
                            t_ref0 = time.perf_counter()
                            x_ref, ref_stats = adam_anchor_refine(
                                x_ref_start,
                                mag_target,
                                pad=pad,
                                steps=int(args.refine_steps),
                                lr=float(args.refine_lr),
                                mag_radius=refine_mag_radius,
                                anchor_weight=float(aw),
                                tv_weight=float(args.refine_tv_weight),
                                huber_delta=huber_delta,
                                clamp_each_step=not bool(args.no_clamp_each_refine_step),
                                log_every=args.log_every,
                            )
                            if int(args.gs_post_steps) > 0:
                                x_ref = gs_magnitude_refine(
                                    x_ref,
                                    mag_target,
                                    pad=pad,
                                    radius=float(args.gs_radius),
                                    steps=int(args.gs_post_steps),
                                )
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            ref_runtime_s = time.perf_counter() - t_ref0
                            tag = f"adam{args.refine_steps}_lr{args.refine_lr:g}_anchor{aw:g}"
                            if args.refine_mag_radius != "full":
                                tag += f"_mr{args.refine_mag_radius}"
                            if int(args.gs_pre_steps) > 0:
                                tag = f"gs_pre{args.gs_pre_steps}_" + tag
                            if int(args.gs_post_steps) > 0:
                                tag += f"_gs_post{args.gs_post_steps}"
                            candidate_versions.append((tag, x_ref, ref_stats, ref_runtime_s))

                        for refine_tag, x_rec, ref_stats, ref_runtime_s in candidate_versions:
                            eval_rows = eval_alignment_rows(
                                x_rec=x_rec,
                                x_gt=x_gt,
                                mag_clean=mag_clean,
                                mag_target=mag_target,
                                pad=pad,
                                score_radius=float(args.score_radius),
                                lpips_model=lpips_model,
                                alignment_modes=alignment_modes,
                            )
                            for erow in eval_rows:
                                row = {
                                    "timestamp": stamp,
                                    "image_basename": image_name,
                                    "image_index_in_split": image_index,
                                    "seed": seed,
                                    "config_tag": cfg["config_tag"],
                                    "score_mode": cfg["score_mode"],
                                    "score_reg_lambda": cfg["score_reg_lambda"],
                                    "score_reg_lambda_schedule": cfg["score_reg_lambda_schedule"],
                                    "variant": variant.name,
                                    "refine_tag": refine_tag,
                                    "np_runtime_s": np_runtime_s,
                                    "refine_runtime_s": ref_runtime_s,
                                    "runtime_s": np_runtime_s + ref_runtime_s,
                                    "num_steps": args.np_steps,
                                    "proj_start": args.late_start,
                                    "num_candidates_soft": args.soft_candidates,
                                    "num_candidates_hard": args.hard_candidates,
                                    "score_radius": args.score_radius,
                                    "proj_radius": args.proj_radius,
                                    "oversample": oversample,
                                    "measurement_noise_std": noise_std,
                                    **selector_stats,
                                    **ref_stats,
                                    **erow,
                                }
                                run_rows.append(row)
                                image_rows_this_condition.append(row)

                            print(
                                f"[np+refine] image={image_name} cfg={cfg['config_tag']} seed={seed} "
                                f"refine={refine_tag} best_align_psnr={max(float(r['psnr']) for r in eval_rows):.2f} "
                                f"np_time={np_runtime_s:.1f}s ref_time={ref_runtime_s:.1f}s",
                                flush=True,
                            )

                for refine_tag in sorted({str(r["refine_tag"]) for r in image_rows_this_condition}):
                    rows = [r for r in image_rows_this_condition if str(r["refine_tag"]) == refine_tag]
                    add_selected_views(
                        image_rows=rows,
                        alignment_modes=alignment_modes,
                        selected_rows=selected_rows,
                    )

    write_csv(os.path.join(run_root, "configs.csv"), config_rows)
    write_csv(os.path.join(run_root, "run_level.csv"), run_rows)
    write_csv(os.path.join(run_root, "selected_image_level.csv"), selected_rows)
    summary = summarize_selected(selected_rows)
    write_csv(os.path.join(run_root, "selected_summary.csv"), summary)

    print(f"Saved: {run_root}")
    print("Selected summary:")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
