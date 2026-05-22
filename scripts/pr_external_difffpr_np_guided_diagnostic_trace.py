#!/usr/bin/env python3
"""Diagnostic NP trace runner for FFHQ phase retrieval.

This script records per-step candidate information for selector diagnostics.  It is
not meant to replace the normal benchmark runner.  The main outputs are:

- candidate_trace.csv.gz: one row per candidate per reconstruction step.
- step_trace.csv.gz: one row per reconstruction step with winner/margin info.
- final_metrics.csv: final reconstruction metrics for each run/alignment.
- configs.csv: run configuration.

The candidate trace intentionally stores candidate metadata and scalar scores, not
full candidate images/tensors.  Storing every candidate image at every step would
be prohibitively large for FFHQ-25.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO, Tuple

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


class StreamingCsvWriter:
    def __init__(self, path: str, fieldnames: List[str]):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if path.endswith(".gz"):
            self.f: TextIO = gzip.open(path, "wt", newline="", encoding="utf-8")
        else:
            self.f = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames, extrasaction="ignore")
        self.writer.writeheader()

    def writerow(self, row: Dict[str, object]) -> None:
        self.writer.writerow(row)

    def close(self) -> None:
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def finite_float(x: torch.Tensor | float | int) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def normalize_scores(values: List[torch.Tensor]) -> torch.Tensor:
    v = torch.stack([x.float().reshape(()) for x in values])
    scale = v.detach().median().abs().clamp_min(1e-8)
    return v / scale


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


def filter_images(image_names: List[str], select_images: str) -> List[str]:
    text = str(select_images or "").strip()
    if not text:
        return image_names
    wanted_raw = [tok.strip() for tok in text.split(",") if tok.strip()]
    wanted = set(wanted_raw)
    wanted_noext = {Path(x).stem for x in wanted_raw}
    out = []
    for name in image_names:
        p = Path(name)
        if name in wanted or p.name in wanted or p.stem in wanted_noext:
            out.append(name)
    if not out:
        raise ValueError(f"select_images={select_images!r} matched no images")
    return out


@torch.no_grad()
def magnitude_scores(
    x: torch.Tensor,
    mag_target: torch.Tensor,
    pad: int,
    score_radius: float,
) -> Dict[str, float]:
    """Return full and low-frequency magnitude residual scores vs observation."""
    mag = base.oversampled_magnitude(x, pad)
    resid = mag - mag_target
    full_mse = torch.mean(resid.square())
    full_l2 = torch.norm(resid)
    _, _, h, w = mag.shape
    mask_hw = base.centered_lowfreq_mask(h, w, score_radius, mag.device)
    mask = mask_hw[None, None, :, :].expand_as(mag)
    lf_resid = resid[mask]
    lf_mse = torch.mean(lf_resid.square())
    lf_l2 = torch.norm(lf_resid)
    return {
        "full_l2": finite_float(full_l2),
        "full_mse": finite_float(full_mse),
        "lf_l2": finite_float(lf_l2),
        "lf_mse": finite_float(lf_mse),
    }


def build_candidate_eps(
    *,
    x0_target: torch.Tensor,
    eps_prev: Optional[torch.Tensor],
    eps_memory: List[torch.Tensor],
    num_candidates: int,
    noise_memory_k: int,
) -> List[Tuple[str, torch.Tensor]]:
    out: List[Tuple[str, torch.Tensor]] = []
    if noise_memory_k > 0:
        for eps in eps_memory:
            out.append(("memory", eps))
            if len(out) >= min(int(noise_memory_k), int(num_candidates)):
                break
    elif eps_prev is not None and num_candidates > 1:
        out.append(("legacy_eps_prev", eps_prev))

    while len(out) < num_candidates:
        out.append(("fresh", torch.randn_like(x0_target)))
    return out


@torch.no_grad()
def trace_reconstruct_one(
    *,
    run_id: str,
    tag: str,
    image_name: str,
    seed: int,
    x_gt: torch.Tensor,
    mag_target: torch.Tensor,
    mag_clean: torch.Tensor,
    pad: int,
    unet,
    scheduler,
    device: torch.device,
    variant,
    num_steps: int,
    score_radius: float,
    proj_radius: float,
    proj_radius_schedule: str | None,
    score_mode: str,
    score_reg_lambda: float,
    score_reg_lambda_schedule: str,
    score_huber_delta: float,
    adaptive_s2_margin: float,
    noise_memory_k: int,
    log_every: int,
    candidate_writer: StreamingCsvWriter,
    step_writer: StreamingCsvWriter,
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
        pre_projection_stage = i < int(variant.proj_start)

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
        current_proj_radius = math.nan
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

        candidate_eps = build_candidate_eps(
            x0_target=x0_hat,
            eps_prev=eps_prev,
            eps_memory=eps_memory,
            num_candidates=k,
            noise_memory_k=noise_memory_k,
        )

        candidate_x0s: List[torch.Tensor] = []
        candidate_eps_out: List[torch.Tensor] = []
        candidate_sources: List[str] = []
        lf_l2s: List[torch.Tensor] = []
        lf_mses: List[float] = []
        full_l2s: List[float] = []
        full_mses: List[float] = []
        prev_regs: List[torch.Tensor] = []
        huber_scores: List[torch.Tensor] = []

        for cand_idx, (source, eps_cand) in enumerate(candidate_eps):
            x_t_cand = sqrt_at * x0_hat + sqrt_1mat * eps_cand
            t_tensor = torch.tensor([t_next_int], device=device, dtype=torch.long)
            eps_pred = unet(x_t_cand, t_tensor).sample
            x0_cand = (x_t_cand - sqrt_1mat * eps_pred) / sqrt_at

            scores = magnitude_scores(x0_cand, mag_target, pad, score_radius)
            prev_reg = torch.sqrt(
                torch.mean((base.to01(x0_cand) - base.to01(x0_hat)) ** 2).clamp_min(1e-12)
            )
            huber_score = base.oversampled_lowfreq_mag_huber(
                x0_cand, mag_target, pad, score_radius, delta=score_huber_delta
            )

            candidate_x0s.append(x0_cand)
            candidate_eps_out.append(eps_cand.detach().clone())
            candidate_sources.append(source)
            lf_l2s.append(torch.tensor(scores["lf_l2"], device=device, dtype=x0_hat.dtype))
            lf_mses.append(scores["lf_mse"])
            full_l2s.append(scores["full_l2"])
            full_mses.append(scores["full_mse"])
            prev_regs.append(prev_reg)
            huber_scores.append(huber_score)

        mode = str(score_mode).lower()
        lf_norm = normalize_scores(lf_l2s)
        prev_norm = normalize_scores(prev_regs)
        adaptive_used = False
        if mode in {"lf", "s1"}:
            final_scores = torch.stack([x.float().reshape(()) for x in lf_l2s])
        elif mode in {"prev_l2", "s2"}:
            final_scores = lf_norm + float(lambda_eff) * prev_norm
        elif mode in {"adaptive_prev_l2", "adaptive_s2", "s2_adaptive"}:
            if k >= 2:
                lf_sorted_norm, _ = torch.sort(lf_norm)
                lf_norm_margin = lf_sorted_norm[1] - lf_sorted_norm[0]
            else:
                lf_norm_margin = torch.tensor(float("inf"), device=device)
            adaptive_used = finite_float(lf_norm_margin) <= float(adaptive_s2_margin)
            final_scores = lf_norm + float(lambda_eff) * prev_norm if adaptive_used else lf_norm
        elif mode in {"consensus_l2", "s3"}:
            mean_x0 = torch.stack(candidate_x0s, dim=0).mean(dim=0)
            consensus_regs = [
                torch.sqrt(torch.mean((base.to01(x0_cand) - base.to01(mean_x0)) ** 2).clamp_min(1e-12))
                for x0_cand in candidate_x0s
            ]
            final_scores = lf_norm + float(lambda_eff) * normalize_scores(consensus_regs)
        elif mode in {"huber_lf", "s4"}:
            final_scores = torch.stack([x.float().reshape(()) for x in huber_scores])
        else:
            raise ValueError(f"Unknown score_mode={score_mode!r}")

        winner_idx = int(torch.argmin(final_scores).item())
        lf_order = sorted(range(k), key=lambda j: lf_l2s[j].detach().cpu().item())
        final_order = sorted(range(k), key=lambda j: final_scores[j].detach().cpu().item())
        lf_best_idx = int(lf_order[0])
        lf_second_idx = int(lf_order[1]) if k >= 2 else -1
        final_second_idx = int(final_order[1]) if k >= 2 else -1

        lf_l2_margin = (
            finite_float(lf_l2s[lf_second_idx] - lf_l2s[lf_best_idx]) if k >= 2 else math.nan
        )
        lf_mse_margin = lf_mses[lf_second_idx] - lf_mses[lf_best_idx] if k >= 2 else math.nan
        final_score_margin = (
            finite_float(final_scores[final_second_idx] - final_scores[winner_idx]) if k >= 2 else math.nan
        )

        winner_lf_l2 = finite_float(lf_l2s[winner_idx])
        winner_lf_mse = lf_mses[winner_idx]
        winner_full_l2 = full_l2s[winner_idx]
        winner_full_mse = full_mses[winner_idx]
        winner_prev_l2 = finite_float(prev_regs[winner_idx])
        winner_final_score = finite_float(final_scores[winner_idx])

        for cand_idx in range(k):
            candidate_writer.writerow(
                {
                    "run_id": run_id,
                    "config_tag": tag,
                    "image_basename": image_name,
                    "seed": seed,
                    "step_index": i,
                    "t_int": t_int,
                    "t_next_int": t_next_int,
                    "pre_projection_stage": int(pre_projection_stage),
                    "projection_applied": int(projection_applied),
                    "proj_radius_active": current_proj_radius,
                    "num_candidates": k,
                    "candidate_idx": cand_idx,
                    "candidate_source": candidate_sources[cand_idx],
                    "is_lf_best": int(cand_idx == lf_best_idx),
                    "is_winner": int(cand_idx == winner_idx),
                    "winner_idx": winner_idx,
                    "lf_l2_vs_observation": finite_float(lf_l2s[cand_idx]),
                    "lf_mse_vs_observation": lf_mses[cand_idx],
                    "full_l2_vs_observation": full_l2s[cand_idx],
                    "full_mse_vs_observation": full_mses[cand_idx],
                    "prev_l2": finite_float(prev_regs[cand_idx]),
                    "huber_lf_score": finite_float(huber_scores[cand_idx]),
                    "final_selection_score": finite_float(final_scores[cand_idx]),
                    "score_mode": score_mode,
                    "lambda_eff": lambda_eff,
                    "score_reg_lambda_schedule": score_reg_lambda_schedule,
                    "adaptive_s2_margin": adaptive_s2_margin,
                    "adaptive_used": int(adaptive_used),
                    "noise_memory_k": int(noise_memory_k),
                    "lf_best_idx": lf_best_idx,
                    "lf_second_idx": lf_second_idx,
                    "lf_l2_margin_best_to_second": lf_l2_margin,
                    "lf_mse_margin_best_to_second": lf_mse_margin,
                    "winner_lf_mse_vs_observation": winner_lf_mse,
                    "winner_full_mse_vs_observation": winner_full_mse,
                }
            )

        step_writer.writerow(
            {
                "run_id": run_id,
                "config_tag": tag,
                "image_basename": image_name,
                "seed": seed,
                "step_index": i,
                "t_int": t_int,
                "t_next_int": t_next_int,
                "pre_projection_stage": int(pre_projection_stage),
                "projection_applied": int(projection_applied),
                "proj_radius_active": current_proj_radius,
                "num_candidates": k,
                "score_mode": score_mode,
                "lambda_eff": lambda_eff,
                "score_reg_lambda_schedule": score_reg_lambda_schedule,
                "adaptive_s2_margin": adaptive_s2_margin,
                "adaptive_used": int(adaptive_used),
                "noise_memory_k": int(noise_memory_k),
                "winner_idx": winner_idx,
                "winner_source": candidate_sources[winner_idx],
                "winner_is_lf_best": int(winner_idx == lf_best_idx),
                "lf_best_idx": lf_best_idx,
                "lf_second_idx": lf_second_idx,
                "lf_l2_margin_best_to_second": lf_l2_margin,
                "lf_mse_margin_best_to_second": lf_mse_margin,
                "final_score_margin_best_to_second": final_score_margin,
                "winner_lf_l2_vs_observation": winner_lf_l2,
                "winner_lf_mse_vs_observation": winner_lf_mse,
                "winner_full_l2_vs_observation": winner_full_l2,
                "winner_full_mse_vs_observation": winner_full_mse,
                "winner_prev_l2": winner_prev_l2,
                "winner_final_selection_score": winner_final_score,
            }
        )

        x_prev = candidate_x0s[winner_idx]
        eps_prev = candidate_eps_out[winner_idx]
        if noise_memory_k > 0:
            eps_memory = [eps_prev.detach().clone()] + [
                eps.detach().clone() for eps in eps_memory[: max(0, int(noise_memory_k) - 1)]
            ]

        if log_every > 0 and (i + 1) % log_every == 0:
            print(
                f"[{tag}] {image_name} seed={seed} step {i+1}/{len(timesteps)-1} "
                f"winner={winner_idx} lf_mse={winner_lf_mse:.4e} "
                f"lf_margin={lf_mse_margin:.4e} adaptive_used={int(adaptive_used)}"
            )

    assert x_prev is not None
    return x_prev.detach()


CANDIDATE_FIELDS = [
    "run_id", "config_tag", "image_basename", "seed", "step_index", "t_int", "t_next_int",
    "pre_projection_stage", "projection_applied", "proj_radius_active", "num_candidates",
    "candidate_idx", "candidate_source", "is_lf_best", "is_winner", "winner_idx",
    "lf_l2_vs_observation", "lf_mse_vs_observation", "full_l2_vs_observation", "full_mse_vs_observation",
    "prev_l2", "huber_lf_score", "final_selection_score", "score_mode", "lambda_eff",
    "score_reg_lambda_schedule", "adaptive_s2_margin", "adaptive_used", "noise_memory_k",
    "lf_best_idx", "lf_second_idx", "lf_l2_margin_best_to_second", "lf_mse_margin_best_to_second",
    "winner_lf_mse_vs_observation", "winner_full_mse_vs_observation",
]

STEP_FIELDS = [
    "run_id", "config_tag", "image_basename", "seed", "step_index", "t_int", "t_next_int",
    "pre_projection_stage", "projection_applied", "proj_radius_active", "num_candidates",
    "score_mode", "lambda_eff", "score_reg_lambda_schedule", "adaptive_s2_margin", "adaptive_used",
    "noise_memory_k", "winner_idx", "winner_source", "winner_is_lf_best", "lf_best_idx", "lf_second_idx",
    "lf_l2_margin_best_to_second", "lf_mse_margin_best_to_second", "final_score_margin_best_to_second",
    "winner_lf_l2_vs_observation", "winner_lf_mse_vs_observation",
    "winner_full_l2_vs_observation", "winner_full_mse_vs_observation", "winner_prev_l2",
    "winner_final_selection_score",
]


def main() -> None:
    p = argparse.ArgumentParser(description="Trace NP candidates/winners for selector diagnostics.")
    p.add_argument("--data_root", required=True)
    p.add_argument("--image_list_file", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--tag", default="diagnostic")
    p.add_argument("--select_images", default="")
    p.add_argument("--guided_model_path", required=True)
    p.add_argument("--guided_diffusion_dir", default=None)
    p.add_argument("--guided_preset", default="difffpr_ffhq_10m")
    p.add_argument("--guided_strict", action="store_true")
    p.add_argument("--variants", default="np_canonical")
    p.add_argument("--seeds", default="100,101")
    p.add_argument("--np_steps", type=int, default=1000)
    p.add_argument("--late_start", type=int, default=300)
    p.add_argument("--fixed_k", type=int, default=5)
    p.add_argument("--score_radius", type=float, default=0.6)
    p.add_argument("--proj_radius", type=float, default=0.2)
    p.add_argument("--proj_radius_schedule", default=None)
    p.add_argument("--soft_candidates", type=int, default=5)
    p.add_argument("--hard_candidates", type=int, default=1)
    p.add_argument("--oversample_values", default="2")
    p.add_argument("--measurement_noise_values", default="0.05")
    p.add_argument("--measurement_noise_seed", type=int, default=20260423)
    p.add_argument("--clip_noisy_magnitude", action="store_true")
    p.add_argument("--alignments", default="raw,rot180,resolve")
    p.add_argument(
        "--score_mode",
        default="lf",
        choices=[
            "lf", "prev_l2", "adaptive_prev_l2", "consensus_l2", "huber_lf",
            "s1", "s2", "adaptive_s2", "s2_adaptive", "s3", "s4",
        ],
    )
    p.add_argument("--score_reg_lambda", type=float, default=0.0)
    p.add_argument(
        "--score_reg_lambda_schedule",
        default="constant",
        choices=["constant", "linear_decay_to_proj_start", "pre_projection_only"],
    )
    p.add_argument("--adaptive_s2_margin", type=float, default=0.05)
    p.add_argument("--score_huber_delta", type=float, default=0.05)
    p.add_argument("--noise_memory_k", type=int, default=0)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--skip_lpips", action="store_true")
    args = p.parse_args()

    image_names = base.collect_images(args.data_root, args.image_list_file)
    image_names = filter_images(image_names, args.select_images)
    seeds = base.parse_int_list(args.seeds)
    oversample_values = [float(x.strip()) for x in args.oversample_values.split(",") if x.strip()]
    noise_values = [float(x.strip()) for x in args.measurement_noise_values.split(",") if x.strip()]
    alignment_modes = [x.strip() for x in args.alignments.split(",") if x.strip()]

    variant_names = [tok.strip() for tok in args.variants.split(",") if tok.strip()]
    variants = base.make_variants(variant_names, late_start=args.late_start, fixed_k=args.fixed_k)
    remapped = []
    for v in variants:
        remapped.append(
            base.NPVariant(
                name=f"{v.name}_soft{args.soft_candidates}_hard{args.hard_candidates}",
                soft=int(args.soft_candidates),
                hard=int(args.hard_candidates),
                proj_start=v.proj_start,
                use_lowfreq_score=v.use_lowfreq_score,
                use_lowfreq_projection=v.use_lowfreq_projection,
            )
        )
    variants = remapped

    os.makedirs(args.outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.outdir, f"diagnostic_{args.tag}_{stamp}")
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
    final_rows: List[Dict[str, object]] = []

    candidate_path = os.path.join(run_root, "candidate_trace.csv.gz")
    step_path = os.path.join(run_root, "step_trace.csv.gz")
    with StreamingCsvWriter(candidate_path, CANDIDATE_FIELDS) as cand_writer, StreamingCsvWriter(step_path, STEP_FIELDS) as step_writer:
        for variant in variants:
            for oversample in oversample_values:
                pad = base.oversample_pad(image_size, oversample)
                for noise_std in noise_values:
                    config_rows.append(
                        {
                            "tag": args.tag,
                            "variant": variant.name,
                            "backend": "guided_diffusion_diagnostic_trace",
                            "guided_preset": args.guided_preset,
                            "guided_model_path": str(Path(args.guided_model_path).expanduser()),
                            "guided_diffusion_dir": args.guided_diffusion_dir or "PYTHONPATH/default",
                            "selected_images": ",".join(image_names),
                            "num_steps": args.np_steps,
                            "score_radius": args.score_radius,
                            "proj_radius": args.proj_radius,
                            "proj_radius_schedule": args.proj_radius_schedule or f"{variant.proj_start}:{args.proj_radius}",
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
                            "lpips_backend": lpips_status,
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
                            run_id = f"{args.tag}__{Path(image_name).stem}__seed{seed}"
                            t0 = time.perf_counter()
                            x_rec = trace_reconstruct_one(
                                run_id=run_id,
                                tag=args.tag,
                                image_name=image_name,
                                seed=seed,
                                x_gt=x_gt,
                                mag_target=mag_target,
                                mag_clean=mag_clean,
                                pad=pad,
                                unet=bundle.unet,
                                scheduler=bundle.scheduler,
                                device=device,
                                variant=variant,
                                num_steps=args.np_steps,
                                score_radius=args.score_radius,
                                proj_radius=args.proj_radius,
                                proj_radius_schedule=args.proj_radius_schedule,
                                score_mode=args.score_mode,
                                score_reg_lambda=args.score_reg_lambda,
                                score_reg_lambda_schedule=args.score_reg_lambda_schedule,
                                score_huber_delta=args.score_huber_delta,
                                adaptive_s2_margin=args.adaptive_s2_margin,
                                noise_memory_k=int(args.noise_memory_k),
                                log_every=args.log_every,
                                candidate_writer=cand_writer,
                                step_writer=step_writer,
                            )
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            runtime_s = time.perf_counter() - t0

                            for alignment in alignment_modes:
                                x_eval = guided.make_alignment(alignment, x_rec, x_gt)
                                final_rows.append(
                                    {
                                        "run_id": run_id,
                                        "config_tag": args.tag,
                                        "variant": variant.name,
                                        "alignment_mode": alignment,
                                        "image_basename": image_name,
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
                                        "proj_start": variant.proj_start,
                                        "num_candidates_soft": variant.soft,
                                        "num_candidates_hard": variant.hard,
                                        "score_radius": args.score_radius,
                                        "proj_radius": args.proj_radius,
                                        "score_mode": args.score_mode,
                                        "score_reg_lambda": args.score_reg_lambda,
                                        "score_reg_lambda_schedule": args.score_reg_lambda_schedule,
                                        "adaptive_s2_margin": args.adaptive_s2_margin,
                                        "noise_memory_k": int(args.noise_memory_k),
                                        "oversample": oversample,
                                        "measurement_noise_std": noise_std,
                                    }
                                )
                            print(
                                f"[diagnostic] {args.tag} {image_name} seed={seed} "
                                f"done in {runtime_s:.1f}s"
                            )

    write_csv(os.path.join(run_root, "configs.csv"), config_rows)
    write_csv(os.path.join(run_root, "final_metrics.csv"), final_rows)
    print(f"Saved diagnostic run: {run_root}")
    print(f"  - {candidate_path}")
    print(f"  - {step_path}")
    print(f"  - {os.path.join(run_root, 'final_metrics.csv')}")
    print(f"  - {os.path.join(run_root, 'configs.csv')}")


if __name__ == "__main__":
    main()
