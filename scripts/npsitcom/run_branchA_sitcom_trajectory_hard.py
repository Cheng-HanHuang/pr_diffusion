#!/usr/bin/env python3
"""Small hard-image SITCOM trajectory instrumentation pass.

The runner uses the external SITCOM_ODE implementation without editing it.  It
records DAPS trajectories, then derives per-step risk features from `x0hat`,
`x0y`, and `xt` in SITCOM's own measurement/operator space.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def finite(xs: Iterable[float]) -> List[float]:
    return [x for x in xs if math.isfinite(x)]


def fmean(xs: Iterable[float]) -> float:
    vals = finite(xs)
    return mean(vals) if vals else math.nan


def psnr01_from_model_range(x: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    x01 = (x.clamp(-1, 1) + 1.0) * 0.5
    gt01 = (gt.clamp(-1, 1) + 1.0) * 0.5
    mse = torch.mean((x01 - gt01) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def centered_lowfreq_mask(h: int, w: int, radius: float, device: torch.device) -> torch.Tensor:
    fy = torch.fft.fftshift(torch.fft.fftfreq(h, d=1.0, device=device))
    fx = torch.fft.fftshift(torch.fft.fftfreq(w, d=1.0, device=device))
    ky, kx = torch.meshgrid(fy, fx, indexing="ij")
    return torch.sqrt(ky**2 + kx**2) <= float(radius)


def batch_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x.detach().float().square(), dim=tuple(range(1, x.ndim))).clamp_min(1e-24))


def residual_features(operator, x: torch.Tensor, y: torch.Tensor, radius: float, prefix: str) -> Dict[str, torch.Tensor]:
    pred = operator(x)
    resid = pred - y
    full = torch.linalg.norm(resid.flatten(1), dim=1)
    _, _, h, w = resid.shape
    mask_hw = centered_lowfreq_mask(h, w, radius, resid.device)
    mask = mask_hw[None, None, :, :].expand_as(resid)
    lf = torch.linalg.norm(resid[mask].reshape(resid.shape[0], -1), dim=1)
    y_full = torch.linalg.norm(y.flatten(1), dim=1).clamp_min(1e-12)
    y_lf = torch.linalg.norm(y[mask].reshape(y.shape[0], -1), dim=1).clamp_min(1e-12)
    return {
        f"{prefix}_full_residual": full,
        f"{prefix}_lowfreq_residual": lf,
        f"{prefix}_full_residual_normed": full / y_full,
        f"{prefix}_lowfreq_residual_normed": lf / y_lf,
    }


def image_to_tensor(path: Path, resolution: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((resolution, resolution))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    x = torch.from_numpy(arr)[None].to(device)
    return x * 2.0 - 1.0


def tensor_to_png(x: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = ((x.detach().cpu().clamp(-1, 1) * 0.5 + 0.5)[0].permute(1, 2, 0).numpy() * 255.0)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def build_hard_image_folder(data_root: Path, image_ids: List[str], outdir: Path) -> List[Dict[str, str]]:
    outdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, str]] = []
    for i, image_id in enumerate(image_ids):
        src = data_root / "00000" / f"{image_id}.png"
        if not src.exists():
            matches = sorted(data_root.rglob(f"{image_id}.png"))
            if not matches:
                raise FileNotFoundError(f"Could not find FFHQ image {image_id}.png under {data_root}")
            src = matches[0]
        dst = outdir / f"{i:05d}_{image_id}.png"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
        rows.append(
            {
                "hard_index": f"{i:05d}",
                "image_id": image_id,
                "source_path": str(src),
                "sitcom_path": str(dst),
            }
        )
    write_csv(outdir / "manifest.csv", rows)
    return rows


def load_external_sitcom(sitcom_root: Path):
    sys.path.insert(0, str(sitcom_root))
    old_cwd = Path.cwd()
    os.chdir(sitcom_root)
    from data import get_dataset
    from eval import Evaluator, get_eval_fn
    from forward_operator import get_operator
    from model import get_model
    from sampler import get_sampler

    return old_cwd, get_dataset, get_eval_fn, Evaluator, get_operator, get_model, get_sampler


def summarize_runs(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, int], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["image_id"]), int(row["run_index"])), []).append(row)
    out: List[Dict[str, object]] = []
    for (image_id, run_i), rs in sorted(grouped.items()):
        final_psnr = float(rs[-1]["final_psnr"])
        early = [r for r in rs if int(r["step"]) < 50]
        out.append(
            {
                "image_id": image_id,
                "run_index": run_i,
                "final_psnr": final_psnr,
                "final_bad_below25": final_psnr < 25.0,
                "final_bad_below20": final_psnr < 20.0,
                "max_x0y_full_residual_normed_early50": max(float(r["x0y_full_residual_normed"]) for r in early),
                "max_x0y_lowfreq_residual_normed_early50": max(float(r["x0y_lowfreq_residual_normed"]) for r in early),
                "max_x0hat_x0y_disagreement_early50": max(float(r["x0hat_x0y_disagreement"]) for r in early),
                "max_xt_step_jump_early50": max(float(r["xt_step_jump"]) for r in early),
                "max_correction_norm_early50": max(float(r["correction_norm"]) for r in early),
                "max_x0y_full_residual_normed_all": max(float(r["x0y_full_residual_normed"]) for r in rs),
                "max_x0hat_x0y_disagreement_all": max(float(r["x0hat_x0y_disagreement"]) for r in rs),
            }
        )
    return out


def render_summary(outdir: Path, run_summary: List[Dict[str, object]], step_rows: List[Dict[str, object]], args: argparse.Namespace) -> None:
    bad = [r for r in run_summary if bool(r["final_bad_below25"])]
    good = [r for r in run_summary if not bool(r["final_bad_below25"])]
    report = [
        "# A5 SITCOM Hard-Image Trajectory Pass",
        "",
        "## Setup",
        "",
        f"- Noise: `{args.noise}`",
        f"- Images: `{','.join(args.image_ids)}`",
        f"- SITCOM runs per image: `{args.num_runs}`",
        f"- Output folder: `{outdir}`",
        f"- GPU: `{args.gpu}`",
        f"- Per-step rows: `{len(step_rows)}`",
        "",
        "## Final Outcomes",
        "",
        f"- Total image-runs: `{len(run_summary)}`",
        f"- Final PSNR below 25 dB: `{len(bad)}`",
        f"- Final PSNR below 20 dB: `{sum(1 for r in run_summary if bool(r['final_bad_below20']))}`",
        f"- Mean final PSNR: `{fmean(float(r['final_psnr']) for r in run_summary):.3f}`",
        "",
        "## Early-Spike Screen",
        "",
        f"- Bad-run early max x0y full residual normed mean: `{fmean(float(r['max_x0y_full_residual_normed_early50']) for r in bad):.6g}`",
        f"- Good-run early max x0y full residual normed mean: `{fmean(float(r['max_x0y_full_residual_normed_early50']) for r in good):.6g}`",
        f"- Bad-run early max disagreement mean: `{fmean(float(r['max_x0hat_x0y_disagreement_early50']) for r in bad):.6g}`",
        f"- Good-run early max disagreement mean: `{fmean(float(r['max_x0hat_x0y_disagreement_early50']) for r in good):.6g}`",
        "",
        "## Interpretation",
        "",
        "Use `run_level_summary.csv` to compare early max residual, correction, and disagreement features between bad and good final runs. A useful risk detector should show separable early feature ranges before final collapse.",
        "",
        "Artifacts:",
        "",
        "- `trajectory_step_metrics.csv`: one row per run/image/annealing step.",
        "- `run_level_summary.csv`: early/all-step maxima plus final PSNR.",
        "- `raw_traces/`: compact tensors per run containing `xt`, `x0hat`, `x0y`, and `sigma`.",
        "- `samples/`: final PNG samples.",
    ]
    write_text(outdir / "SUMMARY.md", "\n".join(report) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sitcom_root", default="/egr/research-pac/huang248/external/SITCOM_ODE")
    ap.add_argument("--data_root", default="/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024")
    ap.add_argument("--image_ids", default="00005,00013,00028,00027,00034")
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--num_runs", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--oversample", type=float, default=2.0)
    ap.add_argument("--score_radius", type=float, default=0.6)
    ap.add_argument("--anneal_steps", type=int, default=200)
    ap.add_argument("--diff_steps", type=int, default=5)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    image_ids = [x.strip() for x in args.image_ids.split(",") if x.strip()]
    args.image_ids = image_ids

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(f"cuda:{args.gpu}")
    device = torch.device(f"cuda:{args.gpu}")

    image_root = outdir / "sitcom_images_hard5"
    manifest = build_hard_image_folder(Path(args.data_root), image_ids, image_root)

    old_cwd, get_dataset, get_eval_fn, Evaluator, get_operator, get_model, get_sampler = load_external_sitcom(Path(args.sitcom_root))
    try:
        data = get_dataset(
            name="image",
            root=str(image_root),
            resolution=args.resolution,
            device="cuda",
            start_id=0,
            end_id=len(image_ids),
        )
        images = data.get_data(len(image_ids), 0)
        operator = get_operator(name="phase_retrieval", sigma=args.noise, oversample=args.oversample)
        y = operator.measure(images)
        sampler = get_sampler(
            latent=False,
            annealing_scheduler_config={
                "num_steps": args.anneal_steps,
                "schedule": "linear",
                "sigma_max": 100,
                "sigma_min": 0.1,
                "sigma_final": 0,
                "timestep": "poly-7",
            },
            diffusion_scheduler_config={
                "num_steps": args.diff_steps,
                "schedule": "linear",
                "sigma_min": 0.01,
                "sigma_final": 0,
                "timestep": "poly-7",
            },
            lgvd_config={"lr": 5.0e-5, "lr_min_ratio": 0.01, "num_steps": 100, "tau": 0.01},
        )
        model = get_model(
            name="ddpm",
            model_config={
                "attention_resolutions": 16,
                "channel_mult": "",
                "class_cond": False,
                "dropout": 0.0,
                "image_size": 256,
                "learn_sigma": True,
                "model_path": "checkpoint/ffhq256.pt",
                "num_channels": 128,
                "num_head_channels": 64,
                "num_heads": 4,
                "num_heads_upsample": -1,
                "num_res_blocks": 1,
                "resblock_updown": True,
                "use_checkpoint": False,
                "use_fp16": False,
                "use_new_attention_order": False,
                "use_scale_shift_norm": True,
            },
        )
        evaluator = Evaluator([get_eval_fn("psnr")])

        step_rows: List[Dict[str, object]] = []
        sample_dir = outdir / "samples"
        raw_dir = outdir / "raw_traces"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for run_i in range(args.num_runs):
            print(f"[trajectory] run {run_i}/{args.num_runs - 1}")
            x_start = sampler.get_start(images)
            samples = sampler.sample(
                model,
                x_start,
                operator,
                y,
                evaluator=evaluator,
                verbose=True,
                record=True,
                gt=images,
            )
            traj = sampler.trajectory.compile()
            final_psnr = psnr01_from_model_range(samples, images)
            torch.save(
                {
                    "tensor_data": traj.tensor_data,
                    "value_data": traj.value_data,
                    "final_psnr": final_psnr.detach().cpu(),
                    "manifest": manifest,
                },
                raw_dir / f"trajectory_run{run_i:04d}.pt",
            )
            for image_i, meta in enumerate(manifest):
                tensor_to_png(samples[image_i : image_i + 1], sample_dir / f"{meta['hard_index']}_{meta['image_id']}_run{run_i:04d}.png")

            xt = traj.tensor_data["xt"].to(device)
            x0hat = traj.tensor_data["x0hat"].to(device)
            x0y = traj.tensor_data["x0y"].to(device)
            sigmas = traj.value_data["sigma"].detach().cpu().numpy().tolist()
            prev_xt = None
            prev_x0y = None
            for step in range(x0y.shape[0]):
                with torch.no_grad():
                    feat_hat = residual_features(operator, x0hat[step], y, args.score_radius, "x0hat")
                    feat_y = residual_features(operator, x0y[step], y, args.score_radius, "x0y")
                    correction = batch_norm(x0y[step] - x0hat[step])
                    disagreement = correction
                    xt_jump = torch.zeros_like(correction) if prev_xt is None else batch_norm(xt[step] - prev_xt)
                    x0y_jump = torch.zeros_like(correction) if prev_x0y is None else batch_norm(x0y[step] - prev_x0y)
                    prev_xt = xt[step]
                    prev_x0y = x0y[step]
                    for image_i, meta in enumerate(manifest):
                        row: Dict[str, object] = {
                            "image_id": meta["image_id"],
                            "hard_index": meta["hard_index"],
                            "run_index": run_i,
                            "step": step,
                            "sigma": float(sigmas[step]),
                            "correction_norm": float(correction[image_i].detach().cpu().item()),
                            "x0hat_x0y_disagreement": float(disagreement[image_i].detach().cpu().item()),
                            "xt_step_jump": float(xt_jump[image_i].detach().cpu().item()),
                            "x0y_step_jump": float(x0y_jump[image_i].detach().cpu().item()),
                            "final_psnr": float(final_psnr[image_i].detach().cpu().item()),
                        }
                        for feats in (feat_hat, feat_y):
                            for key, values in feats.items():
                                row[key] = float(values[image_i].detach().cpu().item())
                        step_rows.append(row)
            del xt, x0hat, x0y
            torch.cuda.empty_cache()

        write_csv(outdir / "trajectory_step_metrics.csv", step_rows)
        run_summary = summarize_runs(step_rows)
        write_csv(outdir / "run_level_summary.csv", run_summary)
        write_text(
            outdir / "config.json",
            json.dumps(
                {
                    "noise": args.noise,
                    "image_ids": image_ids,
                    "num_runs": args.num_runs,
                    "gpu": args.gpu,
                    "seed": args.seed,
                    "oversample": args.oversample,
                    "score_radius": args.score_radius,
                    "anneal_steps": args.anneal_steps,
                    "diff_steps": args.diff_steps,
                    "sitcom_root": args.sitcom_root,
                },
                indent=2,
            )
            + "\n",
        )
        render_summary(outdir, run_summary, step_rows, args)
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
