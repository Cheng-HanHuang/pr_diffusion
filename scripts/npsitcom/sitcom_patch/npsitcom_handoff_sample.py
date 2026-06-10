#!/usr/bin/env python3
"""NP-to-SITCOM sigma-space handoff runner.

Install this file at the root of a copied SITCOM_ODE tree. It loads NP handoff
states exported by pr_diffusion and continues the SITCOM/DAPS annealing loop
from the nearest sigma in the SITCOM annealing schedule.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image

from data import get_dataset
from eval import Evaluator, get_eval_fn
from forward_operator import get_operator
from model import get_model
from sampler import DiffusionSampler, Scheduler, get_sampler


def read_csv(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
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


def cfg_get(args, key: str, default=None):
    return args[key] if key in args and args[key] is not None else default


def metric(results, name: str):
    try:
        return results[name]["sample"][0][0]
    except Exception:
        return math.nan


def image_index_from_row(row, image_name_to_index, fallback: int) -> int:
    if row.get("image_index_in_split", "") != "":
        return int(row["image_index_in_split"])
    base = Path(row.get("image_basename", "")).stem
    return int(image_name_to_index.get(base, fallback))


def nearest_sigma_step(sigma_steps, sigma: float) -> int:
    vals = [float(x) for x in sigma_steps[:-1]]
    return min(range(len(vals)), key=lambda i: abs(vals[i] - float(sigma)))


def continue_sitcom_from_sigma(sampler, model, x_sigma, operator, measurement, sigma: float):
    start_step = nearest_sigma_step(sampler.annealing_scheduler.sigma_steps, sigma)
    n_steps = int(sampler.annealing_scheduler.num_steps)
    x = x_sigma
    for step in range(start_step, n_steps):
        step_sigma = float(sampler.annealing_scheduler.sigma_steps[step])
        diffusion_scheduler = Scheduler(**sampler.diffusion_scheduler_config, sigma_max=step_sigma)
        diffusion_sampler = DiffusionSampler(diffusion_scheduler)
        x0_hat = diffusion_sampler.sample(model, x, operator, measurement, SDE=False, verbose=False)
        x0_y = sampler.lgvd.sample(
            x0_hat, model, operator, measurement, step_sigma, step / n_steps, record=False, verbose=False
        )
        next_sigma = float(sampler.annealing_scheduler.sigma_steps[step + 1])
        x = x0_y + torch.randn_like(x0_y) * next_sigma
    return x, start_step, float(sampler.annealing_scheduler.sigma_steps[start_step])


@hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
def main(args):
    manifest_path = cfg_get(args, "handoff_manifest")
    outdir = Path(cfg_get(args, "handoff_outdir", str(Path(args.save_dir) / args.name)))
    image_manifest_path = cfg_get(args, "handoff_image_manifest", None)
    max_rows = cfg_get(args, "handoff_max_rows", None)
    if manifest_path is None:
        raise ValueError("Pass +handoff_manifest=/path/to/handoff_manifest.csv")

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    torch.cuda.set_device(f"cuda:{int(args.gpu)}")
    device = torch.device(f"cuda:{int(args.gpu)}")

    outdir.mkdir(parents=True, exist_ok=True)
    sample_dir = outdir / "samples"
    sample_dir.mkdir(exist_ok=True)

    data = get_dataset(**args.data)
    images = data.get_data(len(data), 0).to(device)
    operator = get_operator(**args.task.operator)
    sampler = get_sampler(**args.sampler, lgvd_config=args.task.lgvd_config)
    model = get_model(**args.model)
    evaluator = Evaluator([get_eval_fn(name) for name in args.eval_fn_list])

    name_to_index = {}
    if image_manifest_path:
        for item in read_csv(str(image_manifest_path)):
            idx = int(item["index"])
            for key in ("split_entry", "source_path", "sitcom_path"):
                if key in item:
                    name_to_index[Path(item[key]).stem] = idx

    rows = read_csv(str(manifest_path))
    if max_rows is not None:
        rows = rows[: int(max_rows)]

    out_rows = []
    for row_id, row in enumerate(rows):
        state = torch.load(row["state_path"], map_location=device)
        x_sigma = state["x_sigma"].to(device)
        sigma = float(state.get("sigma", row.get("handoff_sigma")))
        image_i = image_index_from_row(row, name_to_index, row_id % len(images))
        gt = images[image_i : image_i + 1]
        measurement = state.get("measurement", None)
        if measurement is None:
            measurement = operator.measure(gt)
        else:
            measurement = measurement.to(device)

        sample, start_step, actual_sigma = continue_sitcom_from_sigma(
            sampler, model, x_sigma, operator, measurement, sigma
        )
        results = evaluator.report(gt, measurement, sample)
        png = sample_dir / f"{row_id:05d}_{Path(row.get('image_basename','img')).stem}_sig{sigma:g}.png"
        save_image((sample * 0.5 + 0.5).clamp(0, 1), str(png))

        out = dict(row)
        out.update(
            candidate_source="npsitcom_handoff",
            selection_method="npsitcom_handoff",
            alignment_mode="resolve",
            image_index_in_split=image_i,
            requested_sigma=sigma,
            start_step=start_step,
            actual_start_sigma=actual_sigma,
            psnr=metric(results, "psnr"),
            ssim=metric(results, "ssim"),
            lpips=metric(results, "lpips"),
            sample_path=str(png),
        )
        out_rows.append(out)
        print(f"[npsitcom] row={row_id} image={out.get('image_basename')} sigma={sigma:g} psnr={out['psnr']}", flush=True)

    write_csv(str(outdir / "run_level.csv"), out_rows)
    (outdir / "config.yaml").write_text(OmegaConf.to_yaml(args))
    (outdir / "summary.json").write_text(json.dumps({"n": len(out_rows)}, indent=2))


if __name__ == "__main__":
    main()
