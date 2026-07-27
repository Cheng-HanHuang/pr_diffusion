#!/usr/bin/env python3
"""Run official one-trajectory SITCOM on the B22.1 locked smoke input."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from b22_smoke_common import (
    cuda_memory_snapshot,
    environment_cuda_visible_devices,
    git_head,
    load_ground_truth,
    load_measurement,
    metric_pair,
    package_versions,
    read_json,
    save_model_range_png,
    tensor_content_sha256,
    validate_reconstruction,
    write_json,
)


def load_external_sitcom(sitcom_root: Path):
    sys.path.insert(0, str(sitcom_root))
    old_cwd = Path.cwd()
    os.chdir(sitcom_root)
    from forward_operator import get_operator
    from model import get_model
    from sampler import get_sampler

    return old_cwd, get_operator, get_model, get_sampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()

    config = read_json(args.config)
    repo_root = Path(args.repo_root).resolve()
    run_root = Path(args.run_root).resolve()
    method_dir = run_root / "sitcom1"
    method_dir.mkdir(parents=True, exist_ok=False)

    manifest = read_json(run_root / "input" / "input_manifest.json")
    method = config["sitcom1"]
    paths = config["pac_paths"]
    sitcom_root = Path(paths["sitcom_root"]).resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("B22.1 SITCOM smoke requires a CUDA GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    seed = int(method["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    measurement = load_measurement(manifest["measurement_path"], device=device)
    if tensor_content_sha256(measurement) != manifest["measurement_content_sha256"]:
        raise RuntimeError("Locked measurement content hash changed before SITCOM execution")
    gt = load_ground_truth(manifest["ground_truth_tensor_path"], device=device)

    old_cwd, get_operator, get_model, get_sampler = load_external_sitcom(sitcom_root)
    try:
        operator = get_operator(
            name="phase_retrieval",
            sigma=float(config["problem"]["sigma_y"]),
            oversample=float(config["problem"]["oversample"]),
        )
        sampler = get_sampler(
            latent=False,
            annealing_scheduler_config={
                "num_steps": int(method["anneal_steps"]),
                "schedule": method["anneal_schedule"],
                "sigma_max": float(method["anneal_sigma_max"]),
                "sigma_min": float(method["anneal_sigma_min"]),
                "sigma_final": 0.0,
                "timestep": method["timestep"],
            },
            diffusion_scheduler_config={
                "num_steps": int(method["diff_steps"]),
                "schedule": method["diff_schedule"],
                "sigma_min": float(method["diff_sigma_min"]),
                "sigma_final": 0.0,
                "timestep": method["timestep"],
            },
            lgvd_config={
                "lr": float(method["lgvd_lr"]),
                "lr_min_ratio": float(method["lgvd_lr_min_ratio"]),
                "num_steps": int(method["lgvd_steps"]),
                "tau": float(method["lgvd_tau"]),
            },
        )

        load_start = time.perf_counter()
        model = get_model(
            name="ddpm",
            model_config={
                "attention_resolutions": 16,
                "channel_mult": "",
                "class_cond": False,
                "dropout": 0.0,
                "image_size": int(config["problem"]["resolution"]),
                "learn_sigma": True,
                "model_path": method["model_path_relative_to_sitcom_root"],
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
        torch.cuda.synchronize(device)
        model_load_s = time.perf_counter() - load_start

        torch.cuda.reset_peak_memory_stats(device)
        run_start = time.perf_counter()
        x_start = sampler.get_start(gt)
        reconstruction = sampler.sample(
            model,
            x_start,
            operator,
            measurement,
            evaluator=None,
            record=False,
            verbose=False,
        )
        torch.cuda.synchronize(device)
        reconstruction_s = time.perf_counter() - run_start
    finally:
        os.chdir(old_cwd)

    validate_reconstruction(reconstruction)
    metrics = metric_pair(reconstruction, gt)
    memory = cuda_memory_snapshot(device)

    with torch.no_grad():
        residual = operator(reconstruction) - measurement
        residual_l2 = float(torch.linalg.norm(residual).detach().cpu().item())
        measurement_l2 = float(
            torch.linalg.norm(measurement).detach().cpu().clamp_min(1.0e-12).item()
        )

    tensor_path = method_dir / "reconstruction.pt"
    png_path = method_dir / "reconstruction.png"
    torch.save({"reconstruction": reconstruction.detach().cpu()}, tensor_path)
    save_model_range_png(reconstruction, png_path)

    result = {
        "schema_version": 1,
        "method": "SITCOM-1",
        "policy_role": "primary executable one-trajectory baseline",
        "image_id": manifest["image_id"],
        "measurement_file_sha256": manifest["measurement_file_sha256"],
        "measurement_raw_content_sha256": manifest["measurement_content_sha256"],
        "measurement_preprocessing": "none",
        "measurement_used_content_sha256": tensor_content_sha256(measurement),
        "ground_truth_tensor_content_sha256": manifest[
            "ground_truth_tensor_content_sha256"
        ],
        "repo_head": git_head(repo_root),
        "sitcom_head": git_head(sitcom_root),
        "model_sha256": manifest["model_sha256"],
        "cuda_visible_devices": environment_cuda_visible_devices(),
        "device_name": torch.cuda.get_device_name(device),
        "environment": package_versions(
            ("torchvision", "numpy", "scipy", "PIL", "hydra", "omegaconf", "piq")
        ),
        "config": method,
        "metrics": metrics,
        "measurement_residual": {
            "l2": residual_l2,
            "normalized_l2": residual_l2 / measurement_l2,
        },
        "timing": {
            "model_load_s": model_load_s,
            "reconstruction_s": reconstruction_s,
            "total_observed_s": model_load_s + reconstruction_s,
        },
        "memory": memory,
        "reconstruction_tensor_path": str(tensor_path),
        "reconstruction_png_path": str(png_path),
        "reconstruction_content_sha256": tensor_content_sha256(reconstruction),
        "finite_output": True,
    }
    write_json(method_dir / "result.json", result)
    print(
        json.dumps(
            {
                "status": "OK",
                "method": "SITCOM-1",
                "image_id": manifest["image_id"],
                "psnr_raw": metrics["psnr_raw"],
                "runtime_s": reconstruction_s,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
