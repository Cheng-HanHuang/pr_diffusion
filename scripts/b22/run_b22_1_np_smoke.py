#!/usr/bin/env python3
"""Run the frozen one-trajectory NP baseline on the B22.1 locked smoke input."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()

    config = read_json(args.config)
    repo_root = Path(args.repo_root).resolve()
    run_root = Path(args.run_root).resolve()
    method_dir = run_root / "np1"
    method_dir.mkdir(parents=True, exist_ok=False)

    manifest = read_json(run_root / "input" / "input_manifest.json")
    method = config["np1"]
    paths = config["pac_paths"]

    if not torch.cuda.is_available():
        raise RuntimeError("B22.1 NP smoke requires a CUDA GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    measurement_raw = load_measurement(manifest["measurement_path"], device=device)
    if tensor_content_sha256(measurement_raw) != manifest["measurement_content_sha256"]:
        raise RuntimeError("Locked measurement content hash changed before NP execution")

    if method["measurement_preprocessing"] != "clamp_min_zero_in_memory":
        raise RuntimeError(
            f"Unexpected frozen NP preprocessing: {method['measurement_preprocessing']}"
        )
    measurement_np = measurement_raw.clamp_min(0.0)
    gt = load_ground_truth(manifest["ground_truth_tensor_path"], device=device)

    selector_path = repo_root / "scripts" / "pr_external_difffpr_np_guided_lf_s2_selector.py"
    selector = load_module("b22_np_selector_module", selector_path)

    load_start = time.perf_counter()
    bundle = selector.load_guided_diffusion_model(
        model_path=paths["model_path"],
        device=device,
        preset=method["guided_preset"],
        guided_diffusion_dir=paths["difffpr_root"],
        strict=bool(method["guided_strict"]),
    )
    torch.cuda.synchronize(device)
    model_load_s = time.perf_counter() - load_start

    variant = selector.base.NPVariant(
        name=method["variant"],
        soft=int(method["soft_candidates"]),
        hard=int(method["hard_candidates"]),
        proj_start=int(method["proj_start"]),
        use_lowfreq_score=True,
        use_lowfreq_projection=True,
    )
    pad = selector.base.oversample_pad(
        int(config["problem"]["resolution"]),
        float(config["problem"]["oversample"]),
    )

    torch.cuda.reset_peak_memory_stats(device)
    run_start = time.perf_counter()
    reconstruction, selector_stats = selector.reconstruct_with_selector_stat(
        measurement_np,
        pad=pad,
        seed=int(method["seed"]),
        unet=bundle.unet,
        scheduler=bundle.scheduler,
        device=device,
        variant=variant,
        num_steps=int(method["num_steps"]),
        score_radius=float(method["score_radius"]),
        proj_radius=float(method["proj_radius"]),
        proj_radius_schedule=method.get("proj_radius_schedule"),
        score_mode=method["score_mode"],
        score_reg_lambda=float(method["score_reg_lambda"]),
        score_reg_lambda_schedule=method["score_reg_lambda_schedule"],
        score_huber_delta=float(method["score_huber_delta"]),
        log_every=int(method["log_every"]),
    )
    torch.cuda.synchronize(device)
    reconstruction_s = time.perf_counter() - run_start

    validate_reconstruction(reconstruction)
    metrics = metric_pair(reconstruction, gt)
    memory = cuda_memory_snapshot(device)

    tensor_path = method_dir / "reconstruction.pt"
    png_path = method_dir / "reconstruction.png"
    torch.save({"reconstruction": reconstruction.detach().cpu()}, tensor_path)
    save_model_range_png(reconstruction, png_path)

    result = {
        "schema_version": 1,
        "method": "NP-1",
        "policy_role": "primary executable one-trajectory baseline",
        "image_id": manifest["image_id"],
        "measurement_file_sha256": manifest["measurement_file_sha256"],
        "measurement_raw_content_sha256": manifest["measurement_content_sha256"],
        "measurement_preprocessing": method["measurement_preprocessing"],
        "measurement_used_content_sha256": tensor_content_sha256(measurement_np),
        "measurement_negative_entries_clipped": int((measurement_raw < 0).sum().item()),
        "ground_truth_tensor_content_sha256": manifest[
            "ground_truth_tensor_content_sha256"
        ],
        "repo_head": git_head(repo_root),
        "difffpr_head": git_head(paths["difffpr_root"]),
        "model_sha256": manifest["model_sha256"],
        "cuda_visible_devices": environment_cuda_visible_devices(),
        "device_name": torch.cuda.get_device_name(device),
        "environment": package_versions(
            ("torchvision", "numpy", "scipy", "PIL", "diffusers")
        ),
        "config": method,
        "pad_pixels_each_side": pad,
        "selector_stats": selector_stats,
        "metrics": metrics,
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
                "method": "NP-1",
                "image_id": manifest["image_id"],
                "psnr_raw": metrics["psnr_raw"],
                "runtime_s": reconstruction_s,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
