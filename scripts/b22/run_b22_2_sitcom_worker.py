#!/usr/bin/env python3
"""Run a resumable SITCOM-1/SITCOM-4S B22.2 shard in the frozen SITCOM environment."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from b22_json_safe import nonfinite_paths, sanitize_for_json
from b22_smoke_common import (
    cuda_memory_snapshot,
    environment_cuda_visible_devices,
    git_head,
    load_measurement,
    metric_pair,
    package_versions,
    read_json,
    save_model_range_png,
    tensor_content_sha256,
    validate_reconstruction,
    write_json,
)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_json(tmp, sanitize_for_json(value))
    os.replace(tmp, path)


def load_gt(path: Path, resolution: int, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
        ]
    )
    return ((transform(Image.open(path).convert("RGB")) * 2.0 - 1.0).unsqueeze(0)).to(device)


def load_external_sitcom(sitcom_root: Path):
    sys.path.insert(0, str(sitcom_root))
    old_cwd = Path.cwd()
    os.chdir(sitcom_root)
    from forward_operator import get_operator
    from model import get_model
    from sampler import get_sampler

    return old_cwd, get_operator, get_model, get_sampler


def valid_candidate(path: Path, row: dict[str, Any], run_index: int, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        result = read_json(path)
        tensor_path = Path(result["reconstruction_tensor_path"])
        return (
            result["status"] == "PASS"
            and result["image_id"] == row["image_id"]
            and int(result["run_index"]) == run_index
            and int(result["seed"]) == seed
            and result["measurement_content_sha256"] == row["measurement_content_sha256"]
            and tensor_path.is_file()
        )
    except Exception:
        return False


def load_reconstruction(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    value = payload["reconstruction"] if isinstance(payload, dict) else payload
    validate_reconstruction(value)
    return value.float()


def run_candidate(
    *,
    sampler,
    model,
    operator,
    measurement: torch.Tensor,
    gt: torch.Tensor,
    row: dict[str, Any],
    method_dir: Path,
    run_index: int,
    seed: int,
    selector_step: int,
    device: torch.device,
    repo_root: Path,
    sitcom_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    candidate_dir = method_dir / "candidates" / f"run{run_index:02d}_seed{seed}"
    result_path = candidate_dir / "result.json"
    if valid_candidate(result_path, row, run_index, seed):
        return read_json(result_path)
    if candidate_dir.exists():
        raise RuntimeError(
            f"Refusing to overwrite incomplete/invalid candidate directory: {candidate_dir}"
        )
    candidate_dir.mkdir(parents=True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    selector_holder: dict[str, float] = {}
    counter = {"step": 0}
    original_record = sampler._record

    def lightweight_record(self, xt, x0y, x0hat, sigma, x0hat_results, x0y_results):
        step = counter["step"]
        if step == selector_step:
            correction = torch.sqrt(
                torch.mean((x0y.detach().float() - x0hat.detach().float()).square()).clamp_min(1e-24)
            )
            selector_holder["correction_norm"] = float(correction.cpu().item())
            selector_holder["sigma"] = float(sigma)
        counter["step"] = step + 1

    sampler._record = types.MethodType(lightweight_record, sampler)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    try:
        x_start = sampler.get_start(gt)
        reconstruction = sampler.sample(
            model,
            x_start,
            operator,
            measurement,
            evaluator=None,
            record=True,
            verbose=False,
        )
        torch.cuda.synchronize(device)
    finally:
        sampler._record = original_record
    reconstruction_s = time.perf_counter() - start

    if counter["step"] != int(config["sitcom"]["anneal_steps"]):
        raise RuntimeError(
            f"SITCOM recorded {counter['step']} annealing steps, expected {config['sitcom']['anneal_steps']}"
        )
    if "correction_norm" not in selector_holder:
        raise RuntimeError(f"SITCOM selector step {selector_step} was not observed")

    validate_reconstruction(reconstruction)
    memory = cuda_memory_snapshot(device)
    reconstruction_cpu = reconstruction.detach().cpu().float()
    gt_cpu = gt.detach().cpu().float()
    metrics = metric_pair(reconstruction_cpu, gt_cpu)

    with torch.no_grad():
        residual = operator(reconstruction) - measurement
        residual_l2 = float(torch.linalg.norm(residual).detach().cpu().item())
        measurement_l2 = float(
            torch.linalg.norm(measurement).detach().cpu().clamp_min(1e-12).item()
        )

    tensor_path = candidate_dir / "reconstruction.pt"
    png_path = candidate_dir / "reconstruction.png"
    torch.save({"reconstruction": reconstruction_cpu}, tensor_path)
    save_model_range_png(reconstruction_cpu, png_path)

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "method_family": "SITCOM",
        "image_id": row["image_id"],
        "row_id": row["row_id"],
        "run_index": run_index,
        "seed": seed,
        "measurement_path": row["measurement_path"],
        "measurement_file_sha256": row["measurement_file_sha256"],
        "measurement_content_sha256": row["measurement_content_sha256"],
        "measurement_preprocessing": "none",
        "ground_truth_content_sha256": row["ground_truth_content_sha256"],
        "repo_head": git_head(repo_root),
        "sitcom_head": git_head(sitcom_root),
        "cuda_visible_devices": environment_cuda_visible_devices(),
        "device_name": torch.cuda.get_device_name(device),
        "environment": package_versions(
            ("torchvision", "numpy", "scipy", "PIL", "hydra", "omegaconf", "piq")
        ),
        "selector": {
            "tau": float(config["sitcom"]["selector_tau"]),
            "step": selector_step,
            "correction_norm": selector_holder["correction_norm"],
            "sigma": selector_holder["sigma"],
        },
        "metrics": metrics,
        "measurement_residual": {
            "l2": residual_l2,
            "normalized_l2": residual_l2 / measurement_l2,
        },
        "timing": {"reconstruction_s": reconstruction_s},
        "memory": memory,
        "reconstruction_tensor_path": str(tensor_path),
        "reconstruction_png_path": str(png_path),
        "reconstruction_content_sha256": tensor_content_sha256(reconstruction_cpu),
        "finite_output": True,
    }
    undefined = nonfinite_paths(result)
    if undefined:
        result["undefined_diagnostics"] = undefined
    atomic_write_json(result_path, result)
    return read_json(result_path)


def materialize_selected(method_dir: Path, label: str, result: dict[str, Any]) -> dict[str, Any]:
    selected_dir = method_dir / "selected"
    selected_dir.mkdir(exist_ok=True)
    source_tensor = Path(result["reconstruction_tensor_path"])
    source_png = Path(result["reconstruction_png_path"])
    target_tensor = selected_dir / f"{label}.pt"
    target_png = selected_dir / f"{label}.png"
    if not target_tensor.exists():
        shutil.copy2(source_tensor, target_tensor)
    if not target_png.exists():
        shutil.copy2(source_png, target_png)
    return {
        "candidate_run_index": result["run_index"],
        "candidate_seed": result["seed"],
        "metrics": result["metrics"],
        "reconstruction_s": result["timing"]["reconstruction_s"],
        "reconstruction_content_sha256": result["reconstruction_content_sha256"],
        "tensor_path": str(target_tensor),
        "png_path": str(target_png),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()

    config = read_json(args.config)
    repo_root = Path(args.repo_root).resolve()
    run_root = Path(args.run_root).resolve()
    stage_root = run_root / args.stage
    shard = read_json(stage_root / "shards" / f"sitcom_shard{args.shard}.json")
    rows = shard["rows"]

    paths = config["pac_paths"]
    sitcom_root = Path(paths["sitcom_root"]).resolve()
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("SITCOM worker requires CUDA")
    torch.cuda.set_device(device)

    old_cwd, get_operator, get_model, get_sampler = load_external_sitcom(sitcom_root)
    worker_start = time.perf_counter()
    try:
        problem = config["problem"]
        method = config["sitcom"]
        operator = get_operator(
            name="phase_retrieval",
            sigma=float(problem["sigma_y"]),
            oversample=float(problem["oversample"]),
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
                "image_size": int(problem["resolution"]),
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

        completed = 0
        for row in rows:
            method_dir = stage_root / "sitcom" / f"row{int(row['row_id']):03d}_{row['image_id']}"
            policy_path = method_dir / "policy.json"
            if policy_path.is_file():
                policy = read_json(policy_path)
                if policy.get("status") == "PASS":
                    completed += 1
                    print(json.dumps({"status": "SKIP", "image_id": row["image_id"]}))
                    continue
                raise RuntimeError(f"Invalid existing policy file: {policy_path}")
            method_dir.mkdir(parents=True, exist_ok=True)

            measurement = load_measurement(row["measurement_path"], device=device)
            if tensor_content_sha256(measurement) != row["measurement_content_sha256"]:
                raise RuntimeError(f"Measurement content changed for {row['image_id']}")
            gt = load_gt(Path(row["ground_truth_path"]), int(problem["resolution"]), device)
            if tensor_content_sha256(gt) != row["ground_truth_content_sha256"]:
                raise RuntimeError(f"Ground truth content changed for {row['image_id']}")

            candidates: list[dict[str, Any]] = []
            for run_index, seed in enumerate(method["seeds"]):
                candidates.append(
                    run_candidate(
                        sampler=sampler,
                        model=model,
                        operator=operator,
                        measurement=measurement,
                        gt=gt,
                        row=row,
                        method_dir=method_dir,
                        run_index=run_index,
                        seed=int(seed),
                        selector_step=int(method["selector_step"]),
                        device=device,
                        repo_root=repo_root,
                        sitcom_root=sitcom_root,
                        config=config,
                    )
                )
                torch.cuda.empty_cache()

            sitcom1 = candidates[int(method["sitcom1_candidate_index"])]
            selected = min(
                candidates,
                key=lambda r: (float(r["selector"]["correction_norm"]), int(r["run_index"])),
            )
            oracle = max(
                candidates,
                key=lambda r: (float(r["metrics"]["psnr_raw"]), -int(r["run_index"])),
            )
            policy = {
                "schema_version": 1,
                "status": "PASS",
                "method_family": "SITCOM",
                "row_id": row["row_id"],
                "image_id": row["image_id"],
                "candidate_count": len(candidates),
                "candidate_seeds": [int(x) for x in method["seeds"]],
                "measurement_content_sha256": row["measurement_content_sha256"],
                "model_load_s_worker_shared": model_load_s,
                "sum_candidate_reconstruction_s": sum(
                    float(r["timing"]["reconstruction_s"]) for r in candidates
                ),
                "policies": {
                    "SITCOM-1": materialize_selected(method_dir, "sitcom1", sitcom1),
                    "SITCOM-4S": materialize_selected(method_dir, "sitcom4s", selected),
                    "SITCOM-oracle4": materialize_selected(method_dir, "sitcom_oracle4", oracle),
                },
                "selector_rule": method["selector_rule"],
                "selector_values": [
                    {
                        "run_index": r["run_index"],
                        "seed": r["seed"],
                        "correction_norm": r["selector"]["correction_norm"],
                    }
                    for r in candidates
                ],
            }
            atomic_write_json(policy_path, policy)
            completed += 1
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "image_id": row["image_id"],
                        "sitcom1_psnr": policy["policies"]["SITCOM-1"]["metrics"]["psnr_raw"],
                        "sitcom4s_psnr": policy["policies"]["SITCOM-4S"]["metrics"]["psnr_raw"],
                        "selected_run": policy["policies"]["SITCOM-4S"]["candidate_run_index"],
                    },
                    sort_keys=True,
                )
            )
            del measurement, gt
            torch.cuda.empty_cache()
    finally:
        os.chdir(old_cwd)

    worker_summary = {
        "schema_version": 1,
        "status": "PASS",
        "method": "SITCOM",
        "stage": args.stage,
        "shard": args.shard,
        "assigned_rows": len(rows),
        "completed_rows": completed,
        "worker_wall_s": time.perf_counter() - worker_start,
        "device_name": torch.cuda.get_device_name(device),
    }
    atomic_write_json(stage_root / "workers" / f"sitcom_shard{args.shard}.json", worker_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
