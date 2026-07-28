#!/usr/bin/env python3
"""Run a resumable NP-1/NP-8-RS B22.2 shard in the frozen NP environment."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_gt(path: Path, resolution: int, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
        ]
    )
    return ((transform(Image.open(path).convert("RGB")) * 2.0 - 1.0).unsqueeze(0)).to(device)


def valid_candidate(
    path: Path,
    row: dict[str, Any],
    config_tag: str,
    seed: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        result = read_json(path)
        tensor_path = Path(result["reconstruction_tensor_path"])
        return (
            result["status"] == "PASS"
            and result["image_id"] == row["image_id"]
            and result["config_tag"] == config_tag
            and int(result["seed"]) == seed
            and result["measurement_raw_content_sha256"] == row["measurement_content_sha256"]
            and tensor_path.is_file()
        )
    except Exception:
        return False


def run_candidate(
    *,
    selector,
    bundle,
    measurement_raw: torch.Tensor,
    measurement_np: torch.Tensor,
    gt: torch.Tensor,
    row: dict[str, Any],
    method_dir: Path,
    cfg: dict[str, Any],
    cfg_index: int,
    seed: int,
    pad: int,
    variant,
    device: torch.device,
    repo_root: Path,
    difffpr_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    config_tag = str(cfg["config_tag"])
    candidate_dir = method_dir / "candidates" / f"cfg{cfg_index}_{config_tag}_seed{seed}"
    result_path = candidate_dir / "result.json"
    if valid_candidate(result_path, row, config_tag, seed):
        return read_json(result_path)
    if candidate_dir.exists():
        raise RuntimeError(
            f"Refusing to overwrite incomplete/invalid candidate directory: {candidate_dir}"
        )
    candidate_dir.mkdir(parents=True)

    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    reconstruction, selector_stats = selector.reconstruct_with_selector_stat(
        measurement_np,
        pad=pad,
        seed=seed,
        unet=bundle.unet,
        scheduler=bundle.scheduler,
        device=device,
        variant=variant,
        num_steps=int(config["np"]["num_steps"]),
        score_radius=float(config["np"]["score_radius"]),
        proj_radius=float(config["np"]["proj_radius"]),
        proj_radius_schedule=config["np"].get("proj_radius_schedule"),
        score_mode=str(cfg["score_mode"]),
        score_reg_lambda=float(cfg["score_reg_lambda"]),
        score_reg_lambda_schedule=str(cfg["score_reg_lambda_schedule"]),
        score_huber_delta=float(config["np"]["score_huber_delta"]),
        log_every=int(config["np"]["log_every"]),
    )
    torch.cuda.synchronize(device)
    reconstruction_s = time.perf_counter() - start

    validate_reconstruction(reconstruction)
    memory = cuda_memory_snapshot(device)
    reconstruction_cpu = reconstruction.detach().cpu().float()
    gt_cpu = gt.detach().cpu().float()
    metrics = metric_pair(reconstruction_cpu, gt_cpu)

    tensor_path = candidate_dir / "reconstruction.pt"
    png_path = candidate_dir / "reconstruction.png"
    torch.save({"reconstruction": reconstruction_cpu}, tensor_path)
    save_model_range_png(reconstruction_cpu, png_path)

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "method_family": "NP",
        "image_id": row["image_id"],
        "row_id": row["row_id"],
        "config_index": cfg_index,
        "config_tag": config_tag,
        "seed": seed,
        "measurement_path": row["measurement_path"],
        "measurement_file_sha256": row["measurement_file_sha256"],
        "measurement_raw_content_sha256": row["measurement_content_sha256"],
        "measurement_used_content_sha256": tensor_content_sha256(measurement_np),
        "measurement_negative_entries_clipped": int((measurement_raw < 0).sum().item()),
        "measurement_preprocessing": config["np"]["measurement_preprocessing"],
        "ground_truth_content_sha256": row["ground_truth_content_sha256"],
        "repo_head": git_head(repo_root),
        "difffpr_head": git_head(difffpr_root),
        "cuda_visible_devices": environment_cuda_visible_devices(),
        "device_name": torch.cuda.get_device_name(device),
        "environment": package_versions(
            ("torchvision", "numpy", "scipy", "PIL", "diffusers")
        ),
        "config": cfg,
        "selector_stats": selector_stats,
        "metrics": metrics,
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
        "candidate_config_index": result["config_index"],
        "candidate_config_tag": result["config_tag"],
        "candidate_seed": result["seed"],
        "selector_stat": result["selector_stats"]["selector_post_winner_lf_mse_mean"],
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
    shard = read_json(stage_root / "shards" / f"np_shard{args.shard}.json")
    rows = shard["rows"]

    paths = config["pac_paths"]
    difffpr_root = Path(paths["difffpr_root"]).resolve()
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("NP worker requires CUDA")
    torch.cuda.set_device(device)

    selector_path = repo_root / "scripts" / "pr_external_difffpr_np_guided_lf_s2_selector.py"
    selector = load_module("b22_2_np_selector_module", selector_path)
    method = config["np"]
    problem = config["problem"]

    load_start = time.perf_counter()
    bundle = selector.load_guided_diffusion_model(
        model_path=paths["model_path"],
        device=device,
        preset=method["guided_preset"],
        guided_diffusion_dir=str(difffpr_root),
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
        int(problem["resolution"]), float(problem["oversample"])
    )

    worker_start = time.perf_counter()
    completed = 0
    for row in rows:
        method_dir = stage_root / "np" / f"row{int(row['row_id']):03d}_{row['image_id']}"
        policy_path = method_dir / "policy.json"
        if policy_path.is_file():
            policy = read_json(policy_path)
            if policy.get("status") == "PASS":
                completed += 1
                print(json.dumps({"status": "SKIP", "image_id": row["image_id"]}))
                continue
            raise RuntimeError(f"Invalid existing policy file: {policy_path}")
        method_dir.mkdir(parents=True, exist_ok=True)

        measurement_raw = load_measurement(row["measurement_path"], device=device)
        if tensor_content_sha256(measurement_raw) != row["measurement_content_sha256"]:
            raise RuntimeError(f"Measurement content changed for {row['image_id']}")
        measurement_np = measurement_raw.clamp_min(0.0)
        gt = load_gt(Path(row["ground_truth_path"]), int(problem["resolution"]), device)
        if tensor_content_sha256(gt) != row["ground_truth_content_sha256"]:
            raise RuntimeError(f"Ground truth content changed for {row['image_id']}")

        candidates: list[dict[str, Any]] = []
        for cfg_index, cfg in enumerate(method["configs"]):
            for seed in method["seeds"]:
                candidates.append(
                    run_candidate(
                        selector=selector,
                        bundle=bundle,
                        measurement_raw=measurement_raw,
                        measurement_np=measurement_np,
                        gt=gt,
                        row=row,
                        method_dir=method_dir,
                        cfg=cfg,
                        cfg_index=cfg_index,
                        seed=int(seed),
                        pad=pad,
                        variant=variant,
                        device=device,
                        repo_root=repo_root,
                        difffpr_root=difffpr_root,
                        config=config,
                    )
                )
                torch.cuda.empty_cache()

        np1 = next(
            r
            for r in candidates
            if r["config_tag"] == method["np1_config_tag"]
            and int(r["seed"]) == int(method["np1_seed"])
        )
        selected = min(
            candidates,
            key=lambda r: (
                float(r["selector_stats"][method["selector_stat"]]),
                int(r["config_index"]),
                int(r["seed"]),
            ),
        )
        oracle = max(
            candidates,
            key=lambda r: (
                float(r["metrics"]["psnr_raw"]),
                -int(r["config_index"]),
                -int(r["seed"]),
            ),
        )
        policy = {
            "schema_version": 1,
            "status": "PASS",
            "method_family": "NP",
            "row_id": row["row_id"],
            "image_id": row["image_id"],
            "candidate_count": len(candidates),
            "candidate_configs": method["configs"],
            "candidate_seeds": [int(x) for x in method["seeds"]],
            "measurement_raw_content_sha256": row["measurement_content_sha256"],
            "measurement_used_content_sha256": tensor_content_sha256(measurement_np),
            "model_load_s_worker_shared": model_load_s,
            "sum_candidate_reconstruction_s": sum(
                float(r["timing"]["reconstruction_s"]) for r in candidates
            ),
            "policies": {
                "NP-1": materialize_selected(method_dir, "np1", np1),
                "NP-8-RS": materialize_selected(method_dir, "np8_rs", selected),
                "NP-oracle8": materialize_selected(method_dir, "np_oracle8", oracle),
            },
            "selector_rule": method["selector_rule"],
            "selector_values": [
                {
                    "config_index": r["config_index"],
                    "config_tag": r["config_tag"],
                    "seed": r["seed"],
                    "selector_post_winner_lf_mse_mean": r["selector_stats"][
                        method["selector_stat"]
                    ],
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
                    "np1_psnr": policy["policies"]["NP-1"]["metrics"]["psnr_raw"],
                    "np8_psnr": policy["policies"]["NP-8-RS"]["metrics"]["psnr_raw"],
                    "selected_config": policy["policies"]["NP-8-RS"]["candidate_config_tag"],
                    "selected_seed": policy["policies"]["NP-8-RS"]["candidate_seed"],
                },
                sort_keys=True,
            )
        )
        del measurement_raw, measurement_np, gt
        torch.cuda.empty_cache()

    worker_summary = {
        "schema_version": 1,
        "status": "PASS",
        "method": "NP",
        "stage": args.stage,
        "shard": args.shard,
        "assigned_rows": len(rows),
        "completed_rows": completed,
        "worker_wall_s": time.perf_counter() - worker_start,
        "model_load_s": model_load_s,
        "device_name": torch.cuda.get_device_name(device),
    }
    atomic_write_json(stage_root / "workers" / f"np_shard{args.shard}.json", worker_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
