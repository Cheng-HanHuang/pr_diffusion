#!/usr/bin/env python3
"""Prepare and validate a deterministic B22.2 smoke or full-panel execution plan.

Preparation is atomic: the visible stage directory is created only after all
identity checks, manifests, shards, and the plan have been written successfully.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
from PIL import Image
import torchvision.transforms as transforms

from b22_smoke_common import (
    git_branch,
    git_head,
    load_measurement,
    read_json,
    sha256_file,
    tensor_content_sha256,
    write_json,
)

MEAS_RE = re.compile(r"^ffhq(?P<image_id>\d+)_phase_noise005_meas5401\.pt$")


def find_gt(root: Path, image_id: str) -> Path:
    value = int(image_id)
    folder = f"{(value // 1000) * 1000:05d}"
    candidates = [
        root / folder / f"{value:05d}.png",
        root / "00000" / f"{value:05d}.png",
        root / f"{value:05d}.png",
    ]
    hits = [path for path in candidates if path.is_file()]
    if not hits:
        hits = list(root.rglob(f"{value:05d}.png"))
    unique = sorted({path.resolve() for path in hits})
    if len(unique) != 1:
        raise FileNotFoundError(
            f"Expected one FFHQ image for {image_id}; found {len(unique)} under {root}"
        )
    return unique[0]


def load_gt(path: Path, resolution: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
        ]
    )
    return (transform(Image.open(path).convert("RGB")) * 2.0 - 1.0).unsqueeze(0)


def git_diff(repo: Path, path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--", path],
        stderr=subprocess.STDOUT,
        text=True,
    ).strip()


def verify_identities(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    paths = config["pac_paths"]
    expected = config["expected_identity"]
    sitcom_root = Path(paths["sitcom_root"])
    difffpr_root = Path(paths["difffpr_root"])
    model_path = Path(paths["model_path"])

    observed = {
        "repo_branch": git_branch(repo_root),
        "repo_head": git_head(repo_root),
        "sitcom_head": git_head(sitcom_root),
        "difffpr_head": git_head(difffpr_root),
        "model_sha256": sha256_file(model_path),
        "sitcom_posterior_sample_diff": git_diff(sitcom_root, "posterior_sample.py"),
    }
    if observed["repo_branch"] != config["branch"]:
        raise RuntimeError(
            f"Worktree branch {observed['repo_branch']!r} does not match {config['branch']!r}"
        )
    for key, expected_key in (
        ("sitcom_head", "sitcom_commit"),
        ("difffpr_head", "difffpr_commit"),
        ("model_sha256", "model_sha256"),
    ):
        if observed[key] != expected[expected_key]:
            raise RuntimeError(
                f"Identity mismatch for {key}: observed {observed[key]}, expected {expected[expected_key]}"
            )
    diff = observed["sitcom_posterior_sample_diff"]
    if "mkdir(parents=True, exist_ok=True)" not in diff or "Path(dir).mkdir()" not in diff:
        raise RuntimeError("Effective SITCOM checkout no longer has the audited mkdir-only patch")
    return observed


def build_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    paths = config["pac_paths"]
    problem = config["problem"]
    measurement_root = Path(paths["locked_measurement_root"])
    files = sorted(measurement_root.glob(problem["measurement_glob"]))
    expected_count = int(problem["expected_measurement_count"])
    if len(files) != expected_count:
        raise RuntimeError(f"Expected {expected_count} locked measurements, found {len(files)}")

    gt_root = Path(paths["ffhq_data_root"])
    resolution = int(problem["resolution"])
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_content_hashes: set[str] = set()
    for row_id, path in enumerate(files):
        match = MEAS_RE.match(path.name)
        if not match:
            raise RuntimeError(f"Unexpected measurement filename: {path.name}")
        image_id = f"{int(match.group('image_id')):05d}"
        if image_id in seen_ids:
            raise RuntimeError(f"Duplicate image ID {image_id}")
        seen_ids.add(image_id)

        measurement = load_measurement(path, device="cpu")
        measurement_content_hash = tensor_content_sha256(measurement)
        if measurement_content_hash in seen_content_hashes:
            raise RuntimeError(f"Duplicate measurement tensor content at {path}")
        seen_content_hashes.add(measurement_content_hash)

        gt_path = find_gt(gt_root, image_id)
        gt = load_gt(gt_path, resolution)
        if tuple(gt.shape) != (1, 3, resolution, resolution):
            raise RuntimeError(f"Unexpected GT shape for {image_id}: {tuple(gt.shape)}")

        rows.append(
            {
                "row_id": row_id,
                "image_id": image_id,
                "measurement_path": str(path.resolve()),
                "measurement_file_sha256": sha256_file(path),
                "measurement_content_sha256": measurement_content_hash,
                "measurement_negative_count": int((measurement < 0).sum().item()),
                "ground_truth_path": str(gt_path),
                "ground_truth_file_sha256": sha256_file(gt_path),
                "ground_truth_content_sha256": tensor_content_sha256(gt),
            }
        )
    return rows


def make_shards(rows: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    if count <= 0:
        raise ValueError("Shard count must be positive")
    return [
        [row for index, row in enumerate(rows) if index % count == shard]
        for shard in range(count)
    ]


def write_stage_atomically(
    stage_root: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    sitcom_shards: list[list[dict[str, Any]]],
    np_shards: list[list[dict[str, Any]]],
    plan: dict[str, Any],
) -> None:
    if stage_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing stage directory: {stage_root}")
    temporary = stage_root.with_name(f".{stage_root.name}.prepare.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary preparation directory already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        write_json(temporary / "manifest.json", manifest)
        write_json(temporary / "config_snapshot.json", config)
        shards_root = temporary / "shards"
        shards_root.mkdir()
        for method, shards in (("sitcom", sitcom_shards), ("np", np_shards)):
            for shard_id, shard_rows in enumerate(shards):
                write_json(
                    shards_root / f"{method}_shard{shard_id}.json",
                    {
                        "schema_version": 1,
                        "stage": manifest["stage"],
                        "method": method,
                        "shard_id": shard_id,
                        "shard_count": len(shards),
                        "rows": shard_rows,
                    },
                )
        write_json(temporary / "plan.json", plan)
        temporary.replace(stage_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    args = parser.parse_args()

    config = read_json(args.config)
    repo_root = Path(args.repo_root).resolve()
    run_root = Path(args.run_root).resolve()
    stage_root = run_root / args.stage

    identities = verify_identities(config, repo_root)
    all_rows = build_rows(config)
    if args.stage == "smoke":
        indices = [int(value) for value in config["execution"]["smoke_row_indices"]]
        if len(set(indices)) != len(indices) or any(
            index < 0 or index >= len(all_rows) for index in indices
        ):
            raise RuntimeError(f"Invalid smoke row indices: {indices}")
        rows = [all_rows[index] for index in indices]
    else:
        rows = all_rows

    sitcom_shards = make_shards(rows, int(config["execution"]["sitcom_shards"]))
    np_shards = make_shards(rows, int(config["execution"]["np_shards"]))
    manifest = {
        "schema_version": 1,
        "stage": args.stage,
        "repo_root": str(repo_root),
        "run_root": str(run_root),
        "identities": identities,
        "rows": rows,
        "all_panel_row_count": len(all_rows),
        "selected_row_count": len(rows),
        "selection_uses_method_outcome": False,
        "selection_rule": (
            "fixed row indices from config"
            if args.stage == "smoke"
            else "all rows in sorted locked-measurement filename order"
        ),
    }
    plan = {
        "stage": args.stage,
        "selected_rows": len(rows),
        "sitcom_shard_sizes": [len(shard) for shard in sitcom_shards],
        "np_shard_sizes": [len(shard) for shard in np_shards],
        "expected_sitcom_candidates": len(rows)
        * int(config["sitcom"]["trajectory_count"]),
        "expected_np_candidates": len(rows)
        * len(config["np"]["seeds"])
        * len(config["np"]["configs"]),
        "full_launch_machine_gate": args.stage == "smoke",
    }
    write_stage_atomically(stage_root, config, manifest, sitcom_shards, np_shards, plan)
    print(json.dumps({"status": "PASS", **plan}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
