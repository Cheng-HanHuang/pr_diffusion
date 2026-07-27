#!/usr/bin/env python3
"""CPU-only integrity preflight and deterministic input selection for B22.1."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torchvision import transforms

from b22_smoke_common import (
    MEASUREMENT_SHAPE,
    git_branch,
    git_head,
    load_measurement,
    read_json,
    save_model_range_png,
    sha256_file,
    tensor_content_sha256,
    write_json,
)


IMAGE_RE = re.compile(r"^ffhq(?P<image_id>\d+)_phase_noise005_meas5401\.pt$")


def require_git_head(repo: Path, expected: str, label: str) -> None:
    observed = git_head(repo)
    if observed != expected:
        raise RuntimeError(f"{label} commit mismatch: observed {observed}, expected {expected}")


def git_status_porcelain(repo: Path) -> List[str]:
    text = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain=v1"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    return [line for line in text.splitlines() if line.strip()]


def assert_sitcom_local_patch(sitcom_root: Path) -> Dict[str, object]:
    status = git_status_porcelain(sitcom_root)
    relevant = [
        line
        for line in status
        if "__pycache__" not in line
        and not line.rstrip().endswith(".pyc")
        and not line.startswith("?? outputs/")
        and not line.startswith("?? checkpoint/")
        and not line.startswith("?? dataset/")
        and not line.startswith("?? run_ffhq")
    ]

    tracked_modified = [
        line[3:]
        for line in relevant
        if len(line) >= 4 and not line.startswith("??")
    ]
    unexpected_tracked = [path for path in tracked_modified if path != "posterior_sample.py"]
    if unexpected_tracked:
        raise RuntimeError(
            "Unexpected tracked SITCOM modifications: " + ", ".join(unexpected_tracked)
        )

    diff = subprocess.check_output(
        ["git", "-C", str(sitcom_root), "diff", "--", "posterior_sample.py"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    if tracked_modified:
        required_added = "Path(dir).mkdir(parents=True, exist_ok=True)"
        required_removed = "Path(dir).mkdir()"
        if required_added not in diff or required_removed not in diff:
            raise RuntimeError(
                "SITCOM posterior_sample.py differs from the audited mkdir-only patch"
            )

    return {
        "status_relevant": relevant,
        "tracked_modified": tracked_modified,
        "posterior_sample_diff": diff,
    }


def find_unique_image(data_root: Path, image_id: str) -> Path:
    matches = sorted(data_root.rglob(f"{image_id}.png"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one FFHQ source image {image_id}.png under {data_root}; "
            f"found {len(matches)}: {[str(x) for x in matches[:10]]}"
        )
    return matches[0]


def build_ground_truth(image_path: Path, resolution: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    value = transform(image).unsqueeze(0).to(dtype=torch.float32)
    value = value * 2.0 - 1.0
    if tuple(value.shape) != (1, 3, resolution, resolution):
        raise RuntimeError(f"Unexpected ground-truth tensor shape: {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError("Ground-truth tensor contains a NaN or infinity")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    repo_root = Path(args.repo_root).resolve()
    run_root = Path(args.run_root).resolve()
    input_dir = run_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    config = read_json(config_path)
    paths = config["pac_paths"]
    expected = config["expected_identity"]
    selection = config["smoke_selection"]

    branch = git_branch(repo_root)
    if branch != config["branch"]:
        raise RuntimeError(
            f"Run from branch {config['branch']!r}; current branch is {branch!r}"
        )

    main_head = git_head(repo_root)
    sitcom_root = Path(paths["sitcom_root"])
    difffpr_root = Path(paths["difffpr_root"])
    model_path = Path(paths["model_path"])
    measurement_root = Path(paths["locked_measurement_root"])
    data_root = Path(paths["ffhq_data_root"])

    require_git_head(sitcom_root, expected["sitcom_commit"], "SITCOM")
    require_git_head(difffpr_root, expected["difffpr_commit"], "DiffFPR")

    model_sha = sha256_file(model_path)
    if model_sha != expected["model_sha256"]:
        raise RuntimeError(
            f"FFHQ model SHA-256 mismatch: observed {model_sha}, "
            f"expected {expected['model_sha256']}"
        )

    sitcom_patch = assert_sitcom_local_patch(sitcom_root)

    measurements = sorted(measurement_root.glob(selection["measurement_glob"]))
    if len(measurements) != int(selection["expected_measurement_count"]):
        raise RuntimeError(
            f"Expected {selection['expected_measurement_count']} locked measurements; "
            f"found {len(measurements)}"
        )

    if selection["rule"] != "lexicographically_first_locked_measurement":
        raise RuntimeError(f"Unsupported B22.1 selection rule: {selection['rule']}")
    selected_measurement = measurements[0]

    match = IMAGE_RE.match(selected_measurement.name)
    if match is None:
        raise RuntimeError(f"Could not parse image ID from {selected_measurement.name}")
    image_id = match.group("image_id")

    measurement = load_measurement(selected_measurement, device="cpu")
    if tuple(measurement.shape) != MEASUREMENT_SHAPE:
        raise RuntimeError("Measurement shape changed after validated load")

    image_path = find_unique_image(data_root, image_id)
    ground_truth = build_ground_truth(image_path, int(config["problem"]["resolution"]))

    gt_tensor_path = input_dir / "ground_truth.pt"
    torch.save({"ground_truth": ground_truth}, gt_tensor_path)
    save_model_range_png(ground_truth, input_dir / "ground_truth.png")

    raw_negative = measurement < 0
    manifest = {
        "schema_version": 1,
        "selection_rule": selection["rule"],
        "selection_uses_method_outcome": False,
        "image_id": image_id,
        "measurement_path": str(selected_measurement),
        "measurement_filename": selected_measurement.name,
        "measurement_file_sha256": sha256_file(selected_measurement),
        "measurement_content_sha256": tensor_content_sha256(measurement),
        "measurement_shape": list(measurement.shape),
        "measurement_dtype": str(measurement.dtype),
        "measurement_min": float(measurement.min().item()),
        "measurement_max": float(measurement.max().item()),
        "measurement_negative_count": int(raw_negative.sum().item()),
        "measurement_numel": int(measurement.numel()),
        "ground_truth_source_path": str(image_path),
        "ground_truth_source_sha256": sha256_file(image_path),
        "ground_truth_tensor_path": str(gt_tensor_path),
        "ground_truth_tensor_content_sha256": tensor_content_sha256(ground_truth),
        "repo_root": str(repo_root),
        "repo_branch": branch,
        "repo_head": main_head,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "sitcom_root": str(sitcom_root),
        "sitcom_head": git_head(sitcom_root),
        "sitcom_effective_patch": sitcom_patch,
        "difffpr_root": str(difffpr_root),
        "difffpr_head": git_head(difffpr_root),
        "model_path": str(model_path),
        "model_sha256": model_sha,
    }
    write_json(input_dir / "input_manifest.json", manifest)

    print(
        json.dumps(
            {
                "status": "OK",
                "image_id": image_id,
                "measurement": selected_measurement.name,
                "measurement_sha256": manifest["measurement_file_sha256"],
                "repo_head": main_head,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
