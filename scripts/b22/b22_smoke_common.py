#!/usr/bin/env python3
"""Shared, dependency-light utilities for the B22.1 fixed-baseline smoke."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image


MODEL_RANGE_SHAPE = (1, 3, 256, 256)
MEASUREMENT_SHAPE = (1, 3, 384, 384)


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def tensor_content_sha256(tensor: torch.Tensor) -> str:
    """Stable hash over dtype, shape, and contiguous CPU tensor bytes."""
    value = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _unwrap_tensor(payload: Any, accepted_keys: Sequence[str]) -> torch.Tensor:
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict):
        for key in accepted_keys:
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(
        "Expected a tensor or a dictionary containing one of "
        f"{list(accepted_keys)}; received {type(payload).__name__}"
    )


def load_measurement(path: str | Path, device: torch.device | str = "cpu") -> torch.Tensor:
    payload = torch.load(str(path), map_location=device)
    value = _unwrap_tensor(payload, ("measurement", "y", "observation"))
    if tuple(value.shape) != MEASUREMENT_SHAPE:
        raise ValueError(
            f"Locked measurement has shape {tuple(value.shape)}, expected {MEASUREMENT_SHAPE}"
        )
    if value.dtype != torch.float32:
        raise TypeError(f"Locked measurement dtype is {value.dtype}, expected torch.float32")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Locked measurement contains a NaN or infinity")
    return value


def load_ground_truth(path: str | Path, device: torch.device | str = "cpu") -> torch.Tensor:
    payload = torch.load(str(path), map_location=device)
    value = _unwrap_tensor(payload, ("ground_truth", "image", "x"))
    if tuple(value.shape) != MODEL_RANGE_SHAPE:
        raise ValueError(
            f"Ground-truth tensor has shape {tuple(value.shape)}, expected {MODEL_RANGE_SHAPE}"
        )
    value = value.to(dtype=torch.float32)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Ground-truth tensor contains a NaN or infinity")
    return value


def validate_reconstruction(tensor: torch.Tensor) -> None:
    if tuple(tensor.shape) != MODEL_RANGE_SHAPE:
        raise ValueError(
            f"Reconstruction has shape {tuple(tensor.shape)}, expected {MODEL_RANGE_SHAPE}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("Reconstruction contains a NaN or infinity")


def psnr01_from_model_range(x: torch.Tensor, gt: torch.Tensor) -> float:
    validate_reconstruction(x)
    validate_reconstruction(gt)
    x01 = (x.clamp(-1, 1) + 1.0) * 0.5
    gt01 = (gt.clamp(-1, 1) + 1.0) * 0.5
    mse = torch.mean((x01 - gt01).square()).clamp_min(1.0e-12)
    return float((10.0 * torch.log10(1.0 / mse)).detach().cpu().item())


def metric_pair(x: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    raw = psnr01_from_model_range(x, gt)
    rotated = psnr01_from_model_range(torch.rot90(x, 2, dims=(-2, -1)), gt)
    return {
        "psnr_raw": raw,
        "psnr_rot180": rotated,
        "psnr_ambiguity_aware": max(raw, rotated),
    }


def save_model_range_png(x: torch.Tensor, path: str | Path) -> None:
    validate_reconstruction(x)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = (
        ((x.detach().cpu().clamp(-1, 1) + 1.0) * 0.5)[0]
        .permute(1, 2, 0)
        .numpy()
    )
    arr = np.rint(arr * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(out)


def git_output(repo: str | Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        stderr=subprocess.STDOUT,
        text=True,
    ).strip()


def git_head(repo: str | Path) -> str:
    return git_output(repo, "rev-parse", "HEAD")


def git_branch(repo: str | Path) -> str:
    return git_output(repo, "branch", "--show-current")


def package_versions(names: Iterable[str]) -> Dict[str, str]:
    result: Dict[str, str] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
    }
    for name in names:
        try:
            module = __import__(name)
            result[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            result[name] = f"IMPORT_ERROR: {type(exc).__name__}: {exc}"
    return result


def cuda_memory_snapshot(device: torch.device) -> Dict[str, int]:
    if device.type != "cuda":
        return {
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "final_allocated_bytes": 0,
            "final_reserved_bytes": 0,
        }
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "final_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "final_reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def require_exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise RuntimeError(f"{label} mismatch: observed {value!r}, expected {expected!r}")


def require_close(value: float, expected: float, label: str, tolerance: float = 1.0e-7) -> None:
    if not math.isclose(float(value), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise RuntimeError(
            f"{label} mismatch: observed {value!r}, expected {expected!r}, "
            f"tolerance={tolerance}"
        )


def ensure_output_directory(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def fail_if_exists(path: str | Path) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {target}")


def environment_cuda_visible_devices() -> str:
    return os.environ.get("CUDA_VISIBLE_DEVICES", "")
