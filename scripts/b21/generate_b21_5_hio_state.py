#!/usr/bin/env python3
"""Generate a clean-free HIO warm state from a locked phase-retrieval measurement.

The generator uses only the saved Fourier-magnitude measurement and known support.
It writes a continuation payload compatible with the already validated B21.3
continuation loader. Ground truth is never read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
import torch


def first_tensor(obj: Any) -> torch.Tensor:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for key in ("measurement", "y", "Y", "data", "meas", "observed", "observation"):
            value = obj.get(key)
            if torch.is_tensor(value):
                return value
        for value in obj.values():
            if torch.is_tensor(value):
                return value
    if isinstance(obj, (list, tuple)):
        for value in obj:
            if torch.is_tensor(value):
                return value
    raise TypeError(f"Could not extract a tensor from measurement object {type(obj)}")


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def fft2c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.fftn(
            torch.fft.ifftshift(x, dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho",
        ),
        dim=(-2, -1),
    )


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(
        torch.fft.ifftn(
            torch.fft.ifftshift(x, dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho",
        ),
        dim=(-2, -1),
    )


def tensor_sha256(x: torch.Tensor) -> str:
    array = x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def save_png(x01: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x01.detach().cpu().clamp(0, 1)
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"Expected [1,C,H,W], got {tuple(x.shape)}")
    array = (x[0].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    Image.fromarray(array).save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement-path", type=Path, required=True)
    parser.add_argument("--out-state", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=240)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--er-every", type=int, default=20)
    parser.add_argument("--final-er", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--inject-step", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if not 0 < args.beta <= 1.5:
        raise ValueError("--beta must lie in (0, 1.5]")
    if args.inject_step < 0:
        raise ValueError("--inject-step must be nonnegative")
    if not args.measurement_path.exists():
        raise FileNotFoundError(args.measurement_path)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    started = time.perf_counter()
    measurement = first_tensor(load_torch(args.measurement_path)).detach().float()
    if measurement.ndim == 3:
        measurement = measurement.unsqueeze(0)
    if measurement.ndim != 4 or measurement.shape[0] != 1:
        raise ValueError(f"Expected measurement [1,C,H,W], got {tuple(measurement.shape)}")
    if measurement.shape[-2] != measurement.shape[-1]:
        raise ValueError(f"Expected square Fourier grid, got {tuple(measurement.shape[-2:])}")

    amplitude = measurement.to(device=device).clamp_min(0.0)
    _, channels, height, width = amplitude.shape
    if height < args.resolution or width < args.resolution:
        raise ValueError(
            f"Fourier grid {height}x{width} is smaller than support {args.resolution}"
        )

    top = (height - args.resolution) // 2
    left = (width - args.resolution) // 2
    bottom = top + args.resolution
    right = left + args.resolution

    support = torch.zeros((1, 1, height, width), dtype=torch.bool, device=device)
    support[..., top:bottom, left:right] = True
    support = support.expand(1, channels, height, width)

    estimate = torch.zeros_like(amplitude)
    estimate[..., top:bottom, left:right] = torch.rand(
        (1, channels, args.resolution, args.resolution),
        device=device,
        dtype=amplitude.dtype,
    )

    for iteration in range(args.iterations):
        spectrum = fft2c(estimate.to(torch.complex64))
        unit_phase = spectrum / spectrum.abs().clamp_min(1e-12)
        projected = amplitude.to(torch.complex64) * unit_phase
        spatial = ifft2c(projected).real

        er_step = args.er_every > 0 and (iteration + 1) % args.er_every == 0
        if er_step:
            estimate = torch.where(support, spatial.clamp(0.0, 1.0), torch.zeros_like(spatial))
        else:
            feasible = support & (spatial >= 0.0) & (spatial <= 1.0)
            estimate = torch.where(feasible, spatial, estimate - args.beta * spatial)
        estimate = torch.nan_to_num(estimate, nan=0.0, posinf=1.0, neginf=-1.0)

    for _ in range(max(0, args.final_er)):
        spectrum = fft2c(estimate.to(torch.complex64))
        unit_phase = spectrum / spectrum.abs().clamp_min(1e-12)
        projected = amplitude.to(torch.complex64) * unit_phase
        spatial = ifft2c(projected).real
        estimate = torch.where(support, spatial.clamp(0.0, 1.0), torch.zeros_like(spatial))

    warm01 = estimate[..., top:bottom, left:right].clamp(0.0, 1.0)
    warm_model = warm01 * 2.0 - 1.0

    padded = torch.zeros_like(amplitude)
    padded[..., top:bottom, left:right] = warm01
    predicted_amplitude = fft2c(padded.to(torch.complex64)).abs()
    squared_loss = float(((predicted_amplitude - measurement.to(device)) ** 2).sum().item())
    y_norm = float(measurement.norm().item())
    relative_residual = math.sqrt(max(squared_loss, 0.0)) / y_norm if y_norm > 0 else float("nan")

    elapsed = time.perf_counter() - started
    payload = {
        "version": 1,
        "x0y": warm_model.detach().cpu(),
        "step": int(args.inject_step),
        "sigma": None,
        "source_seed": int(args.seed),
        "measurement_path": str(args.measurement_path.resolve()),
        "method": "hio_support_positive_box",
        "iterations": int(args.iterations),
        "beta": float(args.beta),
        "er_every": int(args.er_every),
        "final_er": int(args.final_er),
        "resolution": int(args.resolution),
    }

    args.out_state.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out_state)
    save_png(warm01, args.out_png)

    summary = {
        "measurement_path": str(args.measurement_path.resolve()),
        "out_state": str(args.out_state.resolve()),
        "out_png": str(args.out_png.resolve()),
        "seed": args.seed,
        "iterations": args.iterations,
        "beta": args.beta,
        "er_every": args.er_every,
        "final_er": args.final_er,
        "inject_step": args.inject_step,
        "measurement_shape": list(measurement.shape),
        "warm_state_shape": list(warm_model.shape),
        "warm_state_min": float(warm_model.min().item()),
        "warm_state_max": float(warm_model.max().item()),
        "warm_state_mean": float(warm_model.mean().item()),
        "warm_state_finite": bool(torch.isfinite(warm_model).all().item()),
        "warm_state_sha256": tensor_sha256(warm_model),
        "hio_measurement_squared_loss": squared_loss,
        "hio_sqrt_loss_over_y_norm": relative_residual,
        "hio_wall_seconds": elapsed,
        "clean_free": True,
    }
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[write] {args.out_state}")
    print(f"[write] {args.out_png}")
    print(f"[write] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
