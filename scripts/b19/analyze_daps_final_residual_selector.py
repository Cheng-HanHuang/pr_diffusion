#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_png_model_range(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB")).astype("float32") / 255.0
    x01 = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return x01 * 2.0 - 1.0


def psnr_model_range(x: torch.Tensor, gt: torch.Tensor) -> float:
    x01 = (x.clamp(-1, 1) + 1.0) / 2.0
    g01 = (gt.clamp(-1, 1) + 1.0) / 2.0
    mse = (x01 - g01).pow(2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return float((-10.0 * torch.log10(mse))[0].detach().cpu().item())


def run_index_from_name(path: Path) -> int:
    m = re.search(r"run(\d+)", path.name)
    if not m:
        raise ValueError(f"Cannot parse run index from {path}")
    return int(m.group(1))


def center_pad_to(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, _, h, w = x.shape
    if h > target_h or w > target_w:
        raise ValueError(f"Cannot pad sample shape {tuple(x.shape)} to {(target_h, target_w)}")

    pad_h = target_h - h
    pad_w = target_w - w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return torch.nn.functional.pad(x, (left, right, top, bottom), mode="constant", value=0.0)


def phase_retrieval_magnitude(x: torch.Tensor, y_shape: tuple[int, ...]) -> torch.Tensor:
    # x is model range [-1, 1]. The original operators operate on the model tensor.
    # We center-pad to match the locked measurement size, then compare Fourier magnitudes.
    target_h, target_w = int(y_shape[-2]), int(y_shape[-1])
    x_pad = center_pad_to(x, target_h, target_w)
    return torch.fft.fft2(x_pad, dim=(-2, -1)).abs()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", required=True)
    ap.add_argument("--measurement_path", required=True)
    ap.add_argument("--metrics_json", default="")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(args.measurement_path, map_location=device)
    y = payload["measurement"] if isinstance(payload, dict) and "measurement" in payload else payload
    y = y.to(device)

    gt = None
    if isinstance(payload, dict) and "images" in payload:
        gt = payload["images"].to(device)
        if gt.shape[0] > 1:
            gt = gt[:1]

    metric_psnr = {}
    if args.metrics_json and Path(args.metrics_json).exists():
        m = json.loads(Path(args.metrics_json).read_text())
        ps = m["psnr"]["sample"][0]
        metric_psnr = {i: float(v) for i, v in enumerate(ps)}

    rows = []
    for sample_path in sorted(Path(args.samples_dir).glob("*run*.png")):
        run_i = run_index_from_name(sample_path)
        x = load_png_model_range(sample_path).to(device)

        with torch.no_grad():
            pred = phase_retrieval_magnitude(x, tuple(y.shape))
            rel_residual = (
                (pred - y).flatten(1).norm(dim=1)
                / (y.flatten(1).norm(dim=1) + 1e-12)
            )[0]

            row = {
                "run_index": run_i,
                "sample_path": str(sample_path),
                "relative_measurement_residual": float(rel_residual.detach().cpu().item()),
            }

            if gt is not None:
                row["psnr_recomputed_from_png"] = psnr_model_range(x, gt)
            if run_i in metric_psnr:
                row["psnr_metrics_json"] = metric_psnr[run_i]

        rows.append(row)

    rows = sorted(rows, key=lambda r: r["run_index"])
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else [
        "run_index",
        "sample_path",
        "relative_measurement_residual",
    ]

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("ALL CANDIDATES")
    for r in rows:
        print(r)

    if rows:
        best = min(rows, key=lambda r: r["relative_measurement_residual"])
        print("\nSELECT BY MIN RELATIVE MEASUREMENT RESIDUAL")
        print(best)

    if rows and "psnr_metrics_json" in rows[0]:
        oracle = max(rows, key=lambda r: r["psnr_metrics_json"])
        print("\nORACLE BY METRICS JSON")
        print(oracle)

    print("\nwrote", out)


if __name__ == "__main__":
    main()
