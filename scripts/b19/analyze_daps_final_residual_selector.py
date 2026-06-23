#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daps_root", required=True)
    ap.add_argument("--samples_dir", required=True)
    ap.add_argument("--measurement_path", required=True)
    ap.add_argument("--metrics_json", default="")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--oversample", type=float, default=2.0)
    args = ap.parse_args()

    daps_root = Path(args.daps_root).resolve()
    sys.path.insert(0, str(daps_root))

    from forward_operator import get_operator  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(args.measurement_path, map_location=device)
    y = payload["measurement"] if isinstance(payload, dict) and "measurement" in payload else payload
    y = y.to(device)

    gt = None
    if isinstance(payload, dict) and "images" in payload:
        gt = payload["images"].to(device)
        if gt.shape[0] > 1:
            gt = gt[:1]

    operator = get_operator(name="phase_retrieval", sigma=args.noise, oversample=args.oversample)

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
            pred = operator.forward(x) if hasattr(operator, "forward") else operator(x)
            rel_residual = (
                (pred - y).flatten(1).norm(dim=1)
                / (y.flatten(1).norm(dim=1) + 1e-12)
            )[0]

            try:
                loss = operator.loss(x, y)
                if torch.is_tensor(loss):
                    loss = float(loss.detach().flatten()[0].cpu().item())
            except Exception:
                loss = math.nan

            row = {
                "run_index": run_i,
                "sample_path": str(sample_path),
                "relative_measurement_residual": float(rel_residual.detach().cpu().item()),
                "operator_loss": loss,
            }

            if gt is not None:
                row["psnr_recomputed_from_png"] = psnr_model_range(x, gt)
            if run_i in metric_psnr:
                row["psnr_metrics_json"] = metric_psnr[run_i]

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("run_index")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print("ALL CANDIDATES")
    print(df.to_string(index=False))

    print("\nSELECT BY MIN RELATIVE MEASUREMENT RESIDUAL")
    best = df.loc[df["relative_measurement_residual"].idxmin()]
    print(best.to_string())

    if "psnr_metrics_json" in df:
        print("\nORACLE BY METRICS JSON")
        oracle = df.loc[df["psnr_metrics_json"].idxmax()]
        print(oracle.to_string())

    print("\nwrote", args.out_csv)


if __name__ == "__main__":
    main()
