#!/usr/bin/env python3
"""Evaluate one B24 baseline image in a common saved-RGB representation.

Both DAPS and SITCOM are evaluated from their canonical saved 8-bit RGB terminal
PNGs against the same quantized 256x256 ground truth, raw orientation only.
PSNR chooses best-of-four. SSIM and LPIPS are reporting/sensitivity metrics only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from prdiffusion.b24_protocol import classify_good25  # noqa: E402


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_gt(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    value = payload.get("ground_truth") if isinstance(payload, dict) else payload
    if not torch.is_tensor(value) or tuple(value.shape) != (1, 3, 256, 256):
        raise RuntimeError(f"bad GT payload: {path}")
    return value.float().clamp(-1, 1)


def quantized_gt01(gt: torch.Tensor) -> torch.Tensor:
    x01 = (gt + 1.0) * 0.5
    q = torch.round(x01 * 255.0).clamp(0, 255) / 255.0
    return q


def load_png01(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if arr.shape != (256, 256, 3):
        raise RuntimeError(f"unexpected terminal PNG shape {arr.shape}: {path}")
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def psnr01(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = torch.mean((x - y).square()).clamp_min(1.0e-12)
    return float((10.0 * torch.log10(1.0 / mse)).item())


def ssim01(x: torch.Tensor, y: torch.Tensor) -> float:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, kernel_size=11, stride=1, padding=5)
    mu_y = F.avg_pool2d(y, kernel_size=11, stride=1, padding=5)
    sigma_x = F.avg_pool2d(x * x, 11, 1, 5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 11, 1, 5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 11, 1, 5) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return float((num / den.clamp_min(1e-12)).mean().item())


def terminal_png(method: str, terminal_path: str) -> Path:
    terminal = Path(terminal_path)
    if method == "DAPS":
        return terminal
    if terminal.name != "reconstruction.pt":
        raise RuntimeError(f"unexpected SITCOM terminal path: {terminal}")
    return terminal.with_name("reconstruction.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-id", required=True)
    ap.add_argument("--ground-truth", type=Path, required=True)
    ap.add_argument("--daps-group", type=Path, required=True)
    ap.add_argument("--sitcom-group", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    image_id = f"{int(args.image_id):05d}"
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    if not torch.cuda.is_available():
        raise RuntimeError("B24 LPIPS evaluation requires the assigned CUDA GPU")
    device = torch.device("cuda:0")
    try:
        import lpips  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"LPIPS dependency unavailable: {exc}") from exc
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    gt01 = quantized_gt01(load_gt(args.ground_truth))
    gt_lp = (gt01 * 2.0 - 1.0).to(device)
    rows = []
    by_method = {}
    for method, group_path in (("DAPS", args.daps_group), ("SITCOM", args.sitcom_group)):
        group = read_json(group_path)
        candidates = group.get("candidate_rows", [])
        if len(candidates) != 4:
            raise RuntimeError(f"{method} expected four terminal candidates, got {len(candidates)}")
        method_rows = []
        for cand in sorted(candidates, key=lambda r: int(r["rep"])):
            png = terminal_png(method, cand["terminal_path"])
            if not png.is_file():
                raise FileNotFoundError(png)
            rec01 = load_png01(png)
            with torch.no_grad():
                lp = float(lpips_model((rec01 * 2.0 - 1.0).to(device), gt_lp).mean().cpu().item())
            row = {
                "image_id": image_id,
                "method": method,
                "rep": int(cand["rep"]),
                "canonical_seed": int(cand["canonical_seed"]),
                "native_seed": int(cand["native_seed"]),
                "terminal_png": str(png.resolve()),
                "terminal_content_sha256": cand["terminal_content_sha256"],
                "psnr_raw_rgb_db": psnr01(rec01, gt01),
                "ssim_raw_rgb": ssim01(rec01, gt01),
                "lpips_alex_raw_rgb": lp,
                "gpu_active_seconds": float(cand["gpu_active_seconds"]),
                "wall_seconds": float(cand["wall_seconds"]),
                "peak_allocated_mib": float(cand["peak_allocated_mib"]),
                "peak_reserved_mib": float(cand["peak_reserved_mib"]),
            }
            method_rows.append(row)
            rows.append(row)
        best = max(method_rows, key=lambda r: (r["psnr_raw_rgb_db"], -r["rep"]))
        by_method[method] = {
            "best_rep": best["rep"],
            "best_psnr_raw_rgb_db": best["psnr_raw_rgb_db"],
            "best_ssim_raw_rgb": best["ssim_raw_rgb"],
            "best_lpips_alex_raw_rgb": best["lpips_alex_raw_rgb"],
            "good25": best["psnr_raw_rgb_db"] >= 25.0,
            "good26": best["psnr_raw_rgb_db"] >= 26.0,
            "good28": best["psnr_raw_rgb_db"] >= 28.0,
        }

    klass = classify_good25(
        by_method["DAPS"]["best_psnr_raw_rgb_db"],
        by_method["SITCOM"]["best_psnr_raw_rgb_db"],
    )
    payload = {
        "schema_version": "b24.baseline-image-metrics.v1",
        "image_id": image_id,
        "evaluation_representation": "CANONICAL_SAVED_RGB_8BIT_RAW_ORIENTATION_V1",
        "best_of_four_selection_metric": "psnr_raw_rgb_db",
        "class_threshold_db": 25.0,
        "class_label": klass,
        "methods": by_method,
        "terminal_rows": rows,
        "lpips_model": "alex",
    }
    out_json = args.output / "METRICS.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output / "terminal_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({
        "status": "PASS", "image_id": image_id, "class": klass,
        "daps_best_psnr": by_method["DAPS"]["best_psnr_raw_rgb_db"],
        "sitcom_best_psnr": by_method["SITCOM"]["best_psnr_raw_rgb_db"],
        "metrics": str(out_json.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
