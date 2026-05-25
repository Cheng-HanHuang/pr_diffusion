#!/usr/bin/env python3
"""Analyze input-observable hardness properties for phase retrieval reliability.

This script joins reconstruction-trace outcomes with simple features computed
from the original input image.  The goal is to move beyond labels such as
"thin candidate pool" that are only known after reconstruction, and ask whether
there are pre-reconstruction properties that predict difficult images.

Features are deliberately lightweight and dependency-minimal:

- brightness / contrast / colorfulness;
- high-frequency energy ratio and spectral slope;
- edge energy;
- horizontal/vertical symmetry errors;
- autocorrelation peakiness proxy;
- reconstruction-trace outcome labels at thresholds 25/28/30.

These are not meant to be final semantic features.  They are a first practical
bridge toward Bayesian / PAC-Bayesian reliability modeling: an input-level risk
predictor can later be trained using these features plus trace statistics.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

Row = Dict[str, object]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"[WARN] no rows for {path}")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def fget(row: Dict[str, object], key: str, default: float = math.nan) -> float:
    try:
        v = row.get(key, "")
        if v == "" or v is None:
            return default
        return float(v)
    except Exception:
        return default


def image_id(path: str) -> str:
    m = re.search(r"(\d{5})\.png$", path)
    return m.group(1) if m else path


def resolve_image_path(data_root: Path, image_basename: str) -> Optional[Path]:
    p = Path(image_basename)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    candidates.append(data_root / image_basename)
    # Typical traces use images1024x1024/00000/00013.png while DATA_ROOT may be
    # .../images1024x1024 or the parent ffhq-dataset directory.
    parts = list(p.parts)
    if parts and parts[0] == "images1024x1024":
        candidates.append(data_root / Path(*parts[1:]))
    # Try id-based nested path.
    iid = image_id(image_basename)
    if re.fullmatch(r"\d{5}", iid):
        candidates.append(data_root / iid[:5] / f"{iid}.png")
        candidates.append(data_root / "images1024x1024" / iid[:5] / f"{iid}.png")
        candidates.append(data_root / f"{iid}.png")
    for c in candidates:
        if c.exists():
            return c
    return None


def radial_spectrum_features(gray: np.ndarray) -> Dict[str, float]:
    h, w = gray.shape
    x = gray - float(gray.mean())
    fft = np.fft.fftshift(np.fft.fft2(x))
    power = np.abs(fft) ** 2
    yy, xx = np.indices((h, w))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = rr.max()
    low = power[rr <= 0.15 * rmax].sum()
    mid = power[(rr > 0.15 * rmax) & (rr <= 0.35 * rmax)].sum()
    high = power[rr > 0.35 * rmax].sum()
    total = power.sum() + 1e-12

    # Fit log radial power slope excluding the zero/near-zero bin.
    nbins = 32
    bins = np.linspace(0, rmax, nbins + 1)
    centers = []
    vals = []
    for i in range(1, nbins):
        mask = (rr >= bins[i]) & (rr < bins[i + 1])
        if mask.any():
            v = float(power[mask].mean())
            if v > 0:
                centers.append((bins[i] + bins[i + 1]) / 2.0)
                vals.append(v)
    slope = math.nan
    if len(centers) >= 5:
        lx = np.log(np.asarray(centers) + 1e-8)
        ly = np.log(np.asarray(vals) + 1e-12)
        slope = float(np.polyfit(lx, ly, 1)[0])
    return {
        "spectral_low_frac": float(low / total),
        "spectral_mid_frac": float(mid / total),
        "spectral_high_frac": float(high / total),
        "spectral_high_low_ratio": float(high / (low + 1e-12)),
        "spectral_slope_loglog": slope,
    }


def image_features(path: Path, resize: int = 256) -> Dict[str, float]:
    im = Image.open(path).convert("RGB")
    im = im.resize((resize, resize), Image.BICUBIC)
    arr = np.asarray(im).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)

    # Gradients / edge proxy.
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    edge_energy = float(np.mean(gx ** 2) + np.mean(gy ** 2))

    # Symmetry errors.  Face images are usually roughly centered; large asymmetry
    # can indicate pose/background/hair complexity.
    hsym = float(np.mean((gray - gray[:, ::-1]) ** 2))
    vsym = float(np.mean((gray - gray[::-1, :]) ** 2))

    # Colorfulness proxy.
    rg = arr[:, :, 0] - arr[:, :, 1]
    yb = 0.5 * (arr[:, :, 0] + arr[:, :, 1]) - arr[:, :, 2]
    colorfulness = float(np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))

    # Autocorrelation peakiness proxy.  If the autocorrelation has many strong
    # side peaks, phase retrieval can be more ambiguous.
    x = gray - gray.mean()
    ac = np.fft.ifft2(np.abs(np.fft.fft2(x)) ** 2).real
    ac = np.fft.fftshift(ac)
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    center = float(ac[cy, cx]) + 1e-12
    mask = np.ones_like(ac, dtype=bool)
    rad = 5
    mask[cy - rad:cy + rad + 1, cx - rad:cx + rad + 1] = False
    side_max = float(ac[mask].max())
    side_mean_top1 = float(np.sort(ac[mask].reshape(-1))[-max(1, ac.size // 100):].mean())

    feats: Dict[str, float] = {
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "contrast_p95_p05": float(np.quantile(gray, 0.95) - np.quantile(gray, 0.05)),
        "edge_energy": edge_energy,
        "horizontal_symmetry_mse": hsym,
        "vertical_symmetry_mse": vsym,
        "colorfulness": colorfulness,
        "autocorr_side_max_ratio": side_max / center,
        "autocorr_top1pct_side_ratio": side_mean_top1 / center,
    }
    feats.update(radial_spectrum_features(gray))
    return feats


def summarize_trace(rows: List[Dict[str, str]], threshold: float, psnr_key: str, stat_key: str) -> List[Row]:
    by_img: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_img[image_id(str(r.get("image_basename", "")))].append(r)

    out: List[Row] = []
    for iid, gr in sorted(by_img.items()):
        if not gr:
            continue
        psnrs = [fget(r, psnr_key) for r in gr]
        labels = [p >= threshold for p in psnrs]
        seeds = sorted(set(str(r.get("seed", "")) for r in gr))
        seed_success = 0
        for s in seeds:
            vals = [fget(r, psnr_key) for r in gr if str(r.get("seed", "")) == s]
            if vals and max(vals) >= threshold:
                seed_success += 1
        oracle = max(psnrs) if psnrs else math.nan
        selected = min(gr, key=lambda r: fget(r, stat_key))
        selected_psnr = fget(selected, psnr_key)
        out.append({
            "image_id": iid,
            "image_basename": gr[0].get("image_basename", ""),
            "threshold_db": threshold,
            "n_candidates": len(gr),
            "n_seeds": len(seeds),
            "candidate_success_rate": float(sum(labels) / len(labels)) if labels else math.nan,
            "seed_success_rate": float(seed_success / len(seeds)) if seeds else math.nan,
            "oracle_psnr": oracle,
            "selected_min_stat_psnr": selected_psnr,
            "oracle_failure": int(oracle < threshold),
            "selector_failure_if_min_stat": int(oracle >= threshold and selected_psnr < threshold),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute input-observable hardness features joined to reliability outcomes.")
    ap.add_argument("--trace_csv", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--threshold", type=float, default=28.0)
    ap.add_argument("--psnr_key", default="raw_psnr")
    ap.add_argument("--selector_stat", default="post_winner_lf_mse_mean")
    args = ap.parse_args()

    rows = read_csv(Path(args.trace_csv))
    trace_summary = summarize_trace(rows, args.threshold, args.psnr_key, args.selector_stat)
    data_root = Path(args.data_root)
    out: List[Row] = []
    for r in trace_summary:
        path = resolve_image_path(data_root, str(r["image_basename"]))
        row: Row = dict(r)
        row["resolved_image_path"] = str(path) if path is not None else ""
        if path is not None:
            try:
                row.update(image_features(path))
            except Exception as e:
                row["feature_error"] = repr(e)
        else:
            row["feature_error"] = "image_not_found"
        out.append(row)

    write_csv(Path(args.out_csv), out)
    print(f"Wrote {len(out)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
