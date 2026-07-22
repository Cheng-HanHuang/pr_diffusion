#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import torch


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_exact_loss(path: Path) -> float:
    frame = pd.read_csv(path)
    if frame.empty:
        return float("nan")
    row = frame.iloc[0]
    for name in ("exact_operator_loss", "operator_loss", "measurement_loss", "loss"):
        if name in row.index:
            value = pd.to_numeric(row[name], errors="coerce")
            if pd.notna(value):
                return float(value)
    return float("nan")


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32)


def psnr(sample: Path, gt: Path) -> float:
    image = Image.open(sample).convert("RGB")
    x = np.asarray(image, dtype=np.float32)
    y = load_rgb(gt, image.size)
    mse = float(np.mean((x - y) ** 2))
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def read_timings(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                result[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return result


def markdown_table(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| case | base PSNR | HIO raw PSNR | warm PSNR | base good25 | warm good25 | warm-base | cost ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| {case} | {base:.3f} | {hio:.3f} | {warm:.3f} | {bg} | {wg} | {gain:+.3f} | {ratio:.3f} |".format(
                case=int(row["case_id"]),
                base=float(row["base_psnr"]),
                hio=float(row["hio_raw_psnr"]),
                warm=float(row["warm_psnr"]),
                bg=int(row["base_good25"]),
                wg=int(row["warm_good25"]),
                gain=float(row["warm_minus_base_psnr"]),
                ratio=float(row["warm_over_base_wall_ratio"]),
            )
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--case-ids", type=int, nargs="+", required=True)
    parser.add_argument("--inject-step", type=int, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    out = args.out.resolve()
    image_id = f"{int(args.image):05d}"
    gt = Path("/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024/00000") / f"{image_id}.png"
    if not gt.exists():
        raise FileNotFoundError(gt)

    rows: list[dict[str, object]] = []
    for case_id in args.case_ids:
        case_dir = out / "cases" / f"case{case_id:02d}"
        base_png = case_dir / "daps_results/base_full/samples/00000_run0000.png"
        warm_png = case_dir / "daps_results/hio_warm/samples/00000_run0000.png"
        hio_png = case_dir / "hio/hio_raw.png"
        base_csv = case_dir / "metrics/base_full.csv"
        warm_csv = case_dir / "metrics/hio_warm.csv"
        hio_json = case_dir / "hio/hio_summary.json"
        state_path = case_dir / "hio/hio_state.pt"
        timing_path = case_dir / "timings.tsv"
        for path in (base_png, warm_png, hio_png, base_csv, warm_csv, hio_json, state_path):
            if not path.exists():
                raise FileNotFoundError(path)

        state = load_torch(state_path)
        if not isinstance(state, dict) or "x0y" not in state or "step" not in state:
            raise ValueError(f"Invalid continuation payload: {state_path}")
        x0y = state["x0y"]
        if not torch.is_tensor(x0y):
            raise TypeError(f"payload x0y is not a tensor: {state_path}")

        hio_meta = json.loads(hio_json.read_text())
        timings = read_timings(timing_path)
        base_wall = float(timings.get("base_full", float("nan")))
        warm_wall = float(timings.get("hio_warm", float("nan")))
        hio_wall = float(timings.get("hio_generate", hio_meta.get("hio_wall_seconds", float("nan"))))
        cost_ratio = (warm_wall + hio_wall) / base_wall if base_wall > 0 else float("nan")

        base_value = psnr(base_png, gt)
        warm_value = psnr(warm_png, gt)
        hio_value = psnr(hio_png, gt)
        rows.append(
            {
                "case_id": case_id,
                "base_png": str(base_png),
                "warm_png": str(warm_png),
                "hio_png": str(hio_png),
                "base_sha256": sha256(base_png),
                "warm_sha256": sha256(warm_png),
                "hio_sha256": sha256(hio_png),
                "base_exact_operator_loss": read_exact_loss(base_csv),
                "warm_exact_operator_loss": read_exact_loss(warm_csv),
                "base_psnr": base_value,
                "hio_raw_psnr": hio_value,
                "warm_psnr": warm_value,
                "base_good25": int(base_value >= 25.0),
                "warm_good25": int(warm_value >= 25.0),
                "warm_minus_base_psnr": warm_value - base_value,
                "warm_exact_loss_minus_base": read_exact_loss(warm_csv) - read_exact_loss(base_csv),
                "state_step": int(state["step"]),
                "state_shape": "x".join(str(v) for v in x0y.shape),
                "state_finite": int(torch.isfinite(x0y).all().item()),
                "state_min": float(x0y.min().item()),
                "state_max": float(x0y.max().item()),
                "base_wall_seconds": base_wall,
                "hio_wall_seconds": hio_wall,
                "warm_wall_seconds": warm_wall,
                "warm_total_wall_seconds": hio_wall + warm_wall,
                "warm_over_base_wall_ratio": cost_ratio,
                "hio_measurement_squared_loss": float(hio_meta["hio_measurement_squared_loss"]),
                "hio_sqrt_loss_over_y_norm": float(hio_meta["hio_sqrt_loss_over_y_norm"]),
            }
        )

    frame = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    state_gate = bool(
        (frame["state_finite"] == 1).all()
        and (frame["state_step"] == args.inject_step).all()
        and (frame["state_min"] >= -1.000001).all()
        and (frame["state_max"] <= 1.000001).all()
    )
    outputs_unique = int(frame["warm_sha256"].nunique())
    diversity_gate = outputs_unique >= 2
    mean_ratio = float(frame["warm_over_base_wall_ratio"].mean())
    cost_gate = bool(np.isfinite(mean_ratio) and mean_ratio <= 0.70)
    base_good = int(frame["base_good25"].sum())
    warm_good = int(frame["warm_good25"].sum())
    mean_gain = float(frame["warm_minus_base_psnr"].mean())
    quality_signal = bool(warm_good > base_good or mean_gain >= 1.0)
    complete_gate = len(frame) == len(args.case_ids)
    implementation_pass = bool(complete_gate and state_gate and diversity_gate)
    promote_to_pilot = bool(implementation_pass and cost_gate and quality_signal)

    summary = {
        "image_id": image_id,
        "inject_step": args.inject_step,
        "n_cases": len(frame),
        "complete_gate": complete_gate,
        "state_gate": state_gate,
        "warm_unique_hashes": outputs_unique,
        "diversity_gate": diversity_gate,
        "base_good25": base_good,
        "warm_good25": warm_good,
        "mean_base_psnr": float(frame["base_psnr"].mean()),
        "mean_hio_raw_psnr": float(frame["hio_raw_psnr"].mean()),
        "mean_warm_psnr": float(frame["warm_psnr"].mean()),
        "mean_warm_minus_base_psnr": mean_gain,
        "mean_warm_over_base_wall_ratio": mean_ratio,
        "cost_gate_ratio_at_most": 0.70,
        "cost_gate": cost_gate,
        "quality_signal_rule": "warm_good25 > base_good25 OR mean PSNR gain >= 1.0 dB",
        "quality_signal": quality_signal,
        "implementation_pass": implementation_pass,
        "promote_to_five_image_pilot": promote_to_pilot,
    }

    rows_path = out / "hio_warmstart_smoke_rows.csv"
    summary_path = out / "hio_warmstart_smoke_summary.json"
    report_path = repo / "docs/b21/b21_5_hio_warmstart_smoke.md"
    frame.to_csv(rows_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report = [
        "# B21.5 HIO warm-start smoke",
        "",
        f"- image: `{image_id}`",
        f"- cases: `{len(frame)}`",
        f"- injection step: `{args.inject_step}`",
        f"- implementation pass: **{implementation_pass}**",
        f"- promote to five-image pilot: **{promote_to_pilot}**",
        "",
        "The HIO generator uses only the locked measurement and known central support. Ground truth is used only by this checker for offline PSNR evaluation.",
        "",
        "## Results",
        "",
        *markdown_table(frame),
        "",
        "## Summary",
        "",
        f"- base good25: `{base_good}/{len(frame)}`",
        f"- warm good25: `{warm_good}/{len(frame)}`",
        f"- mean base PSNR: `{summary['mean_base_psnr']:.4f}`",
        f"- mean raw-HIO PSNR: `{summary['mean_hio_raw_psnr']:.4f}`",
        f"- mean warm PSNR: `{summary['mean_warm_psnr']:.4f}`",
        f"- mean warm-base PSNR: `{mean_gain:+.4f}`",
        f"- warm/base wall ratio: `{mean_ratio:.4f}`",
        f"- distinct warm hashes: `{outputs_unique}/{len(frame)}`",
        "",
        "## Gates",
        "",
        f"- state validity: `{state_gate}`",
        f"- output diversity: `{diversity_gate}`",
        f"- cost <= 0.70x base: `{cost_gate}`",
        f"- quality signal: `{quality_signal}`",
        "",
        f"Artifacts: `{out}`",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("[write]", rows_path)
    print("[write]", summary_path)
    print("[write]", report_path)
    if not implementation_pass:
        print("[FAIL] HIO warm-start smoke implementation gates failed", file=sys.stderr)
        return 2
    print("[PASS] HIO warm-start mechanism completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
