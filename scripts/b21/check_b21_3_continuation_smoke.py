#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_numeric(row: pd.Series, names: list[str]) -> float:
    for name in names:
        if name in row.index:
            value = pd.to_numeric(row[name], errors="coerce")
            if pd.notna(value):
                return float(value)
    return float("nan")


def read_exact_loss(path: Path) -> float:
    df = pd.read_csv(path)
    if df.empty:
        return float("nan")
    return pick_numeric(
        df.iloc[0],
        ["exact_operator_loss", "operator_loss", "measurement_loss", "loss"],
    )


def find_gt(image_id: str) -> Path:
    root = Path("/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024")
    image_id = f"{int(image_id):05d}"
    candidates = [
        root / "00000" / f"{image_id}.png",
        root / f"{image_id}.png",
        root / "00000" / f"{int(image_id):06d}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"ground truth not found for {image_id}; tried {candidates}")


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32)


def psnr(sample: Path, gt: Path) -> float:
    sample_image = Image.open(sample).convert("RGB")
    x = np.asarray(sample_image, dtype=np.float32)
    y = load_rgb(gt, sample_image.size)
    mse = float(np.mean((x - y) ** 2))
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def pixel_diff(a: Path, b: Path) -> tuple[float, float]:
    aa = np.asarray(Image.open(a).convert("RGB"), dtype=np.int16)
    bb = np.asarray(Image.open(b).convert("RGB"), dtype=np.int16)
    if aa.shape != bb.shape:
        raise ValueError(f"image shape mismatch: {aa.shape} vs {bb.shape}")
    diff = np.abs(aa - bb)
    return float(diff.max()), float(diff.mean())


def markdown_rows(df: pd.DataFrame) -> list[str]:
    lines = [
        "| case | exact operator loss | PSNR | good25 | sha256 |",
        "|---|---:|---:|---:|---|",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| {case} | {loss:.9g} | {psnr:.6f} | {good} | `{sha}` |".format(
                case=row["case"],
                loss=float(row["exact_operator_loss"]),
                psnr=float(row["psnr_recomputed_from_png"]),
                good=int(row["good25"]),
                sha=str(row["sha256"]),
            )
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--split-step", type=int, required=True)
    parser.add_argument("--branch-seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()

    out = args.out.resolve()
    repo = args.repo.resolve()
    save_dir = out / "daps_results"
    metric_dir = out / "metrics"
    gt = find_gt(args.image)

    case_names = ["default_full", "source_full", "cont0_same_seed"] + [
        f"branch_seed{seed}" for seed in args.branch_seeds
    ]

    rows: list[dict[str, object]] = []
    for case_name in case_names:
        sample = save_dir / case_name / "samples" / "00000_run0000.png"
        metric = metric_dir / f"{case_name}.csv"
        if not sample.exists():
            raise FileNotFoundError(sample)
        if not metric.exists():
            raise FileNotFoundError(metric)
        sample_psnr = psnr(sample, gt)
        rows.append(
            {
                "case": case_name,
                "sample_path": str(sample),
                "metric_path": str(metric),
                "sha256": sha256(sample),
                "exact_operator_loss": read_exact_loss(metric),
                "psnr_recomputed_from_png": sample_psnr,
                "good25": int(sample_psnr >= 25.0),
            }
        )

    df = pd.DataFrame(rows)
    by_case = df.set_index("case")

    source_path = Path(by_case.loc["source_full", "sample_path"])
    cont0_path = Path(by_case.loc["cont0_same_seed", "sample_path"])
    max_pixel_diff, mean_pixel_diff = pixel_diff(source_path, cont0_path)

    source_hash = str(by_case.loc["source_full", "sha256"])
    cont0_hash = str(by_case.loc["cont0_same_seed", "sha256"])
    hash_equal = source_hash == cont0_hash

    source_loss = float(by_case.loc["source_full", "exact_operator_loss"])
    cont0_loss = float(by_case.loc["cont0_same_seed", "exact_operator_loss"])
    loss_diff = abs(source_loss - cont0_loss)

    source_psnr = float(by_case.loc["source_full", "psnr_recomputed_from_png"])
    cont0_psnr = float(by_case.loc["cont0_same_seed", "psnr_recomputed_from_png"])
    psnr_diff = abs(source_psnr - cont0_psnr)

    exact_resume_pass = bool(
        hash_equal
        or (
            max_pixel_diff <= 1.0
            and loss_diff <= 1e-5
            and psnr_diff <= 1e-5
        )
    )

    branch_cases = [f"branch_seed{seed}" for seed in args.branch_seeds]
    branch_hashes = [str(by_case.loc[case, "sha256"]) for case in branch_cases]
    branch_unique_hashes = len(set(branch_hashes))
    branch_diversity_pass = branch_unique_hashes >= 2

    state_dir = save_dir / "source_full" / "continuation_states" / "run0000"
    state0 = state_dir / "step0000.pt"
    state_split = state_dir / f"step{args.split_step:04d}.pt"
    state_payloads_pass = state0.exists() and state_split.exists()
    default_off_pass = (
        save_dir / "default_full" / "samples" / "00000_run0000.png"
    ).exists()

    overall_pass = bool(
        default_off_pass
        and state_payloads_pass
        and exact_resume_pass
        and branch_diversity_pass
    )

    summary = {
        "image_id": f"{int(args.image):05d}",
        "run_seed": args.run_seed,
        "split_step": args.split_step,
        "branch_seeds": args.branch_seeds,
        "default_off_pass": default_off_pass,
        "state_payloads_pass": state_payloads_pass,
        "state0_path": str(state0),
        "state_split_path": str(state_split),
        "source_cont0_hash_equal": hash_equal,
        "source_cont0_max_uint8_diff": max_pixel_diff,
        "source_cont0_mean_uint8_diff": mean_pixel_diff,
        "source_cont0_exact_loss_abs_diff": loss_diff,
        "source_cont0_psnr_abs_diff": psnr_diff,
        "exact_resume_pass": exact_resume_pass,
        "branch_unique_hashes": branch_unique_hashes,
        "branch_total": len(branch_cases),
        "branch_diversity_pass": branch_diversity_pass,
        "overall_pass": overall_pass,
    }

    rows_path = out / "continuation_smoke_rows.csv"
    summary_path = out / "continuation_smoke_summary.json"
    report_path = repo / "docs/b21/b21_3_continuation_smoke.md"

    df.to_csv(rows_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report_lines = [
        "# B21.3 continuation smoke",
        "",
        f"- image: `{summary['image_id']}`",
        f"- run seed: `{args.run_seed}`",
        f"- split step: `{args.split_step}`",
        f"- overall pass: **{overall_pass}**",
        "",
        "## Gates",
        "",
        f"- default-off full run completed: `{default_off_pass}`",
        f"- step-0 and split payloads exist: `{state_payloads_pass}`",
        f"- continuation-from-0 pass: `{exact_resume_pass}`",
        f"- source/cont0 PNG hash equal: `{hash_equal}`",
        f"- max/mean uint8 difference: `{max_pixel_diff}` / `{mean_pixel_diff}`",
        f"- exact-loss absolute difference: `{loss_diff}`",
        f"- PSNR absolute difference: `{psnr_diff}`",
        f"- branch diversity: `{branch_unique_hashes}` unique hashes / `{len(branch_cases)}`",
        "",
        "## Candidate rows",
        "",
        *markdown_rows(df),
        "",
        f"Runtime artifacts: `{out}`",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("[write]", rows_path)
    print("[write]", summary_path)
    print("[write]", report_path)

    if not overall_pass:
        print("[FAIL] B21.3 continuation smoke did not pass all gates", file=sys.stderr)
        return 2
    print("[PASS] B21.3 continuation smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
