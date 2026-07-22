#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms


DEFAULT_IMAGE_ROOT = Path(
    "/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024"
)


def first_numeric(row: pd.Series, names: Sequence[str]) -> float:
    for name in names:
        if name not in row.index:
            continue
        value = pd.to_numeric(row[name], errors="coerce")
        if pd.notna(value):
            return float(value)
    return float("nan")


def find_gt(image_root: Path, image_id: str) -> Path:
    value = int(image_id)
    folder = f"{(value // 1000) * 1000:05d}"
    candidates = [
        image_root / folder / f"{value:05d}.png",
        image_root / "00000" / f"{value:05d}.png",
        image_root / f"{value:05d}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    hits = list(image_root.rglob(f"{value:05d}.png"))
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(
        f"ground truth image {image_id} not found under {image_root}; "
        f"candidates={candidates} hits={hits[:5]}"
    )


def load_sample_model_range(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    x01 = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return x01 * 2.0 - 1.0


def load_gt_model_range(path: Path, resolution: int = 256) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
    ])
    return (transform(Image.open(path).convert("RGB")) * 2.0 - 1.0).unsqueeze(0)


def psnr_model_range(sample: torch.Tensor, gt: torch.Tensor) -> float:
    sample01 = (sample.clamp(-1, 1) + 1.0) / 2.0
    gt01 = (gt.clamp(-1, 1) + 1.0) / 2.0
    mse = (sample01 - gt01).pow(2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return float((-10.0 * torch.log10(mse))[0].item())


def read_metric(path: Path, gt: torch.Tensor) -> tuple[float, float, str, str]:
    """Read exact loss and obtain offline PSNR without requiring GT in measurement.

    Older locked-measurement payloads included an ``images`` tensor, so the metric
    CSV contained ``psnr_recomputed_from_png``.  The fresh-validation measurement
    files intentionally contain only the measurement tensor.  In that case the
    exact-loss analyzer correctly omits PSNR, and this final offline analyzer
    recomputes it from the saved sample PNG plus the frozen FFHQ ground truth.
    """
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty metric CSV: {path}")
    row = frame.iloc[0]

    loss = first_numeric(
        row,
        [
            "exact_operator_loss",
            "selector_exact_operator_loss",
            "operator_loss",
            "loss",
        ],
    )
    if not math.isfinite(loss):
        raise ValueError(
            f"no finite exact operator loss in {path}; columns={list(frame.columns)}"
        )

    sample_path_raw = str(row.get("sample_path", "")).strip()
    if not sample_path_raw:
        raise ValueError(
            f"metric CSV lacks sample_path needed for offline PSNR: {path}; "
            f"columns={list(frame.columns)}"
        )
    sample_path = Path(sample_path_raw)
    if not sample_path.exists():
        raise FileNotFoundError(
            f"sample_path recorded in {path} does not exist: {sample_path}"
        )

    psnr = first_numeric(
        row,
        [
            "psnr_recomputed_from_png",
            "selector_psnr_recomputed_from_png",
            "psnr_metrics_json",
            "psnr",
        ],
    )
    if math.isfinite(psnr):
        source = "metric_csv"
    else:
        sample = load_sample_model_range(sample_path)
        if tuple(sample.shape) != tuple(gt.shape):
            raise ValueError(
                f"sample/GT shape mismatch for {sample_path}: "
                f"sample={tuple(sample.shape)} gt={tuple(gt.shape)}"
            )
        psnr = psnr_model_range(sample, gt)
        source = "offline_png_vs_frozen_gt"

    return loss, psnr, source, str(sample_path)


def read_timings(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        key, value = line.split("\t", 1)
        result[key] = float(value)
    return result


def exact_mcnemar_p(rescues: int, harms: int) -> float:
    n = rescues + harms
    if n == 0:
        return 1.0
    k = min(rescues, harms)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def exact_sign_p(positive: int, negative: int) -> float:
    n = positive + negative
    if n == 0:
        return 1.0
    k = min(positive, negative)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def summarize(group: pd.DataFrame, name: str) -> dict[str, object]:
    n = len(group)
    base_good = int(group.base_good25.sum())
    lf_good = int(group.lf_good25.sum())
    hio_good = int(group.hio_good25.sum())
    base_lf_good = int(group.base_lf_good25.sum())
    three_good = int(group.three_arm_good25.sum())
    rescues = int(group.incremental_hio_rescue.sum())
    harms = int(group.incremental_hio_harm.sum())
    oracle_base_lf = int(group.oracle_base_lf_good25.sum())
    oracle_three = int(group.oracle_three_arm_good25.sum())
    oracle_incremental = int(group.oracle_incremental_hio.sum())
    selector_capture = rescues / oracle_incremental if oracle_incremental > 0 else 1.0
    base_wall = float(group.base_wall_seconds.sum())
    lf_wall = float(group.lf_wall_seconds.sum())
    hio_wall = float(group.hio_total_wall_seconds.sum())
    return {
        "group": name,
        "n": n,
        "base_good25": base_good,
        "lf_good25": lf_good,
        "hio_good25": hio_good,
        "base_lf_gated_good25": base_lf_good,
        "three_arm_gated_good25": three_good,
        "three_arm_net_vs_base_lf": three_good - base_lf_good,
        "three_arm_net_vs_base": three_good - base_good,
        "incremental_hio_rescues": rescues,
        "incremental_hio_harms": harms,
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(rescues, harms),
        "accepted_lf": int(group.select_lf.sum()),
        "accepted_hio_after_lf": int(group.select_hio_after_lf.sum()),
        "oracle_base_lf_good25": oracle_base_lf,
        "oracle_three_arm_good25": oracle_three,
        "oracle_incremental_hio": oracle_incremental,
        "selector_capture_fraction_of_oracle_incremental_hio": selector_capture,
        "mean_base_psnr": float(group.base_psnr.mean()),
        "mean_base_lf_psnr": float(group.base_lf_selected_psnr.mean()),
        "mean_three_arm_psnr": float(group.three_arm_selected_psnr.mean()),
        "mean_three_minus_base_lf_psnr": float(group.three_minus_base_lf_psnr.mean()),
        "median_three_minus_base_lf_psnr": float(group.three_minus_base_lf_psnr.median()),
        "mean_lf_wall_over_base": lf_wall / base_wall if base_wall > 0 else float("nan"),
        "mean_hio_wall_over_base": hio_wall / base_wall if base_wall > 0 else float("nan"),
        "three_arm_total_cost_over_base": (
            (base_wall + lf_wall + hio_wall) / base_wall
            if base_wall > 0
            else float("nan")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--theta", type=float, default=0.7)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    args = parser.parse_args()

    repo = args.repo.resolve()
    out = args.out.resolve()
    image_root = args.image_root.resolve()
    manifest = pd.read_csv(args.manifest, sep="\t", dtype={"image_id": str})
    manifest["image_id"] = manifest.image_id.map(lambda x: f"{int(x):05d}")

    gt_cache: dict[str, tuple[Path, torch.Tensor]] = {}
    rows: list[dict[str, object]] = []
    for rec in manifest.itertuples(index=False):
        case_dir = out / "cases" / f"{rec.image_id}_case{int(rec.case_id):02d}"
        if rec.image_id not in gt_cache:
            gt_path = find_gt(image_root, rec.image_id)
            gt_cache[rec.image_id] = (gt_path, load_gt_model_range(gt_path))
        gt_path, gt = gt_cache[rec.image_id]

        base_loss, base_psnr, base_psnr_source, base_sample = read_metric(
            case_dir / "metrics" / "base_full.csv", gt
        )
        lf_loss, lf_psnr, lf_psnr_source, lf_sample = read_metric(
            case_dir / "metrics" / "lf050.csv", gt
        )
        hio_loss, hio_psnr, hio_psnr_source, hio_sample = read_metric(
            case_dir / "metrics" / "hio_warm.csv", gt
        )
        timing = read_timings(case_dir / "timings.tsv")
        hio_summary = json.loads((case_dir / "hio" / "hio_summary.json").read_text())

        state_valid = bool(
            hio_summary.get("warm_state_finite")
            and hio_summary.get("inject_step") == int(rec.inject_step)
            and hio_summary.get("warm_state_shape") == [1, 3, 256, 256]
            and float(hio_summary.get("warm_state_min")) >= -1.00001
            and float(hio_summary.get("warm_state_max")) <= 1.00001
        )

        select_lf = int(lf_loss < base_loss - args.theta)
        current_variant = "lf050" if select_lf else "base_full"
        current_loss = lf_loss if select_lf else base_loss
        current_psnr = lf_psnr if select_lf else base_psnr
        select_hio = int(hio_loss < current_loss - args.theta)
        final_variant = "hio_warm" if select_hio else current_variant
        final_loss = hio_loss if select_hio else current_loss
        final_psnr = hio_psnr if select_hio else current_psnr

        base_good = int(base_psnr >= 25.0)
        lf_good = int(lf_psnr >= 25.0)
        hio_good = int(hio_psnr >= 25.0)
        current_good = int(current_psnr >= 25.0)
        final_good = int(final_psnr >= 25.0)
        oracle_base_lf = int(max(base_psnr, lf_psnr) >= 25.0)
        oracle_three = int(max(base_psnr, lf_psnr, hio_psnr) >= 25.0)
        oracle_incremental_hio = int(not oracle_base_lf and hio_good)

        rows.append({
            "job_id": int(rec.job_id),
            "image_id": rec.image_id,
            "case_id": int(rec.case_id),
            "gpu": str(rec.gpu),
            "base_seed": int(rec.base_seed),
            "hio_seed": int(rec.hio_seed),
            "warm_noise_seed": int(rec.warm_noise_seed),
            "measurement_path": str(rec.measurement_path),
            "gt_path": str(gt_path),
            "base_sample_path": base_sample,
            "lf_sample_path": lf_sample,
            "hio_sample_path": hio_sample,
            "base_psnr_source": base_psnr_source,
            "lf_psnr_source": lf_psnr_source,
            "hio_psnr_source": hio_psnr_source,
            "base_exact_operator_loss": base_loss,
            "lf_exact_operator_loss": lf_loss,
            "hio_exact_operator_loss": hio_loss,
            "base_psnr": base_psnr,
            "lf_psnr": lf_psnr,
            "hio_psnr": hio_psnr,
            "base_good25": base_good,
            "lf_good25": lf_good,
            "hio_good25": hio_good,
            "select_lf": select_lf,
            "base_lf_selected_variant": current_variant,
            "base_lf_selected_loss": current_loss,
            "base_lf_selected_psnr": current_psnr,
            "base_lf_good25": current_good,
            "select_hio_after_lf": select_hio,
            "three_arm_selected_variant": final_variant,
            "three_arm_selected_loss": final_loss,
            "three_arm_selected_psnr": final_psnr,
            "three_arm_good25": final_good,
            "three_minus_base_lf_psnr": final_psnr - current_psnr,
            "incremental_hio_rescue": int(not current_good and final_good),
            "incremental_hio_harm": int(current_good and not final_good),
            "oracle_base_lf_good25": oracle_base_lf,
            "oracle_three_arm_good25": oracle_three,
            "oracle_incremental_hio": oracle_incremental_hio,
            "state_valid": int(state_valid),
            "base_wall_seconds": timing.get("base_full", float("nan")),
            "lf_wall_seconds": timing.get("lf050", float("nan")),
            "hio_generate_wall_seconds": timing.get("hio_generate", float("nan")),
            "hio_warm_wall_seconds": timing.get("hio_warm", float("nan")),
            "hio_total_wall_seconds": (
                timing.get("hio_generate", float("nan"))
                + timing.get("hio_warm", float("nan"))
            ),
            "hio_relative_residual": float(
                hio_summary["hio_sqrt_loss_over_y_norm"]
            ),
            "hio_state_sha256": hio_summary["warm_state_sha256"],
        })

    frame = pd.DataFrame(rows).sort_values(["image_id", "case_id"]).reset_index(drop=True)
    by_image = [
        summarize(group, image)
        for image, group in frame.groupby("image_id", sort=True)
    ]
    overall = summarize(frame, "ALL")
    image_summary = pd.DataFrame(by_image)

    positive_images = int((image_summary.three_arm_net_vs_base_lf > 0).sum())
    negative_images = int((image_summary.three_arm_net_vs_base_lf < 0).sum())
    zero_images = int((image_summary.three_arm_net_vs_base_lf == 0).sum())
    image_sign_p = exact_sign_p(positive_images, negative_images)

    complete_gate = bool(
        len(frame) == len(manifest) == 80
        and frame.state_valid.eq(1).all()
        and frame[["base_psnr", "lf_psnr", "hio_psnr"]]
        .apply(np.isfinite)
        .all()
        .all()
    )
    net_gate = int(overall["three_arm_net_vs_base_lf"]) >= 6
    harm_gate = int(overall["incremental_hio_harms"]) <= 2
    image_spread_gate = positive_images >= 5 and negative_images <= 2
    capture_gate = (
        float(overall["selector_capture_fraction_of_oracle_incremental_hio"])
        >= 0.80
    )
    marginal_hio_gate = float(overall["mean_hio_wall_over_base"]) <= 0.70
    total_cost_gate = float(overall["three_arm_total_cost_over_base"]) <= 1.75
    fresh_validation_pass = bool(
        complete_gate
        and net_gate
        and harm_gate
        and image_spread_gate
        and capture_gate
        and marginal_hio_gate
        and total_cost_gate
    )

    verdict = {
        "question": "Does frozen base+LF050+HIO generalize to fresh official FFHQ validation-split images and new measurements/seeds?",
        "selection_rule": "select LF iff loss_lf < loss_base - 0.7; then HIO iff loss_hio < loss_current - 0.7",
        "frozen_theta": args.theta,
        "expected_cases": 80,
        "complete_cases": len(frame),
        "official_validation_split": "FFHQ ids 60000--69999",
        "psnr_evaluation": "offline saved PNG versus frozen FFHQ ground truth using DAPS preprocessing; selection remains exact-loss-only",
        "overall": overall,
        "positive_image_count": positive_images,
        "negative_image_count": negative_images,
        "zero_image_count": zero_images,
        "image_level_exact_sign_test_two_sided_p": image_sign_p,
        "gates": {
            "complete_80_valid_states_and_finite_psnr": complete_gate,
            "incremental_net_at_least_6": net_gate,
            "incremental_harms_at_most_2": harm_gate,
            "positive_images_at_least_5_and_negative_at_most_2": image_spread_gate,
            "selector_captures_at_least_80pct_oracle_incremental_hio": capture_gate,
            "marginal_hio_cost_at_most_0_70": marginal_hio_gate,
            "total_cost_at_most_1_75": total_cost_gate,
        },
        "fresh_validation_pass": fresh_validation_pass,
        "next_if_pass": "freeze three-arm policy; run larger disjoint official-validation benchmark without retuning",
        "next_if_fail": "retire HIO arm; retain frozen base+LF portfolio",
    }

    analysis_dir = out / f"fresh_analysis_theta{args.theta}"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    rows_path = analysis_dir / "fresh_three_arm_rows.csv"
    summary_path = analysis_dir / "fresh_three_arm_summary_by_image.csv"
    verdict_path = analysis_dir / "fresh_three_arm_verdict.json"
    frame.to_csv(rows_path, index=False)
    image_summary.to_csv(summary_path, index=False)
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")

    lines = [
        "# B21.5 fresh three-arm validation",
        "",
        f"- official FFHQ validation-split images: `{len(image_summary)}`",
        f"- paired cases: `{len(frame)}`",
        f"- frozen margin: `{args.theta}`",
        f"- fresh validation pass: **{fresh_validation_pass}**",
        "- PSNR: offline saved PNG versus frozen FFHQ ground truth; selection remains exact-loss-only.",
        "",
        "## By image",
        "",
        "| image | n | base+LF | three-arm | net | HIO rescues | HIO harms | accepted LF | accepted HIO | oracle incremental HIO | selector capture | total cost/base |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_image:
        lines.append(
            f"| `{row['group']}` | {row['n']} | "
            f"{row['base_lf_gated_good25']} | {row['three_arm_gated_good25']} | "
            f"{int(row['three_arm_net_vs_base_lf']):+d} | "
            f"{row['incremental_hio_rescues']} | {row['incremental_hio_harms']} | "
            f"{row['accepted_lf']} | {row['accepted_hio_after_lf']} | "
            f"{row['oracle_incremental_hio']} | "
            f"{float(row['selector_capture_fraction_of_oracle_incremental_hio']):.3f} | "
            f"{float(row['three_arm_total_cost_over_base']):.3f} |"
        )
    lines += [
        "",
        "## Overall",
        "",
        f"- base+LF gated good25: `{overall['base_lf_gated_good25']}/80`",
        f"- three-arm gated good25: `{overall['three_arm_gated_good25']}/80`",
        f"- incremental HIO net: `{int(overall['three_arm_net_vs_base_lf']):+d}`",
        f"- incremental rescues / harms: `{overall['incremental_hio_rescues']} / {overall['incremental_hio_harms']}`",
        f"- positive / zero / negative images: `{positive_images} / {zero_images} / {negative_images}`",
        f"- case-level McNemar p: `{overall['mcnemar_exact_two_sided_p']}`",
        f"- image-level sign-test p: `{image_sign_p}`",
        f"- selector capture of oracle incremental HIO: `{overall['selector_capture_fraction_of_oracle_incremental_hio']}`",
        f"- total cost/base: `{overall['three_arm_total_cost_over_base']}`",
        "",
        "## Frozen gates",
        "",
    ]
    for key, value in verdict["gates"].items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", f"Artifacts: `{analysis_dir}`", ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))

    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"[write] {rows_path}")
    print(f"[write] {summary_path}")
    print(f"[write] {verdict_path}")
    print(f"[write] {args.report}")
    return 0 if complete_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
