#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms

DEFAULT_IMAGE_ROOT = Path(
    "/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024"
)
EXCLUDED = {
    "62802", "63282", "63803", "65808", "65960", "66452", "66892", "68263", "68924", "69293",
    "60067", "62957", "63135", "63199", "63319", "63368", "63678", "64050", "64116", "64471",
    "64542", "65317", "65656", "66511", "66731", "67092", "67673", "68111", "68922", "69441",
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_gt(root: Path, image_id: str) -> Path:
    value = int(image_id)
    folder = f"{(value // 1000) * 1000:05d}"
    candidates = [
        root / folder / f"{value:05d}.png",
        root / "00000" / f"{value:05d}.png",
        root / f"{value:05d}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    hits = list(root.rglob(f"{value:05d}.png"))
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(f"cannot resolve FFHQ image {image_id} under {root}")


def load_gt(path: Path, resolution: int = 256) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
    ])
    return (transform(Image.open(path).convert("RGB")) * 2.0 - 1.0).unsqueeze(0)


def load_sample(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0


def psnr_model_range(sample: torch.Tensor, gt: torch.Tensor) -> float:
    sample01 = (sample.clamp(-1, 1) + 1.0) / 2.0
    gt01 = (gt.clamp(-1, 1) + 1.0) / 2.0
    mse = (sample01 - gt01).pow(2).flatten(1).mean(1).clamp_min(1e-12)
    return float((-10.0 * torch.log10(mse))[0].item())


def read_metric(path: Path, gt: torch.Tensor) -> tuple[float, float, str]:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"expected exactly one metric row in {path}, got {len(frame)}")
    row = frame.iloc[0]
    loss = float(pd.to_numeric(row["exact_operator_loss"], errors="raise"))
    sample_path = Path(str(row["sample_path"]))
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)
    psnr = psnr_model_range(load_sample(sample_path), gt)
    if not math.isfinite(loss) or not math.isfinite(psnr):
        raise ValueError(f"nonfinite metric for {path}")
    return loss, psnr, str(sample_path.resolve())


def read_timings(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    result: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        key, value = line.split("\t", 1)
        result[key] = float(value)
    return result


def disagreement(first_path: str, second_path: str) -> tuple[float, float]:
    first = np.asarray(Image.open(first_path).convert("RGB"), dtype=np.float32) / 255.0
    second = np.asarray(Image.open(second_path).convert("RGB"), dtype=np.float32) / 255.0
    candidates = [second, np.rot90(second, 2, axes=(0, 1))]
    mse = min(float(np.mean((first - candidate) ** 2)) for candidate in candidates)
    l1 = min(float(np.mean(np.abs(first - candidate))) for candidate in candidates)
    return float(math.sqrt(max(mse, 0.0))), l1


def bootstrap_rate_interval(successes: int, n: int, seed: int = 5401, reps: int = 100_000) -> dict[str, float | int | str]:
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    rng = np.random.default_rng(seed)
    draws = rng.binomial(n, p, size=reps).astype(np.float64) / n
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "method": "deterministic percentile bootstrap of Bernoulli image units",
        "seed": seed,
        "repetitions": reps,
        "estimate": p,
        "lower_95": float(low),
        "upper_95": float(high),
    }


def series_summary(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="raise")
    quantiles = values.quantile([0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0])
    return {
        "min": float(quantiles.loc[0.0]),
        "q05": float(quantiles.loc[0.05]),
        "q10": float(quantiles.loc[0.10]),
        "q25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.50]),
        "mean": float(values.mean()),
        "q75": float(quantiles.loc[0.75]),
        "q90": float(quantiles.loc[0.90]),
        "q95": float(quantiles.loc[0.95]),
        "max": float(quantiles.loc[1.0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--panel-checksum", type=Path, required=True)
    parser.add_argument("--measurement-manifest", type=Path, required=True)
    parser.add_argument("--theta", type=float, default=0.7)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if abs(args.theta - 0.7) > 1e-12:
        raise ValueError("B21.11 theta is frozen to 0.7")

    out = args.out.resolve()
    manifest = pd.read_csv(args.manifest, sep="\t", dtype={"image_id": str})
    panel = pd.read_csv(args.panel_manifest, sep="\t", dtype={"image_id": str})
    manifest["image_id"] = manifest.image_id.map(lambda value: f"{int(value):05d}")
    panel["image_id"] = panel.image_id.map(lambda value: f"{int(value):05d}")

    if len(manifest) != 100 or len(panel) != 100:
        raise ValueError(f"expected 100 execution/panel rows, got {len(manifest)} and {len(panel)}")
    if manifest.image_id.nunique() != 100 or panel.image_id.nunique() != 100:
        raise ValueError("panel images are not distinct")
    overlap = sorted(set(panel.image_id) & EXCLUDED)
    if overlap:
        raise ValueError(f"excluded images appear in panel: {overlap}")

    checksum_tokens = args.panel_checksum.read_text().strip().split()
    if not checksum_tokens:
        raise ValueError("empty panel checksum file")
    expected_checksum = checksum_tokens[0]
    actual_checksum = file_sha256(args.panel_manifest)
    if actual_checksum != expected_checksum:
        raise ValueError(f"panel checksum mismatch: expected {expected_checksum}, actual {actual_checksum}")

    panel_compare = panel[["row_id", "image_id", "seed1", "seed2"]].copy()
    execution_compare = manifest[["row_id", "image_id", "seed1", "seed2"]].copy()
    for column in ["row_id", "seed1", "seed2"]:
        panel_compare[column] = pd.to_numeric(panel_compare[column], errors="raise").astype(int)
        execution_compare[column] = pd.to_numeric(execution_compare[column], errors="raise").astype(int)
    if not panel_compare.equals(execution_compare):
        raise ValueError("execution manifest does not exactly match frozen panel rows/seeds")

    expected_rows = np.arange(100)
    if not np.array_equal(panel_compare.row_id.to_numpy(), expected_rows):
        raise ValueError("row_id must be exactly 0..99 in frozen panel order")
    if not np.array_equal(panel_compare.seed1.to_numpy(), 22000 + expected_rows):
        raise ValueError("trajectory-1 seeds do not match 22000+i")
    if not np.array_equal(panel_compare.seed2.to_numpy(), 23000 + expected_rows):
        raise ValueError("trajectory-2 seeds do not match 23000+i")

    measurement_payload = json.loads(args.measurement_manifest.read_text())
    measurement_rows = measurement_payload.get("rows", [])
    if len(measurement_rows) != 100:
        raise ValueError(f"expected 100 measurement manifest rows, got {len(measurement_rows)}")
    measurement_paths = [str(Path(row["measurement_path"]).resolve()) for row in measurement_rows]
    measurement_hashes = [str(row["sha256"]) for row in measurement_rows]
    if len(set(measurement_paths)) != 100:
        raise ValueError("measurement files are not distinct")
    if len(set(measurement_hashes)) != 100:
        raise ValueError("measurement tensor hashes are not distinct")
    for path in measurement_paths:
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    manifest_measurements = [str(Path(path).resolve()) for path in manifest.measurement_path]
    if set(manifest_measurements) != set(measurement_paths):
        raise ValueError("execution and measurement manifests disagree")

    gt_cache: dict[str, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    for rec in manifest.itertuples(index=False):
        row_id = int(rec.row_id)
        image_id = f"{int(rec.image_id):05d}"
        case_dir = out / "cases" / f"{image_id}_row{row_id:03d}"
        if image_id not in gt_cache:
            gt_cache[image_id] = load_gt(find_gt(args.image_root.resolve(), image_id))
        gt = gt_cache[image_id]

        loss1, psnr1, sample1 = read_metric(case_dir / "metrics" / "base_full.csv", gt)
        loss2, psnr2, sample2 = read_metric(case_dir / "metrics" / "base_extra.csv", gt)
        timings = read_timings(case_dir / "timings.tsv")
        if "base_full" not in timings or "base_extra" not in timings:
            raise KeyError(f"missing candidate timing in {case_dir / 'timings.tsv'}")
        wall1 = float(timings["base_full"])
        wall2 = float(timings["base_extra"])
        if not math.isfinite(wall1) or not math.isfinite(wall2) or wall1 <= 0 or wall2 <= 0:
            raise ValueError(f"invalid timings for row {row_id}")

        accepted = int(loss2 < loss1 - args.theta)
        selected_idx = 1 if accepted else 0
        psnrs = [psnr1, psnr2]
        losses = [loss1, loss2]
        names = ["base_full", "base_extra"]
        selected_psnr = psnrs[selected_idx]
        selected_loss = losses[selected_idx]
        selected_good = int(selected_psnr >= 25.0)
        fresh1_good = int(psnr1 >= 25.0)
        arm2_good = int(psnr2 >= 25.0)
        oracle_good = int(max(psnrs) >= 25.0)
        rescue = int(not fresh1_good and selected_good)
        harm = int(fresh1_good and not selected_good)
        oracle_gap = int(oracle_good - selected_good)
        rmse, l1 = disagreement(sample1, sample2)

        rows.append({
            "row_id": row_id,
            "image_id": image_id,
            "gpu": str(rec.gpu),
            "seed1": int(rec.seed1),
            "seed2": int(rec.seed2),
            "measurement_path": str(Path(rec.measurement_path).resolve()),
            "execution_order": str(rec.execution_order),
            "arm1_name": "base_full",
            "arm1_exact_operator_loss": loss1,
            "arm1_psnr": psnr1,
            "arm1_good25": fresh1_good,
            "arm1_sample_path": sample1,
            "arm1_wall_seconds": wall1,
            "arm2_name": "base_extra",
            "arm2_exact_operator_loss": loss2,
            "arm2_psnr": psnr2,
            "arm2_good25": arm2_good,
            "arm2_sample_path": sample2,
            "arm2_wall_seconds": wall2,
            "arm2_accepted": accepted,
            "fresh2_selected_variant": names[selected_idx],
            "fresh2_selected_loss": selected_loss,
            "fresh2_selected_psnr": selected_psnr,
            "fresh2_selected_good25": selected_good,
            "fresh2_oracle_good25": oracle_good,
            "fresh2_selected_oracle_gap": oracle_gap,
            "fresh2_incremental_rescue": rescue,
            "fresh2_incremental_harm": harm,
            "candidate_good25_discordant": int(fresh1_good != arm2_good),
            "candidate_disagreement_rmse_rot180": rmse,
            "candidate_disagreement_l1_rot180": l1,
            "fresh2_total_wall_seconds": wall1 + wall2,
            "arm2_over_arm1_wall_ratio": wall2 / wall1,
        })

    frame = pd.DataFrame(rows).sort_values("row_id").reset_index(drop=True)
    if len(frame) != 100:
        raise ValueError(f"analyzed {len(frame)} rows instead of 100")

    numeric_columns = [
        column for column in frame.columns
        if column.endswith("_loss")
        or column.endswith("_psnr")
        or column.endswith("_seconds")
        or column.endswith("_ratio")
        or column.startswith("candidate_disagreement_")
    ]
    finite_gate = bool(np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all())
    if not finite_gate:
        raise ValueError("nonfinite analyzed values")

    fresh1_good = int(frame.arm1_good25.sum())
    selected_good = int(frame.fresh2_selected_good25.sum())
    oracle_good = int(frame.fresh2_oracle_good25.sum())
    selected_bad = 100 - selected_good
    rescues = int(frame.fresh2_incremental_rescue.sum())
    harms = int(frame.fresh2_incremental_harm.sum())
    oracle_gap = int(frame.fresh2_selected_oracle_gap.sum())
    accepted = int(frame.arm2_accepted.sum())

    psnr_summary = series_summary(frame.fresh2_selected_psnr)
    arm1_timing = series_summary(frame.arm1_wall_seconds)
    arm2_timing = series_summary(frame.arm2_wall_seconds)
    total_timing = series_summary(frame.fresh2_total_wall_seconds)
    reliability_interval = bootstrap_rate_interval(selected_good, 100)
    observed_full_run_equivalents = float(
        frame.fresh2_total_wall_seconds.mean()
        / pd.concat([frame.arm1_wall_seconds, frame.arm2_wall_seconds], ignore_index=True).mean()
    )

    integrity_gates = {
        "complete_100_rows": len(frame) == 100,
        "100_distinct_images": frame.image_id.nunique() == 100,
        "outside_excluded_panels": not bool(set(frame.image_id) & EXCLUDED),
        "panel_checksum_matches": actual_checksum == expected_checksum,
        "100_distinct_measurement_files": len(set(measurement_paths)) == 100,
        "100_distinct_measurement_hashes": len(set(measurement_hashes)) == 100,
        "200_complete_finite_trajectory_metrics": finite_gate and len(set(frame.arm1_sample_path)) == 100 and len(set(frame.arm2_sample_path)) == 100,
        "offline_selected_psnr_complete": bool(np.isfinite(frame.fresh2_selected_psnr.to_numpy(dtype=float)).all()),
        "frozen_seed_schedule_matches": bool(
            np.array_equal(frame.seed1.to_numpy(), 22000 + expected_rows)
            and np.array_equal(frame.seed2.to_numpy(), 23000 + expected_rows)
        ),
        "no_failed_row_dropped": len(frame) == len(manifest) == 100,
    }
    benchmark_valid = bool(all(integrity_gates.values()))

    support_checks = {
        "selected_good25_at_least_90": selected_good >= 90,
        "selected_oracle_gap_at_most_2": oracle_gap <= 2,
        "selector_harms_at_most_1": harms <= 1,
        "rescues_at_least_5": rescues >= 5,
    }
    deployment_support_statement = bool(all(support_checks.values()))

    summary = {
        "question": "What is the prospective reliability of the frozen two-restart DAPS policy on 100 disjoint official FFHQ validation images?",
        "status": "final_prospective_estimation",
        "policy": {
            "restart_budget": 2,
            "ann_steps": 400,
            "diff_steps": 5,
            "theta": args.theta,
            "lf_enabled": False,
            "hio_enabled": False,
            "conditional_fallback": False,
        },
        "panel": {
            "independent_images": 100,
            "locked_measurements": 100,
            "trajectory_outputs": 200,
            "panel_manifest_sha256": actual_checksum,
        },
        "primary": {
            "fresh1_good25": fresh1_good,
            "fresh2_selected_good25": selected_good,
            "fresh2_selected_bad25": selected_bad,
            "fresh2_oracle_any_good25": oracle_good,
            "selected_good25_rate": selected_good / 100,
            "selected_good25_interval_95": reliability_interval,
            "selected_psnr": psnr_summary,
        },
        "incremental_and_selector": {
            "fresh2_rescues_over_fresh1": rescues,
            "fresh2_harms_over_fresh1": harms,
            "selected_oracle_gap": oracle_gap,
            "trajectory2_accepts": accepted,
            "trajectory2_accept_fraction": accepted / 100,
            "candidate_good25_discordant_rows": int(frame.candidate_good25_discordant.sum()),
        },
        "cost": {
            "trajectory1_wall_seconds": arm1_timing,
            "trajectory2_wall_seconds": arm2_timing,
            "total_fresh2_wall_seconds_per_image": total_timing,
            "mean_observed_full_run_equivalents": observed_full_run_equivalents,
            "mean_arm2_over_arm1_wall_ratio": float(frame.arm2_over_arm1_wall_ratio.mean()),
            "median_arm2_over_arm1_wall_ratio": float(frame.arm2_over_arm1_wall_ratio.median()),
        },
        "integrity_gates": integrity_gates,
        "benchmark_valid": benchmark_valid,
        "descriptive_support_checks": support_checks,
        "deployment_support_statement": deployment_support_statement,
        "interpretation": (
            "This panel is estimation-only. Report the observed reliability and uncertainty without retuning. "
            "Failure of the descriptive support statement does not authorize detector, fallback, or restart-count "
            "search on this panel."
        ),
    }

    analysis_dir = out / "analysis_theta0.7"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    rows_path = analysis_dir / "fresh2_final_rows.csv"
    summary_path = analysis_dir / "fresh2_final_summary.json"
    discordant_path = analysis_dir / "fresh2_selector_discordant_rows.csv"
    psnr_path = analysis_dir / "fresh2_selected_psnr_summary.csv"
    timing_path = analysis_dir / "fresh2_timing_summary.csv"
    panel_checked_path = analysis_dir / "panel_manifest_checked.tsv"

    frame.to_csv(rows_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    discordant = frame.loc[
        frame.candidate_good25_discordant.eq(1)
        | frame.fresh2_selected_oracle_gap.eq(1)
        | frame.fresh2_incremental_rescue.eq(1)
        | frame.fresh2_incremental_harm.eq(1)
    ].copy()
    discordant.to_csv(discordant_path, index=False)
    pd.DataFrame([psnr_summary]).to_csv(psnr_path, index=False)
    pd.DataFrame([
        {"scope": "trajectory1", **arm1_timing},
        {"scope": "trajectory2", **arm2_timing},
        {"scope": "fresh2_total", **total_timing},
    ]).to_csv(timing_path, index=False)
    panel.to_csv(panel_checked_path, sep="\t", index=False)

    lines = [
        "# B21.11 prospective Fresh2 final benchmark",
        "",
        f"- benchmark valid: **{benchmark_valid}**",
        "- independent image/measurement units: `100`",
        f"- Fresh1 good25: `{fresh1_good}/100`",
        f"- Fresh2 selected good25: `{selected_good}/100`",
        f"- Fresh2 selected bad25: `{selected_bad}/100`",
        f"- Fresh2 oracle-any-good25: `{oracle_good}/100`",
        f"- Fresh2 rescues / harms: `{rescues} / {harms}`",
        f"- selected-oracle gap: `{oracle_gap}`",
        f"- trajectory-2 accepts: `{accepted}/100`",
        f"- selected-good rate 95% bootstrap interval: "
        f"`[{reliability_interval['lower_95']:.3f}, {reliability_interval['upper_95']:.3f}]`",
        f"- selected PSNR minimum / median / mean: "
        f"`{psnr_summary['min']:.4f} / {psnr_summary['median']:.4f} / {psnr_summary['mean']:.4f}`",
        f"- mean total wall seconds per image: `{total_timing['mean']:.2f}`",
        f"- observed full-run equivalents: `{observed_full_run_equivalents:.4f}`",
        f"- descriptive deployment-support statement: **{deployment_support_statement}**",
        "",
        "## Integrity gates",
        "",
    ]
    for key, value in integrity_gates.items():
        lines.append(f"- `{key}`: **{value}**")
    lines += ["", "## Descriptive support checks", ""]
    for key, value in support_checks.items():
        lines.append(f"- `{key}`: **{value}**")
    lines += [
        "",
        "This is an estimation benchmark. Its outcomes must be reported without retuning the policy on this panel.",
        "",
        f"Artifacts: `{analysis_dir}`",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[write] {rows_path}")
    print(f"[write] {summary_path}")
    print(f"[write] {discordant_path}")
    print(f"[write] {psnr_path}")
    print(f"[write] {timing_path}")
    print(f"[write] {panel_checked_path}")
    print(f"[write] {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
