#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0


def psnr_model_range(sample: torch.Tensor, gt: torch.Tensor) -> float:
    sample01 = (sample.clamp(-1, 1) + 1.0) / 2.0
    gt01 = (gt.clamp(-1, 1) + 1.0) / 2.0
    mse = (sample01 - gt01).pow(2).flatten(1).mean(1).clamp_min(1e-12)
    return float((-10.0 * torch.log10(mse))[0].item())


def read_metric(path: Path, gt: torch.Tensor) -> tuple[float, float, str]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty metric CSV: {path}")
    row = frame.iloc[0]
    loss = float(pd.to_numeric(row["exact_operator_loss"], errors="raise"))
    sample_path = Path(str(row["sample_path"]))
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)
    psnr = psnr_model_range(load_sample(sample_path), gt)
    return loss, psnr, str(sample_path)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-out", type=Path, required=True)
    parser.add_argument("--fresh-rows", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--theta", type=float, default=0.7)
    parser.add_argument("--target-good", type=int, default=76)
    parser.add_argument("--max-oracle-gap", type=int, default=1)
    parser.add_argument("--max-cumulative-harms", type=int, default=1)
    parser.add_argument("--min-per-image-good", type=int, default=6)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    fresh_out = args.fresh_out.resolve()
    prior = pd.read_csv(args.fresh_rows, dtype={"image_id": str})
    prior["image_id"] = prior.image_id.map(lambda x: f"{int(x):05d}")
    manifest = pd.read_csv(args.manifest, sep="\t", dtype={"image_id": str})
    manifest["image_id"] = manifest.image_id.map(lambda x: f"{int(x):05d}")
    if len(prior) != 80 or len(manifest) != 80:
        raise ValueError(f"expected 80 prior/manifest rows, got {len(prior)} and {len(manifest)}")

    merged = prior.merge(
        manifest[["job_id", "image_id", "case_id", "extra2_seed", "extra3_seed"]],
        on=["image_id", "case_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 80:
        raise ValueError(f"merged row count {len(merged)} != 80")

    gt_cache: dict[str, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    for rec in merged.itertuples(index=False):
        image_id = f"{int(rec.image_id):05d}"
        case_id = int(rec.case_id)
        case_dir = fresh_out / "cases" / f"{image_id}_case{case_id:02d}"
        if image_id not in gt_cache:
            gt_cache[image_id] = load_gt(find_gt(args.image_root.resolve(), image_id))
        gt = gt_cache[image_id]

        extra1_loss, extra1_psnr, extra1_sample = read_metric(
            case_dir / "metrics" / "base_extra.csv", gt
        )
        extra2_loss, extra2_psnr, extra2_sample = read_metric(
            case_dir / "metrics" / "base_extra2.csv", gt
        )
        extra3_loss, extra3_psnr, extra3_sample = read_metric(
            case_dir / "metrics" / "base_extra3.csv", gt
        )
        timings = read_timings(case_dir / "timings.tsv")

        losses = [
            float(rec.base_exact_operator_loss),
            extra1_loss,
            extra2_loss,
            extra3_loss,
        ]
        psnrs = [
            float(rec.base_psnr),
            extra1_psnr,
            extra2_psnr,
            extra3_psnr,
        ]
        names = ["base_full", "base_extra", "base_extra2", "base_extra3"]

        current_idx = 0
        cumulative_harms = 0
        row: dict[str, object] = {
            "job_id": int(rec.job_id),
            "image_id": image_id,
            "case_id": case_id,
            "base_seed": int(rec.base_seed),
            "extra1_seed": int(rec.extra_seed),
            "extra2_seed": int(rec.extra2_seed),
            "extra3_seed": int(rec.extra3_seed),
            "measurement_path": str(rec.measurement_path),
            "base_exact_operator_loss": losses[0],
            "extra1_exact_operator_loss": losses[1],
            "extra2_exact_operator_loss": losses[2],
            "extra3_exact_operator_loss": losses[3],
            "base_psnr": psnrs[0],
            "extra1_psnr": psnrs[1],
            "extra2_psnr": psnrs[2],
            "extra3_psnr": psnrs[3],
            "extra1_sample_path": extra1_sample,
            "extra2_sample_path": extra2_sample,
            "extra3_sample_path": extra3_sample,
            "base_wall_seconds": timings.get("base_full", float("nan")),
            "extra1_wall_seconds": timings.get("base_extra", float("nan")),
            "extra2_wall_seconds": timings.get("base_extra2", float("nan")),
            "extra3_wall_seconds": timings.get("base_extra3", float("nan")),
        }

        previous_selected_good = int(psnrs[current_idx] >= 25.0)
        for k in range(1, 5):
            if k > 1:
                candidate_idx = k - 1
                accept = int(losses[candidate_idx] < losses[current_idx] - args.theta)
                if accept:
                    current_idx = candidate_idx
            else:
                accept = 0

            selected_good = int(psnrs[current_idx] >= 25.0)
            oracle_good = int(max(psnrs[:k]) >= 25.0)
            rescue = int(k > 1 and not previous_selected_good and selected_good)
            harm = int(k > 1 and previous_selected_good and not selected_good)
            cumulative_harms += harm

            row[f"fresh{k}_selected_variant"] = names[current_idx]
            row[f"fresh{k}_selected_loss"] = losses[current_idx]
            row[f"fresh{k}_selected_psnr"] = psnrs[current_idx]
            row[f"fresh{k}_selected_good25"] = selected_good
            row[f"fresh{k}_oracle_good25"] = oracle_good
            row[f"fresh{k}_oracle_gap"] = oracle_good - selected_good
            row[f"fresh{k}_accepted_new"] = accept
            row[f"fresh{k}_incremental_rescue"] = rescue
            row[f"fresh{k}_incremental_harm"] = harm
            row[f"fresh{k}_cumulative_harms"] = cumulative_harms
            previous_selected_good = selected_good

        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(["image_id", "case_id"]).reset_index(drop=True)
    numeric = [
        "base_exact_operator_loss", "extra1_exact_operator_loss",
        "extra2_exact_operator_loss", "extra3_exact_operator_loss",
        "base_psnr", "extra1_psnr", "extra2_psnr", "extra3_psnr",
        "base_wall_seconds", "extra1_wall_seconds", "extra2_wall_seconds", "extra3_wall_seconds",
    ]
    finite_gate = bool(np.isfinite(frame[numeric].to_numpy(dtype=float)).all())

    curve_rows: list[dict[str, object]] = []
    per_image_rows: list[dict[str, object]] = []
    for image_id, group in frame.groupby("image_id", sort=True):
        image_row: dict[str, object] = {"image_id": image_id, "n": len(group)}
        for k in range(1, 5):
            image_row[f"fresh{k}_selected_good25"] = int(group[f"fresh{k}_selected_good25"].sum())
            image_row[f"fresh{k}_oracle_good25"] = int(group[f"fresh{k}_oracle_good25"].sum())
        per_image_rows.append(image_row)
    per_image = pd.DataFrame(per_image_rows)

    total_base_wall = float(frame.base_wall_seconds.sum())
    cumulative_wall = total_base_wall
    selected_harms_cumulative = 0
    for k in range(1, 5):
        if k > 1:
            cumulative_wall += float(frame[f"extra{k-1}_wall_seconds"].sum())
            selected_harms_cumulative += int(frame[f"fresh{k}_incremental_harm"].sum())
        selected_good = int(frame[f"fresh{k}_selected_good25"].sum())
        oracle_good = int(frame[f"fresh{k}_oracle_good25"].sum())
        min_image_good = int(per_image[f"fresh{k}_selected_good25"].min())
        curve_rows.append({
            "k": k,
            "selected_good25": selected_good,
            "selected_bad25": 80 - selected_good,
            "oracle_good25": oracle_good,
            "oracle_bad25": 80 - oracle_good,
            "selected_oracle_gap": oracle_good - selected_good,
            "incremental_selected_gain": (
                selected_good - int(frame[f"fresh{k-1}_selected_good25"].sum()) if k > 1 else selected_good
            ),
            "incremental_oracle_gain": (
                oracle_good - int(frame[f"fresh{k-1}_oracle_good25"].sum()) if k > 1 else oracle_good
            ),
            "incremental_rescues": int(frame[f"fresh{k}_incremental_rescue"].sum()),
            "incremental_harms": int(frame[f"fresh{k}_incremental_harm"].sum()),
            "cumulative_harms": selected_harms_cumulative,
            "accepted_new_arm": int(frame[f"fresh{k}_accepted_new"].sum()),
            "min_per_image_selected_good25": min_image_good,
            "total_wall_over_base": cumulative_wall / total_base_wall if total_base_wall > 0 else float("nan"),
            "meets_reliability_target": bool(
                selected_good >= args.target_good
                and oracle_good - selected_good <= args.max_oracle_gap
                and selected_harms_cumulative <= args.max_cumulative_harms
                and min_image_good >= args.min_per_image_good
            ),
        })

    curve = pd.DataFrame(curve_rows)
    qualifying = curve.loc[(curve.k >= 2) & curve.meets_reliability_target]
    if qualifying.empty:
        selected_k: int | None = None
        decision = "no_k_meets_development_target"
    else:
        selected_k = int(qualifying.k.min())
        decision = f"validate_fresh{selected_k}_on_new_panel"

    verdict = {
        "question": "How many independent full DAPS restarts are needed before fresh validation of a fixed budget?",
        "frozen_theta": args.theta,
        "complete_cases": len(frame),
        "finite_metrics_and_timings": finite_gate,
        "development_target": {
            "selected_good25_at_least": args.target_good,
            "selected_oracle_gap_at_most": args.max_oracle_gap,
            "cumulative_selected_harms_at_most": args.max_cumulative_harms,
            "minimum_selected_good25_per_image_at_least": args.min_per_image_good,
        },
        "curve": curve_rows,
        "selected_k": selected_k,
        "budget_decision": decision,
        "next_if_selected": "freeze selected K and validate unchanged on a disjoint official FFHQ validation panel with new measurements and seeds",
        "next_if_none": "do not scale independent restarts blindly; return to hard-case candidate-generation or adaptive budget design",
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows_path = args.outdir / "freshk_rows.csv"
    curve_path = args.outdir / "freshk_curve.csv"
    image_path = args.outdir / "freshk_summary_by_image.csv"
    verdict_path = args.outdir / "freshk_verdict.json"
    frame.to_csv(rows_path, index=False)
    curve.to_csv(curve_path, index=False)
    per_image.to_csv(image_path, index=False)
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")

    lines = [
        "# B21.8 independent-restart budget curve",
        "",
        f"- paired cases: `{len(frame)}`",
        f"- frozen theta: `{args.theta}`",
        f"- budget decision: **{decision}**",
        "",
        "## Curve",
        "",
        "| K | selected good25 | oracle good25 | selected-oracle gap | incremental selected | rescues | harms | cumulative harms | min image good | cost/base | target |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in curve_rows:
        lines.append(
            f"| {row['k']} | {row['selected_good25']}/80 | {row['oracle_good25']}/80 | "
            f"{row['selected_oracle_gap']} | {row['incremental_selected_gain']:+d} | "
            f"{row['incremental_rescues']} | {row['incremental_harms']} | "
            f"{row['cumulative_harms']} | {row['min_per_image_selected_good25']}/8 | "
            f"{float(row['total_wall_over_base']):.3f} | {row['meets_reliability_target']} |"
        )
    lines += [
        "",
        "## Frozen target",
        "",
        f"- selected good25 >= `{args.target_good}/80`",
        f"- selected-oracle gap <= `{args.max_oracle_gap}`",
        f"- cumulative selected harms <= `{args.max_cumulative_harms}`",
        f"- every image selected good25 >= `{args.min_per_image_good}/8`",
        "",
        f"Artifacts: `{args.outdir}`",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))

    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"[write] {rows_path}")
    print(f"[write] {curve_path}")
    print(f"[write] {image_path}")
    print(f"[write] {verdict_path}")
    print(f"[write] {args.report}")
    return 0 if finite_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
