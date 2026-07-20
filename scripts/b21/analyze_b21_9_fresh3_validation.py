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
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0


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


def exact_mcnemar_two_sided(rescues: int, harms: int) -> float:
    n = rescues + harms
    if n == 0:
        return 1.0
    k = min(rescues, harms)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def summarize_k(frame: pd.DataFrame, k: int) -> dict[str, object]:
    selected = frame[f"fresh{k}_selected_good25"]
    oracle = frame[f"fresh{k}_oracle_good25"]
    previous_selected = (
        pd.Series(np.zeros(len(frame), dtype=int), index=frame.index)
        if k == 1
        else frame[f"fresh{k - 1}_selected_good25"]
    )
    previous_oracle = (
        pd.Series(np.zeros(len(frame), dtype=int), index=frame.index)
        if k == 1
        else frame[f"fresh{k - 1}_oracle_good25"]
    )
    return {
        "k": k,
        "selected_good25": int(selected.sum()),
        "selected_bad25": int(len(frame) - selected.sum()),
        "oracle_good25": int(oracle.sum()),
        "oracle_bad25": int(len(frame) - oracle.sum()),
        "selected_oracle_gap": int(oracle.sum() - selected.sum()),
        "incremental_selected_gain": int(selected.sum() - previous_selected.sum()),
        "incremental_oracle_gain": int(oracle.sum() - previous_oracle.sum()),
        "incremental_rescues": int(frame[f"fresh{k}_incremental_rescue"].sum()),
        "incremental_harms": int(frame[f"fresh{k}_incremental_harm"].sum()),
        "accepted_new_arm": int(frame[f"fresh{k}_accepted_new_arm"].sum()),
        "mean_selected_psnr": float(frame[f"fresh{k}_selected_psnr"].mean()),
        "median_selected_psnr": float(frame[f"fresh{k}_selected_psnr"].median()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--theta", type=float, default=0.7)
    parser.add_argument("--target-good", type=int, default=74)
    parser.add_argument("--max-oracle-gap", type=int, default=2)
    parser.add_argument("--min-incremental-gain", type=int, default=4)
    parser.add_argument("--max-incremental-harms", type=int, default=1)
    parser.add_argument("--max-cumulative-harms", type=int, default=1)
    parser.add_argument("--min-positive-images", type=int, default=4)
    parser.add_argument("--max-negative-images", type=int, default=1)
    parser.add_argument("--min-per-image-good", type=int, default=2)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    out = args.out.resolve()
    manifest = pd.read_csv(args.manifest, sep="\t", dtype={"image_id": str})
    manifest["image_id"] = manifest.image_id.map(lambda x: f"{int(x):05d}")
    if len(manifest) != 80:
        raise ValueError(f"expected 80 manifest rows, got {len(manifest)}")

    gt_cache: dict[str, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    names = ["base_full", "base_extra", "base_extra2"]

    for rec in manifest.itertuples(index=False):
        image_id = f"{int(rec.image_id):05d}"
        case_id = int(rec.case_id)
        case_dir = out / "cases" / f"{image_id}_case{case_id:02d}"
        if image_id not in gt_cache:
            gt_cache[image_id] = load_gt(find_gt(args.image_root.resolve(), image_id))
        gt = gt_cache[image_id]

        metrics = [
            read_metric(case_dir / "metrics" / f"{name}.csv", gt)
            for name in names
        ]
        losses = [item[0] for item in metrics]
        psnrs = [item[1] for item in metrics]
        sample_paths = [item[2] for item in metrics]
        timings = read_timings(case_dir / "timings.tsv")

        current_idx = 0
        cumulative_harms = 0
        row: dict[str, object] = {
            "job_id": int(rec.job_id),
            "image_id": image_id,
            "case_id": case_id,
            "gpu": str(rec.gpu),
            "seed1": int(rec.seed1),
            "seed2": int(rec.seed2),
            "seed3": int(rec.seed3),
            "measurement_path": str(rec.measurement_path),
            "execution_order": str(rec.execution_order),
        }
        for idx, name in enumerate(names):
            arm = idx + 1
            row[f"arm{arm}_name"] = name
            row[f"arm{arm}_exact_operator_loss"] = losses[idx]
            row[f"arm{arm}_psnr"] = psnrs[idx]
            row[f"arm{arm}_good25"] = int(psnrs[idx] >= 25.0)
            row[f"arm{arm}_sample_path"] = sample_paths[idx]
            row[f"arm{arm}_wall_seconds"] = timings.get(name, float("nan"))

        previous_selected_good = 0
        for k in range(1, 4):
            if k == 1:
                accepted = 0
            else:
                candidate_idx = k - 1
                accepted = int(losses[candidate_idx] < losses[current_idx] - args.theta)
                if accepted:
                    current_idx = candidate_idx

            selected_good = int(psnrs[current_idx] >= 25.0)
            oracle_good = int(max(psnrs[:k]) >= 25.0)
            rescue = int(not previous_selected_good and selected_good) if k > 1 else 0
            harm = int(previous_selected_good and not selected_good) if k > 1 else 0
            cumulative_harms += harm

            row[f"fresh{k}_accepted_new_arm"] = accepted
            row[f"fresh{k}_selected_variant"] = names[current_idx]
            row[f"fresh{k}_selected_loss"] = losses[current_idx]
            row[f"fresh{k}_selected_psnr"] = psnrs[current_idx]
            row[f"fresh{k}_selected_good25"] = selected_good
            row[f"fresh{k}_oracle_good25"] = oracle_good
            row[f"fresh{k}_incremental_rescue"] = rescue
            row[f"fresh{k}_incremental_harm"] = harm
            row[f"fresh{k}_cumulative_harms"] = cumulative_harms
            previous_selected_good = selected_good

        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(["image_id", "case_id"]).reset_index(drop=True)
    numeric_required = [
        column
        for column in frame.columns
        if column.endswith("_loss")
        or column.endswith("_psnr")
        or column.endswith("_wall_seconds")
    ]
    finite_gate = bool(np.isfinite(frame[numeric_required].to_numpy(dtype=float)).all())

    curve = [summarize_k(frame, k) for k in (1, 2, 3)]
    curve_frame = pd.DataFrame(curve)

    image_rows: list[dict[str, object]] = []
    for image_id, group in frame.groupby("image_id", sort=True):
        image_rows.append({
            "image_id": image_id,
            "n": len(group),
            "fresh1_selected_good25": int(group.fresh1_selected_good25.sum()),
            "fresh1_oracle_good25": int(group.fresh1_oracle_good25.sum()),
            "fresh2_selected_good25": int(group.fresh2_selected_good25.sum()),
            "fresh2_oracle_good25": int(group.fresh2_oracle_good25.sum()),
            "fresh3_selected_good25": int(group.fresh3_selected_good25.sum()),
            "fresh3_oracle_good25": int(group.fresh3_oracle_good25.sum()),
            "fresh3_minus_fresh2_selected_good25": int(
                group.fresh3_selected_good25.sum() - group.fresh2_selected_good25.sum()
            ),
            "fresh3_incremental_rescues": int(group.fresh3_incremental_rescue.sum()),
            "fresh3_incremental_harms": int(group.fresh3_incremental_harm.sum()),
        })
    image_summary = pd.DataFrame(image_rows)

    overall_k2 = curve[1]
    overall_k3 = curve[2]
    positive_images = int((image_summary.fresh3_minus_fresh2_selected_good25 > 0).sum())
    negative_images = int((image_summary.fresh3_minus_fresh2_selected_good25 < 0).sum())
    tied_images = int((image_summary.fresh3_minus_fresh2_selected_good25 == 0).sum())
    minimum_per_image = int(image_summary.fresh3_selected_good25.min())
    cumulative_harms = int(frame.fresh3_cumulative_harms.sum())
    incremental_rescues = int(frame.fresh3_incremental_rescue.sum())
    incremental_harms = int(frame.fresh3_incremental_harm.sum())
    oracle_incremental = int(overall_k3["incremental_oracle_gain"])
    selector_capture = incremental_rescues / oracle_incremental if oracle_incremental else 1.0

    gates = {
        "complete_80_and_finite": bool(len(frame) == 80 and finite_gate),
        "fresh3_selected_good25_at_least_target": int(overall_k3["selected_good25"]) >= args.target_good,
        "fresh3_selected_oracle_gap_at_most": int(overall_k3["selected_oracle_gap"]) <= args.max_oracle_gap,
        "fresh3_incremental_gain_at_least": int(overall_k3["incremental_selected_gain"]) >= args.min_incremental_gain,
        "fresh3_incremental_harms_at_most": incremental_harms <= args.max_incremental_harms,
        "cumulative_selected_harms_at_most": cumulative_harms <= args.max_cumulative_harms,
        "positive_images_at_least": positive_images >= args.min_positive_images,
        "negative_images_at_most": negative_images <= args.max_negative_images,
        "minimum_per_image_good25_at_least": minimum_per_image >= args.min_per_image_good,
    }
    validation_pass = bool(all(gates.values()))

    verdict = {
        "question": "Does the frozen three-independent-restart policy generalize to a disjoint official FFHQ validation panel?",
        "frozen_k": 3,
        "frozen_theta": args.theta,
        "expected_cases": 80,
        "complete_cases": len(frame),
        "panel_shape": "20 untouched official-validation images x 4 independent measurement/seed cases",
        "curve": curve,
        "fresh3_incremental_rescues": incremental_rescues,
        "fresh3_incremental_harms": incremental_harms,
        "fresh3_oracle_incremental_gain": oracle_incremental,
        "fresh3_selector_capture_of_oracle_incremental": selector_capture,
        "positive_image_count": positive_images,
        "negative_image_count": negative_images,
        "tied_image_count": tied_images,
        "minimum_fresh3_selected_good25_per_image": minimum_per_image,
        "fresh3_vs_fresh2_mcnemar_two_sided_p": exact_mcnemar_two_sided(
            incremental_rescues, incremental_harms
        ),
        "gates": gates,
        "fresh3_validation_pass": validation_pass,
        "next_if_pass": "adopt Fresh3 as the fixed default reliability budget and move to a larger benchmark/adaptive early-stop design without retuning",
        "next_if_fail": "retain Fresh2 as the default fixed budget; do not scale independent restarts further without a new candidate-generation or adaptive-budget idea",
    }

    analysis_dir = out / f"analysis_theta{args.theta}"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    rows_path = analysis_dir / "fresh3_validation_rows.csv"
    curve_path = analysis_dir / "fresh3_validation_curve.csv"
    image_path = analysis_dir / "fresh3_validation_summary_by_image.csv"
    verdict_path = analysis_dir / "fresh3_validation_verdict.json"
    frame.to_csv(rows_path, index=False)
    curve_frame.to_csv(curve_path, index=False)
    image_summary.to_csv(image_path, index=False)
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")

    lines = [
        "# B21.9 Fresh3 disjoint validation",
        "",
        "- panel: `20` untouched official FFHQ validation images x `4` cases",
        "- paired cases: `80`",
        f"- frozen theta: `{args.theta}`",
        f"- validation pass: **{validation_pass}**",
        "",
        "## Cumulative curve",
        "",
        "| K | selected good25 | oracle good25 | gap | incremental selected | rescues | harms | accepted new arm | mean selected PSNR |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in curve:
        lines.append(
            f"| {row['k']} | {row['selected_good25']}/80 | {row['oracle_good25']}/80 | "
            f"{row['selected_oracle_gap']} | {int(row['incremental_selected_gain']):+d} | "
            f"{row['incremental_rescues']} | {row['incremental_harms']} | "
            f"{row['accepted_new_arm']} | {float(row['mean_selected_psnr']):.4f} |"
        )
    lines += [
        "",
        "## Fresh3 versus Fresh2",
        "",
        f"- incremental rescues / harms: `{incremental_rescues} / {incremental_harms}`",
        f"- oracle incremental gain: `{oracle_incremental}`",
        f"- selector capture: `{selector_capture}`",
        f"- positive / tied / negative images: `{positive_images} / {tied_images} / {negative_images}`",
        f"- minimum Fresh3 good25 per image: `{minimum_per_image}/4`",
        f"- exact McNemar two-sided p: `{verdict['fresh3_vs_fresh2_mcnemar_two_sided_p']}`",
        "",
        "## Frozen gates",
        "",
    ]
    for key, value in gates.items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", f"Artifacts: `{analysis_dir}`", ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))

    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"[write] {rows_path}")
    print(f"[write] {curve_path}")
    print(f"[write] {image_path}")
    print(f"[write] {verdict_path}")
    print(f"[write] {args.report}")
    return 0 if gates["complete_80_and_finite"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
