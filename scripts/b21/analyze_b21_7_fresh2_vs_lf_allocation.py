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


def exact_two_sided(x_only: int, y_only: int) -> float:
    n = x_only + y_only
    if n == 0:
        return 1.0
    k = min(x_only, y_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def exact_one_sided_favor(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(wins, n + 1)) / (2**n)


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


def read_extra_metric(path: Path, gt: torch.Tensor) -> tuple[float, float, str]:
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


def summarize(group: pd.DataFrame, name: str) -> dict[str, object]:
    n = len(group)
    fresh2_only = int(group.fresh2_only_win.sum())
    base_lf_only = int(group.base_lf_only_win.sum())
    base_wall = float(group.base_wall_seconds.sum())
    extra_wall = float(group.extra_wall_seconds.sum())
    lf_wall = float(group.lf_wall_seconds.sum())
    return {
        "group": name,
        "n": n,
        "base_good25": int(group.base_good25.sum()),
        "extra_good25": int(group.extra_good25.sum()),
        "lf_good25": int(group.lf_good25.sum()),
        "fresh2_selected_good25": int(group.fresh2_selected_good25.sum()),
        "base_lf_selected_good25": int(group.base_lf_selected_good25.sum()),
        "base_lf_minus_fresh2_selected_good25": int(
            group.base_lf_selected_good25.sum() - group.fresh2_selected_good25.sum()
        ),
        "fresh2_only_wins": fresh2_only,
        "base_lf_only_wins": base_lf_only,
        "mcnemar_exact_two_sided_p": exact_two_sided(fresh2_only, base_lf_only),
        "one_sided_p_favor_fresh2": exact_one_sided_favor(fresh2_only, base_lf_only),
        "one_sided_p_favor_base_lf": exact_one_sided_favor(base_lf_only, fresh2_only),
        "fresh2_oracle_any_good25": int(group.fresh2_oracle_good25.sum()),
        "base_lf_oracle_any_good25": int(group.base_lf_oracle_good25.sum()),
        "base_lf_minus_fresh2_oracle_good25": int(
            group.base_lf_oracle_good25.sum() - group.fresh2_oracle_good25.sum()
        ),
        "accepted_extra": int(group.select_extra.sum()),
        "accepted_lf": int(group.select_lf.sum()),
        "mean_fresh2_selected_psnr": float(group.fresh2_selected_psnr.mean()),
        "mean_base_lf_selected_psnr": float(group.base_lf_selected_psnr.mean()),
        "mean_base_lf_minus_fresh2_psnr": float(
            (group.base_lf_selected_psnr - group.fresh2_selected_psnr).mean()
        ),
        "median_base_lf_minus_fresh2_psnr": float(
            (group.base_lf_selected_psnr - group.fresh2_selected_psnr).median()
        ),
        "extra_wall_over_base": extra_wall / base_wall if base_wall > 0 else float("nan"),
        "lf_wall_over_base": lf_wall / base_wall if base_wall > 0 else float("nan"),
        "fresh2_total_cost_over_base": (base_wall + extra_wall) / base_wall if base_wall > 0 else float("nan"),
        "base_lf_total_cost_over_base": (base_wall + lf_wall) / base_wall if base_wall > 0 else float("nan"),
        "fresh2_over_base_lf_total_cost": (
            (base_wall + extra_wall) / (base_wall + lf_wall)
            if base_wall + lf_wall > 0 else float("nan")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-out", type=Path, required=True)
    parser.add_argument("--fresh-rows", type=Path, required=True)
    parser.add_argument("--extension-manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--theta", type=float, default=0.7)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    fresh_out = args.fresh_out.resolve()
    prior = pd.read_csv(args.fresh_rows, dtype={"image_id": str})
    prior["image_id"] = prior.image_id.map(lambda x: f"{int(x):05d}")
    manifest = pd.read_csv(args.extension_manifest, sep="\t", dtype={"image_id": str})
    manifest["image_id"] = manifest.image_id.map(lambda x: f"{int(x):05d}")

    if len(prior) != 80 or len(manifest) != 80:
        raise ValueError(f"expected 80 prior and manifest rows, got {len(prior)} and {len(manifest)}")

    key_cols = ["image_id", "case_id"]
    merged = prior.merge(
        manifest[["image_id", "case_id", "extra_seed"]],
        on=key_cols,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 80:
        raise ValueError(f"merged row count is {len(merged)}, expected 80")

    rows: list[dict[str, object]] = []
    gt_cache: dict[str, torch.Tensor] = {}
    for rec in merged.itertuples(index=False):
        image_id = f"{int(rec.image_id):05d}"
        case_id = int(rec.case_id)
        case_dir = fresh_out / "cases" / f"{image_id}_case{case_id:02d}"
        if image_id not in gt_cache:
            gt_cache[image_id] = load_gt(find_gt(args.image_root.resolve(), image_id))
        extra_loss, extra_psnr, extra_sample = read_extra_metric(
            case_dir / "metrics" / "base_extra.csv",
            gt_cache[image_id],
        )
        timings = read_timings(case_dir / "timings.tsv")

        base_loss = float(rec.base_exact_operator_loss)
        lf_loss = float(rec.lf_exact_operator_loss)
        base_psnr = float(rec.base_psnr)
        lf_psnr = float(rec.lf_psnr)

        select_extra = int(extra_loss < base_loss - args.theta)
        fresh2_variant = "base_extra" if select_extra else "base_full"
        fresh2_loss = extra_loss if select_extra else base_loss
        fresh2_psnr = extra_psnr if select_extra else base_psnr

        select_lf = int(lf_loss < base_loss - args.theta)
        base_lf_variant = "lf050" if select_lf else "base_full"
        base_lf_loss = lf_loss if select_lf else base_loss
        base_lf_psnr = lf_psnr if select_lf else base_psnr

        base_good = int(base_psnr >= 25.0)
        extra_good = int(extra_psnr >= 25.0)
        lf_good = int(lf_psnr >= 25.0)
        fresh2_good = int(fresh2_psnr >= 25.0)
        base_lf_good = int(base_lf_psnr >= 25.0)

        rows.append({
            "job_id": int(rec.job_id),
            "image_id": image_id,
            "case_id": case_id,
            "base_seed": int(rec.base_seed),
            "extra_seed": int(rec.extra_seed),
            "measurement_path": str(rec.measurement_path),
            "base_exact_operator_loss": base_loss,
            "extra_exact_operator_loss": extra_loss,
            "lf_exact_operator_loss": lf_loss,
            "base_psnr": base_psnr,
            "extra_psnr": extra_psnr,
            "lf_psnr": lf_psnr,
            "base_good25": base_good,
            "extra_good25": extra_good,
            "lf_good25": lf_good,
            "select_extra": select_extra,
            "fresh2_selected_variant": fresh2_variant,
            "fresh2_selected_loss": fresh2_loss,
            "fresh2_selected_psnr": fresh2_psnr,
            "fresh2_selected_good25": fresh2_good,
            "select_lf": select_lf,
            "base_lf_selected_variant": base_lf_variant,
            "base_lf_selected_loss": base_lf_loss,
            "base_lf_selected_psnr": base_lf_psnr,
            "base_lf_selected_good25": base_lf_good,
            "fresh2_only_win": int(fresh2_good and not base_lf_good),
            "base_lf_only_win": int(base_lf_good and not fresh2_good),
            "fresh2_oracle_good25": int(base_good or extra_good),
            "base_lf_oracle_good25": int(base_good or lf_good),
            "extra_sample_path": extra_sample,
            "base_wall_seconds": timings.get("base_full", float("nan")),
            "extra_wall_seconds": timings.get("base_extra", float("nan")),
            "lf_wall_seconds": timings.get("lf050", float("nan")),
        })

    frame = pd.DataFrame(rows).sort_values(key_cols).reset_index(drop=True)
    numeric_required = [
        "base_exact_operator_loss", "extra_exact_operator_loss", "lf_exact_operator_loss",
        "base_psnr", "extra_psnr", "lf_psnr",
        "base_wall_seconds", "extra_wall_seconds", "lf_wall_seconds",
    ]
    finite_gate = bool(np.isfinite(frame[numeric_required].to_numpy(dtype=float)).all())

    by_image = [summarize(group, image) for image, group in frame.groupby("image_id", sort=True)]
    overall = summarize(frame, "ALL")
    summary = pd.DataFrame(by_image)
    positive_base_lf = int((summary.base_lf_minus_fresh2_selected_good25 > 0).sum())
    positive_fresh2 = int((summary.base_lf_minus_fresh2_selected_good25 < 0).sum())
    tied_images = int((summary.base_lf_minus_fresh2_selected_good25 == 0).sum())

    difference = int(overall["base_lf_minus_fresh2_selected_good25"])
    p_base_lf = float(overall["one_sided_p_favor_base_lf"])
    p_fresh2 = float(overall["one_sided_p_favor_fresh2"])
    if difference >= 4 or p_base_lf < 0.05:
        decision = "base_lf_wins"
    elif difference <= -4 or p_fresh2 < 0.05:
        decision = "fresh2_wins"
    else:
        decision = "inconclusive"

    verdict = {
        "question": "At approximately two full-run equivalents, is a matched LF050 arm better than an independent fresh DAPS restart?",
        "selection_rule": "for each policy, choose arm 2 iff its exact operator loss is below base loss minus theta",
        "frozen_theta": args.theta,
        "expected_cases": 80,
        "complete_cases": len(frame),
        "finite_metrics_and_timings": finite_gate,
        "overall": overall,
        "images_favoring_base_lf": positive_base_lf,
        "images_favoring_fresh2": positive_fresh2,
        "tied_images": tied_images,
        "frozen_decision_rule": {
            "base_lf_wins": "selected-good advantage >=4/80 OR one-sided exact McNemar p<0.05 favoring base+LF",
            "fresh2_wins": "selected-good advantage >=4/80 OR one-sided exact McNemar p<0.05 favoring Fresh2",
            "otherwise": "inconclusive",
        },
        "allocation_decision": decision,
        "next_if_base_lf_wins": "retain LF050 as the preferred second arm and move to budget-policy design",
        "next_if_fresh2_wins": "retire LF050 as a default second arm and use independent restarts",
        "next_if_inconclusive": "retain both as optional arms; do not claim one dominates",
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows_path = args.outdir / "fresh2_vs_lf_rows.csv"
    summary_path = args.outdir / "fresh2_vs_lf_summary_by_image.csv"
    verdict_path = args.outdir / "fresh2_vs_lf_verdict.json"
    frame.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    verdict_path.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")

    lines = [
        "# B21.7 equal-cost Fresh2 versus Base+LF allocation",
        "",
        f"- paired cases: `{len(frame)}`",
        f"- frozen theta: `{args.theta}`",
        f"- decision: **{decision}**",
        "",
        "## By image",
        "",
        "| image | n | Fresh2 selected | Base+LF selected | Base+LF net | Fresh2 only | Base+LF only | Fresh2 oracle | Base+LF oracle | Fresh2 cost/base | Base+LF cost/base |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_image:
        lines.append(
            f"| `{row['group']}` | {row['n']} | {row['fresh2_selected_good25']} | "
            f"{row['base_lf_selected_good25']} | {int(row['base_lf_minus_fresh2_selected_good25']):+d} | "
            f"{row['fresh2_only_wins']} | {row['base_lf_only_wins']} | "
            f"{row['fresh2_oracle_any_good25']} | {row['base_lf_oracle_any_good25']} | "
            f"{float(row['fresh2_total_cost_over_base']):.3f} | {float(row['base_lf_total_cost_over_base']):.3f} |"
        )
    lines += [
        "",
        "## Overall",
        "",
        f"- Fresh2 selected good25: `{overall['fresh2_selected_good25']}/80`",
        f"- Base+LF selected good25: `{overall['base_lf_selected_good25']}/80`",
        f"- Base+LF selected net: `{int(overall['base_lf_minus_fresh2_selected_good25']):+d}`",
        f"- Fresh2-only / Base+LF-only wins: `{overall['fresh2_only_wins']} / {overall['base_lf_only_wins']}`",
        f"- exact McNemar two-sided p: `{overall['mcnemar_exact_two_sided_p']}`",
        f"- one-sided p favor Fresh2: `{overall['one_sided_p_favor_fresh2']}`",
        f"- one-sided p favor Base+LF: `{overall['one_sided_p_favor_base_lf']}`",
        f"- Fresh2 oracle any-good: `{overall['fresh2_oracle_any_good25']}/80`",
        f"- Base+LF oracle any-good: `{overall['base_lf_oracle_any_good25']}/80`",
        f"- Fresh2 total cost/base: `{overall['fresh2_total_cost_over_base']}`",
        f"- Base+LF total cost/base: `{overall['base_lf_total_cost_over_base']}`",
        f"- Fresh2/Base+LF cost ratio: `{overall['fresh2_over_base_lf_total_cost']}`",
        f"- image counts favoring Base+LF / tied / favoring Fresh2: `{positive_base_lf} / {tied_images} / {positive_fresh2}`",
        "",
        f"Artifacts: `{args.outdir}`",
        "",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines))

    print(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"[write] {rows_path}")
    print(f"[write] {summary_path}")
    print(f"[write] {verdict_path}")
    print(f"[write] {args.report}")
    return 0 if finite_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
