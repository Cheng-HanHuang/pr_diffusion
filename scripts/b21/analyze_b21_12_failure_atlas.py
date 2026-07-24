#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as transforms


DEFAULT_IMAGE_ROOT = Path(
    "/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024"
)
EXPECTED_COUNTS = {
    "persistent_failure": 8,
    "fresh2_rescue": 12,
    "protected_fresh1_success": 7,
}


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
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0


def psnr_model_range(sample: torch.Tensor, gt: torch.Tensor) -> float:
    sample01 = (sample.clamp(-1, 1) + 1.0) / 2.0
    gt01 = (gt.clamp(-1, 1) + 1.0) / 2.0
    mse = (sample01 - gt01).pow(2).flatten(1).mean(1).clamp_min(1e-12)
    return float((-10.0 * torch.log10(mse))[0].item())


def best_orientation(sample: torch.Tensor, gt: torch.Tensor) -> tuple[float, float, float, bool, torch.Tensor]:
    raw = psnr_model_range(sample, gt)
    rotated = torch.rot90(sample, 2, dims=(-2, -1))
    rot_psnr = psnr_model_range(rotated, gt)
    use_rot = bool(rot_psnr > raw)
    return raw, rot_psnr, max(raw, rot_psnr), use_rot, rotated if use_rot else sample


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (
        ((tensor[0].clamp(-1, 1) + 1.0) / 2.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def fit_text(text: str, width: int, font: ImageFont.ImageFont) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def labeled_tile(
    image: Image.Image,
    label: str,
    tile_size: int,
    selected: bool = False,
) -> Image.Image:
    font = ImageFont.load_default()
    image = image.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
    lines = fit_text(label, tile_size - 12, font)
    label_height = max(28, 6 + 12 * len(lines))
    tile = Image.new("RGB", (tile_size, tile_size + label_height), "white")
    tile.paste(image, (0, 0))
    draw = ImageDraw.Draw(tile)
    if selected:
        draw.rectangle((1, 1, tile_size - 2, tile_size - 2), outline="black", width=5)
    y = tile_size + 4
    for line in lines:
        draw.text((6, y), line, fill="black", font=font)
        y += 12
    return tile


def make_case_panel(
    row: pd.Series,
    gt: torch.Tensor,
    arm1: torch.Tensor,
    arm1_aligned: torch.Tensor,
    arm2: torch.Tensor,
    arm2_aligned: torch.Tensor,
    tile_size: int,
) -> Image.Image:
    selected_variant = str(row["fresh2_selected_variant"])
    tiles = [
        labeled_tile(tensor_to_pil(gt), f"GT image {row.image_id}", tile_size),
        labeled_tile(
            tensor_to_pil(arm1),
            f"Arm1 raw PSNR {row.arm1_psnr:.2f} loss {row.arm1_exact_operator_loss:.2f}",
            tile_size,
            selected_variant == "base_full",
        ),
        labeled_tile(
            tensor_to_pil(arm1_aligned),
            f"Arm1 best orient {row.arm1_best_orientation_psnr:.2f} rot180={bool(row.arm1_best_uses_rot180)}",
            tile_size,
        ),
        labeled_tile(
            tensor_to_pil(arm2),
            f"Arm2 raw PSNR {row.arm2_psnr:.2f} loss {row.arm2_exact_operator_loss:.2f}",
            tile_size,
            selected_variant == "base_extra",
        ),
        labeled_tile(
            tensor_to_pil(arm2_aligned),
            f"Arm2 best orient {row.arm2_best_orientation_psnr:.2f} rot180={bool(row.arm2_best_uses_rot180)}",
            tile_size,
        ),
    ]
    gap = 8
    header_h = 42
    width = sum(tile.width for tile in tiles) + gap * (len(tiles) - 1)
    height = header_h + max(tile.height for tile in tiles)
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    header = (
        f"group={row.atlas_group} row={int(row.row_id):03d} image={row.image_id} "
        f"selected={selected_variant} selected_raw={row.fresh2_selected_psnr:.2f} "
        f"selected_best_orient={row.selected_best_orientation_psnr:.2f}"
    )
    draw.text((6, 6), header, fill="black", font=font)
    draw.text(
        (6, 22),
        f"offline_failure_mode={row.offline_failure_mode}  official benchmark label is unchanged",
        fill="black",
        font=font,
    )
    x = 0
    for tile in tiles:
        panel.paste(tile, (x, header_h))
        x += tile.width + gap
    return panel


def stack_panels(panels: Iterable[Image.Image], gap: int = 12) -> Image.Image:
    panel_list = list(panels)
    if not panel_list:
        return Image.new("RGB", (256, 64), "white")
    width = max(panel.width for panel in panel_list)
    height = sum(panel.height for panel in panel_list) + gap * (len(panel_list) - 1)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for panel in panel_list:
        sheet.paste(panel, (0, y))
        y += panel.height + gap
    return sheet


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--tile-size", type=int, default=192)
    parser.add_argument("--good-threshold", type=float, default=25.0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows_path = args.rows.resolve()
    outdir = args.outdir.resolve()
    image_root = args.image_root.resolve()
    report_path = args.report.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "cases").mkdir(exist_ok=True)
    (outdir / "sheets").mkdir(exist_ok=True)

    frame = pd.read_csv(rows_path, dtype={"image_id": str})
    frame["image_id"] = frame.image_id.map(lambda value: f"{int(value):05d}")
    if len(frame) != 100:
        raise ValueError(f"expected 100 B21.11 rows, got {len(frame)}")
    if frame.row_id.nunique() != 100 or frame.image_id.nunique() != 100:
        raise ValueError("B21.11 rows must have 100 unique row_id and image_id values")

    failure_mask = frame.fresh2_selected_good25.astype(int).eq(0)
    rescue_mask = frame.fresh2_incremental_rescue.astype(int).eq(1)
    protected_mask = (
        frame.arm1_good25.astype(int).eq(1)
        & frame.arm2_good25.astype(int).eq(0)
        & frame.arm2_accepted.astype(int).eq(0)
        & frame.fresh2_selected_good25.astype(int).eq(1)
    )
    if bool((failure_mask & rescue_mask).any() or (failure_mask & protected_mask).any() or (rescue_mask & protected_mask).any()):
        raise ValueError("atlas groups overlap")

    frame["atlas_group"] = "not_in_atlas"
    frame.loc[failure_mask, "atlas_group"] = "persistent_failure"
    frame.loc[rescue_mask, "atlas_group"] = "fresh2_rescue"
    frame.loc[protected_mask, "atlas_group"] = "protected_fresh1_success"
    atlas = frame.loc[frame.atlas_group.ne("not_in_atlas")].copy()

    counts = atlas.atlas_group.value_counts().to_dict()
    for group, expected in EXPECTED_COUNTS.items():
        observed = int(counts.get(group, 0))
        if observed != expected:
            raise ValueError(f"expected {expected} rows for {group}, got {observed}")
    if len(atlas) != 27:
        raise ValueError(f"expected 27 atlas rows, got {len(atlas)}")

    gt_cache: dict[str, torch.Tensor] = {}
    enriched_rows: list[dict[str, object]] = []
    image_payloads: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    for rec in atlas.sort_values(["atlas_group", "row_id"]).itertuples(index=False):
        image_id = f"{int(rec.image_id):05d}"
        if image_id not in gt_cache:
            gt_cache[image_id] = load_gt(find_gt(image_root, image_id))
        gt = gt_cache[image_id]
        arm1 = load_sample(Path(str(rec.arm1_sample_path)))
        arm2 = load_sample(Path(str(rec.arm2_sample_path)))
        a1_raw, a1_rot, a1_best, a1_use_rot, a1_aligned = best_orientation(arm1, gt)
        a2_raw, a2_rot, a2_best, a2_use_rot, a2_aligned = best_orientation(arm2, gt)

        if abs(a1_raw - float(rec.arm1_psnr)) > 1e-4:
            raise ValueError(f"arm1 PSNR recomputation mismatch for row {rec.row_id}")
        if abs(a2_raw - float(rec.arm2_psnr)) > 1e-4:
            raise ValueError(f"arm2 PSNR recomputation mismatch for row {rec.row_id}")

        selected_is_arm2 = str(rec.fresh2_selected_variant) == "base_extra"
        selected_raw = a2_raw if selected_is_arm2 else a1_raw
        selected_rot = a2_rot if selected_is_arm2 else a1_rot
        selected_best = a2_best if selected_is_arm2 else a1_best
        selected_best_uses_rot = a2_use_rot if selected_is_arm2 else a1_use_rot
        any_best = max(a1_best, a2_best)
        official_failure = int(rec.fresh2_selected_good25) == 0
        selected_resolved = official_failure and selected_best >= args.good_threshold
        unselected_resolved = official_failure and selected_best < args.good_threshold and any_best >= args.good_threshold
        persistent_after_rot = official_failure and any_best < args.good_threshold
        if selected_resolved:
            offline_mode = "selected_rot180_resolvable"
        elif unselected_resolved:
            offline_mode = "unselected_candidate_rot180_resolvable"
        elif persistent_after_rot:
            offline_mode = "persistent_after_rot180"
        else:
            offline_mode = "not_applicable"

        row = rec._asdict()
        row.update({
            "arm1_rot180_psnr": a1_rot,
            "arm1_best_orientation_psnr": a1_best,
            "arm1_best_uses_rot180": int(a1_use_rot),
            "arm2_rot180_psnr": a2_rot,
            "arm2_best_orientation_psnr": a2_best,
            "arm2_best_uses_rot180": int(a2_use_rot),
            "selected_rot180_psnr": selected_rot,
            "selected_best_orientation_psnr": selected_best,
            "selected_best_uses_rot180": int(selected_best_uses_rot),
            "portfolio_best_orientation_psnr": any_best,
            "selected_rot180_resolvable_failure": int(selected_resolved),
            "unselected_candidate_rot180_resolvable_failure": int(unselected_resolved),
            "persistent_failure_after_rot180": int(persistent_after_rot),
            "offline_failure_mode": offline_mode,
        })
        enriched_rows.append(row)
        image_payloads[int(rec.row_id)] = (gt, arm1, a1_aligned, arm2, a2_aligned)

    enriched = pd.DataFrame(enriched_rows).sort_values(["atlas_group", "row_id"]).reset_index(drop=True)
    numeric_required = [
        column for column in enriched.columns
        if column.endswith("_psnr")
        or column.endswith("_loss")
        or column.startswith("candidate_disagreement_")
    ]
    if not np.isfinite(enriched[numeric_required].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite numeric value in atlas")

    group_panels: dict[str, list[Image.Image]] = {group: [] for group in EXPECTED_COUNTS}
    case_paths: list[str] = []
    for _, row in enriched.iterrows():
        gt, arm1, a1_aligned, arm2, a2_aligned = image_payloads[int(row.row_id)]
        panel = make_case_panel(row, gt, arm1, a1_aligned, arm2, a2_aligned, args.tile_size)
        case_path = outdir / "cases" / f"{row.atlas_group}_row{int(row.row_id):03d}_ffhq{row.image_id}.png"
        panel.save(case_path)
        case_paths.append(str(case_path))
        group_panels[str(row.atlas_group)].append(panel)

    sheet_paths: dict[str, str] = {}
    for group, panels in group_panels.items():
        sheet = stack_panels(panels)
        sheet_path = outdir / "sheets" / f"{group}.png"
        sheet.save(sheet_path)
        sheet_paths[group] = str(sheet_path)

    atlas_csv = outdir / "failure_atlas_rows.csv"
    enriched.to_csv(atlas_csv, index=False)
    for group in EXPECTED_COUNTS:
        enriched.loc[enriched.atlas_group.eq(group)].to_csv(outdir / f"{group}_rows.csv", index=False)

    failure_rows = enriched.loc[enriched.atlas_group.eq("persistent_failure")].copy()
    manual = failure_rows[[
        "row_id",
        "image_id",
        "fresh2_selected_variant",
        "fresh2_selected_psnr",
        "selected_best_orientation_psnr",
        "portfolio_best_orientation_psnr",
        "offline_failure_mode",
        "candidate_disagreement_rmse_rot180",
        "candidate_disagreement_l1_rot180",
    ]].copy()
    manual["manual_primary_category"] = ""
    manual["manual_secondary_category"] = ""
    manual["manual_notes"] = ""
    manual["reviewed_by"] = ""
    manual["review_date"] = ""
    manual_path = outdir / "manual_failure_labels_template.csv"
    manual.to_csv(manual_path, index=False)

    rot_selected = int(failure_rows.selected_rot180_resolvable_failure.sum())
    rot_unselected = int(failure_rows.unselected_candidate_rot180_resolvable_failure.sum())
    persistent_after = int(failure_rows.persistent_failure_after_rot180.sum())
    summary = {
        "question": "What visual and ambiguity-resolved failure modes remain after the frozen B21.11 Fresh2 policy?",
        "status": "descriptive_zero_gpu_atlas",
        "source_rows": str(rows_path),
        "official_benchmark_unchanged": True,
        "official_fresh2_good25": 92,
        "official_fresh2_bad25": 8,
        "atlas_rows": len(enriched),
        "group_counts": {group: int((enriched.atlas_group == group).sum()) for group in EXPECTED_COUNTS},
        "failure_orientation_diagnostics": {
            "official_failures": len(failure_rows),
            "selected_candidate_resolved_by_rot180": rot_selected,
            "only_unselected_candidate_resolved_by_rot180": rot_unselected,
            "persistent_after_rot180_oracle": persistent_after,
        },
        "artifacts": {
            "atlas_rows_csv": str(atlas_csv),
            "manual_failure_labels_template": str(manual_path),
            "group_sheets": sheet_paths,
            "case_panels": case_paths,
        },
        "interpretation_guardrail": (
            "The 180-degree alignment uses ground truth offline and cannot alter the frozen runtime policy, "
            "the official 92/100 raw-PSNR result, or the final-panel method decision."
        ),
    }
    summary_path = outdir / "failure_atlas_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True) + "\n")

    report_lines = [
        "# B21.12 zero-GPU failure and selector atlas",
        "",
        "## Scope",
        "",
        "This is a descriptive audit of the frozen B21.11 outputs. It performs no solver run, threshold fitting, candidate selection change, or policy tuning.",
        "",
        "## Atlas composition",
        "",
        f"- persistent Fresh2 failures: `{EXPECTED_COUNTS['persistent_failure']}`",
        f"- Fresh2 rescues over Fresh1: `{EXPECTED_COUNTS['fresh2_rescue']}`",
        f"- protected Fresh1 successes where arm 2 was bad and correctly rejected: `{EXPECTED_COUNTS['protected_fresh1_success']}`",
        f"- total atlas cases: `{len(enriched)}`",
        "",
        "## Offline 180-degree ambiguity audit",
        "",
        f"- official raw-PSNR failures: `{len(failure_rows)}`",
        f"- selected candidate becomes good25 after offline rot180 alignment: `{rot_selected}`",
        f"- only the unselected candidate becomes good25 after offline rot180 alignment: `{rot_unselected}`",
        f"- neither candidate reaches good25 under identity or rot180: `{persistent_after}`",
        "",
        "These aligned metrics use ground truth only for interpretation. They do not revise the official B21.11 result or authorize a runtime orientation resolver.",
        "",
        "## Manual review",
        "",
        "Use `manual_failure_labels_template.csv` to record descriptive visual categories for the failures that remain after the objective rot180 audit. Do not derive a runtime trigger or tune a threshold from these 100 final images.",
        "",
        "## Artifacts",
        "",
        f"- summary: `{summary_path}`",
        f"- full rows: `{atlas_csv}`",
        f"- manual labels: `{manual_path}`",
        f"- sheets: `{outdir / 'sheets'}`",
        f"- per-case panels: `{outdir / 'cases'}`",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines))

    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
    print(f"[write] {summary_path}")
    print(f"[write] {atlas_csv}")
    print(f"[write] {manual_path}")
    print(f"[write] {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
