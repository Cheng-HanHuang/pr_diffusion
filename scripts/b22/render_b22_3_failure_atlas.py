#!/usr/bin/env python3
"""Render a zero-GPU visual atlas for the B22.3 failure/complementarity union."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

POLICIES = [
    "Fresh1",
    "Fresh2",
    "SITCOM-1",
    "SITCOM-4S",
    "SITCOM-oracle4",
    "NP-1",
    "NP-8-RS",
    "NP-oracle8",
]


def load_rgb(path: Path, size: int) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, font: ImageFont.ImageFont) -> None:
    draw.text(xy, value, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))


def make_panel(image_id: str, gt_path: Path, rows: pd.DataFrame, out: Path, tile: int = 220) -> None:
    font = ImageFont.load_default()
    header, footer, gap = 42, 45, 4
    columns = [("GT", gt_path, None)]
    for policy in POLICIES:
        row = rows.loc[policy]
        columns.append((policy, Path(row.selected_png_path), row))
    width = len(columns) * tile + (len(columns) - 1) * gap
    canvas = Image.new("RGB", (width, header + tile + footer), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    for index, (label, path, row) in enumerate(columns):
        x = index * (tile + gap)
        canvas.paste(load_rgb(path, tile), (x, header))
        text(draw, (x + 4, 5), label, font)
        if row is None:
            text(draw, (x + 4, header + tile + 8), f"image {image_id}", font)
        else:
            raw = float(row.psnr_raw)
            rot = float(row.psnr_rot180)
            text(draw, (x + 4, header + tile + 5), f"raw {raw:.2f}  {'good25' if raw >= 25 else 'BAD25'}", font)
            text(draw, (x + 4, header + tile + 22), f"rot180 {rot:.2f}", font)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def thumbnail_with_label(path: Path, label: str, size: int = 260) -> Image.Image:
    font = ImageFont.load_default()
    image = Image.open(path).convert("RGB")
    ratio = size / image.width
    resized = image.resize((size, max(1, int(image.height * ratio))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, resized.height + 24), (20, 20, 20))
    canvas.paste(resized, (0, 24))
    text(ImageDraw.Draw(canvas), (4, 4), label, font)
    return canvas


def contact_sheet(panel_paths: list[tuple[Path, str]], out: Path, columns: int = 2) -> None:
    thumbs = [thumbnail_with_label(path, label) for path, label in panel_paths]
    if not thumbs:
        return
    cell_w = max(image.width for image in thumbs)
    cell_h = max(image.height for image in thumbs)
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), (10, 10, 10))
    for index, image in enumerate(thumbs):
        sheet.paste(image, ((index % columns) * cell_w, (index // columns) * cell_h))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--analysis_dir", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    stage_root = run_root / "full"
    analysis = Path(args.analysis_dir).resolve()
    atlas = analysis / "failure_atlas"
    temporary = atlas.with_name(atlas.name + ".tmp")
    if atlas.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite atlas: {atlas} or {temporary}")

    validation = json.loads((stage_root / "validation.json").read_text())
    if validation.get("status") != "PASS":
        raise RuntimeError("Failure atlas requires a validated B22.2 panel")
    paired = pd.read_csv(stage_root / "paired_rows.csv", dtype={"image_id": str})
    paired.image_id = paired.image_id.str.zfill(5)
    manifest = json.loads((stage_root / "manifest.json").read_text())
    manifest_by_id = {str(row["image_id"]).zfill(5): row for row in manifest["rows"]}
    failure_union = pd.read_csv(analysis / "tables" / "main_policy_failure_union.csv", dtype={"image_id": str})
    failure_union.image_id = failure_union.image_id.str.zfill(5)

    temporary.mkdir(parents=True)
    panels = temporary / "panels"
    panels.mkdir()
    index_rows = []
    panel_paths: list[tuple[Path, str]] = []
    for _, row in failure_union.iterrows():
        image_id = str(row["image_id"]).zfill(5)
        rows = paired.loc[paired.image_id.eq(image_id)].set_index("policy")
        missing = [policy for policy in POLICIES if policy not in rows.index]
        if missing:
            raise RuntimeError(f"Missing policies for {image_id}: {missing}")
        panel_path = panels / f"{image_id}.png"
        make_panel(image_id, Path(manifest_by_id[image_id]["ground_truth_path"]), rows, panel_path)

        fresh_bad = float(row["Fresh2"]) < 25
        sitcom_bad = float(row["SITCOM-4S"]) < 25
        np_bad = float(row["NP-8-RS"]) < 25
        categories = []
        if fresh_bad and sitcom_bad and np_bad:
            categories.append("all_three_bad")
        if fresh_bad and (not sitcom_bad or not np_bad):
            categories.append("fresh2_failure_rescued")
        if np_bad and not fresh_bad:
            categories.append("np8_failure_fresh2_good")
        if sitcom_bad and (not fresh_bad or not np_bad):
            categories.append("sitcom4s_failure_other_good")
        index_rows.append({
            "image_id": image_id,
            "panel_path": str(panel_path),
            "categories": ";".join(categories),
            "Fresh2": float(row["Fresh2"]),
            "SITCOM-4S": float(row["SITCOM-4S"]),
            "NP-8-RS": float(row["NP-8-RS"]),
            "best_policy": str(row["best_policy"]),
            "best_raw_psnr": float(row["best_raw_psnr"]),
        })
        panel_paths.append((panel_path, f"{image_id}: {', '.join(categories) or 'failure union'}"))

    with (temporary / "atlas_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    contact_sheet(panel_paths, temporary / "failure_union_contact_sheet.jpg", columns=2)

    groups = {
        "all_three_bad": [],
        "fresh2_failure_rescued": [],
        "np8_failure_fresh2_good": [],
        "sitcom4s_failure_other_good": [],
    }
    for row in index_rows:
        for category in row["categories"].split(";"):
            if category in groups:
                groups[category].append((Path(row["panel_path"]), row["image_id"]))
    for name, values in groups.items():
        contact_sheet(values, temporary / f"{name}.jpg", columns=2)

    (temporary / "README.md").write_text(
        "# B22.3 failure and complementarity atlas\n\n"
        f"- Failure-union images: `{len(index_rows)}`\n"
        "- Columns: GT, Fresh1, Fresh2, SITCOM-1, SITCOM-4S, SITCOM-oracle4, NP-1, NP-8-RS, NP-oracle8.\n"
        "- Raw PSNR is primary; rot180 PSNR is auxiliary. Oracle outputs are diagnostic only.\n\n"
        "Review priorities:\n\n"
        "1. `65003`: all three executable multi-run policies fail.\n"
        "2. Fresh2 catastrophic failures rescued by SITCOM/NP.\n"
        "3. NP-8-RS failures where Fresh2 is good.\n"
        "4. SITCOM selector misses `60140` and `64518`.\n"
        "5. NP selector miss `65269`.\n",
        encoding="utf-8",
    )
    temporary.replace(atlas)
    print(json.dumps({"status": "PASS", "atlas": str(atlas), "panels": len(index_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
