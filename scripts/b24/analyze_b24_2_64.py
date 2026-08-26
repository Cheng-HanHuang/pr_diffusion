#!/usr/bin/env python3
"""Aggregate the completed B24.2 64-image baseline tranche."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runroot", type=Path, required=True)
    args = ap.parse_args()
    root = args.runroot.resolve()

    shard_rows = []
    completions = []
    for shard in range(4):
        sroot = root / f"shard{shard}"
        summary_path = sroot / "SHARD_COMPLETE.json"
        if not summary_path.is_file():
            print(json.dumps({"status": "INCOMPLETE", "missing": str(summary_path)}, sort_keys=True))
            return 2
        summary = read_json(summary_path)
        if summary.get("status") != "PASS" or int(summary.get("completed", -1)) != 16:
            raise RuntimeError(f"bad shard summary: {summary_path}")
        shard_rows.append(summary)
        for path in sorted(sroot.glob("row*/IMAGE_COMPLETE.json")):
            row = read_json(path)
            if row.get("status") != "PASS":
                raise RuntimeError(f"non-PASS image completion: {path}")
            completions.append(row)

    if len(completions) != 64:
        raise RuntimeError(f"expected 64 image completions, got {len(completions)}")
    row_ids = [int(r["row_index"]) for r in completions]
    image_ids = [r["image_id"] for r in completions]
    if sorted(row_ids) != list(range(64)) or len(set(image_ids)) != 64:
        raise RuntimeError("row/image coverage mismatch")
    manifest_shas = {r["manifest_file_sha256"] for r in completions}
    heads = {r["b24_head"] for r in completions}
    if len(manifest_shas) != 1 or len(heads) != 1:
        raise RuntimeError("mixed manifest/head identities across 64-image tranche")

    counts = Counter(r["class_label"] for r in completions)
    if set(counts) - {"A", "B", "C", "D"}:
        raise RuntimeError(f"unknown class labels: {counts}")
    prevalence = {c: counts.get(c, 0) / 64.0 for c in "ABCD"}
    max_shard_wall = max(float(r["shard_wall_seconds"]) for r in shard_rows)
    throughput = 64.0 / (max_shard_wall / 3600.0)

    daps_group = [float(r["daps_group_wall_seconds"]) for r in completions]
    sitcom_group = [float(r["sitcom_group_wall_seconds"]) for r in completions]
    image_wall = [float(r["image_wall_seconds"]) for r in completions]
    daps_psnr = [float(r["daps_best_psnr_raw_rgb_db"]) for r in completions]
    sitcom_psnr = [float(r["sitcom_best_psnr_raw_rgb_db"]) for r in completions]

    naive_n100 = {
        c: (100.0 / prevalence[c] if prevalence[c] > 0 else None)
        for c in "ABCD"
    }
    naive_n200 = {
        c: (200.0 / prevalence[c] if prevalence[c] > 0 else None)
        for c in "ABCD"
    }
    payload = {
        "schema_version": "b24.baseline-64-summary.v1",
        "stage": "B24.2_64",
        "status": "PASS",
        "n_images": 64,
        "manifest_file_sha256": next(iter(manifest_shas)),
        "b24_head": next(iter(heads)),
        "class_counts": {c: counts.get(c, 0) for c in "ABCD"},
        "class_prevalence": prevalence,
        "evaluation_representation": "CANONICAL_SAVED_RGB_8BIT_RAW_ORIENTATION_V1",
        "good25_threshold_db": 25.0,
        "max_shard_wall_seconds": max_shard_wall,
        "observed_images_per_hour_four_gpu": throughput,
        "daps4_group_wall_seconds": {
            "mean": statistics.mean(daps_group), "median": statistics.median(daps_group), "max": max(daps_group)
        },
        "sitcom4_group_wall_seconds": {
            "mean": statistics.mean(sitcom_group), "median": statistics.median(sitcom_group), "max": max(sitcom_group)
        },
        "per_image_wall_seconds": {
            "mean": statistics.mean(image_wall), "median": statistics.median(image_wall), "max": max(image_wall)
        },
        "best_psnr_raw_rgb_db": {
            "DAPS": {"mean": statistics.mean(daps_psnr), "median": statistics.median(daps_psnr)},
            "SITCOM": {"mean": statistics.mean(sitcom_psnr), "median": statistics.median(sitcom_psnr)},
        },
        "naive_total_images_for_100_per_class_from_point_prevalence": naive_n100,
        "naive_total_images_for_200_per_class_from_point_prevalence": naive_n200,
        "planning_note": "Point-prevalence totals are descriptive only; rare-class uncertainty must be included before scaling beyond the reviewed cumulative tranche.",
    }
    out = root / "B24_2_64_SUMMARY.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS", "class_counts": payload["class_counts"],
        "images_per_hour": throughput, "summary": str(out),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
