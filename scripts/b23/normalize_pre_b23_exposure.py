#!/usr/bin/env python3
"""Collapse unresolved measurement tags to conservative image-wide exclusions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLUMNS = (
    "image_id", "measurement_id", "dataset_split", "first_project_stage",
    "roles_seen", "ground_truth_inspected", "artifacts", "exclusion_reason",
    "source_evidence",
)


def merge(values: list[str]) -> str:
    tokens: set[str] = set()
    for value in values:
        tokens.update(item for item in value.split(";") if item)
    return ";".join(sorted(tokens))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("unexpected PRE_B23_EXPOSURE columns")
        source = list(reader)
    unknown_images = {
        row["image_id"] for row in source
        if row["measurement_id"] == "UNKNOWN_ALL_MEASUREMENTS"
        or "DERIVED_SEED_UNRESOLVED" in row["measurement_id"]
    }
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in source:
        measurement = (
            "UNKNOWN_ALL_MEASUREMENTS"
            if row["image_id"] in unknown_images else row["measurement_id"]
        )
        groups.setdefault((row["image_id"], measurement), []).append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for (image_id, measurement), rows in sorted(groups.items()):
            writer.writerow({
                "image_id": image_id,
                "measurement_id": measurement,
                "dataset_split": "FFHQ_PRE_B23_EXPOSED",
                "first_project_stage": min(row["first_project_stage"] for row in rows),
                "roles_seen": "|".join(sorted({token for row in rows for token in row["roles_seen"].split("|") if token})),
                "ground_truth_inspected": "true" if any(row["ground_truth_inspected"] == "true" for row in rows) else "unknown",
                "artifacts": merge([row["artifacts"] for row in rows]),
                "exclusion_reason": "PRE_B23_EXPOSURE_CONSERVATIVE",
                "source_evidence": merge([row["source_evidence"] for row in rows]),
            })
    print(
        f"status=NORMALIZED input_rows={len(source)} output_rows={len(groups)} "
        f"image_wide_unknown={len(unknown_images)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
