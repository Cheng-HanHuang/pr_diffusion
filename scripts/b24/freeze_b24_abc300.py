#!/usr/bin/env python3
"""Freeze the first 100 A, 100 B, and 100 C B24 cases by B24_CLASS_RANK_V1.

This is a zero-GPU registry operation.  It consumes a previously validated
B24_7424_CASE_UNIVERSE.csv and freezes a deterministic 300-case balanced cohort.
The cohort role remains planner-pending; this script does not decide development
versus held-out usage and does not execute any method.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prdiffusion.b24_protocol import class_rank_key  # noqa: E402

N_PER_CLASS = 100
TARGET_CLASSES = ("A", "B", "C")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("refusing empty ABC300 freeze")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-csv", type=Path, required=True)
    ap.add_argument("--universe-summary", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    universe_csv = args.universe_csv.resolve()
    universe_summary_path = args.universe_summary.resolve()
    out_dir = args.out_dir.resolve()
    if not universe_csv.is_file():
        raise FileNotFoundError(universe_csv)
    if not universe_summary_path.is_file():
        raise FileNotFoundError(universe_summary_path)

    summary = json.loads(universe_summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise RuntimeError(f"7424 universe is not PASS: {summary.get('status')}")
    if int(summary.get("row_count", -1)) != 7424:
        raise RuntimeError("7424 universe summary row-count mismatch")
    if summary.get("case_universe_csv_sha256") != sha256_file(universe_csv):
        raise RuntimeError("7424 universe CSV SHA mismatch")

    with universe_csv.open(newline="", encoding="utf-8") as f:
        rows = [dict(row) for row in csv.DictReader(f)]
    if len(rows) != 7424:
        raise RuntimeError(f"expected 7424 universe rows, got {len(rows)}")
    if len({r["image_id"] for r in rows}) != 7424:
        raise RuntimeError("duplicate image IDs in universe")

    counts = Counter(r["class_label"] for r in rows)
    for label in TARGET_CLASSES:
        if counts[label] < N_PER_CLASS:
            raise RuntimeError(
                f"insufficient class {label}: {counts[label]} < {N_PER_CLASS}; refusing cohort freeze"
            )

    # Recompute every stored class-rank key from the frozen domain before selection.
    for row in rows:
        label = row["class_label"]
        expected = class_rank_key(label, row["image_id"])
        if row["class_rank_sha256"] != expected:
            raise RuntimeError(
                f"class-rank mismatch image={row['image_id']} class={label}: "
                f"{row['class_rank_sha256']} != {expected}"
            )

    selected: list[dict] = []
    selected_rank_refs: dict[str, list[dict]] = {}
    for label in TARGET_CLASSES:
        ranked = sorted(
            (r for r in rows if r["class_label"] == label),
            key=lambda r: r["class_rank_sha256"],
        )
        chosen = ranked[:N_PER_CLASS]
        selected_rank_refs[label] = []
        for rank, row in enumerate(chosen, start=1):
            frozen = dict(row)
            frozen["balanced_class_rank"] = rank
            frozen["balanced_cohort_role"] = "FROZEN_ABC300_ROLE_PENDING_PLANNER"
            selected.append(frozen)
            selected_rank_refs[label].append(
                {
                    "rank": rank,
                    "row_index": int(row["row_index"]),
                    "image_id": row["image_id"],
                    "class_rank_sha256": row["class_rank_sha256"],
                }
            )

    if len(selected) != 300:
        raise RuntimeError(f"ABC300 size drift: {len(selected)}")
    if len({r["image_id"] for r in selected}) != 300:
        raise RuntimeError("duplicate images in ABC300")
    selected_counts = Counter(r["class_label"] for r in selected)
    if {label: selected_counts[label] for label in TARGET_CLASSES} != {
        "A": 100,
        "B": 100,
        "C": 100,
    }:
        raise RuntimeError(f"ABC300 class balance drift: {dict(selected_counts)}")

    # Canonical output order is A1..A100, B1..B100, C1..C100.
    selected.sort(key=lambda r: (TARGET_CLASSES.index(r["class_label"]), int(r["balanced_class_rank"])))

    out_dir.mkdir(parents=True, exist_ok=True)
    cohort_csv = out_dir / "B24_ABC300_FROZEN.csv"
    write_csv_atomic(cohort_csv, selected)
    cohort_csv_sha = sha256_file(cohort_csv)

    cohort_json = out_dir / "B24_ABC300_FROZEN.json"
    write_json_atomic(
        cohort_json,
        {
            "schema_version": "b24.abc300-frozen.v1",
            "selection_rule": "first 100 within A/B/C by B24_CLASS_RANK_V1",
            "role": "FROZEN_ABC300_ROLE_PENDING_PLANNER",
            "source_universe_csv_sha256": sha256_file(universe_csv),
            "source_manifest_file_sha256": summary["manifest_file_sha256"],
            "class_counts": {"A": 100, "B": 100, "C": 100},
            "class_rank_domain": "B24_CLASS_RANK_V1",
            "cases": selected,
        },
    )
    cohort_json_sha = sha256_file(cohort_json)

    freeze_summary = {
        "schema_version": "b24.abc300-freeze-summary.v1",
        "status": "PASS",
        "gpu_work_performed": False,
        "measurement_generation_performed": False,
        "reconstruction_performed": False,
        "method_execution_performed": False,
        "planner_role_assignment_performed": False,
        "selection_rule": "first 100 within A/B/C by B24_CLASS_RANK_V1",
        "role": "FROZEN_ABC300_ROLE_PENDING_PLANNER",
        "source_universe_csv": str(universe_csv),
        "source_universe_csv_sha256": sha256_file(universe_csv),
        "source_universe_summary": str(universe_summary_path),
        "source_manifest_file_sha256": summary["manifest_file_sha256"],
        "available_class_counts": summary["class_counts"],
        "frozen_class_counts": {"A": 100, "B": 100, "C": 100},
        "frozen_row_count": 300,
        "cohort_csv": str(cohort_csv),
        "cohort_csv_sha256": cohort_csv_sha,
        "cohort_json": str(cohort_json),
        "cohort_json_sha256": cohort_json_sha,
        "selected_rank_refs": selected_rank_refs,
        "next": "RETURN_TO_PLANNER_FOR_METHOD_PORTFOLIO_AND_ROLE_POLICY",
    }
    freeze_summary_path = out_dir / "B24_ABC300_FREEZE_SUMMARY.json"
    write_json_atomic(freeze_summary_path, freeze_summary)

    print(json.dumps(freeze_summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
