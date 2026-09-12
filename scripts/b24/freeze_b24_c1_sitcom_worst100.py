#!/usr/bin/env python3
"""Freeze B24 C1: the 100 class-C cases with lowest SITCOM best-of-four PSNR.

C1 is a secondary, severity-enriched diagnostic cohort.  It does NOT alter the
frozen A/B/C/D class definition and does NOT replace the primary hash-ranked C100
inside ABC300.  Selection is intentionally based on the already-realized baseline
SITCOM best-of-four raw-orientation RGB PSNR and is therefore outcome-selected.
It is suitable for SITCOM-configuration sensitivity/stress analysis, not as an
unbiased primary benchmark.

Zero-GPU registry operation only; no model load, measurement generation, or
reconstruction is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median

N = 100
ROLE = "FROZEN_C1_SITCOM_WORST100_SECONDARY_DIAGNOSTIC"
SELECTION = "class C only; ascending SITCOM-4 best raw-RGB PSNR; tie by class_rank_sha256 then row_index"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("refusing empty C1 freeze")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def finite_float(value: str, label: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"non-finite {label}: {value!r}")
    return out


def describe(values: list[float]) -> dict:
    vals = sorted(values)
    return {
        "count": len(vals),
        "min": vals[0],
        "median": median(vals),
        "max": vals[-1],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-csv", type=Path, required=True)
    ap.add_argument("--universe-summary", type=Path, required=True)
    ap.add_argument("--abc300-csv", type=Path, required=True)
    ap.add_argument("--abc300-summary", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    universe_csv = args.universe_csv.resolve()
    universe_summary_path = args.universe_summary.resolve()
    abc300_csv = args.abc300_csv.resolve()
    abc300_summary_path = args.abc300_summary.resolve()
    out_dir = args.out_dir.resolve()
    for p in (universe_csv, universe_summary_path, abc300_csv, abc300_summary_path):
        if not p.is_file():
            raise FileNotFoundError(p)

    universe_summary = read_json(universe_summary_path)
    abc300_summary = read_json(abc300_summary_path)
    if universe_summary.get("status") != "PASS":
        raise RuntimeError("universe summary is not PASS")
    if abc300_summary.get("status") != "PASS":
        raise RuntimeError("ABC300 summary is not PASS")
    if int(universe_summary.get("row_count", -1)) != 7424:
        raise RuntimeError("universe row-count mismatch")
    if universe_summary.get("case_universe_csv_sha256") != sha256_file(universe_csv):
        raise RuntimeError("universe CSV SHA mismatch")
    if abc300_summary.get("cohort_csv_sha256") != sha256_file(abc300_csv):
        raise RuntimeError("ABC300 CSV SHA mismatch")
    if abc300_summary.get("frozen_class_counts") != {"A": 100, "B": 100, "C": 100}:
        raise RuntimeError("ABC300 class-count mismatch")

    with universe_csv.open(newline="", encoding="utf-8") as f:
        universe = [dict(r) for r in csv.DictReader(f)]
    if len(universe) != 7424 or len({r["image_id"] for r in universe}) != 7424:
        raise RuntimeError("invalid universe identity/count")

    c_rows = []
    for row in universe:
        if row["class_label"] != "C":
            continue
        daps = finite_float(row["daps_best_psnr_raw_rgb_db"], "DAPS PSNR")
        sitcom = finite_float(row["sitcom_best_psnr_raw_rgb_db"], "SITCOM PSNR")
        # Reassert class-C Good25 semantics from the stored baseline values.
        if not (daps >= 25.0 and sitcom < 25.0):
            raise RuntimeError(
                f"class-C semantic drift image={row['image_id']}: daps={daps} sitcom={sitcom}"
            )
        item = dict(row)
        item["_daps_psnr"] = daps
        item["_sitcom_psnr"] = sitcom
        c_rows.append(item)

    expected_c = int(universe_summary["class_counts"]["C"])
    if len(c_rows) != expected_c:
        raise RuntimeError(f"class-C count drift: {len(c_rows)} != {expected_c}")
    if len(c_rows) < N:
        raise RuntimeError(f"insufficient class C: {len(c_rows)} < {N}")

    c_rows.sort(
        key=lambda r: (
            r["_sitcom_psnr"],
            r["class_rank_sha256"],
            int(r["row_index"]),
        )
    )
    chosen = c_rows[:N]

    with abc300_csv.open(newline="", encoding="utf-8") as f:
        abc300 = [dict(r) for r in csv.DictReader(f)]
    primary_c_ids = {r["image_id"] for r in abc300 if r["class_label"] == "C"}
    if len(primary_c_ids) != 100:
        raise RuntimeError(f"expected 100 primary C IDs, got {len(primary_c_ids)}")

    frozen: list[dict] = []
    for rank, row in enumerate(chosen, start=1):
        out = {k: v for k, v in row.items() if not k.startswith("_")}
        out["c1_sitcom_severity_rank"] = rank
        out["c1_role"] = ROLE
        out["c1_overlaps_primary_c100"] = str(row["image_id"] in primary_c_ids).upper()
        frozen.append(out)

    chosen_ids = {r["image_id"] for r in frozen}
    overlap_ids = sorted(chosen_ids & primary_c_ids)
    if len(frozen) != 100 or len(chosen_ids) != 100:
        raise RuntimeError("C1 identity/count drift")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "B24_C1_SITCOM_WORST100_FROZEN.csv"
    json_path = out_dir / "B24_C1_SITCOM_WORST100_FROZEN.json"
    summary_path = out_dir / "B24_C1_SITCOM_WORST100_FREEZE_SUMMARY.json"

    write_csv_atomic(csv_path, frozen)
    write_json_atomic(
        json_path,
        {
            "schema_version": "b24.c1-sitcom-worst100.v1",
            "role": ROLE,
            "selection_rule": SELECTION,
            "warning": "Outcome-selected severity cohort; secondary diagnostic only; does not replace primary hash-ranked C100.",
            "source_manifest_file_sha256": universe_summary["manifest_file_sha256"],
            "source_universe_csv_sha256": sha256_file(universe_csv),
            "source_abc300_csv_sha256": sha256_file(abc300_csv),
            "available_class_c": len(c_rows),
            "frozen_count": 100,
            "overlap_with_primary_c100_count": len(overlap_ids),
            "overlap_with_primary_c100_image_ids": overlap_ids,
            "cases": frozen,
        },
    )

    all_c_sitcom = [r["_sitcom_psnr"] for r in c_rows]
    c1_sitcom = [r["_sitcom_psnr"] for r in chosen]
    c1_daps = [r["_daps_psnr"] for r in chosen]
    summary = {
        "schema_version": "b24.c1-sitcom-worst100-freeze-summary.v1",
        "status": "PASS",
        "gpu_work_performed": False,
        "measurement_generation_performed": False,
        "reconstruction_performed": False,
        "method_execution_performed": False,
        "primary_class_definition_changed": False,
        "primary_abc300_changed": False,
        "role": ROLE,
        "selection_rule": SELECTION,
        "selection_is_outcome_based": True,
        "use_policy": "SECONDARY_SITCOM_CONFIGURATION_SENSITIVITY_OR_STRESS_TEST_ONLY",
        "available_class_c": len(c_rows),
        "frozen_count": 100,
        "source_manifest_file_sha256": universe_summary["manifest_file_sha256"],
        "source_universe_csv_sha256": sha256_file(universe_csv),
        "source_abc300_csv_sha256": sha256_file(abc300_csv),
        "overlap_with_primary_c100_count": len(overlap_ids),
        "overlap_with_primary_c100_image_ids": overlap_ids,
        "all_class_c_sitcom_psnr": describe(all_c_sitcom),
        "c1_sitcom_psnr": describe(c1_sitcom),
        "c1_daps_psnr": describe(c1_daps),
        "c1_100th_worst_sitcom_psnr_db": c1_sitcom[-1],
        "cohort_csv": str(csv_path),
        "cohort_csv_sha256": sha256_file(csv_path),
        "cohort_json": str(json_path),
        "cohort_json_sha256": sha256_file(json_path),
        "next": "RETURN_TO_PLANNER_WITH_PRIMARY_C100_AND_SECONDARY_C1_SEVERITY100_FROZEN",
    }
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
