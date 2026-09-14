#!/usr/bin/env python3
"""Freeze B24 method-development image roles and the 16-image pilot.

Zero-GPU only. Reads the already-frozen 7424 universe, ABC300, and C1 files.
It never generates a measurement or runs a reconstruction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

SCREEN_MANIFEST_SHA = "b516c8154cbbb790d8a3592b86736bb0d4bd47d0833d85ecf3d6a9d710e950ba"
UNIVERSE_SHA = "4c6eeabe7d73f948ff0820a7f8c87ed091ff94100fc5def32586c5581c449f25"
ABC300_SHA = "4599c2a8c1f4a5922640e0c26d2c1efce7f1996d75dcabbff2e9a1c4b427cbce"
C1_SHA = "9c04994dbd91f6a5bb04280736e4dea346c337508a89f30ae8553a42867576b6"
DEV_PER_CLASS = 20
PILOT_PER_CLASS = 4
TARGET_COUNTS = {"A": 100, "B": 100, "C": 100, "D": 85}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def domain_hash(domain: str, *parts: object) -> str:
    material = "|".join([domain, SCREEN_MANIFEST_SHA, *[str(x) for x in parts]])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def seed63(domain: str, *parts: object) -> int:
    return int(domain_hash(domain, *parts)[:16], 16) & ((1 << 63) - 1)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("refusing empty role freeze")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe-csv", type=Path, required=True)
    ap.add_argument("--abc300-csv", type=Path, required=True)
    ap.add_argument("--c1-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    universe = args.universe_csv.resolve()
    abc300 = args.abc300_csv.resolve()
    c1 = args.c1_csv.resolve()
    out = args.out_dir.resolve()
    for p, expected in ((universe, UNIVERSE_SHA), (abc300, ABC300_SHA), (c1, C1_SHA)):
        if not p.is_file():
            raise FileNotFoundError(p)
        observed = sha256_file(p)
        if observed != expected:
            raise RuntimeError(f"input SHA mismatch: {p}: {observed} != {expected}")

    universe_rows = read_csv(universe)
    abc_rows = read_csv(abc300)
    c1_rows = read_csv(c1)
    if len(universe_rows) != 7424:
        raise RuntimeError(f"expected 7424 universe rows, got {len(universe_rows)}")
    if len(abc_rows) != 300:
        raise RuntimeError(f"expected 300 ABC rows, got {len(abc_rows)}")
    if len(c1_rows) != 100:
        raise RuntimeError(f"expected 100 C1 rows, got {len(c1_rows)}")

    by_class: dict[str, list[dict[str, str]]] = {}
    for label in "ABC":
        vals = [r for r in abc_rows if r["class_label"] == label]
        if len(vals) != 100:
            raise RuntimeError(f"ABC300 class {label} count drift: {len(vals)}")
        by_class[label] = vals
    d_rows = [r for r in universe_rows if r["class_label"] == "D"]
    if len(d_rows) != 85:
        raise RuntimeError(f"D count drift: {len(d_rows)}")
    by_class["D"] = d_rows

    c1_ids = {r["image_id"] for r in c1_rows}
    panel_rows: list[dict] = []
    pilot_rows: list[dict] = []

    for label in "ABCD":
        vals = by_class[label]
        ranked = sorted(vals, key=lambda r: domain_hash("B24_METHOD_ROLE_V1", label, r["image_id"]))
        if len(ranked) != TARGET_COUNTS[label]:
            raise RuntimeError(f"source count drift for {label}: {len(ranked)}")
        dev_ids = {r["image_id"] for r in ranked[:DEV_PER_CLASS]}
        dev_ranked = sorted(
            ranked[:DEV_PER_CLASS],
            key=lambda r: domain_hash("B24_METHOD_PILOT_V1", label, r["image_id"]),
        )
        pilot_ids = {r["image_id"] for r in dev_ranked[:PILOT_PER_CLASS]}

        for class_rank, r in enumerate(ranked, start=1):
            image = r["image_id"]
            role = "DEVELOPMENT" if image in dev_ids else "CONFIRMATION"
            pilot = image in pilot_ids
            if label == "C" and image in c1_ids:
                c1_policy = (
                    "C1_DEVELOPMENT_ALLOWED_BY_PRIMARY_C_ROLE"
                    if role == "DEVELOPMENT"
                    else "C1_CONFIRMATION_LOCKED"
                )
            else:
                c1_policy = "NOT_IN_C1"
            row = {
                "class_label": label,
                "image_id": image,
                "source_row_index": int(r["row_index"]),
                "source_measurement_seed": int(r["measurement_seed"]),
                "method_role_rank": class_rank,
                "method_role_rank_sha256": domain_hash("B24_METHOD_ROLE_V1", label, image),
                "method_role": role,
                "pilot16": str(pilot).upper(),
                "pilot_rank_sha256": domain_hash("B24_METHOD_PILOT_V1", label, image),
                "c1_member": str(image in c1_ids).upper(),
                "c1_policy": c1_policy,
                "dev_measurement_seed": (
                    seed63("B24_METHOD_DEV_MEAS_V1", label, image) if role == "DEVELOPMENT" else ""
                ),
                "np_root_seed_0": (
                    seed63("B24_METHOD_NP_ROOT_V1", label, image, 0) if role == "DEVELOPMENT" else ""
                ),
                "np_root_seed_1": (
                    seed63("B24_METHOD_NP_ROOT_V1", label, image, 1) if role == "DEVELOPMENT" else ""
                ),
                "np_root_seed_2": (
                    seed63("B24_METHOD_NP_ROOT_V1", label, image, 2) if role == "DEVELOPMENT" else ""
                ),
                "np_root_seed_3": (
                    seed63("B24_METHOD_NP_ROOT_V1", label, image, 3) if role == "DEVELOPMENT" else ""
                ),
            }
            if role == "DEVELOPMENT" and int(row["dev_measurement_seed"]) == int(row["source_measurement_seed"]):
                raise RuntimeError(f"new-measurement seed collision for {label}/{image}")
            panel_rows.append(row)
            if pilot:
                pilot_rows.append(dict(row))

    # C1 images outside primary C100 are explicitly development-locked.
    primary_c_ids = {r["image_id"] for r in by_class["C"]}
    outside_c1 = sorted(c1_ids - primary_c_ids)
    if len(outside_c1) != 69:
        raise RuntimeError(f"expected 69 C1 cases outside primary C100, got {len(outside_c1)}")

    if len(panel_rows) != 385 or len({r["image_id"] for r in panel_rows}) != 385:
        raise RuntimeError("primary 385-panel identity/count drift")
    if len(pilot_rows) != 16 or len({r["image_id"] for r in pilot_rows}) != 16:
        raise RuntimeError("pilot16 identity/count drift")

    role_counts = Counter((r["class_label"], r["method_role"]) for r in panel_rows)
    expected_role_counts = {
        ("A", "DEVELOPMENT"): 20, ("A", "CONFIRMATION"): 80,
        ("B", "DEVELOPMENT"): 20, ("B", "CONFIRMATION"): 80,
        ("C", "DEVELOPMENT"): 20, ("C", "CONFIRMATION"): 80,
        ("D", "DEVELOPMENT"): 20, ("D", "CONFIRMATION"): 65,
    }
    if dict(role_counts) != expected_role_counts:
        raise RuntimeError(f"role-count drift: {dict(role_counts)}")
    pilot_counts = Counter(r["class_label"] for r in pilot_rows)
    if dict(pilot_counts) != {"A": 4, "B": 4, "C": 4, "D": 4}:
        raise RuntimeError(f"pilot-count drift: {dict(pilot_counts)}")

    out.mkdir(parents=True, exist_ok=True)
    panel_csv = out / "B24_METHOD_IMAGE_ROLES.csv"
    pilot_csv = out / "B24_METHOD_PILOT16.csv"
    summary_json = out / "B24_METHOD_ROLE_FREEZE_SUMMARY.json"
    panel_json = out / "B24_METHOD_IMAGE_ROLES.json"
    write_csv_atomic(panel_csv, panel_rows)
    write_csv_atomic(pilot_csv, sorted(pilot_rows, key=lambda r: (r["class_label"], r["pilot_rank_sha256"])))
    write_json_atomic(panel_json, {
        "schema_version": "b24.method-image-roles.v1",
        "screen_manifest_sha256": SCREEN_MANIFEST_SHA,
        "role_rank_domain": "B24_METHOD_ROLE_V1",
        "pilot_rank_domain": "B24_METHOD_PILOT_V1",
        "development_measurement_seed_domain": "B24_METHOD_DEV_MEAS_V1",
        "np_root_seed_domain": "B24_METHOD_NP_ROOT_V1",
        "rows": panel_rows,
        "c1_outside_primary_c100": [
            {
                "image_id": image,
                "policy": "C1_SECONDARY_LOCKED_UNTIL_MAIN_METHOD_FREEZE",
            }
            for image in outside_c1
        ],
    })
    summary = {
        "schema_version": "b24.method-role-freeze-summary.v1",
        "status": "PASS",
        "gpu_work_performed": False,
        "measurement_generation_performed": False,
        "method_execution_performed": False,
        "screen_manifest_sha256": SCREEN_MANIFEST_SHA,
        "source_universe_csv_sha256": UNIVERSE_SHA,
        "source_abc300_csv_sha256": ABC300_SHA,
        "source_c1_csv_sha256": C1_SHA,
        "primary_panel_count": 385,
        "development_count": 80,
        "confirmation_count": 305,
        "role_counts": {
            label: {
                "development": role_counts[(label, "DEVELOPMENT")],
                "confirmation": role_counts[(label, "CONFIRMATION")],
            }
            for label in "ABCD"
        },
        "pilot_count": 16,
        "pilot_counts": {label: pilot_counts[label] for label in "ABCD"},
        "c1_count": 100,
        "c1_outside_primary_c100_count": len(outside_c1),
        "c1_outside_primary_policy": "LOCKED_UNTIL_MAIN_METHOD_FREEZE",
        "panel_csv": str(panel_csv),
        "panel_csv_sha256": sha256_file(panel_csv),
        "panel_json": str(panel_json),
        "panel_json_sha256": sha256_file(panel_json),
        "pilot_csv": str(pilot_csv),
        "pilot_csv_sha256": sha256_file(pilot_csv),
        "next": "IMPLEMENT_AND_SMOKE_B24_3_NP_BRANCHING_ON_PILOT16_AFTER_EXPLICIT_GPU_AUTHORIZATION",
    }
    write_json_atomic(summary_json, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
