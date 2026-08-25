#!/usr/bin/env python3
"""Validate the B23.0 repository contract without importing a GPU launcher."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    "docs/b23/00_START_HERE.md",
    "docs/b23/01_PROTOCOL_AND_GATES.md",
    "docs/b23/02_PARENT_SEMANTICS_AND_COMPATIBILITY.md",
    "docs/b23/03_TYPED_STATE_MODULE_ADAPTER_API.md",
    "docs/b23/04_COMPUTE_LEDGER_AND_FRE_SPEC.md",
    "docs/b23/05_REPLAY_DETERMINISM_AND_RNG_POLICY.md",
    "docs/b23/06_PRE_B23_EXPOSURE_MANIFEST.md",
    "docs/b23/07_PAC_INVENTORY_AND_FREEZE.md",
    "docs/b23/08_HISTORICAL_DEAD_DIRECTIONS.md",
    "docs/b23/09_B23_1_REPLAY_RUNBOOK.md",
    "docs/b23/10_B23_2_PREREGISTRATION_TEMPLATE.md",
    "docs/b23/B23_0_CORRECTION_LEDGER.md",
    "configs/b23/fresh1_frozen.yaml",
    "configs/b23/lf_v1_frozen.yaml",
    "configs/b23/np1_frozen.yaml",
    "configs/b23/sitcom1_frozen.yaml",
    "configs/b23/replay_policy.yaml",
    "manifests/b23/PRE_B23_EXPOSURE.csv",
    "manifests/b23/future_split_registry.csv",
    "manifests/b23/b23_0_correction_ledger.json",
    "schemas/b23/compute_ledger.schema.json",
    "schemas/b23/replay_report.schema.json",
    "schemas/b23/future_split_registry.schema.json",
)

EXPOSURE_COLUMNS = (
    "image_id",
    "measurement_id",
    "dataset_split",
    "first_project_stage",
    "roles_seen",
    "ground_truth_inspected",
    "artifacts",
    "exclusion_reason",
    "source_evidence",
)

FUTURE_COLUMNS = (
    "registry_version",
    "split",
    "row_id",
    "image_id",
    "measurement_id",
    "measurement_seed",
    "solver_base_seed",
    "assigned_before_run",
    "pre_b23_exposure_checked",
    "source_manifest_sha256",
    "notes",
)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def check_manifest(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPOSURE_COLUMNS:
            raise ValueError(f"unexpected exposure columns: {reader.fieldnames}")
        rows = list(reader)
    keys: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, start=2):
        key = (row["image_id"], row["measurement_id"])
        if key in keys:
            raise ValueError(f"duplicate exposure key at line {index}: {key}")
        keys.add(key)
        if not row["image_id"] or not row["measurement_id"]:
            raise ValueError(f"blank exposure identity at line {index}")
        if row["ground_truth_inspected"] not in {"true", "false", "unknown"}:
            raise ValueError(f"invalid ground_truth_inspected at line {index}")
        if not row["exclusion_reason"] or not row["source_evidence"]:
            raise ValueError(f"missing exclusion evidence at line {index}")
        if "DERIVED_SEED_UNRESOLVED" in row["measurement_id"]:
            raise ValueError(
                f"unresolved measurement tag must be image-wide UNKNOWN_ALL_MEASUREMENTS at line {index}"
            )
    return {
        "rows": len(rows),
        "images": len({row["image_id"] for row in rows}),
        "truly_resolved_measurement_rows": sum(
            row["measurement_id"] != "UNKNOWN_ALL_MEASUREMENTS" for row in rows
        ),
        "unresolved_measurement_tag_rows": 0,
        "image_wide_unknown_exposure_rows": sum(
            row["measurement_id"] == "UNKNOWN_ALL_MEASUREMENTS" for row in rows
        ),
        "image_ids": sorted({row["image_id"] for row in rows}),
    }


def check_future_registry(path: Path, exposed_image_ids: set[str]) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FUTURE_COLUMNS:
            raise ValueError(f"unexpected future-registry columns: {reader.fieldnames}")
        rows = list(reader)
    if rows:
        raise ValueError("B23.0 future split registry must remain unpopulated")
    from prdiffusion.b23_protocol import validate_future_split_registry
    validate_future_split_registry(rows, exposed_image_ids)
    return {"rows": 0, "status": "EMPTY_AND_EXPOSED_IMAGE_DISJOINT"}


def validate_schema_instance(schema: dict[str, Any], instance: Any) -> str:
    from prdiffusion.b23_schema import validate_with_mode
    return validate_with_mode(instance, schema)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from prdiffusion.b23_protocol import (  # pylint: disable=import-outside-toplevel
        validate_compute_ledger,
        validate_replay_report,
    )

    checks: dict[str, Any] = {}
    missing = [name for name in REQUIRED_FILES if not (repo / name).is_file()]
    if missing:
        raise ValueError(f"missing required B23.0 files: {missing}")
    checks["required_files"] = {"count": len(REQUIRED_FILES), "status": "PASS"}

    config_names = (
        "fresh1_frozen.yaml",
        "lf_v1_frozen.yaml",
        "np1_frozen.yaml",
        "sitcom1_frozen.yaml",
        "replay_policy.yaml",
        "pac_paths.yaml",
        "b23_1_smoke_template.yaml",
    )
    configs = {name: read_json(repo / "configs/b23" / name) for name in config_names}
    if {configs[name]["parent_id"] for name in config_names[:4]} != {
        "Fresh1", "LF-v1", "NP-1", "SITCOM-1"
    }:
        raise ValueError("parent freeze set is incomplete")
    if any(configs[name]["status"] != "FROZEN_NATIVE_PARENT_B23_0" for name in config_names[:4]):
        raise ValueError("a native parent is not frozen")
    if configs["replay_policy.yaml"]["authorization"]["b23_0_gpu_budget"] != 0:
        raise ValueError("B23.0 GPU budget is not zero")
    if configs["replay_policy.yaml"]["authorization"]["b23_1_gpu_authorized"]:
        raise ValueError("B23.1 must not be authorized in B23.0")
    if configs["b23_1_smoke_template.yaml"]["gpu_commands"]:
        raise ValueError("B23.0 smoke template unexpectedly contains GPU commands")
    checks["configs"] = {"count": len(configs), "status": "PASS", "format": "JSON_SUBSET_OF_YAML"}

    exposure_check = check_manifest(
        repo / "manifests/b23/PRE_B23_EXPOSURE.csv"
    )
    exposed_image_ids = set(exposure_check.pop("image_ids"))
    checks["exposure_manifest"] = exposure_check
    checks["future_registry"] = check_future_registry(
        repo / "manifests/b23/future_split_registry.csv", exposed_image_ids
    )
    for smoke_name in (
        "b23_1_one_image_smoke.template.csv",
        "b23_1_four_image_smoke.template.csv",
    ):
        with (repo / "manifests/b23" / smoke_name).open(newline="", encoding="utf-8") as handle:
            if list(csv.DictReader(handle)):
                raise ValueError(f"B23.1 smoke registry must remain empty: {smoke_name}")
    checks["smoke_registries"] = {"rows": 0, "status": "EMPTY_AS_REQUIRED"}
    correction_ledger = read_json(
        repo / "manifests/b23/b23_0_correction_ledger.json"
    )
    invalidated = {
        entry.get("evidence_commit")
        for entry in correction_ledger.get("entries", [])
        if entry.get("disposition") == "INVALID_AS_B23_0_PASS_PRESERVED"
    }
    expected_invalidated = "0d35656b360b4b0d04a28812079f18de8a03a9af"
    if expected_invalidated not in invalidated:
        raise ValueError("B23.0 correction ledger omits the preserved false-PASS evidence commit")
    checks["correction_ledger"] = {
        "status": "PASS",
        "invalidated_evidence_commit": expected_invalidated,
    }

    compute_schema = read_json(repo / "schemas/b23/compute_ledger.schema.json")
    replay_schema = read_json(repo / "schemas/b23/replay_report.schema.json")
    future_schema = read_json(repo / "schemas/b23/future_split_registry.schema.json")
    compute_example = read_json(
        repo / "manifests/b23/examples/compute_ledger.uncalibrated.json"
    )
    replay_example = read_json(
        repo / "manifests/b23/examples/replay_report.not_run.json"
    )
    modes = {
        validate_schema_instance(compute_schema, compute_example),
        validate_schema_instance(replay_schema, replay_example),
        validate_schema_instance(future_schema, []),
    }
    validate_compute_ledger(compute_example)
    validate_replay_report(replay_example)
    checks["schemas"] = {
        "status": "PASS",
        "validation_modes": sorted(modes),
        "custom_validator_scope": "ALL_ASSERTION_KEYWORDS_PRESENT_IN_B23_SCHEMAS",
    }

    b23_text_files = [
        path for root in (repo / "configs/b23", repo / "scripts/b23")
        for path in root.rglob("*") if path.is_file()
    ]
    forbidden_home_prefix = "/" + "home" + "/"
    home_mentions = []
    for path in b23_text_files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if forbidden_home_prefix in text:
            home_mentions.append(str(path.relative_to(repo)))
    if home_mentions:
        raise ValueError(f"forbidden home-directory paths in executable artifacts: {home_mentions}")
    checks["pac_path_policy"] = {"status": "PASS", "root": "/egr/research-pac/huang248"}

    result = {
        "schema_version": "b23.zero-gpu-validation.v1",
        "status": "PASS",
        "gpu_work_performed": False,
        "checks": checks,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
