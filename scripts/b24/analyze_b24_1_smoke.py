#!/usr/bin/env python3
"""Combine DAPS and SITCOM B24.1 smoke results into one fail-closed gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    methods: dict[str, dict[str, Any]] = {}
    for method in ("DAPS", "SITCOM"):
        path = root / method.lower() / "METHOD_SUMMARY.json"
        if not path.is_file():
            print(json.dumps({"status": "RUNNING_OR_INCOMPLETE", "missing": str(path)}, sort_keys=True))
            return 3
        value = read_json(path)
        if value.get("method") != method:
            raise RuntimeError(f"method identity mismatch: {path}")
        methods[method] = value

    same_image = len({value["image_id"] for value in methods.values()}) == 1
    same_measurement = len({
        value["accepted_b23_input"]["measurement_tensor_sha256"] for value in methods.values()
    }) == 1
    overall = same_image and same_measurement and all(bool(value.get("overall_pass")) for value in methods.values())
    summary = {
        "schema_version": 1,
        "stage": "B24.1",
        "overall_pass": overall,
        "same_exposed_image": same_image,
        "same_locked_measurement": same_measurement,
        "image_id": methods["DAPS"]["image_id"],
        "methods": {
            method: {
                "gpu_id": value["preflight"]["gpu_id"],
                "gpu_uuid": value["preflight"]["gpu_uuid"],
                "planned_concurrency": value["planned_concurrency"],
                "exact_terminal_hash_equivalence": value["exact_terminal_hash_equivalence"],
                "memory_pass": value["memory_pass"],
                "serial_wall_seconds": value["serial"]["group_wall_seconds"],
                "concurrent_wall_seconds": value["concurrent"]["group_wall_seconds"],
                "speedup": value["speedup_serial_wall_over_concurrent_wall"],
                "max_observed_concurrent_b24_process_mib": value["concurrent"]["max_observed_b24_process_mib"],
                "conservative_concurrent_peak_reserved_mib": value["conservative_concurrent_peak_reserved_mib"],
            }
            for method, value in methods.items()
        },
        "b24_2_64_gate_recommendation": "PASS_TO_64_BASELINE" if overall else "STOP_REVIEW_B24_1",
        "note": "B24.1 validates concurrent independent-process protocol execution, not NP-native algorithmic batching."
    }
    write_json(root / "B24_1_SUMMARY.json", summary)
    print(json.dumps({
        "status": "PASS" if overall else "FAIL",
        "summary": str(root / "B24_1_SUMMARY.json"),
        "daps_concurrency": methods["DAPS"]["planned_concurrency"],
        "sitcom_concurrency": methods["SITCOM"]["planned_concurrency"],
        "daps_speedup": methods["DAPS"]["speedup_serial_wall_over_concurrent_wall"],
        "sitcom_speedup": methods["SITCOM"]["speedup_serial_wall_over_concurrent_wall"],
        "next": summary["b24_2_64_gate_recommendation"],
    }, sort_keys=True))
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
