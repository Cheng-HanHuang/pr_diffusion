#!/usr/bin/env python3
"""Freeze coupled native-trajectory weights and calibrated B23.1 FRE ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any


PARENTS = ("Fresh1", "LF-v1", "NP-1", "SITCOM-1")
RAW_FIELDS = (
    "denoiser_forward", "denoiser_backward", "denoiser_jvp", "denoiser_vjp",
    "measurement_forward", "measurement_adjoint", "measurement_jvp", "measurement_vjp",
    "fft", "projection", "correction", "optimizer_iterations", "state_conversion",
    "renoising", "random_proposals", "rng_draws",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    native_runs: dict[str, list[dict[str, Any]]] = {}
    wrapper_runs: dict[str, dict[str, Any]] = {}
    for parent in PARENTS:
        key = parent.lower().replace("-", "_")
        native_runs[parent] = [
            read_json(args.run_root / "replay" / key / f"native_{index}" / "RUN.json")
            for index in range(3)
        ]
        wrapper_runs[parent] = read_json(args.run_root / "replay" / key / "wrapper" / "RUN.json")
        if any(run["status"] != "PASS" for run in native_runs[parent]) or wrapper_runs[parent]["status"] != "PASS":
            raise ValueError(f"cannot calibrate failed parent {parent}")

    device_names = {
        run["timing"]["device_name"] for runs in native_runs.values() for run in runs
    }
    cuda_visibility = {
        run["timing"]["cuda_visible_devices"] for runs in native_runs.values() for run in runs
    }
    if len(device_names) != 1 or len(cuda_visibility) != 1:
        raise ValueError("coupled calibration requires one pinned hardware identity")
    inventory = {
        "schema_version": "b23.hardware-identity.v1",
        "device_name": next(iter(device_names)),
        "cuda_visible_devices": next(iter(cuda_visibility)),
        "timer_method": "CUDA_EVENTS_SYNCHRONIZED_PLUS_PERF_COUNTER",
        "calibration_gpu": 0,
    }
    inventory_sha = canonical_sha(inventory)
    identity_id = f"B23.1-GPU0:{inventory['device_name']}"
    write_json(output / "HARDWARE_IDENTITY.json", {**inventory, "inventory_sha256": inventory_sha})

    definitions = {}
    weights = {}
    for parent in PARENTS:
        samples = [float(run["timing"]["gpu_active_seconds"]) for run in native_runs[parent]]
        if any(value <= 0 for value in samples):
            raise ValueError(f"nonpositive coupled timing for {parent}")
        definition = {
            "operation_type": f"native_trajectory::{parent}",
            "parent_id": parent,
            "semantic_unit": "one complete frozen one-terminal native trajectory",
            "operation_count_audit_sha256": native_runs[parent][0]["operation_counts_sha256"],
            "accounting": "non-overlapping coupled block; detailed raw calls are evidence, not double-counted atomic terms",
        }
        definition_sha = canonical_sha(definition)
        definitions[parent] = {**definition, "definition_sha256": definition_sha}
        weights[parent] = {
            "operation_type": definition["operation_type"],
            "definition_sha256": definition_sha,
            "gpu_active_seconds_samples": samples,
            "median_gpu_active_seconds": statistics.median(samples),
            "sample_count": len(samples),
            "timer_method": "CUDA_EVENTS_SYNCHRONIZED_PLUS_PERF_COUNTER",
        }
    weight_registry = {
        "schema_version": "b23.coupled-weight-registry.v1",
        "status": "CALIBRATED",
        "hardware_identity": {"identity_id": identity_id, "inventory_sha256": inventory_sha},
        "calibration_design": {
            "unit": "one complete frozen one-terminal native parent entrypoint including model load",
            "excluded_warmup_iterations": 0,
            "measured_iterations_per_parent": 3,
            "paired_blocks": 3,
            "ordering": "three repeat cycles, each ordered Fresh1, LF-v1, NP-1, SITCOM-1",
            "justification": "No common sound boundary isolates every parent-internal operation without changing native scheduler, optimizer, proposal-set, RNG, or model-load semantics.",
        },
        "weights": weights,
        "definitions": definitions,
        "no_guessed_decomposition": True,
        "no_double_counting": True,
    }
    weight_path = output / "WEIGHT_REGISTRY.json"
    write_json(weight_path, weight_registry)
    weight_sha = file_sha(weight_path)

    reference = {
        "schema_version": "b23.fresh1-reference.v1",
        "parent_id": "Fresh1",
        "image_id": native_runs["Fresh1"][0]["image_id"],
        "measurement_id": native_runs["Fresh1"][0]["measurement_id"],
        "median_gpu_active_seconds": weights["Fresh1"]["median_gpu_active_seconds"],
        "operation_counts_sha256": native_runs["Fresh1"][0]["operation_counts_sha256"],
        "weight_registry_sha256": weight_sha,
    }
    reference_path = output / "FRESH1_REFERENCE.json"
    write_json(reference_path, reference)
    reference_sha = file_sha(reference_path)

    empty_counts = {name: 0 for name in RAW_FIELDS}
    all_reference_blocks = [
        {
            "operation_type": weights[parent]["operation_type"],
            "count": 1 if parent == "Fresh1" else 0,
            "measured_weight_seconds": weights[parent]["median_gpu_active_seconds"],
            "definition_sha256": weights[parent]["definition_sha256"],
        }
        for parent in PARENTS
    ]
    fresh_weight = weights["Fresh1"]["median_gpu_active_seconds"]
    for parent in PARENTS:
        run = wrapper_runs[parent]
        timing = run["timing"]
        policy_blocks = [
            {
                "operation_type": weights[item]["operation_type"],
                "count": 1 if item == parent else 0,
                "measured_weight_seconds": weights[item]["median_gpu_active_seconds"],
                "definition_sha256": weights[item]["definition_sha256"],
            }
            for item in PARENTS
        ]
        work_fre = weights[parent]["median_gpu_active_seconds"] / fresh_weight
        time_fre = float(timing["gpu_active_seconds"]) / fresh_weight
        ledger = {
            "schema_version": "b23.compute-ledger.v2",
            "experiment_id": f"B23.1A-{parent}-wrapper",
            "image_id": run["image_id"],
            "measurement_id": run["measurement_id"],
            "parent_or_policy_id": parent,
            "hardware_identity": {"identity_id": identity_id, "inventory_sha256": inventory_sha},
            "raw_counts": dict(empty_counts),
            "coupled_operation_blocks": policy_blocks,
            "optimizer_iterations_by_type": {},
            "rng_streams": [],
            "branches": {
                "total_created": 1, "max_live": 1, "retained": 1, "terminal_candidates": 1,
                "branch_ids": ["root"], "retained_branch_ids": ["root"], "terminal_branch_ids": ["root"],
            },
            "timing": {
                "gpu_active_seconds": float(timing["gpu_active_seconds"]),
                "wall_seconds": float(timing["wall_seconds"]),
                "timer_method": timing["timer_method"],
            },
            "memory_bytes": {
                "peak_allocated": int(timing["peak_allocated_bytes"]),
                "peak_reserved": int(timing["peak_reserved_bytes"]),
            },
            "overhead_seconds": {},
            "atomic_weights": {},
            "fre": {
                "status": "CALIBRATED",
                "work_FRE": work_fre,
                "time_FRE": time_fre,
                "claim_FRE": max(work_fre, time_fre),
                "reference": {
                    "identity_id": identity_id,
                    "inventory_sha256": inventory_sha,
                    "reference_id": "B23.1-Fresh1-coupled-native-trajectory",
                    "reference_ledger_sha256": reference_sha,
                    "weight_registry_sha256": weight_sha,
                    "raw_counts": dict(empty_counts),
                    "coupled_operation_blocks": all_reference_blocks,
                    "gpu_active_seconds": fresh_weight,
                },
            },
            "notes": [
                "Detailed exact calls are frozen in operation_counts_sha256=" + run["operation_counts_sha256"],
                "All parent-internal calls are accounted once by a typed native-trajectory coupled block; raw atomic counters are zero to prevent double counting.",
            ],
        }
        repo = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo))
        from prdiffusion.b23_protocol import validate_compute_ledger
        validate_compute_ledger(ledger)
        write_json(output / f"COMPUTE_LEDGER_{parent.replace('-', '_')}.json", ledger)
    print(json.dumps({"status": "PASS", "mode": "TYPED_COUPLED_BLOCKS", "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
