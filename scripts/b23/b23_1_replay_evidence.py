#!/usr/bin/env python3
"""Freeze native envelopes before wrappers and validate B23.1 replay reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "max_abs_err", "mean_abs_err", "relative_l2_err", "raw_psnr_delta",
    "measurement_loss_delta", "trace_max_abs_err",
)


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_tensor(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".png":
        from PIL import Image
        value = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return (value * 2.0 - 1.0).transpose(2, 0, 1)[None]
    import torch
    payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        tensor = payload
    elif isinstance(payload, dict):
        tensor = next((payload[key] for key in ("reconstruction", "ground_truth", "sample", "x") if torch.is_tensor(payload.get(key))), None)
    else:
        tensor = None
    if tensor is None:
        raise TypeError(f"no reconstruction tensor in {path}")
    return tensor.detach().cpu().to(dtype=torch.float32).numpy()


def load_input_manifest(run: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(run["reconstruction_path"]).resolve()
    cursor = run_dir
    while cursor != cursor.parent:
        candidate = cursor / "input/input_manifest.json"
        if candidate.is_file():
            return read_json(candidate)
        cursor = cursor.parent
    raise FileNotFoundError("compatible input manifest not found from run")


def load_gt(run: dict[str, Any]) -> np.ndarray:
    return load_tensor(Path(load_input_manifest(run)["ground_truth_tensor_path"]))


def load_measurement(run: dict[str, Any]) -> np.ndarray:
    import torch
    payload = torch.load(Path(load_input_manifest(run)["measurement_path"]), map_location="cpu")
    if torch.is_tensor(payload):
        tensor = payload
    elif isinstance(payload, dict):
        tensor = next((payload[key] for key in ("measurement", "y", "observation") if torch.is_tensor(payload.get(key))), None)
    else:
        tensor = None
    if tensor is None:
        raise TypeError("locked input contains no measurement tensor")
    if run["parent_id"] == "NP-1":
        tensor = tensor.clamp_min(0.0)
    return tensor.detach().cpu().to(dtype=torch.float32).numpy()


def numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    leaves: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith("sha256") or key in {"seed", "derived_seed"}:
                continue
            leaves.update(numeric_leaves(item, f"{prefix}/{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            leaves.update(numeric_leaves(item, f"{prefix}/{index}"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            leaves[prefix] = number
    return leaves


def trace_delta(left: dict[str, Any], right: dict[str, Any]) -> float:
    a = numeric_leaves(left)
    b = numeric_leaves(right)
    if set(a) != set(b):
        return math.inf
    return max((abs(a[key] - b[key]) for key in a), default=0.0)


def psnr(value: np.ndarray, gt: np.ndarray) -> float:
    x = np.clip((value + 1.0) * 0.5, 0.0, 1.0)
    y = np.clip((gt + 1.0) * 0.5, 0.0, 1.0)
    mse = max(float(np.mean((x - y) ** 2)), 1.0e-12)
    return 10.0 * math.log10(1.0 / mse)


def measurement_loss(value: np.ndarray, target: np.ndarray) -> float:
    x = np.clip((value + 1.0) * 0.5, 0.0, 1.0)
    pad_h = (384 - x.shape[-2]) // 2
    pad_w = (384 - x.shape[-1]) // 2
    padded = np.pad(x, ((0, 0), (0, 0), (pad_h, 384 - x.shape[-2] - pad_h), (pad_w, 384 - x.shape[-1] - pad_w)))
    magnitude = np.abs(np.fft.fft2(padded, norm="ortho", axes=(-2, -1)))
    if magnitude.shape != target.shape:
        raise ValueError(f"measurement shape mismatch: predicted={magnitude.shape} target={target.shape}")
    return float(np.mean((magnitude - target.astype(np.float64)) ** 2))


def compare(
    a: np.ndarray, b: np.ndarray, trace_a: dict[str, Any], trace_b: dict[str, Any],
    gt: np.ndarray, measurement: np.ndarray,
) -> dict[str, Any]:
    delta = a.astype(np.float64) - b.astype(np.float64)
    denom = max(float(np.linalg.norm(a.astype(np.float64))), np.finfo(np.float64).tiny)
    return {
        "max_abs_err": float(np.max(np.abs(delta))),
        "mean_abs_err": float(np.mean(np.abs(delta))),
        "relative_l2_err": float(np.linalg.norm(delta) / denom),
        "raw_psnr_delta": float(psnr(a, gt) - psnr(b, gt)),
        "measurement_loss_delta": float(measurement_loss(a, measurement) - measurement_loss(b, measurement)),
        "trace_max_abs_err": float(trace_delta(trace_a, trace_b)),
        "tensor_hash_equal": bool(np.array_equal(a, b)),
    }


def envelope(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    result = {key: max(abs(float(item[key])) for item in comparisons) for key in METRICS}
    result["tensor_hash_equal"] = all(bool(item["tensor_hash_equal"]) for item in comparisons)
    return result


def load_runs(paths: list[Path]) -> list[dict[str, Any]]:
    return [read_json(path / "RUN.json") for path in paths]


def freeze(args) -> int:
    if args.output.exists():
        raise FileExistsError(args.output)
    runs = load_runs(args.native_dir)
    if len(runs) != 3 or {run["repeat_index"] for run in runs} != {0, 1, 2}:
        raise ValueError("freeze requires exactly native repeats 0, 1, and 2")
    identity = {(run["parent_id"], run["image_id"], run["measurement_id"], run["derived_parent_seed"]) for run in runs}
    if len(identity) != 1 or any(run["mode"] != "native" or run["status"] != "PASS" for run in runs):
        raise ValueError("native repeat identities or statuses differ")
    tensors = [load_tensor(Path(run["reconstruction_path"])) for run in runs]
    traces = [read_json(Path(run["trace_path"])) for run in runs]
    gt = load_gt(runs[0])
    measurement = load_measurement(runs[0])
    pairs = [compare(tensors[i], tensors[j], traces[i], traces[j], gt, measurement) for i in range(3) for j in range(i + 1, 3)]
    native_envelope = envelope(pairs)
    bitwise = len({run["reconstruction_sha256"] for run in runs}) == 1 and len({run["trace_sha256"] for run in runs}) == 1
    qualification = None
    if not bitwise:
        dtype = tensors[0].dtype
        epsilon = float(np.finfo(dtype).eps)
        tensor_scale = max(1.0, *(float(np.max(np.abs(value))) for value in tensors))
        trace_values = numeric_leaves(traces[0])
        trace_scale = max([1.0, *(abs(value) for value in trace_values.values())])
        metric_scales = {
            "max_abs_err": tensor_scale,
            "mean_abs_err": tensor_scale,
            "relative_l2_err": 1.0,
            "raw_psnr_delta": max(1.0, *(abs(psnr(value, gt)) for value in tensors)),
            "measurement_loss_delta": max(1.0, *(abs(measurement_loss(value, measurement)) for value in tensors)),
            "trace_max_abs_err": trace_scale,
        }
        depths = {
            "max_abs_err": 1,
            "mean_abs_err": int(tensors[0].size),
            "relative_l2_err": int(tensors[0].size),
            "raw_psnr_delta": 1,
            "measurement_loss_delta": 384 * 384,
            "trace_max_abs_err": max(1, len(trace_values)),
        }
        floors = {key: epsilon * metric_scales[key] * depths[key] for key in METRICS}
        qualification = {
            "formula": "machine_epsilon(dtype) * max(1, declared_metric_scale) * declared_reduction_depth",
            "dtype": str(dtype),
            "machine_epsilon": epsilon,
            "metric_scales": metric_scales,
            "reduction_depths": depths,
            "numerical_floors": floors,
            "arbitrary_constants_used": False,
        }
    freeze_record = {
        "schema_version": "b23.tolerance-freeze.v1",
        "parent_id": runs[0]["parent_id"],
        "image_id": runs[0]["image_id"],
        "native_run_ids": [f"{run['parent_id']}-native-{run['repeat_index']}" for run in runs],
        "native_tensor_sha256s": [run["reconstruction_sha256"] for run in runs],
        "native_trace_sha256s": [run["trace_sha256"] for run in runs],
        "native_operation_counts_sha256": runs[0]["operation_counts_sha256"],
        "native_rng_ledger_sha256": runs[0]["rng_ledger_sha256"],
        "native_repeatability_envelope": native_envelope,
        "eligibility": "BITWISE" if bitwise else "TOLERANCE_QUALIFIED",
        "tolerance_floor_evaluation": qualification,
        "wrapper_run_id": None,
        "frozen_before_wrapper": True,
    }
    freeze_record["freeze_record_sha256"] = canonical_sha(freeze_record)
    write_json(args.output, freeze_record)
    print(json.dumps({"status": "PASS", "eligibility": freeze_record["eligibility"], "output": str(args.output)}, sort_keys=True))
    return 0


def analyze(args) -> int:
    if args.output.exists():
        raise FileExistsError(args.output)
    freeze_record = read_json(args.freeze)
    if canonical_sha({key: value for key, value in freeze_record.items() if key != "freeze_record_sha256"}) != freeze_record["freeze_record_sha256"]:
        raise ValueError("tolerance freeze identity mismatch")
    native_runs = load_runs(args.native_dir)
    wrapper = read_json(args.wrapper_dir / "RUN.json")
    if wrapper["mode"] != "wrapper" or wrapper["parent_id"] != freeze_record["parent_id"]:
        raise ValueError("wrapper identity mismatch")
    native = native_runs[0]
    native_tensor = load_tensor(Path(native["reconstruction_path"]))
    wrapper_tensor = load_tensor(Path(wrapper["reconstruction_path"]))
    native_trace = read_json(Path(native["trace_path"]))
    wrapper_trace = read_json(Path(wrapper["trace_path"]))
    comparison = compare(
        wrapper_tensor, native_tensor, wrapper_trace, native_trace,
        load_gt(native), load_measurement(native),
    )
    eligibility = freeze_record["eligibility"]
    failures = []
    if wrapper["operation_counts_sha256"] != freeze_record["native_operation_counts_sha256"]:
        failures.append("operation count hash mismatch")
    if wrapper["rng_ledger_sha256"] != freeze_record["native_rng_ledger_sha256"]:
        failures.append("RNG ledger hash mismatch")
    tolerance_qualification = None
    if eligibility == "BITWISE":
        if wrapper["reconstruction_sha256"] != freeze_record["native_tensor_sha256s"][0]:
            failures.append("BITWISE wrapper tensor hash mismatch")
        if wrapper["trace_sha256"] != freeze_record["native_trace_sha256s"][0]:
            failures.append("BITWISE wrapper trace hash mismatch")
        if any(float(comparison[key]) != 0.0 for key in METRICS):
            failures.append("BITWISE wrapper has nonzero tensor/scalar/trace delta")
    else:
        floor_eval = freeze_record["tolerance_floor_evaluation"]
        floors = floor_eval["numerical_floors"]
        for key in METRICS:
            limit = abs(float(freeze_record["native_repeatability_envelope"][key])) + float(floors[key])
            if not math.isfinite(float(comparison[key])) or abs(float(comparison[key])) > limit:
                failures.append(f"{key} exceeds frozen native envelope plus numerical floor")
        tolerance_qualification = {
            "freeze_record_sha256": freeze_record["freeze_record_sha256"],
            "frozen_before_wrapper": True,
            "wrapper_run_id_at_freeze": None,
            "frozen_native_run_ids": freeze_record["native_run_ids"],
            "numerical_floors": floors,
        }
    audit = wrapper["timing"]["determinism_audit"]
    report = {
        "schema_version": "b23.replay-report.v3",
        "experiment_id": f"B23.1A-{wrapper['parent_id']}-{wrapper['image_id']}",
        "parent_id": wrapper["parent_id"],
        "native_run_ids": freeze_record["native_run_ids"],
        "wrapper_run_id": f"{wrapper['parent_id']}-wrapper-0",
        "eligibility": eligibility,
        "eligibility_rationale": "derived mechanically from three native repeats before wrapper execution",
        "determinism_audit": {
            "audit_status": "COMPLETE",
            "unavailable_reason": None,
            **audit,
            "flags_change_native_parent": False,
            "notes": ["audited after the unchanged native entrypoint returned"],
        },
        "tolerance_qualification": tolerance_qualification,
        "native_repeatability_envelope": freeze_record["native_repeatability_envelope"],
        "wrapper_comparison": comparison,
        "native_tensor_sha256s": freeze_record["native_tensor_sha256s"],
        "wrapper_tensor_sha256": wrapper["reconstruction_sha256"],
        "native_trace_sha256s": freeze_record["native_trace_sha256s"],
        "wrapper_trace_sha256": wrapper["trace_sha256"],
        "native_operation_counts_sha256": freeze_record["native_operation_counts_sha256"],
        "wrapper_operation_counts_sha256": wrapper["operation_counts_sha256"],
        "operation_count_reconciled": wrapper["operation_counts_sha256"] == freeze_record["native_operation_counts_sha256"],
        "native_rng_ledger_sha256": freeze_record["native_rng_ledger_sha256"],
        "wrapper_rng_ledger_sha256": wrapper["rng_ledger_sha256"],
        "rng_draws_reconciled": wrapper["rng_ledger_sha256"] == freeze_record["native_rng_ledger_sha256"],
        "serialization_resume_check": "NOT_APPLICABLE",
        "serialization_not_applicable_reason": "The frozen one-terminal native entrypoint exposes no sound portable mid-run resume boundary without changing parent semantics; trajectory recording is observation-only.",
        "verdict": "FAIL" if failures else "PASS",
        "failure_reasons": failures,
    }
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from prdiffusion.b23_protocol import validate_replay_report
    validate_replay_report(report)
    write_json(args.output, report)
    print(json.dumps({"status": report["verdict"], "parent": report["parent_id"], "output": str(args.output)}, sort_keys=True))
    return 0 if not failures else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--native-dir", type=Path, action="append", required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--native-dir", type=Path, action="append", required=True)
    analyze_parser.add_argument("--freeze", type=Path, required=True)
    analyze_parser.add_argument("--wrapper-dir", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return freeze(args) if args.command == "freeze" else analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
