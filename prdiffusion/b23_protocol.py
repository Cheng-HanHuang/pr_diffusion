"""B23 protocol primitives that are safe to exercise without a GPU.

This module deliberately models native parent state as a typed envelope with
named payloads.  It does not assign a universal meaning to a field called
``x``.  Cross-parent interpretation belongs in an explicit AdapterSpec.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Generic, Mapping, Sequence, TypeVar


PROTOCOL_VERSION = "b23.0-v3"
SEED_DERIVATION_VERSION = "b23-sha256-v1"

REPLAY_NUMERIC_METRICS = (
    "max_abs_err",
    "mean_abs_err",
    "relative_l2_err",
    "raw_psnr_delta",
    "measurement_loss_delta",
    "trace_max_abs_err",
)

PLACEHOLDER_TIMER_METHODS = {
    "NOTRUN",
    "NONE",
    "NULL",
    "NA",
    "N/A",
    "TBD",
    "TODO",
    "UNKNOWN",
    "UNAVAILABLE",
    "PLACEHOLDER",
}

ParentName = TypeVar("ParentName", bound=str)


class ProtocolError(ValueError):
    """Raised when an object violates a frozen B23 protocol invariant."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def derive_seed(
    base_seed: int,
    *,
    stream_name: str,
    image_id: str,
    measurement_id: str,
    parent_id: str,
    branch_id: str = "root",
    draw_index: int = 0,
) -> int:
    """Derive a stable non-negative 63-bit seed for one named RNG stream.

    The length-prefixed canonical JSON payload prevents concatenation
    ambiguities.  Adding an unrelated stream cannot shift existing streams.
    """

    if base_seed < 0 or draw_index < 0:
        raise ProtocolError("base_seed and draw_index must be non-negative")
    required_strings = {
        "stream_name": stream_name,
        "image_id": image_id,
        "measurement_id": measurement_id,
        "parent_id": parent_id,
        "branch_id": branch_id,
    }
    if any(not isinstance(value, str) or not value for value in required_strings.values()):
        raise ProtocolError("all seed identity fields must be non-empty strings")
    payload = {
        "version": SEED_DERIVATION_VERSION,
        "base_seed": int(base_seed),
        "stream_name": stream_name,
        "image_id": image_id,
        "measurement_id": measurement_id,
        "parent_id": parent_id,
        "branch_id": branch_id,
        "draw_index": int(draw_index),
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


@dataclasses.dataclass(frozen=True)
class RNGStreamState:
    name: str
    base_seed: int
    derived_seed: int
    draw_count: int
    device: str
    serialized_state_b64: str | None = None

    def validate(self) -> None:
        if not self.name or self.base_seed < 0 or self.derived_seed < 0:
            raise ProtocolError("invalid RNG stream identity")
        if self.draw_count < 0 or not self.device:
            raise ProtocolError("invalid RNG stream counter or device")
        if self.serialized_state_b64 is not None:
            try:
                base64.b64decode(self.serialized_state_b64, validate=True)
            except Exception as exc:  # pragma: no cover - exact decoder varies
                raise ProtocolError("serialized RNG state is not valid base64") from exc


@dataclasses.dataclass(frozen=True)
class SemanticCoordinate:
    native_name: str
    native_value: float
    normalized_noise: float
    cumulative_budget: float

    def validate(self) -> None:
        values = (self.native_value, self.normalized_noise, self.cumulative_budget)
        if not all(math.isfinite(value) for value in values):
            raise ProtocolError("semantic coordinates must be finite")
        if not 0.0 <= self.normalized_noise <= 1.0:
            raise ProtocolError("normalized_noise must be in [0, 1]")
        if not 0.0 <= self.cumulative_budget <= 1.0:
            raise ProtocolError("cumulative_budget must be in [0, 1]")


@dataclasses.dataclass
class NativeState(Generic[ParentName]):
    parent_id: ParentName
    source_revision: str
    representation_contract: Mapping[str, str]
    tensor_payloads: Mapping[str, Any]
    coordinate: SemanticCoordinate
    model_identity: Mapping[str, str]
    scheduler_identity: Mapping[str, str]
    measurement_identity: Mapping[str, Any]
    optimizer_state: Mapping[str, Any]
    rng_streams: Sequence[RNGStreamState]
    trace_id: str
    compute_ledger_id: str

    def validate(self) -> None:
        if not self.parent_id or not self.source_revision:
            raise ProtocolError("native state needs parent and source identities")
        if not self.tensor_payloads:
            raise ProtocolError("native state needs at least one named tensor payload")
        if set(self.tensor_payloads) != set(self.representation_contract):
            raise ProtocolError(
                "every tensor payload needs exactly one representation contract"
            )
        if len({stream.name for stream in self.rng_streams}) != len(self.rng_streams):
            raise ProtocolError("RNG stream names must be unique")
        self.coordinate.validate()
        for stream in self.rng_streams:
            stream.validate()
        for identity in (
            self.model_identity,
            self.scheduler_identity,
            self.measurement_identity,
        ):
            if not identity:
                raise ProtocolError("model, scheduler, and measurement identities are required")
        if not self.trace_id or not self.compute_ledger_id:
            raise ProtocolError("trace and compute-ledger identifiers are required")


@dataclasses.dataclass(frozen=True)
class ModuleSpec:
    module_id: str
    input_parent: str
    output_parent: str
    input_contract: Mapping[str, str]
    output_contract: Mapping[str, str]
    valid_noise_interval: tuple[float, float]
    operation_semantics: str
    rng_streams_consumed: tuple[str, ...]
    raw_counter_names: tuple[str, ...]
    serialization_boundary: bool

    def validate(self) -> None:
        if not self.module_id or not self.operation_semantics:
            raise ProtocolError("module identity and semantics are required")
        low, high = self.valid_noise_interval
        if not (0.0 <= low <= high <= 1.0):
            raise ProtocolError("module noise interval must lie in [0, 1]")


@dataclasses.dataclass(frozen=True)
class AdapterSpec:
    adapter_id: str
    source_parent: str
    target_parent: str
    mathematical_conversion: str
    source_coordinate: str
    target_coordinate: str
    discarded_information: tuple[str, ...]
    newly_sampled_information: tuple[str, ...]
    validity_checks: tuple[str, ...]
    round_trip_check: str | None
    raw_counter_names: tuple[str, ...]
    rng_streams_consumed: tuple[str, ...]
    lossy: bool

    def validate(self) -> None:
        if not all(
            (self.adapter_id, self.source_parent, self.target_parent,
             self.mathematical_conversion, self.source_coordinate,
             self.target_coordinate)
        ):
            raise ProtocolError("adapter identities and conversion are required")
        if not self.validity_checks:
            raise ProtocolError("adapter needs at least one validity check")
        if self.lossy and self.round_trip_check == "exact":
            raise ProtocolError("a lossy adapter cannot claim an exact round trip")


RAW_COUNT_FIELDS = (
    "denoiser_forward",
    "denoiser_backward",
    "denoiser_jvp",
    "denoiser_vjp",
    "measurement_forward",
    "measurement_adjoint",
    "measurement_jvp",
    "measurement_vjp",
    "fft",
    "projection",
    "correction",
    "optimizer_iterations",
    "state_conversion",
    "renoising",
    "random_proposals",
    "rng_draws",
)


def calibrated_work(raw_counts: Mapping[str, int], weights: Mapping[str, float]) -> float:
    """Calculate calibrated work, refusing incomplete or unmeasured weights."""

    unknown = sorted(set(raw_counts) - set(RAW_COUNT_FIELDS))
    if unknown:
        raise ProtocolError(f"unknown raw-count fields: {unknown}")
    missing = sorted(name for name, count in raw_counts.items() if count and name not in weights)
    if missing:
        raise ProtocolError(f"missing measured atomic weights: {missing}")
    total = 0.0
    for name, count in raw_counts.items():
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ProtocolError(f"raw count {name!r} must be a non-negative integer")
        weight = float(weights.get(name, 0.0))
        if not math.isfinite(weight) or (count and weight <= 0) or weight < 0:
            raise ProtocolError(
                f"weight {name!r} must be finite and strictly positive when count is nonzero"
            )
        total += count * weight
    return total


def _coupled_work(blocks: Sequence[Mapping[str, Any]]) -> float:
    names: set[str] = set()
    total = 0.0
    for block in blocks:
        required = {"operation_type", "count", "measured_weight_seconds", "definition_sha256"}
        missing = sorted(required - set(block))
        if missing:
            raise ProtocolError(f"coupled operation block missing fields: {missing}")
        name = block["operation_type"]
        count = block["count"]
        weight = block["measured_weight_seconds"]
        digest = block["definition_sha256"]
        if not isinstance(name, str) or not name or name in names:
            raise ProtocolError("coupled operation types must be unique non-empty strings")
        names.add(name)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ProtocolError("coupled operation count must be a non-negative integer")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not math.isfinite(weight):
            raise ProtocolError("coupled operation weight must be finite")
        if weight < 0 or (count and weight <= 0):
            raise ProtocolError("a nonzero coupled operation needs a strictly positive measured weight")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ProtocolError("coupled operation definition needs a SHA-256 identity")
        total += count * float(weight)
    return total


def _validate_raw_counts(raw_counts: Mapping[str, int]) -> None:
    unknown = sorted(set(raw_counts) - set(RAW_COUNT_FIELDS))
    if unknown:
        raise ProtocolError(f"unknown raw-count fields: {unknown}")
    missing = sorted(set(RAW_COUNT_FIELDS) - set(raw_counts))
    if missing:
        raise ProtocolError(f"missing raw-count fields: {missing}")
    for name, count in raw_counts.items():
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ProtocolError(f"raw count {name!r} must be a non-negative integer")


def fre_values(
    *,
    policy_counts: Mapping[str, int],
    reference_counts: Mapping[str, int],
    measured_weights: Mapping[str, float],
    gpu_active_seconds: float,
    paired_reference_median_seconds: float,
    policy_coupled_operations: Sequence[Mapping[str, Any]] = (),
    reference_coupled_operations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, float]:
    """Return work-, time-, and claim-FRE under already frozen weights."""

    policy_work = calibrated_work(policy_counts, measured_weights) + _coupled_work(
        policy_coupled_operations
    )
    reference_work = calibrated_work(reference_counts, measured_weights) + _coupled_work(
        reference_coupled_operations
    )
    if reference_work <= 0:
        raise ProtocolError("Fresh1 calibrated reference work must be positive")
    if not math.isfinite(gpu_active_seconds) or gpu_active_seconds < 0:
        raise ProtocolError("GPU-active seconds must be finite and non-negative")
    if not math.isfinite(paired_reference_median_seconds) or paired_reference_median_seconds <= 0:
        raise ProtocolError("paired Fresh1 median GPU-active seconds must be positive")
    work_fre = policy_work / reference_work
    time_fre = gpu_active_seconds / paired_reference_median_seconds
    return {
        "calibrated_work": policy_work,
        "reference_calibrated_work": reference_work,
        "work_FRE": work_fre,
        "time_FRE": time_fre,
        "claim_FRE": max(work_fre, time_fre),
    }


def validate_compute_ledger(ledger: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "experiment_id",
        "image_id",
        "measurement_id",
        "parent_or_policy_id",
        "hardware_identity",
        "raw_counts",
        "coupled_operation_blocks",
        "optimizer_iterations_by_type",
        "rng_streams",
        "branches",
        "timing",
        "memory_bytes",
        "overhead_seconds",
        "fre",
    }
    missing = sorted(required - set(ledger))
    if missing:
        raise ProtocolError(f"compute ledger missing fields: {missing}")
    if ledger["schema_version"] != "b23.compute-ledger.v2":
        raise ProtocolError("compute ledger must use b23.compute-ledger.v2")
    _validate_raw_counts(ledger["raw_counts"])
    optimizer_by_type = ledger["optimizer_iterations_by_type"]
    if not isinstance(optimizer_by_type, Mapping) or any(
        not isinstance(name, str) or not name or not isinstance(count, int)
        or isinstance(count, bool) or count < 0
        for name, count in optimizer_by_type.items()
    ):
        raise ProtocolError("optimizer_iterations_by_type must map names to non-negative integers")
    if ledger["raw_counts"]["optimizer_iterations"] != sum(optimizer_by_type.values()):
        raise ProtocolError("aggregate optimizer iterations do not reconcile by type")

    streams = ledger["rng_streams"]
    if not isinstance(streams, Sequence) or isinstance(streams, (str, bytes)):
        raise ProtocolError("rng_streams must be an array")
    stream_names: set[str] = set()
    rng_total = 0
    for stream in streams:
        if not isinstance(stream, Mapping):
            raise ProtocolError("each RNG stream must be an object")
        name = stream.get("name")
        if not isinstance(name, str) or not name or name in stream_names:
            raise ProtocolError("RNG stream names must be unique and non-empty")
        stream_names.add(name)
        for field in ("derived_seed", "draw_calls", "values_drawn"):
            value = stream.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ProtocolError(f"rng_streams.{field} must be a non-negative integer")
        if not isinstance(stream.get("device"), str) or not stream["device"]:
            raise ProtocolError("RNG stream device identity is required")
        rng_total += stream["values_drawn"]
    if ledger["raw_counts"]["rng_draws"] != rng_total:
        raise ProtocolError("aggregate RNG draws do not reconcile to unique named streams")

    branches = ledger["branches"]
    for name in ("total_created", "max_live", "retained", "terminal_candidates"):
        if not isinstance(branches.get(name), int) or branches[name] < 0:
            raise ProtocolError(f"branches.{name} must be a non-negative integer")
    for field, count_field in (
        ("branch_ids", "total_created"),
        ("retained_branch_ids", "retained"),
        ("terminal_branch_ids", "terminal_candidates"),
    ):
        values = branches.get(field)
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise ProtocolError(f"branches.{field} must contain unique IDs")
        if len(values) != branches[count_field] or any(not isinstance(x, str) or not x for x in values):
            raise ProtocolError(f"branches.{field} does not reconcile with {count_field}")
    all_ids = set(branches["branch_ids"])
    retained_ids = set(branches["retained_branch_ids"])
    terminal_ids = set(branches["terminal_branch_ids"])
    if not terminal_ids <= retained_ids <= all_ids:
        raise ProtocolError("terminal and retained candidates must be nested subsets")
    if not (branches["terminal_candidates"] <= branches["retained"] <= branches["total_created"]):
        raise ProtocolError("branch candidate counts are inconsistent")
    if branches["max_live"] > branches["total_created"]:
        raise ProtocolError("max_live cannot exceed total_created")
    for name in ("gpu_active_seconds", "wall_seconds"):
        value = ledger["timing"].get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ProtocolError(f"timing.{name} must be finite and non-negative")
    if ledger["timing"]["gpu_active_seconds"] > ledger["timing"]["wall_seconds"]:
        raise ProtocolError("GPU-active seconds cannot exceed wall seconds")
    if not isinstance(ledger["timing"].get("timer_method"), str) or not ledger["timing"]["timer_method"]:
        raise ProtocolError("timing.timer_method is required")
    memory = ledger["memory_bytes"]
    for name in ("peak_allocated", "peak_reserved"):
        if not isinstance(memory.get(name), int) or isinstance(memory[name], bool) or memory[name] < 0:
            raise ProtocolError(f"memory_bytes.{name} must be a non-negative integer")
    if memory["peak_allocated"] > memory["peak_reserved"]:
        raise ProtocolError("peak allocated memory cannot exceed peak reserved memory")
    overhead = ledger["overhead_seconds"]
    if not isinstance(overhead, Mapping):
        raise ProtocolError("overhead_seconds must be an object")
    for name, value in overhead.items():
        if not isinstance(name, str) or not name or not isinstance(value, (int, float)) \
                or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ProtocolError("overhead entries must be named finite non-negative numbers")
    if sum(overhead.values()) > ledger["timing"]["wall_seconds"] + 1e-12:
        raise ProtocolError("declared overhead cannot exceed wall time")

    coupled = ledger["coupled_operation_blocks"]
    _coupled_work(coupled)
    fre = ledger["fre"]
    status = fre.get("status")
    if status not in {"UNCALIBRATED", "CALIBRATED"}:
        raise ProtocolError("fre.status must be UNCALIBRATED or CALIBRATED")
    if status == "UNCALIBRATED":
        if ledger.get("atomic_weights"):
            raise ProtocolError("uncalibrated ledgers must not carry atomic weights")
        if any(ledger["raw_counts"].values()) or any(block["count"] for block in coupled):
            raise ProtocolError("nonzero operations require measured weights before validation")
        if ledger["hardware_identity"] is not None or fre.get("reference") is not None:
            raise ProtocolError("not-run uncalibrated ledgers must not claim hardware/reference identities")
        if any(fre.get(name) is not None for name in ("work_FRE", "time_FRE", "claim_FRE")):
            raise ProtocolError("uncalibrated FRE values must be null")
    else:
        hardware = ledger["hardware_identity"]
        reference = fre.get("reference")
        if not isinstance(hardware, Mapping) or not isinstance(reference, Mapping):
            raise ProtocolError("calibrated ledgers require hardware and frozen reference identities")
        for identity in (hardware, reference):
            if not all(isinstance(identity.get(key), str) and identity[key] for key in ("identity_id", "inventory_sha256")):
                raise ProtocolError("hardware/reference identity and inventory hash are required")
            digest = identity["inventory_sha256"]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ProtocolError("hardware/reference inventory identity must be a SHA-256")
        if hardware["identity_id"] != reference["identity_id"]:
            raise ProtocolError("policy and paired reference hardware identities differ")
        if hardware["inventory_sha256"] != reference["inventory_sha256"]:
            raise ProtocolError("policy and paired reference hardware inventories differ")
        for key in ("reference_ledger_sha256", "weight_registry_sha256"):
            value = reference.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ProtocolError(f"frozen reference {key} is missing or invalid")
        _validate_raw_counts(reference.get("raw_counts", {}))
        weights = ledger.get("atomic_weights", {})
        if not isinstance(weights, Mapping) or set(weights) - set(RAW_COUNT_FIELDS):
            raise ProtocolError("atomic weights may name only fixed raw operation counters")
        reference_blocks = reference.get("coupled_operation_blocks", [])
        _coupled_work(reference_blocks)
        policy_block_map = {block["operation_type"]: block for block in coupled}
        reference_block_map = {block["operation_type"]: block for block in reference_blocks}
        if set(policy_block_map) != set(reference_block_map):
            raise ProtocolError("policy/reference coupled operation registries differ")
        for name in policy_block_map:
            policy_block = policy_block_map[name]
            reference_block = reference_block_map[name]
            if policy_block["definition_sha256"] != reference_block["definition_sha256"] or not math.isclose(
                float(policy_block["measured_weight_seconds"]),
                float(reference_block["measured_weight_seconds"]),
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ProtocolError("policy/reference coupled operation definitions or weights differ")
        reference_seconds = reference.get("gpu_active_seconds")
        if not isinstance(reference_seconds, (int, float)) or isinstance(reference_seconds, bool) \
                or not math.isfinite(reference_seconds) or reference_seconds <= 0:
            raise ProtocolError("paired frozen reference GPU-active seconds must be finite and positive")
        executed_work = any(ledger["raw_counts"].values()) or any(
            block["count"] for block in coupled
        )
        if executed_work:
            if ledger["timing"]["gpu_active_seconds"] <= 0:
                raise ProtocolError(
                    "executed calibrated work requires strictly positive GPU-active time"
                )
            if ledger["timing"]["wall_seconds"] <= 0:
                raise ProtocolError(
                    "executed calibrated work requires strictly positive wall time"
                )
            timer_method = ledger["timing"]["timer_method"]
            normalized_timer = "".join(
                character for character in timer_method.upper() if character.isalnum()
            )
            if normalized_timer in PLACEHOLDER_TIMER_METHODS:
                raise ProtocolError(
                    "executed calibrated work requires a real non-placeholder timer method"
                )
        expected = fre_values(
            policy_counts=ledger["raw_counts"],
            reference_counts=reference["raw_counts"],
            measured_weights=weights,
            gpu_active_seconds=ledger["timing"]["gpu_active_seconds"],
            paired_reference_median_seconds=reference_seconds,
            policy_coupled_operations=coupled,
            reference_coupled_operations=reference_blocks,
        )
        values = [fre.get(name) for name in ("work_FRE", "time_FRE", "claim_FRE")]
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 for value in values):
            raise ProtocolError("calibrated FRE values must be finite non-negative numbers")
        for key in ("work_FRE", "time_FRE", "claim_FRE"):
            if not math.isclose(float(fre[key]), expected[key], rel_tol=1e-12, abs_tol=1e-12):
                raise ProtocolError(f"supplied {key} does not match recomputed frozen-reference value")


def validate_replay_report(report: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "experiment_id",
        "parent_id",
        "native_run_ids",
        "wrapper_run_id",
        "eligibility",
        "determinism_audit",
        "tolerance_qualification",
        "native_repeatability_envelope",
        "wrapper_comparison",
        "operation_count_reconciled",
        "rng_draws_reconciled",
        "verdict",
    }
    missing = sorted(required - set(report))
    if missing:
        raise ProtocolError(f"replay report missing fields: {missing}")
    if report["schema_version"] != "b23.replay-report.v3":
        raise ProtocolError("replay report must use b23.replay-report.v3")
    if report["eligibility"] not in {"BITWISE", "TOLERANCE_QUALIFIED", "UNDETERMINED"}:
        raise ProtocolError("invalid replay eligibility")
    if report["verdict"] not in {"PASS", "FAIL", "NOT_RUN"}:
        raise ProtocolError("invalid replay verdict")
    audit = report["determinism_audit"]
    if not isinstance(audit, Mapping):
        raise ProtocolError("determinism_audit must be an object")
    audit_status = audit.get("audit_status")
    if audit_status not in {"COMPLETE", "JUSTIFIED_UNAVAILABLE"}:
        raise ProtocolError(
            "determinism audit must be COMPLETE or JUSTIFIED_UNAVAILABLE"
        )
    if audit_status == "COMPLETE":
        if audit.get("unavailable_reason") is not None:
            raise ProtocolError("a complete determinism audit cannot claim unavailability")
        for field in (
            "torch_deterministic_algorithms",
            "cudnn_deterministic",
            "cudnn_benchmark",
            "flags_change_native_parent",
        ):
            if not isinstance(audit.get(field), bool):
                raise ProtocolError(f"complete determinism audit requires boolean {field}")
        if not isinstance(audit.get("cublas_workspace_config"), str) or not audit[
            "cublas_workspace_config"
        ].strip():
            raise ProtocolError(
                "complete determinism audit requires CUBLAS_WORKSPACE_CONFIG evidence; use UNSET when absent"
            )
    else:
        reason = audit.get("unavailable_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ProtocolError(
                "JUSTIFIED_UNAVAILABLE determinism audit requires a reason"
            )

    if report["verdict"] == "NOT_RUN":
        if report["eligibility"] != "UNDETERMINED":
            raise ProtocolError("an unexecuted replay must have UNDETERMINED eligibility")
        if report["tolerance_qualification"] is not None:
            raise ProtocolError("an unexecuted replay cannot carry a tolerance freeze")
        return
    if report["verdict"] != "PASS":
        return
    runs = report["native_run_ids"]
    if not isinstance(runs, list) or len(runs) < 3 or len(set(runs)) != len(runs):
        raise ProtocolError("a passing replay needs at least three unique native runs")
    if not isinstance(report["wrapper_run_id"], str) or not report["wrapper_run_id"]:
        raise ProtocolError("a passing replay needs a non-null wrapper run")
    for section in ("native_repeatability_envelope", "wrapper_comparison"):
        comparison = report.get(section)
        if not isinstance(comparison, Mapping):
            raise ProtocolError(f"{section} is missing")
        for metric in REPLAY_NUMERIC_METRICS:
            value = comparison.get(metric)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ProtocolError(f"passing replay requires finite {section}.{metric}")
        if not isinstance(comparison.get("tensor_hash_equal"), bool):
            raise ProtocolError(f"passing replay requires {section}.tensor_hash_equal")
    hash_pairs = (
        ("native_operation_counts_sha256", "wrapper_operation_counts_sha256"),
        ("native_rng_ledger_sha256", "wrapper_rng_ledger_sha256"),
    )
    for native_key, wrapper_key in hash_pairs:
        values = (report.get(native_key), report.get(wrapper_key))
        if any(not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in values):
            raise ProtocolError("passing replay requires valid operation and RNG evidence hashes")
        if values[0] != values[1]:
            raise ProtocolError("operation/RNG evidence hashes must reconcile exactly")
    for native_key, wrapper_key in (
        ("native_tensor_sha256s", "wrapper_tensor_sha256"),
        ("native_trace_sha256s", "wrapper_trace_sha256"),
    ):
        native_hashes = report.get(native_key)
        wrapper_hash = report.get(wrapper_key)
        if not isinstance(native_hashes, list) or len(native_hashes) != len(runs):
            raise ProtocolError("passing replay needs one tensor/trace evidence hash per native run")
        all_hashes = [*native_hashes, wrapper_hash]
        if any(not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in all_hashes):
            raise ProtocolError("passing replay tensor/trace evidence hashes are missing or invalid")
    if report.get("operation_count_reconciled") is not True or report.get("rng_draws_reconciled") is not True:
        raise ProtocolError("a passing replay must reconcile operation and RNG counts")
    serialization = report.get("serialization_resume_check")
    if serialization == "NOT_APPLICABLE":
        reason = report.get("serialization_not_applicable_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ProtocolError("NOT_APPLICABLE serialization requires a justification")
    elif serialization != "PASS":
        raise ProtocolError("passing replay requires serialization PASS or justified NOT_APPLICABLE")
    if report["eligibility"] == "UNDETERMINED":
        raise ProtocolError("a passing replay cannot have UNDETERMINED eligibility")
    tolerance = report["tolerance_qualification"]
    if report["eligibility"] == "TOLERANCE_QUALIFIED":
        if not isinstance(tolerance, Mapping):
            raise ProtocolError(
                "TOLERANCE_QUALIFIED PASS requires a pre-wrapper tolerance freeze identity"
            )
        freeze_hash = tolerance.get("freeze_record_sha256")
        if not isinstance(freeze_hash, str) or len(freeze_hash) != 64 or any(
            character not in "0123456789abcdef" for character in freeze_hash
        ):
            raise ProtocolError("tolerance freeze identity must be a SHA-256")
        if tolerance.get("frozen_before_wrapper") is not True:
            raise ProtocolError("tolerance envelope must be frozen before the wrapper run")
        if tolerance.get("wrapper_run_id_at_freeze") is not None:
            raise ProtocolError("pre-wrapper tolerance freeze cannot include a wrapper run")
        frozen_runs = tolerance.get("frozen_native_run_ids")
        if frozen_runs != runs:
            raise ProtocolError("tolerance freeze native runs do not match the replay report")
        floors = tolerance.get("numerical_floors")
        if not isinstance(floors, Mapping) or set(floors) != set(REPLAY_NUMERIC_METRICS):
            raise ProtocolError("every replay comparison metric needs one declared numerical floor")
        for metric in REPLAY_NUMERIC_METRICS:
            floor = floors[metric]
            if not isinstance(floor, (int, float)) or isinstance(floor, bool) \
                    or not math.isfinite(floor) or floor < 0:
                raise ProtocolError(f"numerical floor for {metric} must be finite and non-negative")
            native_bound = abs(float(report["native_repeatability_envelope"][metric]))
            wrapper_delta = abs(float(report["wrapper_comparison"][metric]))
            if wrapper_delta > native_bound + float(floor):
                raise ProtocolError(
                    f"wrapper {metric} exceeds the frozen native envelope plus numerical floor"
                )
    elif tolerance is not None:
        raise ProtocolError("BITWISE replay must not masquerade as tolerance-qualified")
    if report["eligibility"] == "BITWISE":
        for native_key, wrapper_key in (
            ("native_tensor_sha256s", "wrapper_tensor_sha256"),
            ("native_trace_sha256s", "wrapper_trace_sha256"),
        ):
            if any(value != report[wrapper_key] for value in report[native_key]):
                raise ProtocolError("BITWISE requires matching native and wrapper tensor/trace hashes")
        for section in ("native_repeatability_envelope", "wrapper_comparison"):
            comparison = report[section]
            if comparison["tensor_hash_equal"] is not True:
                raise ProtocolError("BITWISE requires matching tensor hashes")
            if any(float(comparison[key]) != 0.0 for key in REPLAY_NUMERIC_METRICS):
                raise ProtocolError("BITWISE requires zero declared tensor and trace deltas")


def validate_future_split_registry(
    rows: Sequence[Mapping[str, Any]], exposed_image_ids: set[str]
) -> None:
    """Fail closed if a future row reuses any image exposed before B23."""

    required = {
        "registry_version", "split", "row_id", "image_id", "measurement_id",
        "measurement_seed", "solver_base_seed", "assigned_before_run",
        "pre_b23_exposure_checked", "source_manifest_sha256",
    }
    allowed_splits = {
        "DEV-SCREEN", "DEV-MECH", "DEV-NATURAL", "CAL-B2", "TEST-AUDIT",
        "TEST-PROSPECTIVE", "B23.1-SMOKE-1", "B23.1-SMOKE-4",
    }
    keys: set[tuple[str, int]] = set()
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise ProtocolError(f"future-registry row missing fields: {missing}")
        image_id = row.get("image_id")
        if row.get("registry_version") != "b23.future-split.v1":
            raise ProtocolError("invalid future-registry version")
        if row.get("split") not in allowed_splits:
            raise ProtocolError("invalid future-registry split")
        if not isinstance(image_id, str) or len(image_id) != 5 or not image_id.isdigit():
            raise ProtocolError("future-registry image_id must contain five digits")
        for field in ("row_id", "measurement_seed", "solver_base_seed"):
            value = row.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ProtocolError(f"future-registry {field} must be a non-negative integer")
        if not isinstance(row.get("measurement_id"), str) or not row["measurement_id"]:
            raise ProtocolError("future-registry measurement identity is required")
        digest = row.get("source_manifest_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ProtocolError("future-registry source manifest needs a SHA-256")
        if image_id in exposed_image_ids:
            raise ProtocolError(
                f"future registry image {image_id} is in PRE_B23_EXPOSURE; measurement/seed cannot override exclusion"
            )
        key = (row["split"], row["row_id"])
        if key in keys:
            raise ProtocolError(f"duplicate future-registry row identity: {key}")
        keys.add(key)
        if row.get("assigned_before_run") is not True or row.get("pre_b23_exposure_checked") is not True:
            raise ProtocolError("future-registry rows must be assigned and exposure-checked before run")


def _tensor_fingerprint(tensor: Any) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - PAC environment has torch
        raise ProtocolError("state serialization requires torch") from exc
    if not isinstance(tensor, torch.Tensor):
        raise ProtocolError("native tensor payloads must be torch.Tensor instances")
    detached = tensor.detach().cpu().contiguous()
    byte_view = detached.view(torch.uint8)
    digest = hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "sha256": digest,
    }


def serialize_native_state(state: NativeState[Any], directory: str | Path) -> dict[str, Any]:
    """Serialize one state at an explicitly valid native boundary."""

    state.validate()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - PAC environment has torch
        raise ProtocolError("state serialization requires torch") from exc
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    payload_path = destination / "native_payloads.pt"
    torch.save(
        {
            "tensor_payloads": dict(state.tensor_payloads),
            "optimizer_state": dict(state.optimizer_state),
        },
        payload_path,
    )
    fingerprints = {
        name: _tensor_fingerprint(tensor) for name, tensor in state.tensor_payloads.items()
    }
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "parent_id": state.parent_id,
        "source_revision": state.source_revision,
        "representation_contract": dict(state.representation_contract),
        "coordinate": dataclasses.asdict(state.coordinate),
        "model_identity": dict(state.model_identity),
        "scheduler_identity": dict(state.scheduler_identity),
        "measurement_identity": dict(state.measurement_identity),
        "rng_streams": [dataclasses.asdict(stream) for stream in state.rng_streams],
        "trace_id": state.trace_id,
        "compute_ledger_id": state.compute_ledger_id,
        "tensor_fingerprints": fingerprints,
        "payload_file": payload_path.name,
    }
    (destination / "metadata.json").write_bytes(_canonical_json_bytes(metadata) + b"\n")
    return metadata


def load_native_state(directory: str | Path) -> NativeState[str]:
    """Load and fingerprint-check a serialized native state."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - PAC environment has torch
        raise ProtocolError("state serialization requires torch") from exc
    source = Path(directory)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("unsupported native-state protocol version")
    payload = torch.load(
        source / metadata["payload_file"], map_location="cpu", weights_only=False
    )
    actual = {
        name: _tensor_fingerprint(tensor)
        for name, tensor in payload["tensor_payloads"].items()
    }
    if actual != metadata["tensor_fingerprints"]:
        raise ProtocolError("native-state tensor fingerprint mismatch")
    state = NativeState(
        parent_id=metadata["parent_id"],
        source_revision=metadata["source_revision"],
        representation_contract=metadata["representation_contract"],
        tensor_payloads=payload["tensor_payloads"],
        coordinate=SemanticCoordinate(**metadata["coordinate"]),
        model_identity=metadata["model_identity"],
        scheduler_identity=metadata["scheduler_identity"],
        measurement_identity=metadata["measurement_identity"],
        optimizer_state=payload["optimizer_state"],
        rng_streams=[RNGStreamState(**item) for item in metadata["rng_streams"]],
        trace_id=metadata["trace_id"],
        compute_ledger_id=metadata["compute_ledger_id"],
    )
    state.validate()
    return state


__all__ = [
    "AdapterSpec",
    "ModuleSpec",
    "NativeState",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "RAW_COUNT_FIELDS",
    "RNGStreamState",
    "SEED_DERIVATION_VERSION",
    "SemanticCoordinate",
    "calibrated_work",
    "derive_seed",
    "fre_values",
    "load_native_state",
    "serialize_native_state",
    "validate_compute_ledger",
    "validate_replay_report",
]
