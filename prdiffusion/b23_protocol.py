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


PROTOCOL_VERSION = "b23.0-v1"
SEED_DERIVATION_VERSION = "b23-sha256-v1"

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
        if not math.isfinite(weight) or weight < 0:
            raise ProtocolError(f"weight {name!r} must be finite and non-negative")
        total += count * weight
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
) -> dict[str, float]:
    """Return work-, time-, and claim-FRE under already frozen weights."""

    policy_work = calibrated_work(policy_counts, measured_weights)
    reference_work = calibrated_work(reference_counts, measured_weights)
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
        "raw_counts",
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
    _validate_raw_counts(ledger["raw_counts"])
    branches = ledger["branches"]
    for name in ("max_live", "retained", "terminal_candidates"):
        if not isinstance(branches.get(name), int) or branches[name] < 0:
            raise ProtocolError(f"branches.{name} must be a non-negative integer")
    for name in ("gpu_active_seconds", "wall_seconds"):
        value = ledger["timing"].get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ProtocolError(f"timing.{name} must be finite and non-negative")
    fre = ledger["fre"]
    status = fre.get("status")
    if status not in {"UNCALIBRATED", "CALIBRATED"}:
        raise ProtocolError("fre.status must be UNCALIBRATED or CALIBRATED")
    if status == "UNCALIBRATED":
        if ledger.get("atomic_weights"):
            raise ProtocolError("uncalibrated ledgers must not carry atomic weights")
        if any(fre.get(name) is not None for name in ("work_FRE", "time_FRE", "claim_FRE")):
            raise ProtocolError("uncalibrated FRE values must be null")
    else:
        calibrated_work(ledger["raw_counts"], ledger.get("atomic_weights", {}))
        values = [fre.get(name) for name in ("work_FRE", "time_FRE", "claim_FRE")]
        if not all(isinstance(value, (int, float)) and value >= 0 for value in values):
            raise ProtocolError("calibrated FRE values must be non-negative numbers")
        if not math.isclose(fre["claim_FRE"], max(fre["work_FRE"], fre["time_FRE"])):
            raise ProtocolError("claim_FRE must equal max(work_FRE, time_FRE)")


def validate_replay_report(report: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "experiment_id",
        "parent_id",
        "native_run_ids",
        "wrapper_run_id",
        "eligibility",
        "determinism_audit",
        "native_repeatability_envelope",
        "wrapper_comparison",
        "operation_count_reconciled",
        "rng_draws_reconciled",
        "verdict",
    }
    missing = sorted(required - set(report))
    if missing:
        raise ProtocolError(f"replay report missing fields: {missing}")
    if report["eligibility"] not in {"BITWISE", "TOLERANCE_QUALIFIED"}:
        raise ProtocolError("invalid replay eligibility")
    if report["verdict"] not in {"PASS", "FAIL", "NOT_RUN"}:
        raise ProtocolError("invalid replay verdict")
    if report["verdict"] == "PASS" and not (
        report["operation_count_reconciled"] and report["rng_draws_reconciled"]
    ):
        raise ProtocolError("a passing replay must reconcile operation and RNG counts")


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
