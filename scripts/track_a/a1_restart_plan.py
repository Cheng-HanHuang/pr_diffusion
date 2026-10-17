#!/usr/bin/env python3
"""A1.0d synthetic-only restart manifest and fail-closed bookkeeping.

STRICTLY NO native execution. Pure Python stdlib; no dataset, model, GT,
GPU, measurements, sockets, files, process creation or PAC path reads.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from scripts.track_a.a1_protocol import (
    ARMS, ProtocolStop, candidate_matrix, image_id, mixed_np4, mixed_np8,
    seed32_numpy, seed63,
)

SCHEMA = "prdiff.a1.synthetic-restart-manifest.v1"
SYNTHETIC_MARKER = "INVENTED_FAKE_IDS_NO_DATA"
START_SHAPE = (1, 3, 256, 256)
CONFIG = {
    "DAPS": "native_phase_retrieval_edm_daps_anneal400_diff5",
    "SITCOM": "native_sitcom_ode_anneal200_diff5_lgvd100",
    "NP_LF": "np_lf_1000_steps",
    "NP_S2": "np_prev_l2_lam001_preprojection_only_1000_steps",
}
# These are semantic execution contracts, never usable native commands.
SEED_RULE = {
    "DAPS": "argv_seed_and_B21_START_NOISE_SEED_32bit_same",
    "SITCOM": "native_single_run_config_seed_32bit",
    "NP_LF": "native_single_candidate_seed_32bit",
    "NP_S2": "native_single_candidate_seed_32bit",
}


def build_synthetic_plan(image_ids: list[str]) -> dict[str, Any]:
    """Accept only explicitly flagged, invented IDs used in CPU tests."""
    if not image_ids or len(image_ids) > 12:
        raise ProtocolStop("synthetic plan needs 1..12 invented image IDs")
    if len(set(image_ids)) != len(image_ids):
        raise ProtocolStop("duplicate invented image IDs")
    rows: list[dict[str, Any]] = []
    shared: list[dict[str, Any]] = []
    for raw in image_ids:
        identifier = image_id(raw)
        candidates = candidate_matrix(identifier)
        meas_seed = seed32_numpy("A1_MEAS_V1", identifier, 0)
        if meas_seed in {seed32_numpy(c.family, identifier, c.replicate)
                         for c in candidates}:
            raise ProtocolStop("measurement/solver seed collided")
        for candidate in candidates:
            native_seed = seed32_numpy(
                candidate.family, identifier, candidate.replicate)
            rows.append({
                "synthetic_image_id": identifier,
                "candidate_id": candidate.candidate_id,
                "method": candidate.family,
                "config": CONFIG[candidate.family],
                "restart_index": candidate.replicate,
                "seed63_domain_audit": seed63(
                    candidate.family, identifier, candidate.replicate),
                "native_seed32": native_seed,
                "seed_rule": SEED_RULE[candidate.family],
                "independent_process_required": True,
                "measurement_seed32": meas_seed,
                "measurement_identity": "UNASSIGNED_NO_REAL_MEASUREMENT",
                "gt_runtime_access": "FORBIDDEN_EXCEPT_OFFLINE_POSTRUN_EVALUATOR",
                "operator": "OVERSAMPLED_FOURIER_MAGNITUDE_2X_SIGMA_005",
                "initialization_reference": (
                    "MODEL_INPUT_SHAPE_ONLY" if candidate.family == "DAPS"
                    else "EMPTY_REF_SHAPE_ONLY" if candidate.family == "SITCOM"
                    else "NOT_REQUIRED_BY_NP_API"),
                "requested_gpu": "UNALLOCATED",
                "estimated_candidate_seconds": "NOT_YET_MEASURED",
                "native_command": "NOT_READY_SOURCE_PREFLIGHT_PENDING",
            })
        shared.append({
            "image": identifier,
            "measurement_seed32": meas_seed,
            "unique_native_candidates": len(candidates),
            "mixed4_reuses": [c.candidate_id for c in mixed_np4(candidates)],
            "mixed8_reuses": [c.candidate_id for c in mixed_np8(candidates)],
        })
    verify_synthetic_plan(rows, shared)
    result = {
        "schema": SCHEMA, "kind": SYNTHETIC_MARKER, "status": "DRYRUN_NOT_LAUNCHABLE",
        "checks": {
            "cohort_allocated": False,
            "measurement_created": False,
            "model_loaded": False,
            "gpu_work_performed": False,
            "locked_confirmation_accessed": False,
            "native_commands_executable": False,
        },
        "images": shared,
        "candidates": rows,
        "candidate_total": len(rows),
        "np_mixed8_extra_runs": 0,
        "min_candidate_set_size": 16,
    }
    return result


def verify_synthetic_plan(rows: list[dict[str, Any]],
                          shared: list[dict[str, Any]]) -> None:
    if len(rows) != len(shared) * 16:
        raise ProtocolStop("candidate budget does not equal 16 per image")
    ids: set[str] = set()
    for image in shared:
        identifier = image["image"]
        per = [r for r in rows if r["synthetic_image_id"] == identifier]
        if len(per) != 16:
            raise ProtocolStop("incomplete per-image candidate set")
        if Counter(r["method"] for r in per) != Counter({m: 4 for m in ARMS}):
            raise ProtocolStop("not exactly four independent starts per method")
        native_seeds = [r["native_seed32"] for r in per]
        if len(set(native_seeds)) != 16:
            raise ProtocolStop("native seed collision within image")
        measure_seeds = {r["measurement_seed32"] for r in per}
        if measure_seeds != {image["measurement_seed32"]}:
            raise ProtocolStop("measurement seed differs among methods")
        for row in per:
            ident = row["candidate_id"]
            if ident in ids:
                raise ProtocolStop("candidate ID reused in batch")
            ids.add(ident)
            if not row["independent_process_required"]:
                raise ProtocolStop("solver process isolation not required")
            if row["native_command"] != "NOT_READY_SOURCE_PREFLIGHT_PENDING":
                raise ProtocolStop("dry-run manifest must contain no executable command")
            if row["measurement_identity"] != "UNASSIGNED_NO_REAL_MEASUREMENT":
                raise ProtocolStop("dry-run manifest is contaminated with real data")
            if row["native_seed32"] != seed32_numpy(
                row["method"], identifier, row["restart_index"]):
                raise ProtocolStop("native seed drift")
        current_ids = {r["candidate_id"] for r in per}
        if len(image["mixed4_reuses"]) != 4 or len(image["mixed8_reuses"]) != 8:
            raise ProtocolStop("mixed NP candidate counts invalid")
        if len(set(image["mixed8_reuses"])) != 8:
            raise ProtocolStop("mixed8 IDs duplicated")
        if not set(image["mixed8_reuses"]).issubset(current_ids):
            raise ProtocolStop("NP mix references unknown run")
        if image["mixed4_reuses"] != image["mixed8_reuses"][:4]:
            raise ProtocolStop("NP mixed4 allocation drift")


def synthetic_summary() -> dict[str, Any]:
    """Small, invented example only; not a prospective cohort proposal."""
    plan = build_synthetic_plan(["10000", "10001"])
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema": SCHEMA, "status": plan["status"],
        "kind": SYNTHETIC_MARKER,
        "invented_images": len(plan["images"]),
        "unique_candidates": len(plan["candidates"]),
        "per_image_methods": {"DAPS": 4, "SITCOM": 4, "NP_LF": 4, "NP_S2": 4},
        "mixed8_additional_gpu_work": 0,
        "manifest_content_sha256": digest,
        "checks": plan["checks"],
    }


if __name__ == "__main__":
    print(json.dumps(synthetic_summary(), sort_keys=True))
