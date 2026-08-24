#!/usr/bin/env python3
"""Classify B23.1 donors without inventing a cross-family adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PARENTS = ("Fresh1", "LF-v1", "NP-1", "SITCOM-1")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-report", type=Path, action="append", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = {value["parent_id"]: value for value in map(read_json, args.replay_report)}
    if set(reports) != set(PARENTS) or any(value["verdict"] != "PASS" for value in reports.values()):
        raise ValueError("donor classification requires four passing native replays")
    smoke_runs = []
    for row in range(4):
        for parent in PARENTS:
            key = parent.lower().replace("-", "_")
            run_path = args.run_root / "smoke" / f"row{row}" / key / "RUN.json"
            run = read_json(run_path)
            if (
                run.get("status") != "PASS"
                or run.get("mode") != "smoke"
                or int(run.get("row_id", -1)) != row
                or run.get("parent_id") != parent
                or int(run.get("terminal_candidates", -1)) != 1
            ):
                raise ValueError(f"invalid heterogeneous-smoke record: {run_path}")
            smoke_runs.append(run)
    smoke_images = sorted({run["image_id"] for run in smoke_runs})
    if len(smoke_runs) != 16 or len(smoke_images) != 4:
        raise ValueError("donor classification requires 16 passing runs on four heterogeneous images")
    rows = [
        {
            "operation_id": "Fresh1.complete_native_sequence",
            "source_parent": "Fresh1", "target_family": "DAPS", "classification": "DAPS-NATIVE-DONOR",
            "adapter": "identity", "reason": "same typed DAPS state and frozen native module sequence replayed",
        },
        {
            "operation_id": "LF-v1.early_lf_then_native_sequence",
            "source_parent": "LF-v1", "target_family": "DAPS", "classification": "DAPS-NATIVE-DONOR",
            "adapter": "identity", "reason": "same typed DAPS state; LF intervention and ordinary late DAPS replayed together",
        },
        {
            "operation_id": "NP-1.proposal_ranking_projection",
            "source_parent": "NP-1", "target_family": "cross-family", "classification": "BASELINE-ONLY",
            "adapter": None, "reason": "native replay passed, but a DAPS/SITCOM boundary would discard proposal-set, scheduler, and RNG semantics; no lossless adapter was demonstrated",
        },
        {
            "operation_id": "SITCOM-1.triple_consistency_block",
            "source_parent": "SITCOM-1", "target_family": "cross-family", "classification": "BASELINE-ONLY",
            "adapter": None, "reason": "native replay passed, but LGVD optimizer state and forward-noised coordinate have no qualified DAPS/NP adapter",
        },
        {
            "operation_id": "historical.NP_to_SITCOM_direct_handoff",
            "source_parent": "NP-1", "target_family": "SITCOM-1", "classification": "REJECTED-PROTOTYPE",
            "adapter": "historical lossy clean-estimate handoff", "reason": "preserved negative evidence; direct handoff violates native-state semantics",
        },
    ]
    result = {
        "schema_version": "b23.donor-compatibility.v1",
        "status": "PASS",
        "rows": rows,
        "cross_family_adapter_qualified_donor_count": 0,
        "heterogeneous_smoke_audit": {
            "status": "PASS",
            "parent_runs": len(smoke_runs),
            "unique_images": smoke_images,
            "terminal_candidates_per_run": 1,
        },
        "b23_1_return_recommendation": "CONTINUE DAPS-NATIVE ONLY UNDER NARROWED CLAIM",
        "b23_2_authorized": False,
        "adaptive_schedules_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "recommendation": result["b23_1_return_recommendation"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
