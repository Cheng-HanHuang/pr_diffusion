#!/usr/bin/env python3
"""Render B23.1 command records without launching or touching CUDA."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PARENTS = ("Fresh1", "LF-v1", "NP-1", "SITCOM-1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    registry = args.registry or repo / "manifests/b23/future_split_registry.csv"
    with registry.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    rendered = {
        "schema_version": "b23.b23-1-dry-run.v1",
        "authorized": False,
        "gpu_work_performed": False,
        "registry": (
            registry.resolve().relative_to(repo).as_posix()
            if registry.resolve().is_relative_to(repo)
            else "PAC_CONTRACT:" + str(registry.resolve())
        ),
        "registry_rows": len(rows),
        "parents": list(PARENTS),
        "commands": [],
        "command_templates": {
            "Fresh1": "CUDA_VISIBLE_DEVICES=<AUTHORIZED_GPU> <DAPS_PYTHON> <DAPS_ROOT>/posterior_sample.py <FROZEN_FRESH1_ARGS_FROM_CONFIG> <SIGNED_REGISTRY_ROW>",
            "LF-v1": "CUDA_VISIBLE_DEVICES=<AUTHORIZED_GPU> B20_LF_ENABLE=1 B20_LF_ALPHA=0.50 B20_LF_FRAC=0.35 B20_LF_RADIUS_FRAC=0.12 <DAPS_PYTHON> <DAPS_ROOT>/posterior_sample.py <FROZEN_LF_V1_ARGS> <SIGNED_REGISTRY_ROW>",
            "NP-1": "CUDA_VISIBLE_DEVICES=<AUTHORIZED_GPU> <NP_PYTHON> scripts/b22/run_b22_1_np_smoke.py <FROZEN_NP1_ARGS> <SIGNED_REGISTRY_ROW>",
            "SITCOM-1": "CUDA_VISIBLE_DEVICES=<AUTHORIZED_GPU> <SITCOM_PYTHON> scripts/b22/run_b22_1_sitcom_smoke.py <FROZEN_SITCOM1_ARGS> <SIGNED_REGISTRY_ROW>"
        },
        "blockers": [
            "B23.0 planner/user sign-off has not been recorded",
            "separate B23.1 GPU authorization has not been received",
            "the B23.1 signed smoke registry is intentionally empty",
            "native repeatability tolerances and atomic operation weights are not yet measured"
        ],
        "note": "Templates are documentation, not executable schedules. Exact commands are rendered only after a signed registry and authorization exist."
    }
    output = args.output or repo / "manifests/b23/b23_1_dry_run.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rendered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "DRY_RUN_ONLY", "commands": 0, "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
