#!/usr/bin/env python3
"""Fail-closed B24.0 validator. No model or GPU access."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from prdiffusion.b24_protocol import B24_BASE_HEAD, validate_pre_b24


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--contract", type=Path, default=Path("configs/b24/b24_0_contract.json"))
    parser.add_argument("--exposure", type=Path, default=Path("manifests/b24/PRE_B24_EXPOSURE.csv"))
    parser.add_argument("--allow-missing-pre-b24", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    head = git(repo, "rev-parse", "HEAD")
    ancestry = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", B24_BASE_HEAD, head],
        check=False,
    ).returncode == 0
    if not ancestry:
        raise SystemExit(f"STOP: current head {head} does not descend from {B24_BASE_HEAD}")

    contract = json.loads((repo / args.contract).read_text(encoding="utf-8"))
    if contract.get("gpu_authorized") is not False:
        raise SystemExit("STOP: B24.0 contract must set gpu_authorized=false")
    future = contract.get("future_stages_authorized", {})
    if any(bool(v) for v in future.values()):
        raise SystemExit("STOP: B24.0 may not authorize a future GPU/method stage")

    exposure = repo / args.exposure
    exposure_status = "PASS"
    exposure_count = None
    if exposure.exists():
        exposure_count = len(validate_pre_b24(exposure))
    elif args.allow_missing_pre_b24:
        exposure_status = "BLOCKED_PENDING_HASH_FROZEN_PAC_IMPORT"
    else:
        raise SystemExit("STOP: PRE_B24_EXPOSURE.csv missing; run the zero-GPU PAC import")

    result = {
        "head": head,
        "base_ancestry": "PASS",
        "contract": "PASS",
        "gpu_authorized": False,
        "pre_b24_exposure": exposure_status,
        "pre_b24_unique_images": exposure_count,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
