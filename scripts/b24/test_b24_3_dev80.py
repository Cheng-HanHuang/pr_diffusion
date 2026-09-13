#!/usr/bin/env python3
"""Zero-GPU contract tests for the authorized B24.3 DEV80 stage."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "configs/b24/b24_3_dev80_overnight.json"
EPP = REPO / "configs/b24/b24_3_epp321_refinement.json"
SCREEN_SHA = "b516c8154cbbb790d8a3592b86736bb0d4bd47d0833d85ecf3d6a9d710e950ba"


def seed63(domain: str, label: str, image: str, rep: int) -> int:
    material = "|".join([domain, SCREEN_SHA, label, image, str(rep)])
    return int(hashlib.sha256(material.encode()).hexdigest()[:16], 16) & ((1 << 63) - 1)


def main() -> int:
    spec = json.loads(SPEC.read_text())
    epp = json.loads(EPP.read_text())
    assert spec["status"] == "AUTHORIZED_DEVELOPMENT_ONLY"
    assert spec["development_panel"] == {
        "total": 80,
        "screening_strata": {"A": 20, "B": 20, "C": 20, "D": 20},
        "pilot16_reused": 16,
        "new_nonpilot_development_images": 64,
        "confirmation_images": 0,
    }
    assert spec["execution_scope"]["fresh_daps4_images"] == 80
    assert spec["execution_scope"]["fresh_sitcom4_images"] == 80
    assert spec["execution_scope"]["new_np_execution_images"] == 64
    assert not spec["execution_scope"]["confirmation_images_authorized"]
    assert not spec["execution_scope"]["confirmation_measurement_generation_authorized"]

    np4 = spec["np_arms_all80"]["NP4_INDEPENDENT"]
    main_arm = spec["np_arms_all80"]["NP_EPP_321"]
    rand = spec["np_arms_all80"]["NP_EPP_321_RANDOM_PRUNE"]
    no_realloc = spec["np_arms_all80"]["NP_EPP_321_NO_REALLOCATION"]
    assert np4["total_unet_evals"] == main_arm["total_unet_evals"] == rand["total_unet_evals"] == 8800
    assert no_realloc["total_unet_evals"] == 6900
    assert epp["arms"]["NP_EPP_321"]["estimated_total_unet_evals_including_initial"] == 8800

    for domain in ("B24_DEV80_DAPS_SOLVER_V1", "B24_DEV80_SITCOM_SOLVER_V1"):
        seeds = [seed63(domain, "D", "30740", r) for r in range(4)]
        assert len(set(seeds)) == 4
        assert len({x % (2**32) for x in seeds}) == 4
        assert seeds == [seed63(domain, "D", "30740", r) for r in range(4)]
    assert seed63("B24_DEV80_DAPS_SOLVER_V1", "D", "30740", 0) != seed63(
        "B24_DEV80_SITCOM_SOLVER_V1", "D", "30740", 0
    )

    assert spec["evaluation_policy"]["good25_threshold_db"] == 25.0
    assert spec["resource_contract"]["hard_process_or_group_ceiling_mib"] == 52452
    assert spec["resource_contract"]["min_free_mib"] == 10240
    print("B24_3_DEV80_ZERO_GPU_TESTS_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
