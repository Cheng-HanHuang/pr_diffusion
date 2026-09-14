#!/usr/bin/env python3
"""Bounded B24.3 EPP321 Pilot16 refinement.

Adds only the planner-authorized refinement arms on top of the frozen B24.3
branching runner:
  NP_EPP_321
  NP_EPP_321_RANDOM_PRUNE
  NP_EPP_321_NO_REALLOCATION

All runtime survival decisions remain clean-free.  The main and random-prune
arms are exactly matched to NP4/current-EPP dominant NP compute: 8796 proposal
UNet evaluations + 4 initial root evaluations = 8800 total evaluations.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PARENT_PATH = REPO / "scripts" / "b24" / "run_b24_3_np_branching.py"
REFINEMENT_SPEC_PATH = REPO / "configs" / "b24" / "b24_3_epp321_refinement.json"


def load_parent():
    spec = importlib.util.spec_from_file_location("b24_3_branching_parent", PARENT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import parent runner from {PARENT_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_parent()
base.SPEC_PATH = REFINEMENT_SPEC_PATH

ARM_ORDER = (
    "NP_EPP_321",
    "NP_EPP_321_RANDOM_PRUNE",
    "NP_EPP_321_NO_REALLOCATION",
)

EXPECTED_COUNTS = {
    "NP_EPP_321": {"proposal_evals": 8796, "total_unet_evals": 8800, "terminals": 4},
    "NP_EPP_321_RANDOM_PRUNE": {"proposal_evals": 8796, "total_unet_evals": 8800, "terminals": 4},
    "NP_EPP_321_NO_REALLOCATION": {"proposal_evals": 6896, "total_unet_evals": 6900, "terminals": 4},
}


def prune_random_321(ctx, branches, keep, checkpoint, events):
    ranked = sorted(
        branches,
        key=lambda b: (
            base.domain_hash(
                "B24_METHOD_EPP321_RANDOM_PRUNE_V1",
                ctx.image_id,
                checkpoint,
                b.lineage,
            ),
            b.lineage,
        ),
    )
    survivors = ranked[:keep]
    events.append(
        base.checkpoint_event(
            branches,
            survivors,
            checkpoint,
            ctx,
            "domain_hash_random_epp321",
        )
    )
    return survivors


def run_epp321_common(ctx, roots, prune_mode: str, reallocate: bool):
    branches, initial_total = [], 0
    for r, seed in enumerate(roots):
        b, initial = base.initialize_branch(ctx, seed, f"root{r}")
        branches.append(b)
        initial_total += initial

    proposals, events = 0, []
    pruner = base.prune_measurement if prune_mode == "measurement" else prune_random_321

    # 4 roots x 5 proposals x 72 transitions = 1440.
    for i in range(72):
        for b in branches:
            proposals += base.step_greedy(ctx, b, i, 5)

    # Conservative first pruning: 4 -> 3.
    branches = pruner(ctx, branches, 3, 72, events)

    # Matched main/random: 3 x 8 x 76 = 1824.
    # No-reallocation: 3 x 5 x 76 = 1140.
    for i in range(72, 148):
        for b in branches:
            proposals += base.step_greedy(ctx, b, i, 8 if reallocate else 5)

    # 3 -> 2.
    branches = pruner(ctx, branches, 2, 148, events)

    # Matched main/random: 2 x 9 x 152 = 2736.
    # No-reallocation: 2 x 5 x 152 = 1520.
    for i in range(148, 300):
        for b in branches:
            proposals += base.step_greedy(ctx, b, i, 9 if reallocate else 5)

    # 2 -> 1 immediately before projection begins.
    branches = pruner(ctx, branches, 1, 300, events)

    descendants = base.hard_fork(
        ctx,
        branches[0],
        4,
        "B24_METHOD_EPP321_HARD_FORK_V1",
    )
    for i in range(300, 999):
        for b in descendants:
            proposals += base.step_greedy(ctx, b, i, 1)

    return descendants, events, proposals, initial_total + proposals


def run_epp321(ctx, roots):
    return run_epp321_common(ctx, roots, "measurement", True)


def run_epp321_random(ctx, roots):
    return run_epp321_common(ctx, roots, "random", True)


def run_epp321_no_reallocation(ctx, roots):
    return run_epp321_common(ctx, roots, "measurement", False)


def install_refinement() -> None:
    # The parent main/finalization path remains authoritative.  Only the arm
    # registry, accounting contracts, and spec identity are extended here.
    base.ARM_ORDER = ARM_ORDER
    base.EXPECTED_COUNTS.update(EXPECTED_COUNTS)
    base.RUNNERS.update(
        {
            "NP_EPP_321": run_epp321,
            "NP_EPP_321_RANDOM_PRUNE": run_epp321_random,
            "NP_EPP_321_NO_REALLOCATION": run_epp321_no_reallocation,
        }
    )


def main() -> int:
    install_refinement()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
