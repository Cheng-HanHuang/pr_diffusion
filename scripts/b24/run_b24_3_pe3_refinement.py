#!/usr/bin/env python3
"""Final B24.3 protected-explorer (PE3) method definitions.

This module is intentionally small and installs exactly two planner-authorized
arms into the tested B24.3 NP runner:
  NP_PE3_SCORE
  NP_PE3_RANDOM

Both are exactly 8,800 total same-model UNet evaluations per image.  There is
one and only one lineage-role decision, immediately before transition 72.
No pruning occurs at 148 or 300.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PARENT_PATH = REPO / "scripts" / "b24" / "run_b24_3_np_branching.py"
PE3_SPEC_PATH = REPO / "configs" / "b24" / "b24_3_pe3_final_dev.json"

ARM_ORDER = ("NP_PE3_SCORE", "NP_PE3_RANDOM")
EXPECTED_COUNTS = {
    "NP_PE3_SCORE": {"proposal_evals": 8796, "total_unet_evals": 8800, "terminals": 4},
    "NP_PE3_RANDOM": {"proposal_evals": 8796, "total_unet_evals": 8800, "terminals": 4},
}

base: Any | None = None


def load_parent():
    spec = importlib.util.spec_from_file_location("b24_3_pe3_parent", PARENT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import B24.3 parent from {PARENT_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _require_base():
    if base is None:
        raise RuntimeError("PE3 is not installed")
    return base


def role_assignment(ctx, branches, mode: str, events: list[dict]):
    """Return explorer, protected roots after the only PE3 role decision."""
    b = _require_base()
    if len(branches) != 4:
        raise RuntimeError(f"PE3 checkpoint 72 expected four roots, got {len(branches)}")
    if mode == "score":
        ranked = sorted(branches, key=lambda x: (x.trailing_score(), x.lineage))
        rule = "pe3_trailing32_lf_mse_role_assignment"
        uses_measurement = True
    elif mode == "random":
        ranked = sorted(
            branches,
            key=lambda x: (
                b.domain_hash("B24_METHOD_PE3_RANDOM_ROLE_V1", ctx.image_id, x.lineage),
                x.lineage,
            ),
        )
        rule = "pe3_domain_hash_random_role_assignment"
        uses_measurement = False
    else:
        raise ValueError(mode)

    explorer = ranked[0]
    protected = ranked[1:3]
    survivors = [explorer, *protected]
    dropped = ranked[3:]
    event = b.checkpoint_event(branches, survivors, 72, ctx, rule)
    event.update(
        {
            "role_assignment_uses_measurement": uses_measurement,
            "explorer_lineage": explorer.lineage,
            "protected_lineages": [x.lineage for x in protected],
            "dropped_lineages": [x.lineage for x in dropped],
            "later_pruning": False,
        }
    )
    events.append(event)
    return explorer, protected


def run_pe3_common(ctx, roots, mode: str):
    b = _require_base()
    branches = []
    initial_total = 0
    for r, seed in enumerate(roots):
        branch, initial = b.initialize_branch(ctx, seed, f"root{r}")
        branches.append(branch)
        initial_total += initial

    proposals = 0
    events: list[dict] = []

    # 72 * 4 * 5 = 1,440 proposal evaluations.
    for i in range(72):
        for branch in branches:
            proposals += b.step_greedy(ctx, branch, i, 5)

    explorer, protected = role_assignment(ctx, branches, mode, events)

    # 228 * (10 + 5 + 5) = 4,560 proposal evaluations.
    for i in range(72, 300):
        proposals += b.step_greedy(ctx, explorer, i, 10)
        for branch in protected:
            proposals += b.step_greedy(ctx, branch, i, 5)

    # Preserve native RNG continuation for all three retained lineages, then
    # add one independently-seeded hard-phase explorer fork.  No lineage is
    # pruned at projection start.
    extra = b.hard_fork(ctx, explorer, 1, "B24_METHOD_PE3_EXTRA_HARD_FORK_V1")[0]
    terminals = [explorer, *protected, extra]
    if len({x.lineage for x in terminals}) != 4:
        raise RuntimeError("PE3 terminal lineage collision")

    # 699 * 4 = 2,796 proposal evaluations.
    for i in range(300, 999):
        for branch in terminals:
            proposals += b.step_greedy(ctx, branch, i, 1)

    total = initial_total + proposals
    if proposals != 8796 or total != 8800:
        raise RuntimeError(f"PE3 accounting drift proposals={proposals} total={total}")
    return terminals, events, proposals, total


def run_pe3_score(ctx, roots):
    return run_pe3_common(ctx, roots, "score")


def run_pe3_random(ctx, roots):
    return run_pe3_common(ctx, roots, "random")


def install_pe3(target_base=None):
    """Install PE3 into a supplied tested B24.3 base runner."""
    global base
    base = target_base if target_base is not None else load_parent()
    base.SPEC_PATH = PE3_SPEC_PATH
    base.ARM_ORDER = ARM_ORDER
    base.EXPECTED_COUNTS.update(EXPECTED_COUNTS)
    base.RUNNERS.update(
        {
            "NP_PE3_SCORE": run_pe3_score,
            "NP_PE3_RANDOM": run_pe3_random,
        }
    )
    return base


def main() -> int:
    b = install_pe3()
    return b.main()


if __name__ == "__main__":
    raise SystemExit(main())
