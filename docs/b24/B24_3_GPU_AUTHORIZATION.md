# B24.3 GPU authorization

Status: **AUTHORIZED**

This records the user/planner authorization given after the zero-GPU B24.3 method-role freeze passed.

## Authorized now

1. Implement the frozen B24.3 NP branching runner and its fail-closed launch/status tooling.
2. Materialize the prospectively frozen new development measurement for exactly one already-frozen Pilot16 image.
3. Run a **one-image full-method engineering/calibration smoke** on that image for the frozen Pilot16 arms:
   - `NP1`
   - `NP4_INDEPENDENT`
   - `NP_EPP`
   - `NP_EPP_RANDOM_PRUNE`
   - `NP_EPP_NO_REALLOCATION`
   - `NP_DPS`
4. The one-image smoke must verify native continuation, branch/RNG lineage, exact proposal/model-evaluation accounting, finite terminal outputs, clean-free selection, wall/GPU timing, and measured memory under the global 52,452-MiB B24 process/group ceiling.
5. If and only if all one-image integrity/resource gates pass, the already-frozen **Pilot16** (4 A + 4 B + 4 C + 4 D) is authorized on its one new locked development measurement per image. The one-image calibration result is part of Pilot16 and should be reused rather than silently discarded or regenerated.

## Not authorized

- No execution on the remaining 64 non-pilot development images.
- No confirmation measurement materialization or confirmation method execution.
- No C1-only extra method exposure outside the primary role policy.
- No expansion of the frozen arm set or checkpoint/score search.
- No post-hoc change of A/B/C/D screening strata.
- No merge/rebase/squash/retarget/force-push/history rewrite.
- Do not modify PR #37.

## Frozen scientific specification

The algorithms, budgets, role policy, and seed domains remain those in:

`configs/b24/b24_3_method_dev_spec.json`

The role-freeze inputs are anchored by:

- method-role panel CSV SHA-256: `7c05dd67cc39263db74e58b7255297d014e7d674da0c0bebdee317bcbff809bf`
- Pilot16 CSV SHA-256: `124d3759e4fd540d2e870618dde59ff73d02cbb798d773a785296da5b140e98a`
- source ABC300 CSV SHA-256: `4599c2a8c1f4a5922640e0c26d2c1efce7f1996d75dcabbff2e9a1c4b427cbce`
- source C1 CSV SHA-256: `9c04994dbd91f6a5bb04280736e4dea346c337508a89f30ae8553a42867576b6`

The first one-image smoke is an engineering/calibration gate, not a method-selection result. Ground truth may be used after runtime decisions for offline terminal metrics/oracle diagnostics; it must never drive proposal retention, pruning, branch allocation, or executable terminal selection.
