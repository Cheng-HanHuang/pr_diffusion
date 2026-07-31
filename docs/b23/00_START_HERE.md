# B23 execution start here

## Status and boundary

B23.0 is the only authorized stage. Its GPU budget is zero. This branch freezes repository and PAC
identities, native parent semantics, leakage controls, typed interfaces, compute accounting,
replay/RNG policy, evidence packaging, and dry-run-only B23.1 preparation.

B23.1 replay, B23.2 schedules, large panels, Track B, and Track C remain unauthorized. Nothing in
`docs/b23`, `configs/b23`, or `scripts/b23` grants that authorization.

The execution branch is `codex/b23-execution`, created from operational handoff
`d1119e37fa688ac07f48ffc87ce19b13dbfb1c27`. That handoff descends from accepted scientific plan
`ed4f46e8f116648eda76d387388d762d7cb8f3d7`; the frozen B22 scientific base is
`ba78c06e0c5eac0c915263e4faed0b262d5e917a`.

## Reading order

1. `01_PROTOCOL_AND_GATES.md`
2. `02_PARENT_SEMANTICS_AND_COMPATIBILITY.md`
3. `03_TYPED_STATE_MODULE_ADAPTER_API.md`
4. `04_COMPUTE_LEDGER_AND_FRE_SPEC.md`
5. `05_REPLAY_DETERMINISM_AND_RNG_POLICY.md`
6. `06_PRE_B23_EXPOSURE_MANIFEST.md`
7. `07_PAC_INVENTORY_AND_FREEZE.md`
8. `08_HISTORICAL_DEAD_DIRECTIONS.md`
9. `09_B23_1_REPLAY_RUNBOOK.md`
10. `10_B23_2_PREREGISTRATION_TEMPLATE.md`

Machine-readable contracts are under `configs/b23`, `manifests/b23`, and `schemas/b23`. Protocol
primitives are in `prdiffusion/b23_protocol.py`; tests are under `tests/b23`.

## B23.0 checkpoint sequence

1. Push the reviewable pre-run freeze commit to `codex/b23-execution`.
2. Create or update only the clean PAC worktree at
   `/egr/research-pac/huang248/pr_diffusion_b23`.
3. Run `scripts/b23/run_b23_0_zero_gpu.sh` with `CUDA_VISIBLE_DEVICES` cleared and the explicit
   `/egr/research-pac/huang248/conda-envs/daps/bin/python`.
4. Preserve full logs and the extracted capsule under
   `/egr/research-pac/huang248/outputs/pr_diffusion/b23`.
5. Commit the transparent evidence and the explicitly approved, size-capped `.tar.gz` exception;
   push without force to the execution branch.
6. Return the exact protocol `PLANNER_RETURN` and stop for planner/user sign-off.

The archive is transport, not the scientific database. The same critical files remain extracted in
GitHub, while large model/data/historical outputs stay on PAC and appear by absolute path and hash
in `ARTIFACT_MANIFEST.tsv`.

## Current recommendation

Before the PAC zero-GPU collector and post-run evidence commit: `COMPLETE B23.0 PAC FREEZE`.

After those checks pass: `REQUEST PLANNER REVIEW OF B23.0`; this is not self-authorization for
B23.1.
