# B23 execution start here

## Status and boundary

B23.0 is signed off. The bounded B23.1A/B scientific execution is accepted at
`3ffb237818e1bfa4921b3f4f8bc9a3bd24b7e406`, with repaired packaging at
`fad055d40d5bd0eaf4c9471359177c321958d2d7`. Its zero-GPU evidence-publication closeout is complete
at evidence commit `d158d05a3b85c0eb23524a5e0ab3c81eaf286145`; it launched no parent, generated no
measurement, reconstructed no image, and performed no correction GPU work.

Cross-family H0 failed: zero NP/SITCOM adapters qualified. B23.2, B24 execution, large panels, and
adaptive schedules remain unauthorized. Nothing in `docs/b23`, `configs/b23`, or `scripts/b23`
promotes a donor, defines an adaptive schedule, or grants later-stage authorization.

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

## Completed evidence checkpoint sequence

1. Push the reviewable evidence-closeout pre-run commit to `codex/b23-execution`.
2. Create or update only the clean PAC worktree at
   `/egr/research-pac/huang248/pr_diffusion_b23`.
3. Run `scripts/b23/run_b23_1_evidence_closeout.sh` with `CUDA_VISIBLE_DEVICES` cleared and the explicit
   `/egr/research-pac/huang248/conda-envs/daps/bin/python`.
4. Preserve full logs and the extracted capsule under
   `/egr/research-pac/huang248/outputs/pr_diffusion/b23`.
5. Commit the extracted compact summaries, checkpoint report, and PAC artifact manifest; keep the
   compact `.tar.gz` transport on PAC and push the evidence commit without force.
6. Return the exact protocol `PLANNER_RETURN` and stop for planner/user sign-off.

The archive is transport, not the scientific database. The same critical files remain extracted in
GitHub, while large model/data/historical outputs stay on PAC and appear by absolute path and hash
in `ARTIFACT_MANIFEST.tsv`.

## Current recommendation

The planner accepted all 32 B23.1A/B trajectories, four BITWISE replay reports, calibrated compute
ledgers, the recovered Fresh1 trajectory, the four-image smoke, and donor classifications. The
evidence-only correction used pre-run commit `4d77e7d478c5562626e13c7f0fd71e09a953fba9` and evidence
commit `d158d05a3b85c0eb23524a5e0ab3c81eaf286145`. The PAC closeout recorded 87/87 tests and all three
zero-GPU steps `PASS/0`. Scientific execution was not rerun.

Current extracted evidence is at `docs/b23/evidence/B23_0_return_20260824T190122Z/`; its transport
archive is `docs/b23/evidence/capsules/B23_0_return_20260824T190122Z.tar.gz` with SHA-256
`fdbe5fc3cc3c17fb2086c788240a8b67be099746b4ee20f347b1fe5ec88f368c`. Prior evidence, including
the invalid return preserved in `B23_0_CORRECTION_LEDGER.md`, remains immutable and cannot authorize
later stages.

The accepted full transport is `B23_1_return_20260825T184922Z.tar.gz`, SHA-256
`5731e6b0c20be940ae8a1e8b1326b3668111f2291550280bfeb0013408257469`. The closeout contract is
`manifests/b23/b23_1_evidence_closeout_contract.json`. The extracted compact evidence is at
`docs/b23/evidence/B23_1_closeout_return_20260825T194434Z/`; its PAC transport archive is
`/egr/research-pac/huang248/outputs/pr_diffusion/b23/B23_1_closeout_return_20260825T194434Z.tar.gz`
with SHA-256 `1d434faab58c8e1590196a1b585113bc998e190b8b9d3775f6427a0728c8b9a2`.

The scientific conclusion is intentionally narrower than the original cross-family hypothesis:
`CONTINUE DAPS-NATIVE ONLY UNDER NARROWED CLAIM`. B23.2 remains **CLOSED** pending a separate planner
decision; the present closeout cannot authorize it.

Current recommendation: `PASS_RECOMMEND_B23_1A_B_SIGNOFF_ONLY`. Return to the planner and stop;
do not begin B23.2, B24 execution, a large panel, or an adaptive schedule without a new explicit
authorization.
