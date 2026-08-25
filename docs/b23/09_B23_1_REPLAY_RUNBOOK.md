# B23.1A/B authorized bounded replay runbook

## Authorization stop

The two required decisions are now recorded:

1. B23.0 sign-off accepted on final status head `3829d52e6489ec5c90d5bc0ce404253e71131fda`;
2. B23.1A/B-only GPU authorization received on 2026-08-24.

The authorization is bounded by `configs/b23/b23_1a_b_execution.yaml`. B23.2, large panels, B24,
and adaptive schedules remain closed.

## Signed selection

- replay image: `65082`;
- four-image smoke: `61492`, `62959`, `66821`, `68142`;
- selection: minimum SHA-256 rank after image-level exposure exclusion, with one selection in each
  of four FFHQ-validation ID strata; no image content or method outcome was inspected;
- canonical signed registry: `manifests/b23/b23_1_signed_registry.csv`.

The earlier header-only templates and dry render remain preserved as B23.0 historical artifacts.

The first authorized launch at pre-run head `6e3e08eba2621f903bd33cd8d818442a34158318`
stopped in CUDA-hidden `prerun_validate` before input generation because its signed rows carried the
pre-closeout byte identity of `PRE_B23_EXPOSURE.csv`. The accepted closeout manifest has the same
328 exposed image identities but SHA-256 `a513cb4e3b79b39700ff1d623cb4b2eaf496bc2d6d0fe58bd963709e6a56d288`.
Recomputation preserves all five selected images and proves zero intersection. The false start and
correction are preserved in `manifests/b23/b23_1_correction_ledger.json`; GPU work performed was
`NO` and the parent-trajectory count was zero.

The next launch at pre-run head `5f937c26ab67783ff810b51070bf34cb8a9b3a07` passed preflight
and generated all five locked inputs, then stopped before the first Fresh1 random draw because the
canonical 63-bit seed exceeded the legacy NumPy RandomState uint32 range. No parent trajectory or
denoiser call completed. The correction preserves the canonical seed and freezes
`native_entrypoint_seed = canonical_parent_seed modulo 2**32` for every parent. The next launch must
use `--reuse-inputs` to validate and reuse the five existing measurements instead of regenerating
them. This partial run is also preserved in the correction ledger.

The following launch completed Fresh1 native repeat 0, including its 400-step native trace, but the
post-run audit rejected 40,402 observed RNG calls against an incomplete 40,401 source formula. The
pinned DAPS `DiffusionData.get_data(size, sigma=0)` implementation still calls
`torch.randn_like(data)` before multiplying by zero. Fresh1 and LF-v1 therefore have 40,402
process-wide RNG calls: one zero-sigma dataset-setup call plus 40,401 stochastic trajectory calls
(one native start, 40,000 Langevin, 400 forward re-noising). The setup call advances the unchanged
native global stream and is part of exact replay accounting even though it does not perturb the
image tensor.

The correction must pass that exact partial directory through `--recover-fresh1-native0`. Recovery
validates the locked input, successful CUDA timing, terminal sample, raw trajectory, 400-step trace,
seed, and hashes with CUDA hidden. It references that trajectory in the new return and runs only the
remaining 31 parent trajectories; it must never rerun the completed trajectory.

The frozen recovery source is
`/egr/research-pac/huang248/outputs/pr_diffusion/b23/B23_1_run_20260825T025410Z/replay/fresh1/native_0`
at physical execution head `45c6b6107e2bf2b100eac6b771ea0d1004f19a20`. Its returned timing, compact
trace, input-manifest, terminal-sample hashes, and 943,734,389-byte raw trajectory size are frozen in
the execution config and append-only correction ledger. Recovery additionally recomputes all 1,200
recorded tensor hashes from the raw trajectory on CPU before accepting it.

The return capsule must transport `FINAL_STATUS.tsv` and the recovered run's timing, compact trace,
input manifest, and available frozen runtime config/log. The 943 MB raw trajectory remains on PAC,
but the packager re-hashes it and records its path, size, and digest in
`RECOVERY_ARTIFACT_INDEX.json`. Repackaging an already-completed run is evidence-only, uses CUDA
hidden, and must not launch any parent trajectory.

## Required pre-run freeze after authorization

- signed one-image and four-image registries, disjoint from `PRE_B23_EXPOSURE.csv`;
- immutable parent configs and exact worktree/source/model/environment hashes;
- derived measurement, parent, module, and adapter RNG seeds;
- expected raw operation counts and terminal candidates;
- native-repeat count, complete-or-justified-unavailable deterministic-flag audit, pre-wrapper
  tolerance-freeze record identity, per-metric numerical floors, and replay comparison fields;
- typed atomic-or-coupled microbenchmark plan and maximum authorized GPU budget;
- exact commands, output roots, refusal-to-overwrite behavior, and stop rules;
- pushed pre-run commit before a GPU is reached.

## B23.1A sequence

For Fresh1, LF-v1, NP-1, then SITCOM-1:

1. run unchanged native repeats on the one-image smoke row;
2. freeze bitwise eligibility or write and hash the tolerance envelope/floor record before any
   wrapper run;
3. run the trace wrapper;
4. reconcile intermediate tensors, final metrics, exact operation counts, and every named RNG draw;
5. test serialization/resume only at a valid native boundary;
6. microbenchmark every nonzero operation as an isolated atomic unit where sound, or as an explicit
   non-overlapping typed coupled block where isolation would change native semantics, then freeze weights;
7. record `PASS`, `FAIL`, or `BASELINE-ONLY` without changing the parent.

Only after all four native wrappers pass may the four-image heterogeneous smoke run.

## B23.1B donor sequence

1. replay the Fresh1 DAPS-native module sequence;
2. replay the LF-v1 DAPS-native module sequence;
3. audit NP proposal/ranking/projection as a coupled native contract;
4. audit SITCOM LGVD/correction/forward-noising boundaries;
5. test any proposed cross-parent adapter for validity, information loss, round trip where meaningful,
   cost, RNG, and module replay;
6. classify every operation with the final-plan donor classes.

Do not force NP or SITCOM through an adapter to make H0 pass.

## Authorized execution entrypoint

The pushed pre-run commit is executed with:

```text
bash scripts/b23/run_b23_1a_b.sh \
  --repo /egr/research-pac/huang248/pr_diffusion_b23 \
  --output-root /egr/research-pac/huang248/outputs/pr_diffusion/b23 \
  --expected-head <PUSHED_B23_1_PRERUN_SHA> \
  --reuse-inputs /egr/research-pac/huang248/outputs/pr_diffusion/b23/B23_1_run_20260825T015257Z/inputs \
  --gpus "0 1 2 3"
```

The entrypoint refuses a dirty worktree, wrong branch/head, changed exposure manifest, duplicate or
exposed smoke image, non-derived seed, missing frozen environment, unexpected trajectory count,
wrapper execution before the tolerance freeze, parent replay failure, or overwrite. It packages
summary JSON and terminal reconstructions while leaving large raw DAPS trajectories on PAC under
their indexed and hashed paths.

The old `render_b23_1_dry_runs.py` output remains a B23.0 record and is not the authorized launcher.

## B23.1 gates

- all four native wrappers replay;
- Fresh1 and LF-v1 DAPS-native module sequences replay;
- exact compute and RNG reconciliation;
- trustworthy atomic weights/time reference;
- at least one NP/SITCOM adapter-qualified donor for cross-family Track A;
- no hidden retry/branch/candidate;
- separate planner decision before B23.2.

Possible returns are the four decisions in the final plan. None is preselected here.
