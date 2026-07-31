# B23.1 native replay runbook — preparation only

## Authorization stop

This runbook is non-executable preparation. B23.1 GPU work requires both:

1. planner/user sign-off on the completed B23.0 post-run evidence commit; and
2. a second explicit user authorization for B23.1.

Neither condition is currently recorded. Do not assign GPUs, populate the smoke registry, or replace
placeholders in command templates.

## Required pre-run freeze after authorization

- signed one-image and four-image registries, disjoint from `PRE_B23_EXPOSURE.csv`;
- immutable parent configs and exact worktree/source/model/environment hashes;
- derived measurement, parent, module, and adapter RNG seeds;
- expected raw operation counts and terminal candidates;
- native-repeat count, deterministic-flag audit, and replay comparison fields;
- atomic microbenchmark plan and maximum authorized GPU budget;
- exact commands, output roots, refusal-to-overwrite behavior, and stop rules;
- pushed pre-run commit before a GPU is reached.

## B23.1A sequence

For Fresh1, LF-v1, NP-1, then SITCOM-1:

1. run unchanged native repeats on the one-image smoke row;
2. freeze bitwise eligibility or tolerance envelope;
3. run the trace wrapper;
4. reconcile intermediate tensors, final metrics, exact operation counts, and every named RNG draw;
5. test serialization/resume only at a valid native boundary;
6. microbenchmark nonzero atomic operations and freeze weights;
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

## Dry-run rendering

The B23.0 command is safe because it emits documentation only:

```text
CUDA_VISIBLE_DEVICES='' /egr/research-pac/huang248/conda-envs/daps/bin/python \
  scripts/b23/render_b23_1_dry_runs.py \
  --repo /egr/research-pac/huang248/pr_diffusion_b23 \
  --output /egr/research-pac/huang248/outputs/pr_diffusion/b23/B23_1_dry_run_<timestamp>.json
```

With the registry empty, the output has `authorized=false`, `commands=[]`, four non-executable
parent templates, and explicit blockers. Exact executable commands do not exist until a signed row,
measured cost expectations, and B23.1 authorization exist.

## B23.1 gates

- all four native wrappers replay;
- Fresh1 and LF-v1 DAPS-native module sequences replay;
- exact compute and RNG reconciliation;
- trustworthy atomic weights/time reference;
- at least one NP/SITCOM adapter-qualified donor for cross-family Track A;
- no hidden retry/branch/candidate;
- separate planner decision before B23.2.

Possible returns are the four decisions in the final plan. None is preselected here.
