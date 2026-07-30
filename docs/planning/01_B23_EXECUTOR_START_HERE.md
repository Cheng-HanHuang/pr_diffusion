# B23 executor start here

Date: 2026-07-30

Status: executor handoff for the first post-review implementation checkpoint. This document does not itself authorize execution; begin only after the user explicitly approves the reviewed B23 plan.

Repository: `Cheng-HanHuang/pr_diffusion`

Planning branch: `codex/post-b22-reliability-plan`

Planning PR: `#36`

## 1. Preconditions before starting

The executor must receive an explicit statement from the user that:

1. the B23 plan and amendment have been reviewed and accepted;
2. required review changes, if any, have been incorporated;
3. B23.0 and B23.1 are authorized;
4. no B23.2 schedule experiment or large GPU panel is authorized yet.

If any condition is absent, stop after reading and report the missing authorization.

## 2. Required reading and precedence

Read every file before proposing implementation, in this order:

1. `docs/planning/00_B23_REVIEW_START_HERE.md`
2. `docs/planning/2026-07-30_b23_modular_fixed_budget_amendment.md`
3. `docs/planning/2026-07-29_post_b22_reliability_research_plan.md`
4. `docs/b22/b22_3_scientific_closeout.md`
5. `docs/checkpoints/2026-07-27_b21_to_b22/01_PROJECT_CHECKPOINT.md`
6. `docs/b21/b21_registry.md`
7. parent-method source/report files required to replay Fresh1, LF, NP-1, and SITCOM-1.

Precedence:

- the 2026-07-30 amendment overrides the 2026-07-29 plan where they differ;
- the review entrypoint specifies reading and gate policy but does not alter scientific hypotheses;
- historical reports provide evidence and exact parent behavior but cannot reopen rejected directions without a new approved hypothesis;
- no executor may tune on the B21/B22 final panel.

## 3. First executor scope

The first executor is responsible for **B23.0 and B23.1 only**.

### B23.0 — integration, inventory, and protocol freeze

Deliverables:

1. inspect the open PR/branch stack and report the exact dependency structure;
2. do not merge, retarget, squash, force-push, or rewrite protected history without an explicit human decision;
3. identify the accepted B22 scientific head and accepted B23 planning head;
4. create or propose one clean B23 execution branch from the approved integration point;
5. create a clean PAC worktree under `/egr/research-pac/huang248`;
6. create an exclusion manifest containing every B21/B22 image ID;
7. inventory exact source revisions, external solver revisions, environments, model, dataset, measurements, and output roots;
8. write the common-state/module API specification;
9. write the machine-readable compute-ledger schema;
10. write smoke, validation, and artifact-retention runbooks.

No GPU launch is required for B23.0.

### B23.1 — parent module extraction and exact replay

Implement parent algorithms through the common interface in this order:

1. Fresh1/DAPS;
2. LF-guided parent;
3. NP-1 with its actual delayed/scoring/resampling semantics preserved;
4. SITCOM-1 only after state conversion and RNG behavior are explicitly audited.

Required replay evidence for each parent:

- one-image exact or tolerance-qualified replay;
- intermediate-state/checkpoint comparison;
- random-generator and random-consumption audit;
- output tensor hash where deterministic replay is expected;
- final metric agreement within a frozen tolerance;
- identical or fully reconciled operation counts;
- measured GPU time, wall time, and peak memory;
- serializable/resumable module boundaries;
- no hidden extra diffusion or measurement evaluations.

After one-image replay, run a four-image heterogeneous smoke only. Do not run a schedule screen or full panel.

## 4. Required technical artifacts

At minimum, B23.0/B23.1 should add:

```text
docs/b23/00_START_HERE.md
docs/b23/01_PROTOCOL_AND_GATES.md
docs/b23/02_COMMON_STATE_AND_MODULE_API.md
docs/b23/03_COMPUTE_LEDGER_SPEC.md
docs/b23/04_PARENT_REPLAY_RUNBOOK.md
docs/b23/05_PAC_INVENTORY_AND_FREEZE.md
configs/b23/...
scripts/b23/...
tests/b23/...
```

The exact file split may change, but the information and gates may not be omitted.

## 5. Common-state design requirements

The common state must expose, at minimum:

- current sample/latent and its representation contract;
- diffusion time/noise/schedule position;
- measurement and operator handles;
- RNG/generator state;
- model and solver-state references;
- accumulated diagnostics;
- explicit conversion metadata;
- optional candidate/branch metadata;
- operation counters for the compute ledger.

Do not pretend heterogeneous solvers share identical state semantics. Every conversion must be explicit, reversible where required, and tested.

## 6. Compute-ledger requirements

Record at least:

- diffusion-network evaluations;
- measurement forward evaluations;
- measurement adjoint evaluations;
- gradient/projection/correction calls;
- inner optimizer iterations;
- stochastic branches;
- retained candidates;
- GPU-seconds;
- wall time;
- peak allocated/reserved memory;
- preprocessing or conversion overhead.

The ledger must support normalization to one frozen Fresh1-equivalent run while preserving raw counts and measured time.

## 7. Prohibited actions

The first executor must not:

- run B23.2 schedule search;
- invent hybrid schedules beyond interface smokes;
- launch a large GPU panel;
- tune on B21/B22 final images;
- change the B1/B2 compute caps;
- add NP/LF/SITCOM work on top of a full DAPS trajectory without subtracting cost elsewhere;
- implement a learned controller;
- train an image classifier;
- use ground-truth-dependent runtime features;
- run hidden best-of-k candidate banks;
- treat NP as a generic gradient step if that changes its semantics;
- merge or rewrite the historical PR stack without explicit human authorization.

## 8. B23.1 sign-off gate

B23.1 passes only if all four parents replay correctly through the common interface and the compute ledger is trustworthy.

```text
Fresh1 replay: REQUIRED
LF replay: REQUIRED
NP-1 replay: REQUIRED
SITCOM-1 replay: REQUIRED
one-image parent checks: REQUIRED
four-image heterogeneous smoke: REQUIRED
intermediate/RNG audit: REQUIRED
compute ledger: REQUIRED
module serialization/resume: REQUIRED
B23.2 authorization: SEPARATE DECISION
```

If one parent cannot be represented faithfully, do not conceal the mismatch. Report whether:

- the module boundary is wrong;
- state conversion is lossy;
- RNG semantics differ;
- the parent method must remain an external whole-solver baseline;
- the modular-composition hypothesis needs revision.

## 9. Executor return package

Return:

1. branch and exact head SHA;
2. PR or commit list;
3. source/environment/PAC inventory;
4. common-state API and compute-ledger specification;
5. replay table for all four parents;
6. one-image and four-image logs;
7. tensor/metric/checkpoint comparisons;
8. operation-count and timing tables;
9. unresolved semantic or implementation risks;
10. explicit recommendation: authorize B23.2A, revise B23.1, or stop modular composition.

## 10. Initial instruction to the executor

Use the following instruction when handing off after plan approval:

> Work as the execution lead for B23.0 and B23.1 in `Cheng-HanHuang/pr_diffusion`. Start from `docs/planning/01_B23_EXECUTOR_START_HERE.md`, read every required file in its stated precedence order, and do not propose or run B23.2 schedules. First consolidate the repository/inventory/protocol, then implement a common modular interface that faithfully replays Fresh1, LF, NP-1, and SITCOM-1 with an audited compute ledger. Commit every implementation and runbook change before PAC execution, use smoke gates, and return compact evidence for sign-off.
