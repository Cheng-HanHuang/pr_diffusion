# B23 executor start here

Date: 2026-07-30

Status: executor handoff for B23.0 and preparation of B23.1. This document does not authorize
execution.

Repository: `Cheng-HanHuang/pr_diffusion`

Planning branch: `codex/post-b22-reliability-plan`

Planning PR: `#36`

## 1. Authorization rule

By default, the first executor may perform **B23.0 only** after the user explicitly says that:

1. the final B23 plan is accepted;
2. B23.0 is authorized;
3. no GPU execution is authorized;
4. no B23.2 schedule work is authorized.

The executor may prepare B23.1 code, tests, and runbooks during B23.0, but must stop before any
B23.1 GPU launch.

B23.1 GPU replay requires a second explicit approval after the B23.0 return package is reviewed.

If any authorization is missing, read and report only.

## 2. Required reading

Read every file in this order:

1. `docs/planning/00_B23_REVIEW_START_HERE.md`
2. `docs/planning/2026-07-30_b23_final_research_plan.md`
3. `docs/planning/2026-07-30_b23_supersession_ledger.md`
4. `docs/b22/b22_3_scientific_closeout.md`
5. `docs/b22/b22_3_visual_failure_taxonomy.csv`
6. `docs/checkpoints/2026-07-27_b21_to_b22/01_PROJECT_CHECKPOINT.md`
7. `docs/b21/b21_registry.md`
8. `docs/b21/b21_11_fresh2_final_benchmark.md`
9. `docs/b21/b21_7_fresh2_vs_lf_decision.md`
10. `docs/b21/b21_10_detector_screen_decision.md`
11. `docs/current_experiment_plan.md`
12. `docs/branch_B_fixed_budget_population_selector.md`
13. exact parent source and runbooks identified during inventory.

The July 29 plan and July 30 amendment are historical. Do not merge their stage plans with the final
plan.

## 3. B23.0 scope

### 3.1 Repository and PAC inventory

1. inspect the open branch/PR stack;
2. identify the immutable B22 scientific base and current accepted B23 planning head;
3. do not merge, retarget, squash, force-push, or rewrite protected history;
4. propose or create one clean execution branch only after the user approves the base;
5. use a clean worktree under `/egr/research-pac/huang248`, never `/home`;
6. inventory repository, external DAPS, NP, SITCOM, and DiffFPR revisions;
7. inventory dirty diffs and patch hashes;
8. inventory environments, model, data, measurements, outputs, hardware, driver, CUDA, cuDNN, and
   PyTorch.

Use documented manifests and targeted paths. Do not recursively scan large output, data, or model
trees without explicit user approval.

### 3.2 Freeze the parent definitions

For each of Fresh1, LF-v1, NP-1, and SITCOM-1, identify:

- exact source entrypoint;
- exact commit and local diff;
- exact config and defaults;
- model and scheduler;
- operator and measurement preprocessing;
- solver and measurement seed derivation;
- native state and time/noise coordinate;
- native module boundaries, if any;
- expected outputs and retained B22 artifacts.

Do not treat `prdiffusion/algorithms/hybrid_np_sitcom.py` or historical handoff code as the native
SITCOM parent.

### 3.3 Exposure and split protocol

Create a machine-readable `PRE_B23_EXPOSURE.csv` that covers B19, B20, early NP/SITCOM Branch A/B,
B21, B22, and manually inspected examples.

Required columns:

```text
image_id
measurement_id
dataset_split
first_project_stage
roles_seen
ground_truth_inspected
artifacts
exclusion_reason
source_evidence
```

If exposure is uncertain, exclude the row.

Define but do not populate future B23 split manifests beyond what is needed for a non-experimental
smoke plan.

### 3.4 Typed state and compatibility specification

Write a typed API specification containing:

- `NativeState[parent]`;
- `Module[input_type -> output_type]`;
- `Adapter[parent_A -> parent_B]`;
- semantic noise/log-SNR coordinate;
- cumulative budget coordinate;
- validity, information-loss, round-trip, serialization, and resume checks;
- explicit failure states that classify a parent as `BASELINE-ONLY`.

Do not implement one generic `state.x` abstraction that hides parent semantics.

### 3.5 Compute ledger and FRE specification

Write a machine-readable schema for:

- model forwards/backwards/JVPs/VJPs;
- measurement forwards/adjoints/JVPs/VJPs;
- FFT/projection/correction calls;
- optimizer iterations;
- conversions and re-noising;
- named RNG draws;
- live/retained branches and terminal candidates;
- GPU-active and wall seconds;
- memory;
- overhead.

Define the atomic-operation microbenchmark procedure and how it will freeze `w_j` before hybrid
execution.

Implement formulas for:

```text
work_FRE
time_FRE
claim_FRE = max(work_FRE, time_FRE)
```

Do not invent the numerical weights before measuring them.

### 3.6 Replay and RNG policy

Write:

- native repeatability procedure;
- bitwise replay eligibility;
- tolerance-freeze procedure;
- deterministic-flag audit;
- named RNG stream and seed-derivation scheme;
- draw-count and serialization checks;
- wrapper-versus-native comparison format.

The native parent determines whether bitwise replay is realistic. Do not change the parent simply to
force a hash.

### 3.7 Historical dead-direction ledger

The B23 docs must explicitly include:

- LF-v1 loses to a fresh restart as the default second arm;
- Fresh3 scaling and shared-prefix branching were rejected;
- scalar/pairwise detector thresholds were rejected;
- direct NP-to-SITCOM handoff was noncompetitive;
- executable `3S+1NP` routing was worse than `4S`;
- residual is not a correctness certificate;
- existing hybrid/handoff code is prototype evidence only.

## 4. Required B23.0 artifacts

At minimum:

```text
docs/b23/00_START_HERE.md
docs/b23/01_PROTOCOL_AND_GATES.md
docs/b23/02_PARENT_SEMANTICS_AND_COMPATIBILITY.md
docs/b23/03_TYPED_STATE_MODULE_ADAPTER_API.md
docs/b23/04_COMPUTE_LEDGER_AND_FRE_SPEC.md
docs/b23/05_REPLAY_DETERMINISM_AND_RNG_POLICY.md
docs/b23/06_PRE_B23_EXPOSURE_MANIFEST.md
docs/b23/07_PAC_INVENTORY_AND_FREEZE.md
docs/b23/08_HISTORICAL_DEAD_DIRECTIONS.md
docs/b23/09_B23_1_REPLAY_RUNBOOK.md
docs/b23/10_B23_2_PREREGISTRATION_TEMPLATE.md
configs/b23/...
scripts/b23/...
tests/b23/...
```

Machine-readable files should include:

```text
manifests/b23/PRE_B23_EXPOSURE.csv
schemas/b23/compute_ledger.schema.json
schemas/b23/replay_report.schema.json
configs/b23/fresh1_frozen.yaml
configs/b23/lf_v1_frozen.yaml
configs/b23/np1_frozen.yaml
configs/b23/sitcom1_frozen.yaml
configs/b23/replay_policy.yaml
```

Exact paths may change only if the same information and gates remain easy to find.

## 5. B23.0 validation

Run only safe, no-GPU checks:

- schema validation;
- config parsing;
- manifest uniqueness and overlap tests;
- source/hash inventory checks;
- unit tests for ledger arithmetic;
- unit tests for seed derivation and state serialization using synthetic tensors;
- dry-run command rendering;
- no experiment launcher.

## 6. B23.0 return package

Return:

1. execution branch and exact head SHA;
2. PR or commit list;
3. branch-stack recommendation without performing history changes;
4. PAC/source/environment/hardware inventory;
5. unresolved artifact or source identity gaps;
6. exposure-manifest coverage and unresolved IDs;
7. parent-semantics table;
8. typed API;
9. compute ledger and FRE procedure;
10. replay/RNG policy;
11. proposed one-image and four-image B23.1 smoke manifests;
12. exact B23.1 commands as dry runs;
13. explicit recommendation:
   - authorize B23.1;
   - revise B23.0;
   - stop because parent identity or protocol is not recoverable.

Stop and wait for planner/user sign-off.

## 7. B23.1 scope after separate authorization

### 7.1 Native parent replay

In order:

1. Fresh1;
2. LF-v1;
3. NP-1;
4. SITCOM-1.

For each:

- native repeatability calibration;
- unchanged native reference run;
- instrumented-wrapper run;
- intermediate trace comparison;
- RNG draw reconciliation;
- exact raw operation counts;
- GPU-active, wall-time, and memory measurement;
- bitwise or tolerance-qualified report;
- serialization/resume check where meaningful.

### 7.2 Four-image heterogeneous smoke

Use only four new preregistered images. The smoke may test engineering heterogeneity but may not be
used to choose:

- module boundaries;
- schedule windows;
- thresholds;
- parent configs;
- B23.2 candidates.

### 7.3 Donor extraction

Mandatory:

- DAPS-native Fresh1 module sequence replay;
- DAPS-native LF-v1 module sequence replay.

Audited:

- NP proposal/ranking donor;
- SITCOM triple-consistency donor.

Classify every donor as:

```text
NATIVE-REPLAYED
DAPS-NATIVE-DONOR
ADAPTER-QUALIFIED-DONOR
BASELINE-ONLY
REJECTED-PROTOTYPE
```

Do not force NP or SITCOM through a lossy adapter merely to pass H0.

## 8. B23.1 sign-off

Required for all:

- all four native wrappers replay;
- Fresh1 and LF-v1 DAPS-native replay;
- exact operation-count reconciliation;
- trustworthy RNG and compute ledgers;
- no hidden conversion or model call;
- parent-specific state validity.

Required for cross-family Track A:

- at least one NP or SITCOM `ADAPTER-QUALIFIED-DONOR`.

Return one:

- `AUTHORIZE B23.2 PREREGISTRATION`;
- `CONTINUE DAPS-NATIVE ONLY UNDER NARROWED CLAIM`;
- `REVISE B23.1`;
- `STOP TRACK A AND PIVOT TO TRACK B`.

No B23.2 command, config, manifest, or GPU screen is authorized yet.

## 9. Prohibited actions

The executor must not:

- launch GPU work during B23.0;
- launch B23.1 without the second approval;
- implement or run B23.2 schedules;
- tune on any pre-B23 image;
- treat the B22 panel as schedule-selection data;
- change B1/B2 caps;
- append a donor to a full parent without cost subtraction;
- create hidden branches or candidates;
- reduce NP to a generic gradient step;
- label SITCOM as late polish without semantic evidence;
- revive NP-to-SITCOM handoff;
- add DiffStateGrad or another fifth family;
- implement a learned controller;
- merge or rewrite protected history;
- place work under `/home`.

## 10. Initial executor instruction

> Work as the B23.0 execution lead for `Cheng-HanHuang/pr_diffusion`. Start at
> `docs/planning/01_B23_EXECUTOR_START_HERE.md` and follow its reading order. Perform repository,
> PAC, artifact, parent-semantics, exposure, typed-state, compute-ledger, replay-policy, and dry-run
> preparation only. Do not launch a GPU job and do not implement B23.2 schedules. Return the full
> B23.0 package and stop for sign-off before B23.1.
