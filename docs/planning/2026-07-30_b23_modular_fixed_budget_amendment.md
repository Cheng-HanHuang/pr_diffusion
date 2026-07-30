> [!IMPORTANT]
> **Superseded on 2026-07-30.** The authoritative plan is
> `docs/planning/2026-07-30_b23_final_research_plan.md`; section-level status is in
> `docs/planning/2026-07-30_b23_supersession_ledger.md`. This file is retained as historical
> planning rationale and does not authorize execution.

# B23 amendment: fixed-budget modular solver synthesis

Date: 2026-07-30

Status: authoritative amendment to `2026-07-29_post_b22_reliability_research_plan.md`.

This file supersedes the earlier plan wherever that plan treats whole-solver fallback as the primary B23 contribution. The earlier evidence summary, data-separation rules, leakage rules, and repository safety requirements remain in force.

## 0. Revised executive decision

B22 shows that several mathematically natural phase-retrieval solvers reach different basins. Because magnitude measurements do not determine a unique phase, no scalar measurement residual can certify that one reconstructed phase is the desired solution. The next study should therefore not be limited to choosing among finished solvers.

The primary B23 direction is now:

> **Budget-Normalized Modular Solver Synthesis for Reliable Diffusion Phase Retrieval**

The goal is to expose the useful operations inside DAPS/Fresh, LF guidance, NP, and SITCOM through a common trajectory interface, then construct one solver whose trajectory can use different operations at different diffusion stages under a fixed compute budget.

Whole-solver fallback remains:

- an equal-budget comparison;
- a safety architecture if modular composition fails;
- a possible later deployment layer.

It is no longer the central scientific contribution.

The main claim must come from better allocation of a fixed amount of computation, not from increasing the number of independent candidates.

## 1. Scientific hypothesis

Each existing method expresses a different defensible inductive bias:

- ordinary DAPS/Fresh trajectories repeatedly combine the diffusion prior with strong measurement correction;
- LF changes the early measurement geometry toward coarse structure;
- NP delays direct measurement enforcement and uses measurement agreement to rank or preserve promising states;
- SITCOM supplies a different correction/consistency dynamic;
- independent restarts supply basin diversity without changing the update law.

The new hypothesis is:

> Reliability may improve when these operations are composed at the stages where their assumptions are most useful, rather than running each complete solver independently and selecting only at the end.

Examples of the intended structure include:

```text
soft or low-frequency early guidance
    -> prior-dominated diffusion evolution
    -> progressively stronger measurement correction
    -> late exact-consistency polishing
```

or a clean-free controller that chooses one of several update modules at a small number of frozen checkpoints.

This is **operator composition**, not naive image averaging. Convexly averaging two phase-retrieval reconstructions can destroy both solutions and has no general justification under phase ambiguity. The first study should choose or sequence update modules, not blend final images pixelwise.

## 2. Revised project hierarchy

### Track A — primary: within-trajectory modular composition

Construct a common state and module interface and study fixed-budget hybrid trajectories.

### Track B — secondary: adaptive whole-solver escalation

Retain Fresh2-to-NP escalation as a benchmark and possible safety layer. It becomes primary only if Track A fails its gates.

### Track C — later: shared candidate-generation failures

Study cases like `65003`, where every present method and oracle fails. This remains separate from Track A/B until the fixed-budget composition question is resolved.

## 3. Hard compute contract

### 3.1 Budget units

Define `1.0 FRE` as the measured cost of one frozen Fresh1 trajectory under the current environment.

Every experiment must record:

- diffusion-network evaluations;
- measurement-operator forward/adjoint evaluations;
- gradient/projection/correction evaluations;
- GPU-seconds;
- wall time;
- peak memory;
- number of stochastic branches and retained candidates.

FRE is the primary readable budget unit, but claims must also report the underlying counts and measured GPU time.

### 3.2 Primary deployment budgets

The main study has two budget tiers:

```text
B1: at most 1.10 FRE per image
B2: at most 2.10 FRE per image
```

Interpretation:

- B1 asks whether a single modular trajectory can improve Fresh1 without materially increasing cost.
- B2 asks whether two-FRE computation can beat Fresh2 through better step allocation rather than more blind restarts.

The small 10% tolerance covers implementation overhead and measurement-module cost mismatch. Any larger excess requires an explicit matched-cost baseline.

### 3.3 Prohibited budget arguments

The primary B23 result may not rely on:

- increasing `k` until best-of-k succeeds;
- always-on NP-8 or SITCOM-4S;
- a hidden large candidate bank followed by cheap simulated selection;
- comparing a hybrid with more diffusion evaluations against a cheaper baseline;
- appending a module without removing or shortening another part of the trajectory;
- reporting only oracle quality while omitting execution cost.

NP-8, SITCOM-4S, and cross-method best-of-k remain diagnostic upper bounds only.

### 3.4 Cost-neutral composition rule

Adding an operation must be paid for by at least one of:

- replacing an existing operation;
- reducing the number of operations elsewhere;
- stopping another module early;
- sharing a state without duplicating diffusion-network evaluations;
- reducing the number of independent trajectories.

The experiment config must contain a machine-readable compute ledger before launch.

## 4. Modular solver abstraction

### 4.1 Common state

Every module should consume and return a common trajectory state such as:

```text
state = {
    current sample / latent,
    diffusion time and schedule position,
    measurement and operator handles,
    stochastic-generator state,
    accumulated diagnostics,
    optional candidate/branch metadata
}
```

The exact tensor representation may differ across external solvers. The interface must make every conversion explicit and testable.

### 4.2 Module classes

Initial module library:

1. **Prior evolution module**
   - one or more ordinary diffusion/denoising transitions.

2. **DAPS-style measurement correction module**
   - the frozen correction/projection behavior used by Fresh trajectories.

3. **LF consistency module**
   - low-frequency or coarse measurement guidance.

4. **NP soft/delayed interaction module**
   - delayed direct enforcement, score/ranking, resampling, or state-preservation operations extracted according to NP's actual semantics.
   - NP must not be falsely reduced to a gradient step if its useful operation is candidate scoring or resampling.

5. **SITCOM-style consistency module**
   - its distinct correction/likelihood operation, exposed only after exact replay is possible.

6. **Diversity module**
   - controlled stochastic refresh or independent restart, with explicit cost.

7. **Decision/acceptance module**
   - clean-free logic that selects the next module or accepts/rejects a proposed transition.

### 4.3 Exact-replay requirement

Before hybrid experiments, module sequences must exactly or numerically replay the parent algorithms:

- Fresh1/DAPS;
- LF-guided trajectory;
- NP-1;
- SITCOM-1.

Required checks include:

- output tensor hash when deterministic replay is expected;
- metric agreement within frozen tolerance;
- identical random-number consumption where required;
- matching step counts and compute ledger;
- matching intermediate checkpoints on a one-image smoke.

A module that cannot reproduce its parent solver is not authorized for composition.

## 5. Schedule grammar

The first composition study must remain low-dimensional and interpretable.

### 5.1 Frozen trajectory windows

Divide a trajectory into at most four preregistered windows:

```text
early -> early-middle -> late-middle -> late
```

Use schedule fractions or diffusion-noise ranges defined before outcome inspection.

### 5.2 Initial schedule restrictions

A candidate schedule may have:

- at most three module switches;
- no per-step learned controller;
- no continuous learned mixture weights;
- no reinforcement learning;
- no arbitrary architecture search;
- no more than 12 preregistered open-loop schedules in the first screen.

### 5.3 Initial schedule families

The initial library should include parent controls and a small set of mechanistic hybrids:

1. DAPS/Fresh1 control.
2. LF-early -> DAPS-late.
3. NP-soft-early -> DAPS-late.
4. NP-soft-early -> LF-middle -> DAPS-late.
5. prior-dominant early -> SITCOM-style middle/late correction.
6. LF-early -> SITCOM-style late correction.
7. DAPS early -> NP checkpoint decision -> DAPS late.
8. one independent stochastic refresh inside a hybrid trajectory, paid for by fewer later steps.

Exact schedules should be finalized only after module replay and a compute audit. The list is a grammar, not permission to improvise all combinations.

## 6. Revised hypotheses

### H1 — one-FRE hybrid trajectory

At least one frozen open-loop modular trajectory at `<=1.10 FRE` improves over Fresh1 on a held-out natural panel by satisfying all of:

- raw Good25 improvement of at least 5 percentage points;
- Bad20 reduction of at least 25% relative;
- mean raw PSNR no worse than Fresh1 by more than 0.20 dB;
- no more than 1 percentage point of Good25 harms;
- q05 no worse than Fresh1.

### H2 — two-FRE modular policy

A policy using `<=2.10 FRE` improves over Fresh2 by satisfying all of:

- raw Good25 improvement of at least 3 percentage points;
- threshold harms at most 1 percentage point;
- mean raw PSNR no worse than Fresh2 by more than 0.25 dB;
- q05 improvement of at least 2 dB;
- no increase in Bad10 count;
- matched paired uncertainty analysis supports the reliability improvement.

Authorized B2 structures include:

- one ordinary DAPS trajectory plus one complementary hybrid trajectory;
- two independently initialized copies of the same frozen hybrid;
- one trajectory with two internally budgeted branches if the total compute remains matched.

### H3 — module complementarity, not candidate-count scaling

The hybrid's gain must remain after matching:

- total diffusion evaluations;
- total GPU-seconds within tolerance;
- number of retained final candidates.

If the gain disappears under these controls, the result is candidate-count scaling rather than solver composition.

### H4 — adaptive checkpoint controller

Only after an open-loop hybrid passes H1 or shows clear module complementarity may a clean-free controller be developed.

The controller may choose modules at no more than four frozen checkpoints. It must outperform the best fixed schedule under the same budget and use only diagnostics available at that checkpoint.

### H5 — prospective reproducibility

After the schedule/controller and compute ledger are frozen, the policy must be evaluated without generating unselected future modules or unused candidate branches.

## 7. Development protocol

B21/B22 final-panel images remain excluded from schedule, threshold, controller, and stopping-rule development.

### B23.0 — integration and protocol freeze

Deliverables:

- resolve or explicitly preserve the PR stack;
- record the immutable B22 scientific head;
- create one B23 execution branch/worktree;
- image-exclusion manifest;
- module API specification;
- compute-ledger schema;
- no GPU experiment.

### B23.1 — module extraction and parent replay

Deliverables:

- common state interface;
- parent-solver module sequences;
- one-image replay tests;
- four-image heterogeneous smoke;
- intermediate-state and RNG audit;
- compute ledger for every parent solver.

Gate:

- Fresh1, LF, NP-1, and SITCOM-1 replay correctly;
- no hidden data conversion or extra model evaluation;
- all module boundaries are serializable and resumable.

### B23.2A — small equal-budget mechanism screen

Use a preregistered enriched development set of approximately:

```text
30 historical/new hard cases
30 matched ordinary controls
```

The B22 final panel may be used only for retrospective sanity checks after schedules are frozen, never for selection.

Run at most 12 B1 schedules with paired seeds and exact compute matching.

Pruning rule:

- retain at most three schedules;
- require improvement on both hard cases and ordinary controls;
- reject any schedule whose gain comes only from extra compute or candidate count;
- reject schedules with concentrated rescue on fewer than three distinct images.

### B23.2B — natural confirmation

Evaluate the retained schedules on at least 120 new natural images.

Freeze one B1 hybrid only if H1 is plausible and the gain is not driven by a single visual category.

If no schedule survives, stop open-loop composition and record the negative result before attempting a learned controller.

### B23.3 — equal-budget two-FRE construction

Compare at matched cost:

- Fresh2;
- Fresh1 + frozen hybrid candidate;
- two frozen hybrid restarts if scientifically justified;
- the best fixed within-trajectory B2 composition.

Use a new calibration panel to freeze the B2 policy. No B22 rows may be used for choice.

### B23.4 — limited adaptive checkpoint control

Authorized only if B23.2 or B23.3 establishes real fixed-schedule complementarity.

Controller restrictions:

- logistic regression, shallow tree, or monotone rule;
- at most four decision checkpoints;
- no reconstructed-image deep classifier initially;
- every input feature declared in the leakage manifest;
- no access to diagnostics from a module that has not been executed;
- same B1/B2 compute caps.

Primary comparison is against the best fixed schedule, not against a weak parent baseline.

### B23.5 — untouched full-candidate audit

Evaluate frozen B1 and B2 policies on at least 200 untouched images with complete audit candidates only where needed to measure selector/controller regret.

Primary gates are H1/H2, with reliability, PSNR distribution, catastrophic floor, and compute all reported.

### B23.6 — prospective policy-only execution

Run at least 150 additional untouched images while executing only the modules selected by the frozen schedule/controller.

Unused branches must not be generated.

This is the final deployment-style result.

## 8. Role of whole-solver fallback after amendment

The earlier Fresh2 -> conditional NP plan remains useful as:

1. a fixed baseline for B2;
2. a fallback if modular composition fails;
3. a later safety layer around a modular solver;
4. a way to quantify how much value is lost when modules cannot share one trajectory.

But it is no longer the first implementation target.

A conditional NP cascade may be claimed only if its **average** compute is matched to the modular B2 budget or is reported as a separate higher-cost frontier point. Always-on NP-8 is diagnostic, not a deployment target.

## 9. New dead directions and stop rules

In addition to the previously rejected paths, do not pursue:

- naive pixelwise or latent averaging of final solver outputs;
- exhaustive combinatorial schedule search;
- reinforcement-learning control before fixed schedules establish complementarity;
- a controller that simply learns which completed full solver won after all are run;
- hidden best-of-k banks;
- claims based on a growing candidate count;
- adding NP/LF/SITCOM steps on top of a full DAPS run without cost subtraction;
- choosing schedules on the B22 final panel;
- calling a sequence a unified solver when it is only independent full solvers plus final selection.

Stop modular B23 if:

1. parent algorithms cannot be faithfully expressed through a common interface;
2. all equal-budget hybrids fail to improve Fresh1 on natural confirmation;
3. gains vanish after exact compute and candidate-count matching;
4. hybrid schedules harm ordinary controls to rescue a few historical failures;
5. the adaptive controller cannot outperform the best fixed schedule;
6. B2 cannot beat or materially improve the tail of Fresh2 at matched cost.

A valid negative conclusion would be:

> The useful solver biases are complementary only at the candidate level; composing their operations within one fixed-budget trajectory does not preserve that complementarity.

That result would justify returning to risk-controlled whole-solver escalation.

## 10. Claims strategy

### Target primary claim

> A compute-normalized modular diffusion trajectory composes complementary measurement-interaction operations across diffusion stages and improves catastrophic phase-retrieval reliability without increasing the standard two-run budget.

### Stronger adaptive claim, only if supported

> A clean-free low-capacity controller selects among measurement-interaction modules at frozen trajectory checkpoints, improving the reliability-cost frontier over every fixed schedule.

### Claims to avoid

- robustness obtained by unbounded restart scaling;
- best-of-k as an executable solver;
- measurement residual uniquely identifies the correct phase;
- causal claims about a module without equal-budget ablation;
- generality across operators/noise/datasets without new validation;
- calling a full-solver portfolio a step-composed solver.

## 11. Immediate next action

No GPU run is authorized yet.

Proceed in this order:

1. accept or revise this amendment;
2. resolve the repository integration point and freeze B22;
3. write the common state/module API specification;
4. implement exact Fresh1 replay first;
5. add LF and NP replay modules;
6. add SITCOM only after the state conversion is audited;
7. produce the parent compute ledger;
8. approve the 60-case B23.2A schedule screen only after B23.1 sign-off.

The first executor handoff should cover B23.0 and B23.1 only. It should not include schedule search or a full GPU panel.
