# B23 final research plan: compatibility-gated, fixed-budget solver synthesis

Date: 2026-07-30

Status: final planner revision for review and explicit human authorization.

Repository: `Cheng-HanHuang/pr_diffusion`

Planning branch: `codex/post-b22-reliability-plan`

Planning PR: `#36`

Frozen B22 scientific base: `ba78c06e0c5eac0c915263e4faed0b262d5e917a`

This file is the single authoritative B23 scientific plan. It supersedes:

- `docs/planning/2026-07-30_b23_modular_fixed_budget_amendment.md`;
- the project-direction, hypotheses, data protocol, checkpoint plan, and immediate-action sections of
  `docs/planning/2026-07-29_post_b22_reliability_research_plan.md`.

The exact section-by-section status is recorded in
`docs/planning/2026-07-30_b23_supersession_ledger.md`.

No implementation or GPU execution is authorized by this document alone.

## 0. Executive decision

B22 is complete and frozen. The reliability problem is not.

The primary B23 question is:

> Under a fixed, auditable compute budget, can mathematically distinct operations from the
> Fresh/DAPS, LF, NP, and SITCOM lines be composed inside a solver trajectory so that reliability
> improves without relying on more terminal candidates, hidden best-of-k search, or B22-panel
> tuning?

The primary direction is therefore:

> **Compatibility-Gated, Budget-Normalized Solver Synthesis for Reliable Diffusion Phase
> Retrieval**

The phrase **compatibility-gated** is essential. The parent methods do not automatically share one
state representation, one time coordinate, one RNG-consumption pattern, or one notion of a
"correction step." B23 will not flatten them into generic interchangeable gradients.

The project hierarchy is:

```text
Track A — primary
typed, within-trajectory operator composition at fixed compute

Track B — secondary and conditional
whole-solver portfolio or escalation at an explicitly matched or separately reported budget

Track C — later
new candidate generation for failures shared by all current families, such as 65003
```

Track A begins with native parent replay, trace instrumentation, and a donor-compatibility audit.
Only operations that pass those gates may enter a hybrid schedule.

## 1. Ultimate goal and B23 scope

The ultimate project goal remains a diffusion-prior phase-retrieval method that:

- returns one reliable reconstruction without ground-truth access;
- sharply reduces catastrophic failures rather than improving only mean PSNR;
- uses a modest, explicit execution budget;
- is compared fairly with DAPS/Fresh, SITCOM, NP, DPS, DiffFPR, and other current baselines;
- eventually transfers across noise levels, datasets, and measurement settings;
- ultimately supports a principled reliability explanation or guarantee where the theory permits.

B23 is deliberately narrower. Its main development and claims are restricted to:

- FFHQ;
- the frozen B22 phase-retrieval operator and preprocessing;
- `sigma_y = 0.05`;
- the frozen model family and evaluation contract;
- new image and measurement splits that exclude all pre-B23 exposure.

Cross-noise, ImageNet, other operators, and formal reliability theory are follow-up work. They are
not to be added to B23 before the fixed-budget question is answered.

## 2. Evidence that B23 must preserve

### 2.1 Frozen B22 executable frontier

| Policy | Mean raw PSNR | Median | Min | q05 | raw Good25 | raw Bad20 | Mean GPU-s/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fresh1 | 27.232 | 30.605 | 6.482 | 9.614 | 80/100 | 18 | 156.1 |
| Fresh2 | **29.299** | **30.899** | 6.360 | 15.860 | 92/100 | 8 | 312.6 |
| SITCOM-1 | 22.972 | 27.038 | 6.699 | 7.791 | 71/100 | 25 | 49.3 |
| SITCOM-4S | 26.442 | 27.166 | 6.428 | 23.415 | 93/100 | 4 | 196.1 |
| NP-1 | 25.585 | 29.390 | 6.062 | 10.435 | 75/100 | 24 | 52.7 |
| NP-8-RS | 29.269 | 29.941 | **10.995** | **25.146** | **95/100** | **3** | 411.8 |

The correct B22 interpretation is a quality-reliability-cost frontier:

- Fresh2 is the central-quality leader.
- NP-8-RS is the lower-tail reliability leader at the highest cost.
- SITCOM-4S is a cheaper stabilizer with lower ordinary-image PSNR.
- No executable policy dominates all three axes.

The diagnostic best of `{Fresh2, SITCOM-4S, NP-8-RS}` reaches `99/100` Good25. Fresh2 is the
highest-PSNR method on 91 images, NP-8-RS on nine, and SITCOM-4S on zero. Image `65003` is the
only shared failure.

This proves candidate-level complementarity. It does **not** prove that a clean-free selector or a
within-trajectory composition can preserve that complementarity.

### 2.2 Frozen Fresh/DAPS facts

- Fresh2 uses two independent full DAPS trajectories with `ann400`, `diff5`, LF disabled, and HIO
  disabled.
- The second trajectory is selected only if its exact operator loss improves by more than `0.7`.
- Fresh2 improves Good25 from `80/100` to `92/100`, with 12 rescues, zero harms, and zero
  two-candidate selector-oracle gap on B21.11/B22.
- A fresh second trajectory beat LF as the equal-cost default second arm.
- Fresh3 added only one success on a disjoint validation panel; blind restart scaling was stopped.
- Branching from a shared step-200 prefix underperformed an independent restart.

### 2.3 Frozen failure anatomy

Fresh2's eight failures consist of:

- three 180-degree orientation failures;
- two chromatic or illumination overlays;
- two structured twin or ghost mixtures;
- one high-complexity shared collapse (`65003`).

NP and SITCOM have additional selector misses, quality-floor failures, and method-specific
candidate-generation collapses. The mechanisms are heterogeneous. A single scalar residual is not
a correctness certificate.

## 3. Parent-method semantics

External or public names are not enough. B23 must preserve the exact frozen implementation used in
this repository.

### 3.1 Fresh1 / DAPS

Fresh1 is one frozen DAPS trajectory. DAPS alternates a diffusion-prior prediction with a
measurement-conditioned prediction-distribution step and re-noising along a decoupled annealing
schedule. The exact B22 implementation, external commit, patch state, `ann400`, `diff5`, operator,
and RNG behavior must be pinned in B23.0.

Eligible donor operations are not assumed in advance. Candidate boundaries include:

- prior prediction;
- measurement-conditioned correction or optimization;
- re-noising / transition;
- acceptance and trace operations.

Their exact boundaries must be recovered from the native code and replayed.

### 3.2 LF-v1

LF-v1 is already an early-DAPS intervention, not an independent full solver concept:

- it acts on the post-measurement-update `x0y`;
- it preserves current Fourier phase;
- it blends low-frequency Fourier magnitudes toward the measurement;
- the blend strength decays linearly over the first frozen fraction of the DAPS trajectory;
- it then returns to ordinary DAPS behavior.

Therefore the phrase `LF early -> DAPS late` describes the existing LF parent control. It is not a
new B23 hybrid claim.

A new LF-derived schedule must differ mechanistically, for example through a preregistered
noise-coupled bandwidth or strength law, and must be compared with both LF-v1 and a cost-matched
DAPS control.

### 3.3 NP

The repository's NP line is a noise-proposal and ranking process, not a generic measurement
gradient:

- it proposes multiple noise candidates at a timestep;
- it evaluates the resulting denoised estimates using a frozen measurement-side score, commonly a
  low-frequency magnitude score;
- it preserves the selected proposal;
- it may apply a later low-frequency magnitude projection;
- proposal count, score evaluations, and retained states are part of its cost and semantics.

The exact B22 NP-1 and NP-8-RS implementations must be identified in B23.0. A proposed
`NP module` is eligible only if it preserves the corresponding proposal, score, selection,
projection, and RNG semantics. It may not be renamed as a cheap "soft step."

### 3.4 SITCOM

SITCOM enforces step-wise triple consistency, including backward diffusion consistency obtained by
optimizing the input to the pretrained model at each sampling step. Its inner optimization,
network-gradient behavior, forward noising, time schedule, and correction statistic are coupled.

A `SITCOM late polish` is therefore a conjectured truncation, not an established parent operation.
It becomes an eligible donor only if:

1. the native parent replays;
2. the proposed extracted block has a precise input/output contract;
3. the truncation or adapter preserves the claimed consistency semantics;
4. the block passes state-validity and module-replay tests.

Otherwise SITCOM remains a whole-solver baseline.

## 4. Historical dead-direction and non-novelty ledger

The following constraints are binding.

### 4.1 Prior B21/B22 negatives

Do not repeat as minor variants:

1. LF-v1 as the default equal-cost second arm.
2. Blind Fresh3/Fresh4/Fresh5 scaling.
3. Same-prefix step-200 DAPS branching as a substitute for an independent restart.
4. HIO warm-start replacement or the observed high-cost HIO auxiliary policy.
5. The rejected raw-loss, normalized-residual, and two-run-disagreement fallback thresholds.
6. NP-1 as a generally robust replacement solver.
7. Always-on NP-8-RS as a modest-cost default.
8. Selector mining on B21/B22 final images.

### 4.2 Earlier NP/SITCOM negatives that the initial B23 amendment omitted

The pre-B21 Branch-B work already established:

- direct NP-to-SITCOM sigma-state handoff was technically functional but not competitive;
- a good NP clean estimate could be damaged by SITCOM continuation;
- `3S+1NP` had an interesting oracle candidate set, but the executable health-to-NP rule was worse
  than the `4S` SITCOM baseline;
- lower Fourier-magnitude residual did not safely justify replacement.

Consequences:

- a direct `NP prefix -> SITCOM suffix` schedule, or a close state-handoff variant, is prohibited in
  the first B23 schedule screen;
- reviving it requires a separate written hypothesis that identifies what is materially different
  from B3-B8, includes the historical implementation as a control, and receives separate planner
  approval;
- `prdiffusion/algorithms/hybrid_np_sitcom.py` and the historical handoff scripts are prototypes and
  evidence sources, not accepted parent implementations.

### 4.3 Scope additions rejected for the initial screen

The AI-assisted reports suggested DiffStateGrad-style projected correction as another module.
That method is relevant literature, but it is not supported by the B22 complementarity evidence and
would add a fifth method family before the four current parents are understood.

Therefore:

- DiffStateGrad is not an initial B23 donor or schedule;
- new external module families require a later amendment after B23.2;
- recent frequency-continuation work motivates a hypothesis but does not validate it for nonlinear
  Fourier phase retrieval.

## 5. Scientific architecture

### 5.1 Typed native state, not one falsely universal tensor

Every parent retains a native state type:

```text
NativeState = {
    parent_id and source revision,
    tensor payloads and representation contract,
    native diffusion / annealing coordinate,
    model and scheduler identity,
    measurement and operator identity,
    solver-specific optimizer state,
    named RNG streams and counters,
    trace and compute ledger
}
```

A module declares:

```text
Module = {
    required state type,
    produced state type,
    valid native-time or noise domain,
    operation semantics,
    RNG streams consumed,
    raw operation counters,
    serialization contract
}
```

A cross-parent boundary requires an explicit adapter:

```text
Adapter[A -> B] = {
    mathematical conversion,
    source and target noise coordinates,
    information discarded or newly sampled,
    validity checks,
    round-trip check when meaningful,
    cost and RNG accounting
}
```

No adapter may silently:

- reinterpret a clean estimate as a valid noisy state;
- change the forward operator, normalization, or measurement;
- reset optimizer or RNG state without recording it;
- generate and discard candidate branches outside the ledger;
- claim reversibility when the conversion is lossy.

### 5.2 Shared semantic coordinates

Raw step indices are not comparable across DAPS, NP, and SITCOM.

B23 must record two coordinates:

1. a **semantic noise coordinate**, preferably native `sigma` or log-SNR mapped monotonically to
   `[0,1]`;
2. a **cumulative budget coordinate**, equal to the fraction of the schedule's preregistered
   work-FRE consumed.

Cross-parent switches may occur only at preregistered semantic-noise boundaries supported by a
validated adapter. Cost matching is performed on the budget coordinate.

### 5.3 Donor eligibility classes

After B23.1, every parent operation is classified as:

- `NATIVE-REPLAYED`: parent and trace wrapper replay, but no cross-parent use is claimed;
- `DAPS-NATIVE-DONOR`: composable inside the DAPS state without a cross-parent adapter;
- `ADAPTER-QUALIFIED-DONOR`: cross-parent input/output and semantics pass all compatibility gates;
- `BASELINE-ONLY`: parent remains a whole-solver baseline;
- `REJECTED-PROTOTYPE`: known-invalid, historically failed, or semantically unfaithful.

Failure of one parent to become a donor does not invalidate the other parents. However, a
cross-family Track-A claim requires at least one of NP or SITCOM to become an
`ADAPTER-QUALIFIED-DONOR`.

## 6. Hypotheses and falsifiers

### H0 — feasibility and cross-family compatibility

**Hypothesis.** The four native parents can be instrumented without changing their frozen behavior,
and at least one non-DAPS operation from NP or SITCOM can be extracted or adapted without semantic
loss large enough to invalidate its parent interpretation.

**Pass evidence.**

- native parent replay for Fresh1, LF-v1, NP-1, and SITCOM-1;
- DAPS-native module replay for Fresh1 and LF-v1;
- at least one NP or SITCOM donor passes state, adapter, RNG, and module-replay gates.

**Falsifier.**

- neither NP nor SITCOM yields a faithful donor;
- apparent compatibility depends on an invalid clean-to-noisy handoff;
- the wrapper changes parent outcomes beyond the frozen replay envelope.

If H0 fails, stop cross-family Track A. A DAPS-native frequency-schedule study may continue only
under a narrower claim, or the project may pivot to Track B.

### H1 — one-FRE fixed modular trajectory

**Hypothesis.** One frozen, open-loop, one-terminal-candidate schedule at B1 improves Fresh1 through
better operator allocation.

**Final practical gate on the frozen evaluation panels.**

- hard per-image work-FRE `<= 1.10`;
- paired median time-FRE `<= 1.10`;
- raw Good25 improvement at least `+5` percentage points;
- Good25 harms at most `1` percentage point;
- raw mean PSNR delta at least `-0.20 dB`;
- q05 no worse than Fresh1;
- one-sided 95% paired-bootstrap lower bound for Good25 delta above zero;
- no hidden terminal-candidate increase.

### H2 — two-FRE fixed modular policy

**Hypothesis.** A fixed policy with at most two terminal candidates at B2 improves the
quality-reliability frontier over Fresh2 through module complementarity rather than blind restart
count.

**Final practical gate.**

- hard per-image work-FRE `<= 2.10`;
- paired median time-FRE `<= 2.10`;
- raw Good25 improvement at least `+3` percentage points;
- Good25 harms at most `1` percentage point;
- raw mean PSNR delta at least `-0.25 dB`;
- q05 improvement at least `+2 dB`;
- no increase in Bad10;
- one-sided 95% paired-bootstrap lower bound for Good25 delta above zero;
- paired McNemar exact result reported;
- retained terminal-candidate count matched to the strongest two-candidate control.

### H3 — mechanism, not compute or candidate scaling

**Hypothesis.** The gain remains after matching:

- calibrated work-FRE;
- measured GPU-active time;
- retained terminal candidates;
- solver and measurement seeds;
- deleted/reallocated baseline operations.

Each promoted schedule must beat:

1. its full parent control;
2. a cost-matched shortened or reallocated DAPS control;
3. a module-removal ablation;
4. a candidate-count-matched control.

If the gain disappears, the result is compute allocation, random refresh, or candidate scaling—not
evidence for the claimed donor mechanism.

### H4 — optional limited adaptation

Adaptive control is not part of the first authorized B23 program.

It may be proposed only if H1 or H2 has a frozen fixed-schedule success. A later amendment must:

- use at most two decision checkpoints;
- use low-capacity, clean-free rules;
- expose no feature from a module or future branch that has not been executed;
- preserve the same hard per-image budget and terminal-candidate cap;
- compare against the best fixed schedule, not only Fresh1;
- use new development and calibration data.

Failure to beat the best fixed schedule ends the adaptive claim.

### H5 — prospective reproducibility

After code, adapters, schedules, seeds, and budgets are frozen:

- the full-candidate audit and policy-only execution must agree within preregistered uncertainty;
- no unused branch may be generated in the policy-only run;
- actual time and work-FRE must match the frozen budget prediction;
- the direction of the Good25, harm, mean-PSNR, and lower-tail effects must be consistent across the
  two disjoint final panels.

## 7. Compute contract

### 7.1 Frozen reference

`1.0 Fresh1-equivalent run (FRE)` is anchored to one frozen B22 Fresh1 trajectory:

```text
method: Fresh1 / DAPS
configuration: ann400, diff5, LF off, HIO off
model/operator/noise/preprocessing: exact frozen B22 setting
source/environment/hardware: pinned in B23.0
terminal candidates: 1
```

The exact source SHA, external DAPS SHA and diff, environment lock, model SHA-256, operator config,
GPU model, driver, CUDA, and PyTorch versions are mandatory B23.0 fields. They are not left to
executor judgment.

### 7.2 Raw operation ledger

Every executed policy records per image:

- denoiser/model forward calls;
- denoiser/model backward, JVP, or VJP calls;
- measurement forward calls;
- measurement adjoint, JVP, or VJP calls;
- FFT, projection, and correction calls;
- inner optimizer steps by optimizer type;
- re-noising and state-conversion calls;
- random proposals and named RNG draws;
- live and retained branches;
- retained terminal candidates;
- GPU-active seconds;
- wall seconds;
- peak allocated and reserved memory;
- serialization, preprocessing, and conversion overhead.

### 7.3 Calibrated work-FRE and time-FRE

Raw counts from heterogeneous solvers are not directly commensurate, so B23 does not use the
maximum of unweighted raw-count ratios as its sole cost definition.

Before hybrid execution, B23.0/B23.1 microbenchmarks each atomic operation class on the pinned
hardware and freezes its median active-GPU cost `w_j`.

For policy `s` on image `i`:

```text
calibrated_work(s,i) = sum_j count_j(s,i) * w_j
work_FRE(s,i) = calibrated_work(s,i) / calibrated_work(Fresh1 reference)
time_FRE(s,i) = GPU_active_seconds(s,i) / paired_median_GPU_active_seconds(Fresh1)
claim_FRE(s,i) = max(work_FRE(s,i), time_FRE(s,i))
```

The main-claim eligibility rule is:

- hard per-image `work_FRE <= 1.10` for B1 or `<= 2.10` for B2;
- paired median `time_FRE` within the same cap;
- time-FRE q90 no more than `0.10` above the cap;
- raw counts and both ratios reported.

If a policy exceeds a cap, it may be reported as a separate higher-cost frontier point. It may not
be used for H1/H2.

### 7.4 Candidate and branching contract

- B1 retains exactly one terminal candidate.
- B2 retains at most two terminal candidates.
- Every intermediate NP proposal or stochastic branch is counted, even if discarded.
- A "shared prefix" receives no free credit: all executed work is counted.
- Offline generation of branches that a claimed policy would not execute is allowed only in a
  clearly labeled audit panel and cannot be charged as deployment compute.
- Best-of-k and oracle curves are diagnostic only.

### 7.5 Cost-matched scientific comparisons

For a module-effect claim, operation-weighted work must match within `5%`, and paired median
GPU-active time must match within `10%`. If this is infeasible, report a cost-response curve rather
than a causal module claim.

## 8. Replay, determinism, and RNG contract

### 8.1 Artifact pinning

B23.0 must pin:

- repository and external-solver SHAs;
- every applied patch and dirty diff hash;
- Python environment lock or package inventory;
- model checkpoint SHA-256;
- dataset manifest and checksums;
- measurement operator, oversampling, noise, clipping, normalization, and seed policy;
- GPU model, driver, CUDA, cuDNN, PyTorch, precision, and batch size;
- parent configs and solver seeds.

### 8.2 Two-track replay

The replay gate distinguishes:

1. **bitwise replay**
   - required only when the native parent is bitwise repeatable under the pinned environment;
   - verified by tensor/checkpoint hashes;
2. **tolerance-qualified replay**
   - used for a native GPU path that is not bitwise repeatable;
   - thresholds are frozen from repeated native-parent runs before wrapper comparisons;
   - reports `max_abs_err`, `mean_abs_err`, relative error, metric deltas, and trace deltas.

`torch.use_deterministic_algorithms(True)`, cuDNN flags, and CUBLAS workspace settings must be
recorded and tested. They are not forced if enabling them changes the frozen parent algorithm or
makes the reference path invalid. Such a mismatch is reported.

No cross-release or cross-hardware bitwise claim is required.

### 8.3 Replay tolerances

B23.0 defines the procedure; B23.1 supplies the numbers.

For each parent:

1. run the unchanged native implementation repeatedly on one preregistered smoke case;
2. measure its native repeatability envelope;
3. freeze tensor and scalar tolerances before testing the wrapper;
4. require wrapper error to stay inside that envelope plus a declared numerical floor;
5. require exact operation-count reconciliation.

Final raw PSNR, measurement loss, and other scalar deltas must be reported even when tensors pass.

### 8.4 RNG audit

Each parent and module uses named RNG streams. The audit records:

- base seed and derivation rule;
- device generator;
- draw shape, dtype, device, and count;
- proposal/branch identifier;
- serialization and resume state;
- whether a conversion introduces new randomness.

The requirement is:

- wrapper-versus-native random consumption must reconcile for parent replay;
- different parents need not consume the same number of random values;
- a hybrid must isolate streams so adding an unrelated module does not silently shift all later
  randomness.

## 9. Data separation and leakage control

### 9.1 Pre-B23 exposure manifest

The exclusion rule is broader than the initial amendment.

B23.0 creates `PRE_B23_EXPOSURE.csv` containing every image ID and measurement ID found in:

- B19 and B20 experiments;
- the earlier NP/SITCOM Branch A and Branch B work;
- B21 development, validation, benchmark, and atlas work;
- B22 fixed-baseline evaluation;
- manually inspected failure or rescue examples;
- any attached or local planning artifact used to form a B23 hypothesis.

If exposure status is uncertain, the item is treated as exposed.

Exposed data may be used for:

- historical evidence;
- debugging file readers;
- explicitly labeled retrospective sanity checks after a choice is frozen.

It may not be used for:

- schedule or boundary selection;
- donor acceptance thresholds;
- controller features or thresholds;
- stopping rules;
- promotion decisions;
- final claims.

### 9.2 New split registry

All B23 image IDs and measurement seeds are assigned once in a machine-readable registry before
the corresponding stage starts.

| Split | Minimum size | Purpose | May select methods? |
|---|---:|---|---|
| `DEV-SCREEN` | 600 | Fresh1-only natural screening and hard/control construction | baseline stratification only |
| `DEV-MECH` | 60 | enriched fixed-schedule mechanism screen | yes |
| `DEV-NATURAL` | 120 | natural-prevalence B1 confirmation and freeze | yes, freeze at most one B1 |
| `CAL-B2` | 150 | compare and freeze at most one B2 fixed policy | yes |
| `TEST-AUDIT` | 250 | untouched frozen-policy audit | no |
| `TEST-PROSPECTIVE` | 150 | policy-only execution | no |

All rows are disjoint from pre-B23 exposure. `DEV-MECH` is the declared 60-image subset derived
from `DEV-SCREEN`; every other named split is image-disjoint and measurement-disjoint from
`DEV-SCREEN`/`DEV-MECH` and from one another.

### 9.3 Enriched mechanism set construction

Run only Fresh1 on `DEV-SCREEN`.

If at least 30 Fresh1 Bad25 cases are observed, construct `DEV-MECH` as:

- 30 Fresh1 Bad25 cases selected by a preregistered deterministic stratification over baseline
  PSNR severity;
- 30 Fresh1 Good25 controls sampled with a frozen seed and matched on declared input/measurement
  strata.

If fewer than 30 failures occur, expand `DEV-SCREEN` before schedule evaluation. Do not redefine
Good25 or select historical B22 failures.

The enriched set is for mechanism screening only. Its Good25 rate is not a deployment estimate.

### 9.4 Repeated-measurement diagnostic

Before any adaptive controller is authorized, add a second locked measurement to 50 calibration
images. This tests measurement-seed stability. It is unnecessary for selecting a fixed open-loop
schedule but remains a required diagnostic for any later routing rule.

## 10. Evaluation contract

### 10.1 Primary deployment metrics

- raw RGB PSNR;
- raw Good25;
- raw Bad20 and Bad10;
- mean, median, q01, q05, q10, and bottom-five mean;
- paired rescues and harms;
- work-FRE and time-FRE;
- retained terminal-candidate count.

### 10.2 Auxiliary metrics

- SSIM and LPIPS;
- 180-degree-rotation-minimized PSNR;
- method-specific oracle and selector regret;
- failure taxonomy and visual atlas.

The established project contract remains:

- raw orientation is primary because the deployed solver must return one image in the required
  orientation;
- ground-truth-assisted 180-degree alignment is auxiliary only;
- measurement residual is not a correctness certificate;
- no ground-truth symmetry resolution is allowed at runtime.

No new global-phase, translation, or twin-image adjustment becomes a primary metric unless the
forward-operator audit proves it is an unavoidable equivalence and a separate protocol amendment
is approved before evaluation.

### 10.3 Statistical reporting

For each frozen comparison:

- image-level paired bootstrap 95% intervals;
- paired Good25 delta and one-sided lower confidence bound;
- exact McNemar test on threshold discordances;
- paired mean and quantile differences;
- rescues, harms, and their image IDs;
- no-dropped-row and missing-candidate audit.

The primary inferential population is the preregistered union of `TEST-AUDIT` and
`TEST-PROSPECTIVE` (`n >= 400`) under one unchanged frozen policy. The two panels are also reported
separately. The prospective panel must agree in effect direction and must not reveal a gate-breaking
harm or catastrophic-floor regression.

## 11. Schedule grammar

No concrete B23.2 schedule is authorized by this plan. Exact schedules are frozen only after
B23.1 in a separately reviewed `B23.2_PREREGISTRATION.md`.

The grammar is nevertheless fixed now.

### 11.1 Restrictions

- at most six hybrid schedules in the first screen;
- at most two cross-family switch boundaries;
- boundaries defined in semantic noise/log-SNR coordinates, not arbitrary raw step indices;
- open-loop only;
- one terminal candidate at B1;
- at most two terminal candidates at B2;
- no continuous learned mixture;
- no RL or architecture search;
- no post-outcome boundary tuning;
- every added operation paid for by deletion or shortening elsewhere;
- every schedule paired with a module-removal and cost-reallocation control.

### 11.2 Eligible schedule families after compatibility sign-off

1. **DAPS-native measurement-timing family**
   - prior-dominant or softened measurement interaction early;
   - ordinary frozen DAPS correction later;
   - includes a shortened-DAPS cost control.

2. **DAPS-native frequency-continuation family**
   - a preregistered bandwidth/strength law coupled to the DAPS noise coordinate;
   - materially different from LF-v1;
   - compared with LF-v1 and ordinary DAPS.

3. **NP-proposal substitution family**
   - replaces, rather than supplements, a frozen DAPS window with an adapter-qualified NP
     proposal/ranking kernel;
   - proposal count and all candidate evaluations counted;
   - no retained hidden branch.

4. **SITCOM-consistency substitution family**
   - replaces a frozen correction window with an adapter-qualified SITCOM consistency block;
   - permitted only if the block preserves the stated triple-consistency semantics;
   - not described as "polish" without replay evidence.

### 11.3 Prohibited first-screen schedules

- direct NP-state-to-SITCOM continuation;
- `LF early -> DAPS late` relabeled as novel rather than the LF-v1 control;
- shared-prefix multi-branch DAPS;
- a stochastic refresh that retains both old and new terminal candidates at B1;
- DAPS followed by a full extra solver;
- DiffStateGrad or any fifth external module family;
- a checkpoint selector based on the rejected B21 scalar/pairwise triggers.

## 12. Stage plan and gates

### B23.0 — repository, protocol, and semantics freeze

GPU budget: `0`.

Deliverables:

1. exact repository/PR dependency map and approved execution base;
2. clean B23 branch/worktree under `/egr/research-pac/huang248`;
3. pinned parent sources, patches, environments, model, operator, hardware, and configs;
4. `PRE_B23_EXPOSURE.csv` and future split-registry schema;
5. parent-semantics table and historical dead-direction ledger;
6. typed-state/module/adapter API specification;
7. raw compute-ledger schema and atomic-cost microbenchmark plan;
8. two-track replay and RNG policy;
9. smoke, validation, artifact-retention, and stop runbooks;
10. a B23.2 preregistration template with empty schedule slots;
11. no B23.1 GPU launch.

Gate:

- all mandatory identifiers resolved or explicitly reported missing;
- any missing parent identity, operator identity, model hash, or critical source diff blocks B23.1;
- no history rewrite or merge performed without human authorization;
- no universal-state assumption;
- no schedule candidates invented;
- planner/user signs off before B23.1 GPU work.

### B23.1A — native parent replay and trace instrumentation

Run unchanged native and instrumented versions of:

1. Fresh1;
2. LF-v1;
3. NP-1;
4. SITCOM-1.

Evidence:

- one-image native repeatability calibration;
- one-image wrapper replay;
- intermediate trace comparison;
- exact operation-count reconciliation;
- named RNG ledger;
- work/time microbenchmarks;
- serialization/resume checks where the native algorithm exposes a valid boundary.

Then run a four-image heterogeneous smoke on four new, preregistered images only.

### B23.1B — donor extraction and compatibility classification

Required:

- Fresh1 DAPS-native module sequence replay;
- LF-v1 DAPS-native module sequence replay;
- NP donor audit;
- SITCOM donor audit;
- adapter validity table;
- historical NP-to-SITCOM prototype comparison where relevant;
- classification of every candidate operation using Section 5.3.

Gate:

```text
all four native parent wrappers replay: required
Fresh1 DAPS-native module replay: required
LF-v1 DAPS-native module replay: required
at least one NP/SITCOM adapter-qualified donor: required for cross-family Track A
compute and RNG ledgers trustworthy: required
B23.2 schedule authorization: separate decision
```

Possible returns:

- `AUTHORIZE B23.2 PREREGISTRATION`;
- `CONTINUE DAPS-NATIVE ONLY UNDER NARROWED CLAIM`;
- `REVISE B23.1`;
- `STOP TRACK A AND PIVOT TO TRACK B`.

### B23.2A — screen construction

Separate authorization required.

- freeze `DEV-SCREEN`;
- run Fresh1 only;
- construct `DEV-MECH` exactly as Section 9.3;
- write and approve `B23.2_PREREGISTRATION.md`;
- no schedule execution before the preregistration commit.

### B23.2B — enriched mechanism screen

- execute parent controls, cost controls, and at most six hybrid schedules on `DEV-MECH`;
- use paired solver seeds and locked measurements;
- retain at most two schedules;
- require gains on multiple distinct hard images and no ordinary-control harm concentration;
- reject gains explained by compute, terminal-candidate count, or one visual category.

Promotion minimum:

- at least three distinct Good25 rescues over the relevant baseline;
- at most one Good25 harm among the 30 controls;
- nonnegative direction on q05 and mean-PSNR guardrails;
- H3 ablations directionally support the claimed donor.

These are development gates, not final claims.

### B23.2C — natural B1 confirmation

- evaluate at most two retained B1 schedules on `DEV-NATURAL`;
- compare against Fresh1, LF-v1, and cost-matched controls;
- freeze at most one B1 schedule;
- require H1 to be plausible in point estimates and no gate-breaking harm;
- if neither survives, stop B1 fixed composition.

### B23.3 — fixed B2 construction and freeze

Use `CAL-B2` only.

At most four B2 candidates may be compared:

1. Fresh2;
2. Fresh1 plus the frozen B1 hybrid as the second candidate;
3. two independent frozen B1 hybrid trajectories, if justified;
4. one internally composed B2 schedule, if donor semantics support it.

All must satisfy the two-terminal-candidate cap and cost matching.

Freeze at most one B2 policy. If no candidate plausibly improves Fresh2's tail without ordinary
quality harm, retain Fresh2 and record a negative B2 result.

### B23.4 — optional adaptation

Not automatically authorized.

It requires:

- fixed-schedule success from B23.2 or B23.3;
- a new amendment, new development/calibration data, and new user approval;
- the H4 restrictions.

Track-B whole-solver escalation also requires a new amendment. The superseded 2026-07-29 router
plan is background, not execution permission.

### B23.5 — untouched full-candidate audit

Run frozen B1 and B2 policies on `TEST-AUDIT`.

- no schedule, threshold, seed order, or adapter update;
- generate only the audit arms preregistered as necessary to compute fixed-policy regret;
- report every executable baseline and diagnostic oracle separately;
- create complete failure and complementarity atlases.

Failure of H1/H2 gates is a negative result. Do not revise the policy.

### B23.6 — prospective policy-only execution

Run `TEST-PROSPECTIVE` with:

- only the modules and branches executed by the frozen policy;
- no unused audit candidates;
- actual work/time ledger;
- no silent retry or fallback;
- frozen failure log.

This is the deployment-style closeout and the H5 test.

## 13. Stop and pivot rules

Stop or narrow Track A if:

1. native parent wrappers cannot replay;
2. neither NP nor SITCOM yields a faithful donor;
3. the only working schedules reproduce previously rejected NP-to-SITCOM handoff, LF-v1, or
   same-prefix branching;
4. no B1 schedule survives natural confirmation;
5. B2 cannot improve Fresh2 at matched cost and candidate count;
6. gains disappear under H3 controls;
7. ordinary controls are harmed to rescue a small historical-looking subset;
8. final frozen tests fail the declared practical or statistical gates.

Valid negative conclusions include:

> The parent solvers are complementary only at the terminal-candidate level; their useful
> operations are not semantically compatible enough for within-trajectory composition.

or:

> DAPS-native frequency scheduling is feasible, but cross-family module composition does not
> preserve the B22 complementarity at fixed compute.

After a negative Track-A result:

- Track B may study a risk-controlled whole-solver cascade on entirely new data;
- Track C may study new candidate generation for shared hard cases;
- neither pivot inherits implementation authorization from B23.

## 14. Claim and publication policy

### Target claim if H0-H3 and H5 pass

> A compatibility-audited, compute-normalized solver composes measurement/prior operations across
> diffusion stages and improves phase-retrieval tail reliability over matched Fresh baselines
> without increasing terminal candidate count.

### Stronger B2 claim if supported

> At the standard two-run budget, a fixed modular policy improves the catastrophic reliability
> floor over Fresh2 while preserving ordinary-image quality.

### Claims to avoid

- universal phase-retrieval robustness;
- correctness certification by measurement residual;
- state-of-the-art without a current, matched benchmark audit;
- causal module claims without H3 controls;
- transfer beyond FFHQ and `sigma_y=0.05` without new experiments;
- calling full-solver final selection a within-trajectory solver;
- treating BIPSDA, frequency-continuation, or DiffStateGrad literature as proof that this hybrid
  must work.

### Follow-up external-validity stage

Only after B23 succeeds should a new plan test:

- `sigma_y in {0, 0.01, 0.05}` or another preregistered noise grid;
- ImageNet or another prior/dataset;
- current DPS, DAPS, SITCOM, DiffFPR, and relevant 2026 baselines;
- robustness to measurement seeds and operator variants;
- theory or probabilistic reliability statements.

## 15. Repository and execution policy

Before any GPU work:

1. preserve the stacked PR history unless the user explicitly chooses integration;
2. record the immutable B22 scientific base;
3. create a clean B23 worktree under `/egr/research-pac/huang248`, never `/home`;
4. commit configs, manifests, launchers, validators, and runbooks before execution;
5. use manual-only GPU launchers and smoke gates;
6. keep environments, caches, data, models, and outputs under research-pac;
7. return compact evidence archives to the planner;
8. build inventories from documented manifests and targeted paths; do not recursively scan large
   output/data/model trees without explicit user approval;
9. stop on source, operator, measurement, or semantic incompatibility rather than improvising.

## 16. Authorization boundary

```text
B22 scientific state: COMPLETE AND FROZEN
B23 final plan: READY FOR ACCEPTANCE DECISION
B23.0: NOT AUTHORIZED UNTIL EXPLICIT USER APPROVAL
B23.1 GPU replay: NOT AUTHORIZED UNTIL B23.0 SIGN-OFF
B23.2 and later: NOT AUTHORIZED
Large GPU panels: NOT AUTHORIZED
Track B portfolio: NOT AUTHORIZED
Track C hard-case work: NOT AUTHORIZED
```

The first executor should receive B23.0 only. The same executor may prepare B23.1 code and runbooks,
but must stop before B23.1 GPU execution until the B23.0 return package is reviewed.

## 17. Primary external references and interpretation

- DAPS: Zhang et al., *Improving Diffusion Inverse Problem Solving with Decoupled Noise
  Annealing*, CVPR 2025.
- SITCOM: Alkhouri et al., *SITCOM: Step-wise Triple-Consistent Diffusion Sampling For Inverse
  Problems*, ICML 2025.
- DPS: Chung et al., *Diffusion Posterior Sampling for General Noisy Inverse Problems*, ICLR 2023.
- BIPSDA: Crafts and Villa, *Benchmarking Diffusion Annealing-Based Bayesian Inverse Problem
  Solvers*, 2025.
- DiffStateGrad: Zirvi et al., *Diffusion State-Guided Projected Gradient for Inverse Problems*,
  ICLR 2025.
- Tian et al., *Stabilizing Diffusion Posterior Sampling by Noise-Frequency Continuation*, 2026
  preprint.
- PyTorch, *Reproducibility* documentation.

These references motivate decoupled annealing, triple consistency, modular design, careful
measurement timing, and tolerance-aware replay. They do not establish that an arbitrary
cross-parent splice is valid or that the proposed B23 hypotheses are true.
