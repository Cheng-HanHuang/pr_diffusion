> [!IMPORTANT]
> **Superseded as an execution plan on 2026-07-30.** The authoritative B23 plan is
> `docs/planning/2026-07-30_b23_final_research_plan.md`. The portfolio material in this file is
> retained as historical Track-B rationale only; it does not authorize implementation.

# Post-B22 research decision and detailed reliability roadmap

Date: 2026-07-29

Status: strategic planning document; no new experiment is authorized by this file alone.

Base scientific checkpoint: B22 fixed-baseline evaluation at commit
`ba78c06e0c5eac0c915263e4faed0b262d5e917a`.

## 0. Executive decision

B22 is complete. The broader reliability problem is not.

The repo now establishes three facts:

1. two independent full DAPS trajectories with the frozen exact-loss margin selector
   are the strongest simple default discovered so far (`Fresh2`);
2. NP is not a convincing single-run replacement solver, but an NP population supplies
   complementary candidates that rescue most catastrophic Fresh2 failures;
3. the main remaining systems problem is not another selector on the B22 panel, but a
   clean-free, cost-aware policy that decides when to accept DAPS and when to escalate
   to a complementary population.

The recommended immediate project is therefore:

> **B23: Risk-Controlled Adaptive Portfolio for Reliable Diffusion Phase Retrieval**

The primary runtime architecture should be a staged policy centered on Fresh2 with
conditional NP escalation. SITCOM remains an important fixed baseline and possible
lower-cost auxiliary arm, but it is not the primary complement because, on B22, it
adds no unique good25 rescue beyond the Fresh2+NP pair.

A second, later research track should target shared candidate-generation failures such
as image `65003`. It must not be mixed into the first B23 routing study until the
adaptive-portfolio question has been answered cleanly.

## 1. What the completed repo teaches us

### 1.1 Independent full-trajectory diversity is the clearest positive result

The adopted Fresh2 policy runs two independent complete DAPS trajectories and selects
trajectory 2 only when its exact operator loss improves by more than `0.7`.

On the frozen prospective 100-image panel:

- Fresh1: `80/100` raw good25;
- Fresh2: `92/100` raw good25;
- rescues: `12`;
- harms: `0`;
- selected-oracle gap: `0`.

The remaining eight failures are candidate-generation failures under the two-run DAPS
population. This means the exact-loss selector is adequate for the present two-DAPS
population; the dominant unresolved issue is generating a good candidate in the first
place.

### 1.2 Objective/guidance diversity is real but weaker than a fresh restart

LF guidance generated some genuine rescues and demonstrated that changing the early
optimization geometry can enter different basins. However, at matched wall cost, a
second ordinary independent DAPS trajectory was superior:

- Fresh2: `70/80` good25;
- Base+LF: `63/80` good25;
- Fresh2-only wins: `10`;
- Base+LF-only wins: `3`.

The same advantage appeared under oracle selection, so this was a candidate-generation
result rather than a selector artifact.

Conclusion: LF is not the default second arm. Preserve it only as a diagnostic,
feature source, or optional portfolio component if a new development study shows a
specific cost-adjusted role.

### 1.3 Blind restart scaling has strong diminishing returns

Fresh3 looked promising on a development curve, but on a disjoint validation panel it
added only one success beyond Fresh2 and improved only one of twenty images. Blind
Fresh4/Fresh5 scaling was therefore correctly stopped.

Conclusion: two full DAPS trajectories are the default budget. Additional DAPS
trajectories require a conditional trigger or a new mechanism, not blind scaling.

### 1.4 Mid-trajectory continuation does not substitute for independent diversity

Continuation state capture was technically successful, but equal-cost branching from
step 200 underperformed fresh independent restarts.

Conclusion: trajectory independence matters. Reusing a shared prefix does not provide
sufficient basin diversity under the tested protocol.

### 1.5 HIO warm starts are not a default solution

HIO replacement candidates underperformed ordinary DAPS. An HIO auxiliary arm produced
some rescues, but they were concentrated in very few images and the total cost was too
large.

Conclusion: HIO is retained for forensic or mechanistic study only. It should not be
reintroduced as a default or conditional arm without a genuinely new hypothesis and a
new preregistered development split.

### 1.6 The tested scalar and pairwise failure detectors are not deployable

Retrospective detector work found useful rankings but failed deployment constraints:

- raw exact loss ranked failures well but transferred with excessive false positives;
- measurement-normalized residual lost useful information and missed persistent cases;
- rotation-aware candidate disagreement was informative but insufficiently precise and
  selective.

Conclusion: the exact tested thresholds are dead. The general problem of clean-free
risk estimation remains open, but it must use a new data protocol, richer runtime
features, calibration, and prospective evaluation. It must not be presented as a
minor retuning of the rejected thresholds.

### 1.7 NP is a complementary population mechanism, not a robust standalone solver

On B22:

- NP-1: `75/100` raw good25;
- NP-8-RS: `95/100` raw good25;
- NP-8-RS has the strongest executable q05 and removes sub-10 dB failures;
- Fresh2 still has higher raw PSNR on `91/100` images;
- NP-8-RS is the most expensive executable policy.

NP therefore does not support a claim that delayed measurement enforcement and early
measurement-consistency ranking produce a generally superior one-run solver. Its
validated value is different:

- it reaches basins that ordinary DAPS often misses;
- it rescues seven of the eight Fresh2 raw failures;
- its population residual is a useful but imperfect candidate-ranking feature;
- most remaining NP failures are candidate-generation or quality-floor failures, not
  selector failures.

The knowledge to carry forward is the value of **delayed/soft measurement interaction
as a diversity mechanism**, not NP-1 as the final solver.

### 1.8 SITCOM is a useful cost/reliability baseline but not the primary complement

SITCOM-4S reaches `93/100` raw good25 at lower cost than Fresh2 or NP-8-RS, but its
ordinary-image PSNR is lower. On the B22 cross-method panel:

- Fresh2 is the highest-PSNR method on 91 images;
- NP-8-RS is highest on nine;
- SITCOM-4S is highest on zero;
- Fresh2+NP-8-RS already reaches the `99/100` cross-method oracle good25;
- adding SITCOM does not increase that oracle rate.

Conclusion: SITCOM-4S remains a required baseline and may be tested as a cheaper
escalation option. It is not the central arm for the first adaptive-router project,
because it contributes no unique B22 rescue beyond NP.

### 1.9 Failure mechanisms are heterogeneous

Fresh2's eight failures exactly reproduce the frozen B21/B22 taxonomy:

- three 180-degree orientation failures;
- two chromatic/illumination overlays;
- two structured twin/ghost mixtures;
- one high-complexity shared collapse.

The failures are not one repeated attractor. This argues against one universal scalar
repair. It supports either adaptive routing across genuinely different solver families
or mechanism-specific candidate generation.

## 2. Direction classification

### 2.1 Adopted foundations

These should be treated as fixed starting points, not reopened casually:

| Component | Status | Reason |
|---|---|---|
| Fresh2 | adopted backbone | strongest simple quality/reliability default at 2 full-run equivalents |
| independent full trajectories | adopted diversity principle | consistently stronger than same-prefix branching and LF allocation |
| exact-loss margin selector within Fresh2 | adopted for two-DAPS population | zero selector-oracle gap on the frozen panel |
| transparent candidate accounting | required infrastructure | reliability claims depend on no dropped rows, exact hashes, and cost accounting |
| raw PSNR primary / rot180 auxiliary | adopted evaluation contract | preserves deployment orientation while exposing phase ambiguity |
| failure-union visual atlas | required closeout tool | numerical good25 alone hides distinct mechanisms |

### 2.2 Retained but secondary

| Direction | Status | Allowed future role |
|---|---|---|
| LF guidance | retained diagnostic | diversity feature or specialized arm only after new dev evidence |
| NP mechanism | retained primary complement | conditional candidate generator; not standalone solver claim |
| SITCOM-4S | retained baseline / cheap arm | benchmark and possible lower-cost escalation candidate |
| candidate disagreement | retained feature | multivariate risk input, not standalone trigger |
| HIO | forensic only | mechanism study, not default policy |
| continuation interface | infrastructure only | useful for future mechanistic tests, not current allocation |

### 2.3 Rejected for the current setting

These should not be rerun as minor variants:

1. LF as the default second arm at equal cost.
2. Blind Fresh3/Fresh4/Fresh5 as the default reliability strategy.
3. Step-200 shared-prefix branching as a replacement for fresh restarts.
4. HIO warm-start replacement as the default solver.
5. The tested HIO auxiliary policy at its observed cost.
6. Raw-loss, normalized-residual, or disagreement thresholds copied from B21.
7. NP-1 as a robust replacement solver.
8. Always-on NP-8-RS as a modest-cost default.
9. Additional selector mining on the B22 100-image panel.
10. Ground-truth-assisted orientation correction as a deployable method.

"Rejected" here means rejected under the current FFHQ, `sigma_y=0.05`, operator,
model, and budget evidence. It is not a theorem that the broad idea can never work.
Any revival requires a materially new hypothesis, protocol, and untouched data.

### 2.4 Open questions

1. Can runtime-available features identify when Fresh1 or Fresh2 is unsafe?
2. Can NP escalation be triggered on a small enough fraction of cases to improve the
   reliability floor without exceeding always-on NP cost?
3. Can a clean-free cross-family selector avoid replacing a good Fresh2 output by an
   inferior NP output?
4. Can the NP population be reduced or stopped progressively while retaining most
   rescue opportunity?
5. Do risk features transfer across measurement seeds for the same image?
6. Do the conclusions transfer to other noise levels, datasets, or measurement
   operators?
7. What new candidate-generation mechanism can address shared failures such as
   `65003` when every present candidate family fails?

## 3. Recommended main project: B23

### 3.1 Working title

**Risk-Controlled Adaptive Solver Portfolio for Reliable Diffusion Phase Retrieval**

### 3.2 Primary scientific question

Can a clean-free, staged policy approach the `99/100` Fresh2+NP oracle reliability on
new untouched data while spending substantially less than running NP-8 on every case
and while preserving Fresh2's ordinary-image quality?

### 3.3 Runtime policy family

The initial policy family should be intentionally narrow:

```text
Stage 1: run Fresh1
    |
    +-- high-confidence safe --> return Fresh1
    |
    +-- otherwise --> run the second independent DAPS trajectory
                         |
                         +-- high-confidence safe --> return Fresh2
                         |
                         +-- high-risk --> run a frozen NP escalation budget
                                              |
                                              +-- clean-free cross-family decision
                                                  between Fresh2 and NP candidate
```

This staged family includes two nested research questions:

- **R1:** can computation sometimes stop after Fresh1 without harming reliability?
- **R2:** can high-risk Fresh2 cases be selectively escalated to NP and safely
  replaced?

R2 is the primary objective. R1 should be attempted only after R2 is supported, because
premature Fresh1 acceptance risks sacrificing the central-quality gain already
established by Fresh2.

### 3.4 Why Fresh2+NP is the core pair

On B22, the Fresh2+NP-8-RS diagnostic oracle reaches `99/100` raw good25. The pair
therefore contains almost all currently observed rescue opportunity. SITCOM adds no
unique good25 rescue on that panel.

This does not prove that SITCOM is useless. It means the first adaptive policy should
minimize complexity and focus on the pair with demonstrated unique complementarity.
SITCOM remains a required comparison and may enter later if a development study shows
that its lower cost yields a better risk-cost frontier.

## 4. Hypotheses and preregistered success criteria

### H1: Fresh2 risk estimation

A low-capacity clean-free risk model using runtime trajectory and pairwise features can
capture at least `80%` of Fresh2 failures while flagging no more than `20%` of natural
calibration cases.

This gate is deliberately stronger than the rejected B21 disagreement detector.

### H2: Safe NP escalation

A frozen two-sided decision rule—Fresh2 judged risky and NP judged sufficiently
credible—can improve raw good25 over Fresh2 by at least `3` percentage points on a new
natural test panel, with:

- at most `1` percentage point of threshold harms;
- mean raw PSNR no worse than Fresh2 by more than `0.25 dB`;
- mean compute no greater than `2.75` Fresh1 full-run equivalents;
- all selection features available without ground truth.

### H3: Reduced NP budget

A development-selected NP prefix or progressive stopping policy using at most four NP
candidates on average retains at least `90%` of the NP-8 oracle rescue opportunities
observed on the development set.

If this gate fails, use the full NP-8 escalation only for the conditional study rather
than tuning an unstable stopping rule.

### H4: Prospective reproducibility

After the router, thresholds, NP candidate order, and stopping logic are frozen, a
prospective policy-only run on a new untouched panel reproduces the counterfactual
full-candidate-bank evaluation within the declared statistical uncertainty.

### H5: Measurement-seed stability

Risk probabilities and decisions remain stable enough across repeated measurements of
the same development image that no more than `10%` of image pairs cross the final
accept/escalate threshold solely because of measurement-seed scale effects.

If H5 fails, feature normalization and calibration must be redesigned before final
prospective testing.

## 5. Data protocol

The B21/B22 final panel is evidence and an audit set. It must not be used to select B23
features, thresholds, model hyperparameters, NP order, or stopping rules.

### 5.1 Development screening pool

- `600` new FFHQ images, disjoint from B21/B22 official images.
- one locked `sigma_y=0.05` measurement per image.
- run Fresh1 only.
- use development ground truth to label failures and construct an enriched training
  bank.

Purpose: obtain enough natural Fresh1 failures without paying NP-8 cost on all 600.

### 5.2 Enriched development bank

Construct after Fresh1 screening:

- include every Fresh1 bad25 case;
- include an equal-size random sample of Fresh1 good25 controls;
- group all selection and cross-validation by image;
- run second DAPS and the complete frozen NP candidate bank on these cases.

Expected size is approximately 200--260 cases if the Fresh1 failure rate remains near
20%.

This bank is for feature/model development. Its prevalence is artificial and must not
be used to report deployment reliability directly.

### 5.3 Natural calibration panel

- `150` new images, image-disjoint from screening and development;
- one locked measurement per image;
- run complete Fresh2 and complete frozen NP candidate bank;
- add a second independent measurement for a preregistered subset of `50` images.

Use this panel to calibrate probabilities, freeze thresholds, freeze candidate order,
and estimate natural flagged fraction and expected cost.

No model-class changes are allowed after calibration begins.

### 5.4 Full-candidate test panel

- `250` untouched new images;
- one locked measurement per image;
- complete Fresh2 and NP candidate bank are generated for audit, but the router remains
  frozen;
- report both the simulated policy from frozen runtime features and all fixed baselines.

The complete bank permits exact counterfactual audit and oracle-gap analysis.

### 5.5 Prospective policy-only panel

If the full-candidate test gates pass:

- `150` additional untouched images;
- execute only the arms selected by the frozen staged policy;
- do not generate unused candidates;
- compare observed compute and reliability with the prediction from the test panel.

This is the strongest deployment-style evidence and should be the primary final result.

## 6. Runtime feature contract

Every feature must be computable from information available at runtime. Ground truth,
PSNR, oracle identity, or manual visual labels are forbidden inputs.

### 6.1 Measurement features

- measurement energy and robust scale statistics;
- clipped-negative fraction and energy fraction where applicable;
- radial/frequency-band energy ratios;
- dynamic range and concentration statistics;
- operator-normalized quantities defined before any outcome inspection.

### 6.2 Fresh1 trajectory features

- final exact measurement/operator loss;
- selected intermediate loss values at frozen checkpoints;
- loss slope, curvature, and late-stage variance;
- projection/correction norms where available;
- low- and high-frequency residual components;
- reconstruction intensity/color moments;
- trajectory stability summaries.

### 6.3 Fresh2 pair features

- exact-loss values and margin;
- identity/rot180-minimized candidate disagreement;
- low-frequency disagreement;
- chromatic-histogram disagreement;
- edge/high-frequency disagreement;
- trajectory-curve divergence;
- agreement of independent clean-free quality indicators.

### 6.4 NP population features

- frozen residual selector statistics;
- gap between best and second-best residual candidates;
- candidate-to-candidate agreement and clustering;
- stability of the residual winner as candidates are added;
- consistency between full-band and low-frequency residual rankings;
- evidence of orientation/twin disagreement.

### 6.5 Leakage audit

The feature extractor must produce a machine-readable declaration for every column:

- source tensor/file;
- time at which it becomes available;
- whether it depends on ground truth;
- whether it depends on candidates that the simulated policy would not have run.

Any undeclared or post-decision feature blocks the checkpoint.

## 7. Model and decision-rule restrictions

The dataset will contain relatively few true failures. High-capacity deep classifiers
are inappropriate initially.

Authorized model classes:

1. preregistered scalar baselines from past work;
2. regularized logistic regression;
3. shallow monotone or low-depth gradient-boosted trees;
4. calibrated probability models using a held-out calibration partition;
5. conformal or risk-controlling threshold selection if implemented with strict data
   separation.

Unauthorized before a low-capacity baseline is complete:

- large neural networks;
- end-to-end image classifiers on reconstructed images;
- architecture search;
- thresholds selected from the test panel;
- manual rules designed after inspecting test failures.

The primary rule should be interpretable:

```text
escalate_to_np = risk_fresh2 >= tau_risk
replace_with_np = escalate_to_np and confidence_np >= tau_np
```

The default action after NP escalation remains Fresh2 unless the frozen replacement
rule is satisfied. This asymmetric design protects ordinary Fresh2 quality.

## 8. Checkpoint plan

### B23.0 — repository consolidation and protocol freeze

Deliverables:

- human decision on integrating PRs #30--#35 in dependency order or creating an
  explicit reviewed squash/integration branch;
- immutable tag or commit label for the B22 scientific state;
- new B23 branch from the integrated scientific head;
- data exclusion manifest containing every B21/B22 image ID;
- preregistered B23 protocol, metrics, and stop rules;
- no GPU launch yet.

Gate:

- one authoritative source head;
- source snapshot and environments identified;
- no B22 final-panel row permitted in training/tuning tables.

### B23.1 — feature substrate and zero-GPU replay

Deliverables:

- unified feature schema;
- extractors for Fresh1, Fresh2, and NP records;
- leakage declaration;
- deterministic replay on historical outputs;
- missing-feature and schema-version tests;
- one compact analysis archive.

Gate:

- bitwise-stable feature extraction where inputs are unchanged;
- no GT-dependent runtime feature;
- every simulated decision uses only candidates already available at that stage.

### B23.2 — Fresh1 screening run

Deliverables:

- locked 600-image manifest;
- one measurement per image;
- Fresh1 outputs and trajectory summaries;
- natural Fresh1 failure-rate report;
- enriched-bank selection manifest frozen from the screening results.

Stop rule:

- if fewer than `60` Fresh1 failures are obtained, expand screening before building a
  classifier rather than training on an underpowered sample.

### B23.3 — enriched candidate-bank generation

Deliverables:

- second DAPS trajectory for enriched cases;
- complete NP candidate population;
- fixed SITCOM-4S run on a preregistered 25% stratified subset only;
- candidate-arm and rescue-overlap tables;
- NP prefix/oracle budget curve on development data.

SITCOM expansion gate:

Expand SITCOM to the full development/calibration protocol only if it provides either:

- at least one unique good25 rescue not available from Fresh2+NP; or
- a materially better cost-adjusted escalation frontier than NP.

Otherwise retain SITCOM only as a fixed final baseline.

### B23.4 — risk-model development

Deliverables:

- scalar baselines;
- grouped cross-validation results for authorized low-capacity models;
- feature stability across measurement scales;
- failure recall versus flagged-fraction curves;
- no frozen deployment threshold yet.

Gate:

At least one model must exceed the B21 relative-disagreement baseline under grouped
cross-validation and plausibly reach H1. Otherwise terminate adaptive routing before
running a large calibration panel.

### B23.5 — natural calibration and policy freeze

Deliverables:

- 150-image natural calibration bank;
- repeated-measurement subset;
- calibrated risk probabilities;
- frozen risk threshold, NP confidence threshold, NP order, and stopping budget;
- expected mean and tail compute;
- signed policy specification containing code hash and all constants.

Hard gates:

- Fresh2-failure recall at least `0.80`;
- flagged fraction at most `0.20`;
- replacement harms at most `1%`;
- projected mean cost at most `2.75` Fresh1 equivalents;
- no meaningful repeated-measurement instability beyond H5.

If no rule satisfies all gates, B23 adaptive routing is a negative result. Do not relax
the gates after seeing calibration outcomes.

### B23.6 — full-candidate frozen test

Deliverables:

- 250 untouched cases;
- frozen-policy outcomes;
- Fresh1, Fresh2, NP fixed-policy, and relevant SITCOM baselines;
- reliability, PSNR distribution, q05, catastrophic floor, compute, and paired tests;
- cross-method oracle reported only as diagnostic;
- complete failure union and visual atlas.

Primary success gate:

- policy improves Fresh2 raw good25 by at least `3` percentage points;
- threshold harms at most `1` percentage point;
- mean PSNR delta versus Fresh2 at least `-0.25 dB`;
- mean cost at most `2.75` Fresh1 equivalents;
- result survives paired uncertainty analysis.

No model or threshold update is permitted after this checkpoint starts.

### B23.7 — prospective policy-only execution

Authorized only after B23.6 passes.

Deliverables:

- 150 new untouched cases;
- actual conditional execution with unused arms omitted;
- realized versus predicted compute;
- final prospective reliability result;
- operational failure log and no-silent-fallback audit.

This is the deployment-style closeout.

## 9. Metrics

### 9.1 Quality and reliability

- raw good25 count and interval;
- raw bad20 and bad10 counts;
- minimum, q01, q05, q10, median, mean, and upper quantiles;
- ambiguity-aware metrics as auxiliary only;
- paired rescues and harms relative to Fresh2;
- image-level bootstrap intervals;
- exact McNemar tests for threshold changes.

### 9.2 Compute

- actual GPU-seconds per executed arm;
- mean, median, q90, and maximum policy compute;
- full-run equivalents normalized to Fresh1;
- escalation fraction;
- NP candidate count distribution;
- wall-clock and peak-memory accounting.

### 9.3 Router quality

- AUROC and AUPRC are descriptive, not sufficient;
- recall and precision at the frozen threshold;
- flagged fraction;
- calibration error and reliability plot;
- failure rate among accepted cases;
- replacement precision: fraction of replacements that improve PSNR or rescue good25;
- repeated-measurement decision stability.

## 10. Stop rules and negative-result value

Stop B23 adaptive routing if any of the following occurs:

1. no development model materially improves over the rejected B21 detector frontier;
2. natural calibration requires flagging more than 20--25% of cases to capture failures;
3. NP replacement harms cannot be controlled without eliminating most rescues;
4. projected adaptive cost is not lower than always-on NP-8-RS;
5. decisions are unstable across measurement seeds;
6. prospective full-candidate testing fails the frozen success gate.

A negative result remains scientifically useful:

> Candidate complementarity exists, but the available clean-free signals are
> insufficient to exploit it safely and cheaply.

That conclusion would directly motivate either learned priors/confidence estimation or
new candidate-generation mechanisms rather than more threshold mining.

## 11. Secondary track: shared hard-case candidate generation

This track begins only after B23.5 either freezes a viable router or declares routing a
negative result.

### 11.1 Target

Cases like `65003` where Fresh2, SITCOM-4S, NP-8-RS, and their method-specific oracles
all fail. These cannot be fixed by better selection among the current candidates.

### 11.2 Required first step

Perform a focused literature and operator audit before implementation. The audit must
separate:

- phase/orientation symmetry;
- prior support mismatch;
- high-frequency/saturated-content mismatch;
- measurement-likelihood scheduling;
- clipping/preprocessing effects;
- diffusion-model domain limitations.

### 11.3 Authorized intervention classes

Only one class should be tested at a time:

1. multiscale/coarse-to-fine measurement coupling;
2. symmetry-aware candidate representation or canonicalization;
3. alternative prior/checkpoint with better support for complex content;
4. candidate-generation diversification that is not merely another random seed;
5. learned or adaptive measurement schedule trained only on development data.

### 11.4 Gate

An intervention must generate a good candidate on a preregistered hard-case development
set without degrading a matched ordinary-image control set and must then validate on a
new hard-case panel. Success on `65003` alone is not sufficient.

## 12. Claims strategy

### Supported now

- Fresh2 substantially improves DAPS single-run reliability.
- Independent complete trajectories are more useful than the tested same-cost LF and
  shared-prefix alternatives.
- NP supplies complementary rescue basins but is not a strong one-run replacement.
- Solver families occupy a quality/reliability/cost frontier.
- most residual failures are candidate-generation failures.

### Target B23 claim

> A clean-free calibrated cascade allocates expensive complementary diffusion
> trajectories only to risky measurements, improving the catastrophic reliability
> floor at controlled average compute.

### Claims to avoid

- universal robustness beyond FFHQ and `sigma_y=0.05`;
- "measurement residual certifies correctness";
- SOTA without a current literature and benchmark audit;
- near-perfect reliability based on a ground-truth oracle;
- causal claims about LF/NP mechanisms without isolated ablations;
- routing claims tuned on B22.

## 13. Repository and execution policy

Before B23 GPU work:

1. resolve the stacked PR history explicitly; do not silently rewrite
   `b19_solver_integration`;
2. tag or record the B22 scientific head;
3. create a clean B23 worktree under `/egr/research-pac/huang248`;
4. keep caches, environments, and temporary files under research-pac, never `/home`;
5. commit every config, manifest, launcher, validator, and runbook before PAC execution;
6. use smoke gates before any full run;
7. return compact archives for planner analysis;
8. stop on source/operator/measurement incompatibility rather than improvising.

## 14. Immediate next action

The next action is planning and repository consolidation, not a GPU experiment.

Recommended sequence:

1. human review of PRs #30--#35 and explicit integration decision;
2. freeze/tag the integrated B22 head;
3. create the B23 protocol branch and data-exclusion manifest;
4. implement B23.1 feature extraction and leakage audit;
5. run zero-GPU replay on preserved B21/B22 artifacts;
6. only after B23.1 sign-off, lock the 600-image screening manifest and authorize
   Fresh1 screening.

No NP, SITCOM, or large candidate-bank run should start before the feature contract and
data split are frozen.
