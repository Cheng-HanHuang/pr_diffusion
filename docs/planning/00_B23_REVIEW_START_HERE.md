# B23 review start here

Date: 2026-07-30

Status: final review entrypoint. This file does not authorize implementation or GPU execution.

Repository: `Cheng-HanHuang/pr_diffusion`

Planning branch: `codex/post-b22-reliability-plan`

Planning PR: `#36`

## 1. Current state

- B21 method development is complete.
- B22 fixed-baseline evaluation is complete and frozen.
- The split July 29/July 30 planning documents have been reconciled into one canonical final plan.
- Historical NP-to-SITCOM and same-budget NP/SITCOM negative results are now binding.
- B23 implementation remains subject to explicit user authorization.

## 2. Required reading and precedence

Read in this order:

1. `docs/planning/2026-07-30_b23_final_research_plan.md`
   - single authoritative scientific plan;
   - typed parent states and compatibility gates;
   - H0-H5, compute contract, data protocol, stages, and stop rules.
2. `docs/planning/2026-07-30_b23_supersession_ledger.md`
   - maps the older plans and AI-assisted recommendations to adopted, modified, deferred, or
     superseded status.
3. `docs/b22/b22_3_scientific_closeout.md`
4. `docs/b22/b22_3_visual_failure_taxonomy.csv`
5. `docs/checkpoints/2026-07-27_b21_to_b22/01_PROJECT_CHECKPOINT.md`
6. `docs/b21/b21_registry.md`
7. `docs/b21/b21_11_fresh2_final_benchmark.md`
8. `docs/b21/b21_7_fresh2_vs_lf_decision.md`
9. `docs/b21/b21_10_detector_screen_decision.md`
10. `docs/current_experiment_plan.md`
11. `docs/branch_B_fixed_budget_population_selector.md`
12. `docs/planning/01_B23_EXECUTOR_START_HERE.md`
13. `docs/planning/02_B23_PAC_EXECUTION_AND_RETURN_PROTOCOL.md`

Historical only:

- `docs/planning/2026-07-30_b23_modular_fixed_budget_amendment.md`
- `docs/planning/2026-07-29_post_b22_reliability_research_plan.md`

They provide rationale but do not govern execution.

## 3. What materially changed after review

1. The plan no longer assumes all parents must fit one universal state.
2. Native parent replay is separated from donor extraction.
3. A parent may remain `BASELINE-ONLY` without invalidating other donors.
4. Cross-family Track A requires at least one faithful NP or SITCOM donor.
5. Direct NP-to-SITCOM handoff is excluded from the first screen because it already failed.
6. LF-early-to-DAPS-late is recognized as the LF-v1 control, not a new schedule.
7. FRE is now a raw ledger plus calibrated work-FRE and measured time-FRE.
8. GPU replay is bitwise only when the native parent is bitwise; otherwise it is
   tolerance-qualified against a frozen native repeatability envelope.
9. The leakage manifest covers all known pre-B23 exposure, not only the B21/B22 final panels.
10. Exact B23.2 schedules are deliberately deferred to a separately reviewed preregistration after
    B23.1 compatibility evidence.

## 4. Hypotheses at a glance

| Hypothesis | Question | Falsifier |
|---|---|---|
| H0 | Can parents replay, and can at least one NP/SITCOM operation become a faithful donor? | no cross-family donor passes |
| H1 | Can one fixed one-candidate schedule beat Fresh1 at B1? | final practical/statistical gates fail |
| H2 | Can a fixed two-candidate-max policy beat Fresh2 at B2? | tail gain vanishes or quality/cost gate fails |
| H3 | Is gain due to the claimed module rather than compute/candidate scaling? | matched ablations remove the gain |
| H4 | Can limited adaptation beat the best fixed schedule? | optional; requires a later amendment |
| H5 | Does the frozen policy reproduce prospectively without unused branches? | audit/policy-only results or costs disagree |

## 5. Reviewer questions

1. Is compatibility-gated operator composition the correct primary contribution?
2. Does typed native state avoid false semantic homogenization?
3. Are Fresh1, LF, NP, and SITCOM described faithfully?
4. Does the historical ledger prevent repeated NP-to-SITCOM, LF, branching, detector, and
   best-of-k work?
5. Are work-FRE, time-FRE, raw counts, and candidate caps sufficient to prevent compute laundering?
6. Is the replay/RNG policy both rigorous and attainable?
7. Is pre-B23 exposure excluded broadly enough?
8. Are H1/H2 practical and statistical gates defensible?
9. Is it correct to postpone exact schedules until donor compatibility is known?
10. Are B23.0, B23.1, and B23.2 separated by sufficiently explicit authorization gates?

## 6. Requested verdict

Return exactly one:

- `ACCEPT`
- `ACCEPT WITH REQUIRED CHANGES`
- `REVISE BEFORE IMPLEMENTATION`

Then provide:

1. numbered required changes;
2. optional improvements separately;
3. concerns about leakage, compute, statistics, or parent semantics;
4. whether B23.0 may begin;
5. whether B23.1 GPU replay must wait for B23.0 sign-off;
6. confirmation that B23.2 remains separately gated.

## 7. Current gate

```text
B22: COMPLETE AND FROZEN
B23 final plan: READY FOR ACCEPTANCE DECISION
B23.0: NOT AUTHORIZED UNTIL EXPLICIT USER APPROVAL
B23.1 GPU replay: NOT AUTHORIZED UNTIL B23.0 SIGN-OFF
B23.2+: NOT AUTHORIZED
Large GPU panels: NOT AUTHORIZED
```
