# B23 review start here

Date: 2026-07-30

Status: review entrypoint for the post-B22 research plan. This file does not authorize implementation or GPU execution.

Repository: `Cheng-HanHuang/pr_diffusion`

Planning branch: `codex/post-b22-reliability-plan`

Planning PR: `#36` — **Plan fixed-budget modular solver synthesis after B22**

## 1. Current scientific status

- B21 method development is complete.
- B22 fixed-baseline evaluation is complete and frozen.
- B23 is a proposed next research program and remains under review.
- No B23 implementation or GPU experiment is authorized by the planning documents alone.

## 2. Precedence order

Read the following in this exact order. Later items provide evidence and historical context; they do not override earlier planning files.

1. `docs/planning/2026-07-30_b23_modular_fixed_budget_amendment.md`
   - **Authoritative B23 plan where any conflict exists.**
   - Makes fixed-budget within-trajectory modular composition the primary direction.
   - Defines B1/B2 compute caps, module-replay gates, schedule grammar, hypotheses, checkpoints, and stop rules.

2. `docs/planning/2026-07-29_post_b22_reliability_research_plan.md`
   - Background evidence synthesis and the earlier adaptive whole-solver portfolio plan.
   - Its data-separation, leakage, negative-result, repository-safety, and historical-evidence sections remain active unless superseded by item 1.

3. `docs/b22/b22_3_scientific_closeout.md`
   - Final B22 quantitative and visual scientific conclusion.
   - Establishes the quality–reliability–cost frontier and the Fresh2/NP/SITCOM complementarity evidence.

4. `docs/b22/b22_3_visual_failure_taxonomy.csv`
   - Manual per-image failure taxonomy for the 16-image B22 failure/complementarity union.

5. `docs/checkpoints/2026-07-27_b21_to_b22/01_PROJECT_CHECKPOINT.md`
   - B21 scientific checkpoint, adopted Fresh2 policy, negative-results map, repository stack, and retained artifacts.

6. `docs/b21/b21_registry.md`
   - Authoritative registry of adopted, rejected, superseded, and diagnostic B21 methods/policies.

7. `docs/b21/b21_11_fresh2_final_benchmark.md`
   - Prospective 100-image Fresh2 benchmark.

8. `docs/b21/b21_7_fresh2_vs_lf_decision.md`
   - Equal-cost evidence that a fresh DAPS restart beats LF as the default second arm.

9. `docs/b21/b21_10_detector_screen_decision.md`
   - Negative result for the tested clean-free scalar/pairwise fallback triggers.

## 3. Review modes

### Fast strategic review

Read items 1–3 above. Evaluate:

- whether modular solver synthesis is the right primary contribution;
- whether whole-solver fallback is correctly placed as a secondary benchmark/safety layer;
- whether the B1/B2 compute caps prevent best-of-k scaling from masquerading as algorithmic progress;
- whether B23.0–B23.1 are sufficient before any schedule screen.

### Full scientific review

Read items 1–9. Evaluate:

- whether each proposed module corresponds faithfully to the parent algorithm's actual mathematical operation;
- whether the plan repeats any rejected B21 idea under a new name;
- whether hypotheses H1–H5 and stop rules are strong enough;
- whether the data split prevents B22 leakage;
- whether the target claim is supported by the proposed experiments.

### Implementation-feasibility review

Read items 1, 2, 5, and `docs/planning/01_B23_EXECUTOR_START_HERE.md`. Evaluate:

- whether Fresh1, LF, NP-1, and SITCOM-1 can share a common state interface without changing their semantics;
- whether exact replay and RNG/state-conversion checks are technically feasible;
- whether the compute ledger can compare heterogeneous parent solvers fairly;
- where module boundaries should be drawn.

## 4. Questions reviewers should explicitly answer

1. Is the primary scientific contribution correctly defined as fixed-budget operator/module composition rather than fallback among completed solvers?
2. Are `B1 <= 1.10 FRE` and `B2 <= 2.10 FRE` appropriate main-claim budgets?
3. Is the proposed common state expressive enough without falsely homogenizing NP or SITCOM?
4. Are exact parent replay and compute matching sufficient gates before hybridization?
5. Is the initial schedule grammar small and mechanistically interpretable enough?
6. Are any schedules or controller ideas still too broad or under-specified?
7. Are the success and stop criteria sufficiently resistant to post-hoc continuation?
8. Should any direction be removed, demoted, or added before implementation begins?

## 5. Review output requested

A useful review should return:

- `ACCEPT`, `ACCEPT WITH REQUIRED CHANGES`, or `REVISE BEFORE IMPLEMENTATION`;
- numbered required changes;
- optional improvements separated from blockers;
- any concern about scientific leakage, compute mismatch, or parent-solver semantic mismatch;
- a final statement on whether B23.0 and B23.1 may begin.

## 6. Gate

```text
B22 scientific state: COMPLETE AND FROZEN
B23 planning: UNDER REVIEW
B23.0 repository/protocol work: NOT YET AUTHORIZED
B23.1 module extraction/replay: NOT YET AUTHORIZED
B23.2 schedule experiments: NOT AUTHORIZED
Large GPU panels: NOT AUTHORIZED
```
