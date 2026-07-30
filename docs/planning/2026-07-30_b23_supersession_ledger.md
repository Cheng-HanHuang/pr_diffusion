# B23 planning supersession and research-synthesis ledger

Date: 2026-07-30

Status: authoritative companion to
`docs/planning/2026-07-30_b23_final_research_plan.md`.

This ledger prevents an executor from combining incompatible parts of the July 29 portfolio plan,
the July 30 modular amendment, and the later review reports.

## 1. Authority order

1. `docs/planning/2026-07-30_b23_final_research_plan.md`
2. this supersession ledger
3. frozen B22/B21 scientific evidence
4. historical pre-B21 NP/SITCOM reports and source
5. `2026-07-30_b23_modular_fixed_budget_amendment.md` — historical planning rationale only
6. `2026-07-29_post_b22_reliability_research_plan.md` — historical Track-B rationale only
7. external AI-assisted reviews — advisory only

No lower item overrides a higher item.

## 2. July 30 modular amendment

| Amendment section | Status | Final-plan replacement or interpretation |
|---|---|---|
| §0 Revised executive decision | adopted with revision | Final plan §§0, 5: compatibility-gated modular synthesis remains primary |
| §1 Scientific hypothesis | adopted with caution | Final plan §§3–6; parent operations are conjectured donors until compatibility passes |
| §2 Project hierarchy | adopted | Final plan §0 |
| §3 Hard compute contract | superseded | Final plan §7 defines raw ledger, calibrated work-FRE, time-FRE, and candidate caps |
| §4 Common state | superseded | Final plan §5 uses typed native states and explicit adapters |
| §4 Module classes | superseded | Final plan §§3, 5; no generic NP gradient or assumed SITCOM polish |
| §4 Exact replay | superseded/refined | Final plan §8 uses native replay plus bitwise/tolerance tracks |
| §5 Schedule grammar | superseded | Final plan §11; prior concrete schedule list is not authorized |
| §6 H1–H5 | superseded/refined | Final plan §6 separates H0 feasibility, H1/H2 claims, H3 mechanism, H4 optional adaptation, H5 prospective reproduction |
| §7 Development protocol | superseded | Final plan §§9, 12 |
| §8 Whole-solver fallback | retained as conditional background | Final plan Track B; requires a new amendment |
| §9 Dead directions and stop rules | retained and expanded | Final plan §§4, 13 |
| §10 Claims | retained and narrowed | Final plan §14 |
| §11 Immediate action | superseded | Final plan §§12, 16 |

## 3. July 29 adaptive-portfolio plan

| July 29 section | Status | Final-plan replacement or interpretation |
|---|---|---|
| §0 Executive decision | superseded as primary | Whole-solver cascade is Track B only |
| §1 Completed-repo evidence | active | Final plan §2 |
| §2 Direction classification | active unless expanded | Final plan §§2, 4 |
| §3 Recommended main project | superseded | Final plan §§0, 5 |
| §4 Router H1–H5 | historical Track-B only | No implementation authority |
| §5 Data protocol | superseded | Final plan §9 uses a broader pre-B23 exposure manifest and new split registry |
| §6 Runtime features | historical Track-B only | May inform a future amendment; not B23 Track-A inputs |
| §7 Models/rules | historical Track-B only | Adaptive work is conditional under final-plan H4 |
| §8 Checkpoint plan | superseded | Final plan §12 |
| §9 Metrics | partly active | Final plan §10 controls |
| §10 Router stop rules | active only for a future Track-B study | New amendment required |
| §11 Shared hard cases | retained as Track C | Not authorized by B23 |
| §12 Claims | partly active | Final plan §14 |
| §13 Repository policy | active | Final plan §15 |
| §14 Immediate action | superseded | Final plan §§12, 16 |

## 4. Decisions on the AI-assisted reports

### Adopted

- one reviewable canonical specification;
- spec-first, budget-first, replay-first staging;
- a machine-readable raw compute ledger;
- an operational FRE definition rather than an undefined label;
- bitwise-versus-tolerance replay modes;
- correction/supersession and leakage manifests;
- operational descriptions of NP, LF, and SITCOM;
- fixed schedules before any adaptive controller;
- predeclared phase-ambiguity evaluation conventions.

### Adopted with modification

1. **Component-wise FRE.**
   The reports proposed the maximum of normalized NFE, measurement, inner-optimization, and GPU-time
   ratios. The final plan instead uses a calibrated work cost plus measured time because raw
   heterogeneous counts are not directly commensurate.

2. **Common state.**
   The reports proposed one shared trajectory state. The final plan uses typed native states and
   explicit adapters because DAPS, NP, and SITCOM do not automatically share representation or
   timing semantics.

3. **Schedule count.**
   The reports proposed ten, then six-to-eight schedules. The final first screen allows at most six
   after compatibility sign-off.

4. **Deterministic flags.**
   The reports suggested forcing deterministic algorithms. The final plan records and tests the
   flags but does not change the frozen parent solely to obtain a hash.

### Rejected or deferred

- exact calendar dates and GPU-hour forecasts before PAC inventory;
- direct NP-prefix-to-SITCOM-suffix as an initial schedule;
- treating LF-early-to-DAPS-late as novel rather than the LF-v1 parent;
- adding DiffStateGrad as a fifth initial donor;
- treating recent linear-restoration frequency-continuation evidence as validation for nonlinear
  phase retrieval;
- making ambiguity-aligned PSNR primary and thereby changing the established raw-orientation
  deployment metric;
- authorizing B23.1 GPU work before B23.0 sign-off.

## 5. Newly recovered historical constraints

The reports did not fully incorporate the repository's earlier Branch-B evidence:

- NP-to-SITCOM handoff B3-B8 was noncompetitive;
- `3S+1NP` oracle complementarity did not yield an executable policy;
- existing hybrid/handoff code is historical prototype code;
- lower residual could select worse reconstructions.

These findings are binding through final-plan §4.2.

## 6. Correction protocol

If a later reviewer finds an error:

1. add a dated entry here;
2. identify the exact final-plan section;
3. label the correction as factual, protocol, semantic, statistical, or execution;
4. state whether prior results are invalidated;
5. update the final plan in the same reviewed commit;
6. never silently reinterpret an older plan.

## 7. Current correction entries

| Date | Type | Correction | Effect |
|---|---|---|---|
| 2026-07-30 | access | The first external review could not retrieve the authoritative plans; this was a tooling limitation, not evidence that the committed files were absent | access blocker resolved in the final synthesis |
| 2026-07-30 | compute | `FRE` was named but not operationally defined | replaced by final-plan §7 |
| 2026-07-30 | replay | GPU replay was phrased too close to universal exactness | replaced by final-plan §8 |
| 2026-07-30 | semantics | One common state could falsely homogenize NP and SITCOM | replaced by typed states/adapters in final-plan §5 |
| 2026-07-30 | history | Initial amendment omitted failed NP-to-SITCOM and same-budget executable hybrid evidence | added to final-plan §4.2 |
| 2026-07-30 | novelty | `LF early -> DAPS late` is already LF-v1 behavior | retained as a parent control, not a novel schedule |
| 2026-07-30 | scope | DiffStateGrad was suggested without B22-specific evidence | deferred beyond initial B23 |
