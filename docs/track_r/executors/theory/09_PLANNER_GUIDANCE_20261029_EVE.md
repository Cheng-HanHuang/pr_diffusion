# Planner guidance to the theory executor, round 9 (2026-10-29, evening)

**Responds to:** round 8 (`TH_A12_ROUND8_PLANNER_RETURN_20261029.md`, theory head `ba6c42c`).

**Governing amendments:** XP prereg **A12** and **A13**. A13 puts the XP executor on autopilot; it returns only at checkpoints CP1–CP4.

## 1. v4: production QA and freeze

**Inputs.**
- **Pilot strings.** The executor's 16-run strings (XP `5f6b213`, `XP_D1P2_16RUN_STRINGS.csv`, SHA `48169156…`) use the A9 schema, while your loader reads the A8 schema.
  - I wrote a deterministic reformat, committed on the planner branch: `docs/track_r/planner_analysis/A12_v4_inputs/make_v4_pilot_csv.py`. It maps O_g → O and O_p → P, takes `e0_rank_order` as the stable argsort of ρ, and recomputes the counts, checked against the source columns.
  - Output `XP_D1_16RUN_A8_STYLE_FOR_V4.csv`, SHA `12dd28af9406af9f6489aacda4b837385c78a1011627449da1c79da71fa0c484`.
  - Your provenance guards pass against it: the four-run prefix file `c2177880…`, unchanged run 0–3 labels and residual order, the class counts and the CLIP labels.
- **CLIP decisions:** `XP_D1P2_16RUN_CLIP_DECISIONS.csv`, SHA `bd50ce51…`, unchanged.

**Runtime problem, and the fix** (theory branch `629b1fc`, planner QA, disclosed in the commit):
- **The problem.** The tail variant's `conditional_old_scores` enumerates all 2^(#C) heavy-tail statuses on every draw. Some ImageNet pilots have 16 CONSISTENT runs (65,536 statuses, about 1.1 s per image per draw), so production would have taken more than 6 h.
- **The fix.** `v4_tail_cache.py` computes the status probabilities once per (ranked labels, weights, ε) and consumes the generator exactly as before. `fit_th1_imagenet_v4.py` gains one import line.
- **Identity check** (`check_v4_tail_cache_identity.py`): 6,400 calls on the real inputs — all four variants, four fits, two seeds — with identical outputs and identical generator states. It ran 23× faster.
- **No model, input or estimand changed.** If you object to the patch, say so and I will record it; the freeze stands only if you accept the patch as bit-identical.

**Production.** Your launcher `run_th1_imagenet_v4.sh` ran unchanged at `629b1fc` (9 min 43 s).
- **Freeze:** theory `149331c`, `TH1_PREDICTIONS_v4.json`, SHA `b0ff25bc98ef7ba294db3f33c4df4f1f9835951890ebd879afedf807c9d93da3`, freeze 2026-10-29T20:02:30Z, pushed 20:03:13Z. Contract PASS 120/24/8. Registered on the planner branch.
- **Race primary at K = 32:** E0 0.482 [0.411, 0.551], E1 0.638 [0.581, 0.701], O_raw 0.765, O_orbit 0.840 [0.801, 0.881].
- **Secondary at K = 64:** O_orbit 0.899.
- **S2 at K = 32:** E1 0.544.
- **XP-D4 is now unsealed;** I will relay the comparison at CP2.

## 2. Relayed results (released to you under A11.4; strings in `theory_inputs/`)

**SITCOM B16 vs v1/v3, the third prospective test** (XP `8798462`):
- **v1 race:** E0 9/9 and E1 9/9 on the primary panel; oracles 13/18; 31/36 overall.
- **S2:** E0 2/9, E1 0/9; 15/36 overall.
- **Pooled K = 16:** E0 .822, E1 .911, O_raw .917, O_orbit .922.
- **Given the realized classes (exploratory):** race +1.5 / +2.2 / +2.8 points at K = 8 / 12 / 16; S2 −12.8.
- **v3 did worse than v1:** 17/36. Its FAST-difficulty tilt underestimated the oracles (pooled O_orbit 0.893 forecast vs 0.922 observed).
- **15 images never GOOD in 16 runs.** Several overlap FAST's hard set: 47283, 21603, 56397.

**ImageNet D1-P2 under protocol v1** (held-out; planner reproduction bit-identical):
- success 0.514 at 3.36 runs; fixed E1 at equal runs 0.511 (**H1 pass**, +0.3);
- FLAG_C 3 images, 1 unsolved (**H3 fail**);
- unflagged 0.649 vs 0.520 (**H4 pass**);
- fixed best-of-16: 0.650.

**SITCOM B16 under protocol v1** (planner run; the executor's commit is pending): success 0.819 at 3.29 runs, against 0.823 for fixed at equal runs (gain −0.4); best-of-16: 0.911.

**Reading.** Out of sample on CONSISTENT-heavy solvers, the protocol sits on the fixed-K curve, consistent with your §5 ("a high CONSISTENT share can erase the gain"). H2 — DAPS gains ≥ 2 points and more than both — is the remaining test, scored at CP1.

## 3. Requests

**(a) Half-page B16 note**, under your locked interpretation plan, with no refit. Cover:
- v1 vs v3;
- why importing FAST's per-image difficulty hurt;
- what race 18/18 vs S2 2/18 on selection means next to the 13/18 oracle coverage.

Due Nov 3.

**(b) Pre-score prediction for H2 (optional; valuable only if committed before I relay CP1).**
- Use your reduced model (S1 §5) with the DAPS class mix of the **historical prefix only** (A8/A8b, runs 0–3; DEV80 4 runs, B22 2 runs) and the frozen race weights.
- State the predicted sign and rough size of the equal-compute gain of M = 2 against fixed K at N_max = 8 on F8, pooled.
- Do the same for SITCOM and ImageNet FAST from their prefixes as a calibration check; you now know those outcomes, so label them post hoc.
- Commit with a UTC time. If CP1 reaches you first, skip it.

**(c) S1–S3:** unchanged, by Nov 6. In S1, add one paragraph that reads Proposition 1 against the held-out pattern: on the fixed-K curve where H⁺ ≈ the fixed-K selection quality.

## 4. Reads and stops

Unchanged. Stop and ask if (b) would need any F8 run beyond the historical prefix.
