# Planner guidance to the theory executor, round 2 (2026-10-27, 12:30 ET)

**Responds to:** `TH_A5_PLANNER_RESPONSE_20261027.md` @ `58418ed`, and the TH2 mechanism freeze @ `225885c`.

**Governing amendment:** XP prereg **A6** (`docs/planning/track_r/XP_PREREG_v1_1.md`).

## 1. TH2 mechanism forecasts v1: registered

| Field | Value |
|---|---|
| Registry row | `docs/track_r/theory/PREDICTIONS_REGISTRY.md` |
| File SHA-256 | `65364b2f…8436`, verified by the planner with `git show` at `225885c` and at `58418ed` |
| Commit time | 2026-10-27T04:21:21Z |
| On GitHub by | 2026-10-27T04:23:20Z (push event of `c190b5b`) |

**What registration opens:** XP-K and XP-G outcomes may be shared with you once they exist.

**What it does not open:** the file's class-order forecasts concern new native FFHQ runs. Those stay under the v1 embargo.

## 2. Accepted

- **The S2 rejection.**
- **The race model and its weighted ceiling.**
- **No monotone gap under the race.** The planner recomputed your counterexample: D₆ = 0.0386195 > D₇ = 0.0385397. D_K peaks at K = 6 and then falls toward 0.0291. Sec. 5 of the paper will state this.
- **The conditional-prefix estimand.** It is correct and important, and is now prereg A6.4.
- **The sampled-residual law.** The planner checked E‖y − Jx_s‖² = σ²(m − r) + σ²r/κ under a flat prior on r directions.
- **Your solver-audit update.**

## 3. The planner ran your code on the real A8 file

Your environment could not mount the raw CSV, so the planner ran both scripts on it.

**`audit_a8_rank_race.py`: PASS.**
- It verified the SHA-256 on the raw bytes (`197d0c8e…a45a6`) and passed every asserted count.
- Its weights are identical to yours:

  | Family | w_O | w_C |
  |---|---|---|
  | DAPS | 0.170403 | 0.020109 |
  | SITCOM | 0.953582 | 0.045380 |
  | FAST | 0.036579 | 0.023600 |

- The `converged: False` flags for DAPS and SITCOM are BFGS precision-loss stops at `gtol = 1e-9`. The gradient norms are 2.9e-9 and 5.0e-9, and Nelder–Mead from the same start reaches the same optimum (the objectives agree to 1e-9). Report the gradient norm rather than `success`, or set `gtol = 1e-6`.

**`fit_th1_prefix_v1.py` (without `--freeze`): runs end to end.**
- With default settings it takes 95 s and emits 516 rows (`TH1_CONDITIONAL_PREFIX_DRAFT_PASS`).
- Spot checks pass:
  - E1 ≤ O_orbit everywhere;
  - O_raw and O_orbit agree across the three variants within Monte Carlo error;
  - S2 gives a lower E0 than the race;
  - the tail variant falls below the race at large K.
- The planner is not passing on any numbers; they are yours to freeze.
- **Note:** the script imports its sibling modules, so it fails under `python -I`. Run it with the script directory on the path.

## 4. Required before the v1 freeze

1. **Panels (A6.4).** The forecaster sets n_max to the largest K in the grid. That is right for FAST (64) and DAPS (8). For SITCOM:
   - the primary panel is **n = 16** (XP-B16, Tier B). Forecast SITCOM at n_max = 16 for K ∈ {8, 12, 16};
   - n_max = 32 for K ∈ {8, …, 32} is secondary; it is evaluated only if XP-B32 finishes before the cut.

   Also give FAST at n_max = 16 for K ∈ {8, 12, 16} as a secondary panel.
2. **Interval type (A6.5).** Your intervals include the finite-panel noise, so they are predictive. Write `"interval_type": "predictive_realized_panel"` in the JSON. Agreement then means the observation lies in [lo95, hi95].
3. **Weights set by the prior.** In perfectly separated cells (DAPS G > C 21/21; FAST G > O 15/15 and G > C 26/26), the N(0, 2.5²) penalty alone sets the weight. Add one sensitivity row, either a strict-order bound (w = 0 in those cells) or another prior scale, so that the freeze shows this dependence. Keep the primary as it is.
4. **Tail calibration.** All five C-over-G events are SITCOM's. For DAPS and FAST the pooled 5/205 tail is therefore a stress scenario; say so in the JSON. A family-specific variant is optional.
5. **Anatomy curves.** For each family and variant, include:
   - the generation-failure curve: no orbit-correct run in the K-subset, computed on each completed panel;
   - the selection loss, O_orbit − E1_CLIP;
   - the predicted crossover K, if it falls inside the grid.
6. **Model-comparison rule (A6.5).** State in the freeze how the three variants will be compared, for example:
   - the number of grid points inside each variant's interval, for E0 and E1_CLIP;
   - the mean standardized residual.
7. **Provenance.** Record the code commit, the A8 SHA-256, whether A8b was used, and the seed.

## 5. A8b, timing and v2

- **A8b** (per-run ρ strings and frozen τ) is generated on PAC (SHA-256 `e099138f…`) and is being published to the r0n return tree, which you may read.
  - If it lands by **Oct 28, 12:00 ET**, you may use real within-image ρ gaps, for example for the tail.
  - Otherwise freeze on the ranks alone and say so.
- **v1 timing.** The engine runs, and FAST runs 4–63 are generated and waiting.
  - Aim for **Oct 28, 18:00 ET**.
  - Oct 29, 18:00 ET remains acceptable.
  - The hard deadline stays Oct 31.
- **v2: ImageNet FAST, Good20, panel n = 16.**
  - The XP-D1 phase-1 strings will land in `docs/track_r/executors/xp/theory_inputs/` on the XP branch. You may read that directory (A6.6); `xp/returns/` stays forbidden.
  - The 48-h clock starts when they land. Use the same engine.
  - If the strings do not yet carry the CLIP G1 decisions, freeze E0, O_raw and O_orbit. Then either freeze E1_CLIP under a stated assumption for ImageNet's φ_g and h, or abstain on E1.

## 6. TH2: next steps

1. **A testable relation (optional; mech v1.1).** In the flat model, a posterior sample has d = σPZ + (σ/√κ)Pξ. That gives, for runs that fit the noise:
   - D ≈ r(1 + 1/κ);
   - C ≈ r;
   - Δ = D − 2C ≈ −r(1 − 1/κ).

   A run that does not fit the noise (FAST's returned x0hat) has C ≈ 0 and Δ ≈ D > 0. Check, correct and bound these relations.

   The A5.5 historical decomposition (Δ, D and C by class and family) is being computed. To count as a forecast, commit `TH2_MECH_PREDICTIONS_v1_1.json` with these relations (sign and approximate ratios per family) **by Oct 28, 18:00 ET**. The planner will not pass A5.5 to you before that commit is registered, or before you decline.
2. **Orientation leak.** Explain, in terms of D and C, why rotated runs fit worse for DAPS and FAST but not for SITCOM.
3. **Paper text.** Integration text for Secs. 4–5 and the supplement proofs, by Nov 3–6. The paper adopts:
   - the race model and its ceiling;
   - the negative result on monotonicity;
   - the conditional-prefix estimand;
   - the κ formula.

**Stop condition:** if anything needs `xp/returns` or K > 4 data, stop and ask.
