# Theory guidance, round 23 (planner, 2026-11-06, morning)

**Context.** A third AI review (weak reject; `docs/track_r/reviews/AI_REVIEW_3_ASTRA_20261106.md`) found real overclaims in the main text. The planner fixed them (triage: `AI_REVIEW_3_TRIAGE_20261106.md`). Round 21 (naming pass) and round 22 (five sentences) stand. Please also do the items below, by Nov 12, 12:00 ET.

## 1. Wording to check (main text)

1. **Prop. 1 framing (Sec. 4):** "The proposition is a decomposition, not a comparison with fixed best-of-K; heterogeneity of s alone does not make stopping better (supplement, counterexample). Empirically, stopping gained when consistency predicted correctness."
   - Is the counterexample in S1 stated so that this sentence is exact?
   - If you can state a clean sufficient condition for a gain at equal mean runs, with explicit assumptions, propose it. Do not let it delay round 21.
2. **Notation (Sec. 5):** κ = q + c replaces s in Eq. (2) and the race ceiling. s stays the pass probability of Prop. 1. Please align S1 and S3 with this (κ for class-based, s for threshold-based).
3. **Noise identity (Sec. 4):** "the noise energy ‖η‖² cancels, and a cross term with the difference of the two fits remains. Since the runs depend on η, this motivates the comparison but does not guarantee it." Correct?

## 2. Exploratory refit (reviewer Q7)

The race lost test 1 because the shared generation component overestimated coverage.
- **Refit:** add a near-zero-success component (a spike, or a two-component Beta mixture) to the per-image mixture. Fit it post hoc on the same prefixes as tests 1, 5 and 6 (FFHQ FAST runs 0–3; ImageNet FAST runs 0–15; ImageNet′ runs 0–3).
- **Score:** the selected-run and oracle forecasts, with coverage, mean absolute error and the 95% interval score (Gneiting–Raftery), as in supplement Tab. 16 and `planner_analysis/A27_review3_checks/code/forecast_interval_scores.py`.
- **Label:** explicitly post hoc. It cannot replace the registered results.
- **Outcomes:** whether it helps or not, it goes to S3 as one paragraph and one small table.

## 3. Optional: a forecast for XP-NF

If Cheng gives GO for XP-NF (prereg A26: frozen protocol on 100 new FFHQ validation images for DAPS-2K and DAPS-4K), you may register a population-transfer forecast of the protocol gain and of E1 vs O_orbit before any XP-NF run is generated. Register it with a hash, as for v5. The planner's own forecast is +1 to +5 points.

## Unchanged

- No reads of `xp/returns`.
- Only the planner edits `PREDICTIONS_REGISTRY.md`.
- Double-blind.

## Planner's view

The review accepted the theory as sound. It objected only where the text drew stronger conclusions than the statements give. The fixes narrow the claims; none removes a result.
