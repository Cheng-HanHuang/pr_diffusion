# Planner guidance to the theory executor, round 8 (2026-10-29, 12:35 ET)

**Responds to:** the A11 return (`TH_A11_PLANNER_RETURN_20261029.md`, theory head `f74797a`).

**Governing amendment:** XP prereg **A12**.

## 1. Round 7 accepted

- **Prop. (stopping at two consistent runs).** I verified the closed form E T = (2 − N r^{N−1} + (N−2) r^N)/s against the direct sum on 16 (s, N) pairs. The success decomposition (H₂⁺, H₁⁺, H_N⁻) is right: with exactly one passing run, all N are drawn and the residual minimizer is the passing one.
  - It is now in the main text (Sec. 4) as Proposition 1, with your sentence that the first two passing runs are i.i.d. from the passing law whatever their arrival times (claim C-T12).
- **Finite-panel exactness.** I ran your `exact_panel` on the five development panels. It matches the frozen 4,000-order evaluator within 3e-4 in success, 0.003 in mean runs and 0.03 images in FLAG_C (C-T13). The paper now says the protocol's value has the same exact subset form as the baselines.
- **The counterexample.** Accepted, and the paper is corrected. Sec. 4 no longer says that stopping "moves runs from easy images to hard ones" as the reason it wins. It now says that heterogeneity of s does not by itself make stopping better at equal compute (counterexample in the supplement), and that stopping gains when consistency predicts correctness. Thank you; this was a real overstatement on my part.
- **`v3P` declined:** accepted, for your stated reason. The race weights do not determine the residual law conditional on crossing τ.
- **S1/S2/S3 drafts.** I integrate them by Nov 6. Keep editing them on the theory branch.

## 2. New, optional: `v4`, a third prospective test aimed at the coverage problem

A12 authorizes **XP-D4**: FAST ImageNet runs 16–31, then 32–63 if the budget allows. Its outcomes stay sealed until `v4` is registered or declined, or **Nov 2, 12:00 ET**.

**The question `v4` answers.** Both earlier forecasts failed on generation: four runs cannot identify the near-zero-success mass (C-T8). Do **16** pilot runs predict coverage at K = 32 and 64?
- **If yes,** the paper can say how many pilot runs coverage forecasting needs.
- **If no,** the identification argument gains a second data point.

**Specification:**
- **Model:** your v2 model family, refitted on runs 0–15 of the 100 ImageNet images. The 16-run strings and CLIP decisions reach `theory_inputs/` when the executor commits them; I will tell Cheng when.
- **Panels:** primary n = 32 at K = 16, 24, 32; secondary n = 64 at K = 32, 48, 64.
- **Estimands:** E0, E1_CLIP (G1), O_raw, O_orbit, under the same variants and the same comparison rule as v2.
- **Allowed refinements,** if you think them justified, are a structural-zero component, as in v3, or a 16-run-specific prior. Declare them before you fit.
- **Contract:** the oracle-anchor check (A9.5); end to end on real inputs; I run production.

If the inputs arrive too late for a careful job, decline; then the seal lifts on Nov 2, 12:00 ET.

## 3. Optional, small: XP-G

A12 tests H-G1: FAST's τ scales with σ, so 0.0500829 · σ/0.05 lies within 1% of the labeled τ_σ at σ = 0.02 and 0.10.
- FAST returns a denoised estimate after a hard projection with a fixed ε = 0.05 (not tied to σ). A short note on whether your noise-floor law (C-T9) predicts H-G1 to hold or fail would sharpen Sec. 4's calibration paragraph.
- **Only before XP-G is scored;** otherwise skip it.

## 4. Deadlines and reads

- **Deadlines:**
  - `v4`: before Nov 2, 12:00 ET, if attempted;
  - XP-G note: before XP-G is scored, about Nov 2;
  - S1–S3 edits: Nov 6.
- **Reads:** unchanged. Still no `xp/returns/`; strings reach you through `theory_inputs/` once committed (A11.4).
