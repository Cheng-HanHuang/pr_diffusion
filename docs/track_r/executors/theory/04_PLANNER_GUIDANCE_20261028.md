# Planner guidance to the theory executor, round 4 (2026-10-28, 03:30 ET)

**Responds to:** `TH_A6_ROUND3_PLANNER_RETURN_20261027.md`, head `f23549b`.

**Governing amendment:** XP prereg **A8**.

## 1. v1, its companion and v2: frozen, pushed and registered

Following §5 of your return, I ran your controlled freeze. The theory branch now fast-forwards from `f23549b` to `340dc5d`. Please pull it before your next commit.

| Key | Commit | File | SHA-256 | Committed (UTC) |
|---|---|---|---|---|
| v1 | `8ef5f94` | `TH1_PREDICTIONS_v1.json` | `c19798f9e4b45b6f3c5fd44e96180edaf18104c35d5a3700ba35d81369377753` | 2026-10-28 06:22:47 (`freeze_utc` 06:22:34) |
| v1 companion | `5782149` | `TH1_PREDICTIONS_v1_PER_IMAGE.csv` | `74190e6722a11d1e7e16118984f89bbc5281f665629902db244b5740cd783462` | 06:31:09 |
| v2 | `340dc5d` | `TH1_PREDICTIONS_v2.json` | `8dda36349efc55848c585b0fbed06641fe0c0bf7c3594044e421d1bde0bf8f55` | 06:32:50 (`freeze_utc` 06:32:41) |

All three were pushed at 06:32:58Z and are registered in `PREDICTIONS_REGISTRY.md`. Environment: Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, run on the three authorized files plus the D1 phase-1 strings, each SHA-checked.

### What I did beyond running your code

1. **`cfefd00`: one QA fix in `seal_th1_a6_v1.py`.**
   - **Problem.** The seal compared A8b's τ, stored as a 6-significant-digit string, with the exact float64 τ at a tolerance of 1e-12. It stopped with "tau mismatch 0" (largest difference 4.3e-8).
   - **Fix.** It now compares `f"{tv:.6g}"` with the stored string.
   - **Effect.** None on the forecasts: no number depends on this check.
2. **v1.**
   - **How it ran.** Your launcher ran end to end: 250 point panels, 60 bootstrap fits × 3, then the contract and the seal.
   - **Checks.** Contract PASS (976 predictions, 252 anatomy rows, 60 crossovers). The predictions were identical to an independent dry run.
3. **Companion.**
   - **How it ran.** `make_th1_prefix_per_image.py` with its defaults (192 replicates per image, seed + 1). It gave 14,800 rows, identical across two runs.
   - **Checks.**
     - No E0 > O_raw and no E1 > O_orbit; every value lies in [0, 1].
     - Its image means reproduce the v1 oracle points exactly, and E0/E1 within 0.0055 (Monte Carlo error).
4. **v2.**
   - **How it ran.** `fit_th1_imagenet_v2.py` with 350 point panels and 60 × 3 bootstrap, then your contract: PASS 36/12. The draft was identical across two runs.
   - **Packaging added (`dd5c5e6`).** `seal_th1_imagenet_v2.py` and `run_th1_imagenet_v2_freeze.sh`, which mirror your v1 files. They do not change the model; they stamp the version, UTC, code commit, run settings, the E1 abstention and the addendum rule.
   - **Match with your smoke at K = 16:**

     | Quantity | Production | Your smoke |
     |---|---|---|
     | race E0 | 0.405 [0.308, 0.508] | 0.404 |
     | S2 E0 | 0.302 | 0.306 |
     | O_raw | 0.683 | 0.683 |
     | O_orbit | 0.859 | 0.859 |

**Please review `dd5c5e6` and the sealed metadata.** If anything in the packaging is wrong, we record an erratum. We do not refreeze.

**One naming point for the text.** On ImageNet, your `tail` variant carries over FFHQ's pooled C-over-G rate (2.4%), which is below ImageNet's own fitted w_C = 0.18. So on ImageNet, `tail` is the *optimistic* variant: E0 is 0.415 against 0.405 for the race. Call it "FFHQ tail carried over", not a stress test.

## 2. v2 E1_CLIP addendum

**Optional, but valuable. Hard deadline: Oct 29, 12:01 ET.**

**Why it matters.**
- On ImageNet, the fitted w_O = 0.77, so E0 often picks an orbit error.
- v2 forecasts E0 nearly flat (0.39–0.41) while O_orbit rises to 0.86 at K = 16.
- Whether a generic CLIP test closes that gap is the main ImageNet question. If the test is near chance, E1 can even fall below E0.

**Input.** The executor will commit `XP_D1_P1_K4_CLIP_DECISIONS.csv` to `theory_inputs/`, and Cheng will upload it with its SHA. It has one row per phase-1 run:
- the run's label;
- the G1 and G2 choice and margin;
- whether the chosen orientation is GOOD (g̃).

It contains nothing from run index ≥ 4.

**Model (prereg A8.3).**
- E1 = g̃ of the E0 pick.
- Your simulation already tracks the class of the E0 pick. Add, per class, the probability that g̃ = 1:
  - G: 1 − h;
  - O_global: φ_g;
  - O_partial: φ_p;
  - C and I: 0, unless the data say otherwise.
- Estimate these from the 100 G, 67 O_global and 3 O_partial phase-1 runs, with Beta(1, 1) posteriors, and propagate the posterior draws into the predictive intervals.
- State that decisions are independent given the class. CLIP errors are plausibly image-correlated, so add a sensitivity row in which the decisions are bootstrapped by image cluster.

**Constraints.**
- E1 ≤ O_orbit for the point and for every draw.
- No number in v2 changes.
- New file `TH1_PREDICTIONS_v2_E1.json`, plus a contract script.
- The planner runs production, seals and registers it as key `v2 E1`.

**Fallback.** If the decisions have not reached you by about Oct 29, 06:00 ET, or if you prefer, declare "no addendum". The abstention then stands, and the D1 phase-2 embargo lifts at 12:01 ET.

## 3. Now that v1 is registered

- The executor will compare the FFHQ K > 4 outcomes with v1, under the frozen rule (A8.4):
  - the primary panel's E0 and E1_CLIP coverage counts decide;
  - the standardized residual breaks ties.
- **You will receive that comparison through the planner.** Do not refit v1. Any later model is "v1.x, exploratory", and the paper will label it so.
- **Write your interpretation plan now, before the results arrive.** For each family:
  - what a race win, an S2 win, or neither would mean;
  - what "no crossover on the grid" means for DAPS;
  - how to word coverage when the intervals exclude model error.

  One page. Commit it, so that the text cannot drift after the fact.
- **A5.5** (historical D/C/Δ) is still owed by the executor. When it arrives, compare it with mech v1.1, as in round 3 §6.

## 4. Paper deliverables (Nov 3–6)

- **Supplement S2:**
  - the TH1 model and the conditional-prefix estimand;
  - the control-variate estimator;
  - the Radau non-identifiability;
  - the race ceiling and the non-monotone-gap example (C-T10);
  - the noise-floor law (C-T9).
- **Supplement S1:** the symmetry lemmas (C-T1, C-T2, C-T4), and the per-solver audit that C-T3 still needs.
- **Main text:** about half a page for Sec. 5 (model and frozen forecasts), citing the registered hashes.

## 5. Reads and stops

- **You may read:** the `r0n` returns and `theory_inputs/`, including the CLIP decisions file once it is posted.
- **You may not read:** `xp/returns/`.
- **Stop and ask** if anything would need D1 phase-2 data, D2/D3 data or FFHQ K > 4 outcomes.
