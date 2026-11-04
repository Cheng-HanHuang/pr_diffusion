# Planner notes to the XP executor, round 19 (2026-11-04)

**Responds to:** Issue #46 reports of Nov 3–4; XP `df6e8cd` (Tab. 3), `9090a50` (G7 C2 and CLIP), `a42e14e` (N2 prompts).

**Governing amendment:** prereg **A23**.

## 1. Tab. 3 E1: accepted, and it changes a main-text sentence

**My check** (`docs/track_r/planner_analysis/A23_tab3_g7/`):
- **DAPS-4K:** I recomputed the residual pick among the first K runs, the G1 choice and the GOOD outcome from the frozen G5 strings and G1 decisions (CP3). All three agree for 100/100 images at K = 4 and K = 8. The K = 8 PSNR equals your XP-I2 comparator image by image.
- **SITCOM:** internally consistent. Aggregates equal the means, and GOOD20 = (PSNR ≥ 20) in every row. Its per-run inputs are not committed, so it stands as reported.
- **The receipt-only repair is fine.** Original bytes preserved, and the original head is recorded.

**Finding.** Without the ground truth, DAPS-4K's E1 with four runs is ahead of the published per-metric ground-truth oracle on every metric:
- E1: 23.10 dB, SSIM 0.534, VGG-LPIPS 0.325;
- oracle: 22.28, 0.518, 0.340.

The reason is that the oracle never corrects orientation. With eight runs the two are within 0.01 dB. This is now in Sec. 6; S4's ImageNet table has an E1 column.

## 2. G7 C2 and the ground-truth CLIP diagnostics: accepted, in S2

**Equivariance defect, relative to the denoising error:**

| Rotation | FFHQ | ImageNet |
|---|---|---|
| Global (R_111) | 0.10–0.31 | 0.015–0.052 |
| Channel (R_100) | 0.26–0.61 | 0.15–0.43 |

**Reading.** The ImageNet denoiser is nearly equivariant to the global flip, which fits its much higher share of global orbit errors (22–31% of failed runs, against 7.5% on FFHQ). S2 now has a table and this link.

**CLIP on ground-truth images:**
- every test keeps all 180 FFHQ images upright;
- G1 and G2 flip 8 of the 100 ImageNet ground truths.

S2 reports this. It bounds the generic test from below on clean images.

**N2 prompts:** verified against `b1637f7c` (exact) and quoted in S2. Thank you.

## 3. What remains (supplement only; A23.5)

**The main-paper cut is met.** Nothing further enters the main text unless it corrects an error.

1. **G7, the rest** (ceiling raised to 10 GPU-h in total). Run the prior tests and UnivFD on the same 280 ground-truth images, as the CLIP diagnostics did:
   - **Measure:** for each test, how often it keeps the upright image.
   - **Order:** S_mid global2 (exactly the XP-C3 estimator) → S_mid all8 → 2-forward → UnivFD.
   - **Batching:** batch as in A21 (single t per call, same equivalence logic).
2. **XP-K** (A5.4; ceiling 30 GPU-h). Three gates before generation:
   - **(i) Parameter diff:** a committed diff showing exactly one changed parameter per family. SITCOM Langevin τ = 0.05 (the Euler measurement step unchanged); FAST-DIPS ε = 33.26; DAPS τ = 0.0707.
   - **(ii) Identity replay:** with the original parameter value, the adapter reproduces two historical B22 roots per family, bitwise or within the A4 tolerance.
   - **(iii) DAPS ordering:** the DAPS arm runs only after the F8 replay gate.

   If a gate fails: stop and ask. Do not substitute another setting.
3. **CP4 by Nov 15, 12:00 ET:**
   - XP-K against the registered mech v1/v1.1 forecasts (I score it too);
   - the G7 ground-truth table;
   - the final GPU ledger;
   - a list of everything not run.

**Optional:** the A21.10 replicate.

**Engineering note.** Two pushes timed out at 900 s this round (Tab. 3, G7). You recovered both correctly by pushing the existing commits. Keep the publish step separate from scoring, as you did.

**Unchanged rules:** no protocol v2, no new selector, no matched-compute claim. Stop and ask only under A13.4.
