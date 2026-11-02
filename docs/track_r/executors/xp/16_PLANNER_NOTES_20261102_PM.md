# Planner notes to the XP executor, round 16 (2026-11-02, afternoon)

**Responds to:** the XP-C3 A13.4 definition gate (`a27ef76`) and the Nov 4 CPU process logs (PAC, relayed by Cheng).

**Governing amendment:** prereg **A20**.

## 1. XP-C3: use option B, S_mid global2 (A20)

- **The ambiguity was mine.** A16.4 and A19 merged two tests from the v1.1 XP-C row. Good call to stop instead of guessing.
- **Decision.** S_mid global2 is the **sole** prior test, and nothing else is run as a secondary.
  - It is the lower-variance estimate of the prior's ordering, which is what Sec. 3 argues for.
  - It is the `mid` estimator your frozen R0 scorer already defines.
  - The decision uses no data; no XP-C3 outcome exists.
- **Exact definition** (A20.3):
  - **For each candidate x and p ∈ {000, 111}:** L_p = mean over t ∈ {300, 400, 500, 600} and j ∈ {0, 1, 2, 3} of the per-pixel mean of (ε̂(√ᾱ_t R_p x + √(1−ᾱ_t) R_p ε_tj, t) − R_p ε_tj)².
  - **Noise:** ε_tj from `prior_seed(uid, t, j)`, rotated with the image (paired).
  - **Schedule:** `frozen_alpha_schedule()`.
  - **Denoiser:** the D1 FAST-DIPS ImageNet checkpoint, for all three panels.
  - **uid** = `XPC3|<panel>|<image_id>|<run>`.
  - **`prior_margin`** = L_000 − L_111. **`prior_choice`** = 111 if the margin > 1e-7, else 000.
  - **Columns:** `image_id, run, label, prior_choice, prior_margin, prior_chosen_good`, with the panel named in the filename and the receipt.
- **Budget** (A20.4):
  - 32 forward passes per candidate, so the ceiling rises to **8 GPU-h** inside G7.
  - Smoke first and project the cost.
  - If the projection exceeds 8, score XP-N (6,400), then G5 (800), then XP-D4 (6,400), and stop at the cap with complete panels only.
- **Freeze, then score.** Push the code, the uid and seed rule, the checkpoint SHA-256 and the source manifest **before scoring any candidate**.
- **Return** to `theory_inputs/` and `returns/`. **Due Nov 5, 12:00 ET** for the main text; I compute E1-prior against E1-CLIP.

## 2. Nov 4 CPU items: please publish the outputs

Your logs show the work finished:
- 7/7 FAST FFHQ panels;
- 8/8 XP-D4 panels;
- 180 images and 11,520 face-CLIP decisions (CSV `3595676c…`).

I need the files to use them. Please commit to `returns/`:
1. **The 15 PNGs.** Each holds ground truth, the lowest-residual run and the best orbit-oracle run, with PSNR, at 256 px; at most 2 MB each; no ids drawn in the images. The receipt maps file to image id and records each SHA-256 and the SHA-256 selection rule for XP-D4.
2. **The face-CLIP decision CSV** (11,520 rows: `image_id, run, label, f_choice, f_margin, f_chosen_good, cohort`). Copy it to `theory_inputs/` too.
3. **The A9.2(d) outputs:** FAST FFHQ E1 (face-CLIP) exact U-statistic curves per cohort and pooled, and the comparison with v1's registered E1_CLIP forecasts (coverage per variant). This fills test 1's E1 column.

I reproduce all of it, then build Fig. 3 from the PNGs.

## 3. Unchanged

- XP-I2 continues. Do not change its HEAD or launch a duplicate supervisor.
- **Then:** Tab. 3 E1 columns (A7 ZIP, Nov 6), XP-K, the rest of G7, and CP4 (Nov 15).
- No protocol v2, no new selector, no matched-compute claim.
- Stop and ask only for the A13.4 conditions.
