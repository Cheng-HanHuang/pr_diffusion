# Planner notes to the XP executor, round 12 (2026-10-31, 14:25 ET)

**Responds to:** the CP3 return (`1981dea`, `returns/XP_A17_CP3_20261031T173255Z/`).

**Governing amendment:** prereg **A16** (planner `21babef`). Autopilot (A13) continues.

## 1. CP3 accepted and closed

- **Reproduction.** All seven per-cohort protocol outputs (G2 at σ = 0.02 and 0.10, B22 and DEV80; G4 B22 and DEV80; G5) were rerun from `c598176` on your strings and decisions, with every SHA256SUMS line verified. The maximum absolute difference is **0** in every field.
- **XP-I.** GOOD25 shares reproduced exactly from `G3_STRINGS.csv` and `G3_CLIP.csv` (0.7903 / 0.8278 / 0.8444). The paired GOOD25 interval matches within Monte Carlo error.
- **Two small notes, no re-return needed:**
  1. `XP_A17_CP3_A15_DESCRIPTIVE.json` still says `clean_wall_clock_seconds: PENDING…`. Your timing receipt and compute table supersede it, and A16.1 records that.
  2. At σ = 0.02, FLAG_C fired on 180/180 images (13 unsolved). It is in the paper.
- **Outcomes stand as in A15.2.** Your digest was complete and within 20 lines. Thank you.

## 2. What A16 changes for you

The other A16 analyses (decomposition, budget frontier, confidence-conditioned stopping, cross-fitted label-free τ, gain intervals) are planner CPU work on your committed files. Nothing for you to do.

**New work, in this priority order** (GPU and CPU in parallel as usual):

| # | Item | Type | Due (ET) |
|---|---|---|---|
| 1 | **Nov 4 CPU items (unchanged):** the unsolved-images PNGs (FAST FFHQ: the 7 never-GOOD images; XP-D4: up to 8 of the 22, chosen by SHA-256 of the image id), and A9.2(d), FAST FFHQ face-CLIP E1 | CPU | Nov 4, 12:00 |
| 2 | **XP-I per-particle metric table:** `image_id, particle, label, PSNR, SSIM, VGG_LPIPS, rho, face_choice, face_margin` (evaluation values only; no tensors or paths), copied to `returns/` | CPU | Nov 4, 12:00 |
| 3 | **XP-N** (A16.5), **only after Cheng's GO** (see §3) | GPU, cap 20 GPU-h | runs 0–15 scored by Nov 5, 12:00 |
| 4 | **XP-I2** (unchanged; A15.5) | GPU | main paper if scored by Nov 5, 12:00 |
| 5 | **XP-C3** (A16.4): the XP-C prior orientation test (2-forward S_mid, global2) on every XP-D4 candidate (6,400) and every G5 candidate (800). Rows `image_id, run, label, prior_choice, prior_margin, prior_chosen_good`, copied to `theory_inputs/`. First free GPU after XP-N generation starts; never delays XP-I2 | GPU, ≤ 2 GPU-h inside G7 | Nov 5, 12:00 |
| 6 | **E1 metric columns for Tab. 3** (ImageNet, 4 runs): PSNR / SSIM / VGG-LPIPS of the E1 (G1) pick for DAPS-4K (D2) and SITCOM (D3), via the A7 ZIP route to Cheng (not committed) | CPU | Nov 6 |
| 7 | **XP-K** (tail T1, unchanged) | GPU | Tier C |
| 8 | **G7 remainder** (XP-C, XP-C2) and **CP4** | GPU/CPU | Nov 15 |

## 3. XP-N in practice

**What Cheng decides.** Cheng says GO or no-GO for XP-N and confirms where ImageNet validation images sit on PAC. External downloads are his: the files are untrusted, so read them with `python -I` from their own directory. If either is missing on **Nov 2, 12:00 ET**, record `XP_N_LAPSED` in your next receipt and continue the queue. Do not wait.

**Once GO:**
1. **Images.**
   - Use ILSVRC2012 validation `00048900`–`00048999`.
   - Assert that none is among the 100 D1 ids (D1 is `00049000`–`00049099`; check against your D1 manifest, not by assumption).
   - Write a manifest of id, SHA-256 of the source file and SHA-256 of the 256² GT tensor.
2. **Measurements.** Use D1's exact preprocessing, operator, σ = 0.05 and signed y, with noise seeds from D1's rule under the family tag `N`. Run the GT/y identity checks you ran for D1.
3. **Replay.** A 2-root replay of D1 on the GPU class you will use.
4. **Generation.**
   - FAST-DIPS ImageNet, D1's configuration and checkpoint, seeds `XP|FAST|imagenet_n|<uid>|<r>`.
   - Runs 0–15 on all 100 images first, then runs 16–63.
   - Co-tenancy as in A10.1.
5. **Scoring.** Labels as D1 (GT for evaluation only), G1 decisions for every candidate, and XP-C3 prior decisions once they exist.
6. **Seal.**
   - **Commit runs 0–3 first:** strings and G1 decisions, copied to `theory_inputs/`.
   - **Keep runs 4–63 local** until the registry on the planner branch lists `v5` as registered or declined, or until **Nov 4, 12:00 ET**. Check the registry yourself; no planner message is needed.
7. **Protocol.** Run v1 at `c598176`, unchanged, with τ = 0.0500829 and G1:
   - runs 0–15 with `--run-range 0:16 --nmax 16`;
   - runs 0–63 with `--nmax 64`;
   - secondary: `--tau 0.050434426327793` (the label-free τ from D1 runs 0–3) at N_MAX = 16.
8. **CP-N return (≤ 20 lines).**
   - N-H1a, N-H1b and N-H4, each held or failed;
   - the share of images with no orbit-GOOD run at K = 16 and K = 64;
   - each planner forecast in `planner_analysis/A16_cp3_and_exploratory/out/XP_N_PLANNER_FORECAST.json`, inside or outside its interval;
   - strings and decisions copied to `theory_inputs/`.

   The planner reproduces it.

**Cap.** 20 GPU-h (about 14 expected), separate from the A12 and tail caps. The 400 GPU-h project cap binds.

## 4. Unchanged

- No protocol v2.
- No matched-compute claim.
- Stop and ask only for the A13.4 conditions. A data-governance question about XP-N counts as one: anything that touches FFHQ, the locked holdout or a bundled dataset.
- CP4 by Nov 15.
