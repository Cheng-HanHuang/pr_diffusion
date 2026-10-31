# Planner notes to the XP executor, round 11 (2026-10-31, 01:20 ET)

**Responds to:** CP2 (`ef1cf10`) and the CP3 stop/ask (`020fc68`).

**Governing amendment:** prereg **A15**. Autopilot (A13) continues.

## 1. CP2 accepted, with one correction to its reading

I reproduced both XP-D4 protocol outputs exactly (max diff 0.0). Under the A12.4 hypotheses, as frozen:
- **H1 FAILS** at N_MAX = 32 (−0.0179) and at N_MAX = 64 (−0.0344);
- **H3 FAILS** at 32 (1/3) and is not evaluable at 64 (one flag);
- **H4 PASSES** at both (+0.131 and +0.136).

Your README did not state the hypothesis outcomes. Please add them, as listed above, in the CP3 return digest.

**Mechanism (my decomposition, descriptive).** 4–6 images with fewer than 5 passing runs absorb about 43% of all runs; the other images stop after 2.6–3.0 runs.

**v4.** Selection 18/18 (race) against 1/18 (S2). Oracles: race O_orbit 0/6 and O_raw 2/6; coverage plateaus at 0.75–0.78 against a forecast of 0.84–0.90.

## 2. CP3 disposition: your option 1 (A15.2–A15.4)

- **G2, G4 and G5 are accepted as frozen-confirmatory:**
  - H-G1 FAIL ×2;
  - H-G2 PASS ×2, with no savings at σ = 0.02;
  - G4 H1 PASS (+3.93);
  - G5 H1 PASS (+0.10).
- **XP-I.** The matched-NFE rule in A12.4 was my design error: the protocol's minimum of two runs already exceeds one bag. The rule is **withdrawn and recorded as unresolved**, not satisfied. Report two descriptive analyses instead:
  - **(a) Same particles:** PC-DAPS native mean over particles, against E0 and E1 over the same particles. PSNR, SSIM, VGG-LPIPS and GOOD25; paired over the 180 images, with a paired bootstrap 95% CI on the E1 − native difference.
  - **(b) Compute-labeled comparison:** one PC-DAPS bag (E1) against DAPS-4K run 0, DAPS-4K fixed E1 at K = 1 and 2, and the frozen protocol's pooled E′8 point. Give each policy's compute per image as:
    - **nominal NFE.** State DAPS-4K's resolved annealing × ODE steps and its Langevin steps **from the E′ run receipts**. A2 specifies 400 × 10; your CP3 note says 400 × 5 — which is right? For PC-DAPS, give the logged lower bound.
    - **clean wall-clock seconds.** Extend G6 by one PC-DAPS bag on 3 ids, on a clean GPU.

  **No matched-compute claim** may appear in the return.
- **The CP3 return is authorized.** Contents:
  1. a digest of at most 20 lines, with every hypothesis outcome (both H-G1 failures, σ = 0.02's zero savings, G5's marginal gain, the XP-I limitation);
  2. per-panel protocol JSONs, plus strings and orientation decisions for G2, G4, G5 and the XP-I particles, copied to `theory_inputs/`;
  3. the A8.6 numbers: the ImageNet GT/y identity result; DAPS-4K per-metric best of 4 and of 8 (PSNR / SSIM / VGG-LPIPS) on FFHQ and ImageNet, next to DAPS Table 1 (FFHQ 30.63 / .851 / .139; ImageNet 25.78 / .743 / .254) and PC-DAPS's rerun (29.66 / .770 / .177; 20.98 / .461 / .387).

  I will reproduce every protocol output.

## 3. Queue changes

- **Tail reorder (A15.5).** T2 XP-I2 (PC-DAPS on ImageNet, one bag, analyses (a) and (b) against D2-8) runs **before** T1 XP-K. It is main-paper material only if it is scored by **Nov 5, 12:00 ET**.
- **G6:** add one PC-DAPS bag (3 ids, clean GPU).
- **Still owed, CPU, needed for the main paper by Nov 4, 12:00 ET:**
  1. **Unsolved-images PNGs:**
     - FAST FFHQ: the 7 never-GOOD images in 64 runs;
     - FAST ImageNet XP-D4: the 22 images never orbit-GOOD in 64 — show up to 8 of them, chosen by SHA-256 of the image id, not by looks.

     For each image: ground truth, lowest-residual run, best orbit-oracle run, with PSNR. Tiles of 256 px, files of at most 2 MB, no ids.
  2. **A9.2(d):** FAST FFHQ face-CLIP E1 against v1. It is test 1's missing E1 column; the 11,520 decisions are needed.

## 4. Unchanged

- No protocol v2.
- **CP4** (Tier C) by Nov 15.
- Stop and ask only for the A13.4 conditions.
