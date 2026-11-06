# Planner notes to the XP executor, round 22 (2026-11-06)

**Responds to:** no new return. Cheng ran two AI reviews of the Nov 5 draft; the triage is in `docs/track_r/reviews/AI_REVIEWS_20261105_TRIAGE.md`.

**Governing amendment:** prereg **A25**. It adds CPU-only supplement checks. No gate, budget or hypothesis changes.

## 1. Priority is unchanged

1. **XP-K,** under the three A23.5b gates. **Stop and ask if a gate fails.**
2. **CP4** by Nov 15, 12:00 ET.

The tasks below come after CP4. You may run them earlier only on idle CPU, and only if they never delay XP-K or CP4.

## 2. New CPU tasks (A25.3)

- **X1, per-run metrics.** For every run of these panels, read the stored reconstruction and write PSNR (RGB8) for all eight orbit members R_b, plus SSIM and VGG-LPIPS for b = 000 and for the best member:
  - FAST-DIPS FFHQ 64;
  - DAPS-2K FFHQ (historical runs plus F8);
  - DAPS-4K FFHQ and ImageNet (E′ + G4, D2 + G5);
  - FAST-DIPS ImageNet 64 (D1 + D4);
  - ImageNet′ 64 (XP-N);
  - the σ = 0.02 and 0.10 panels.

  Key the rows by (panel, image_id, run). Include `rho_signed`, so rows join the committed strings. No paths, no images.
- **X2, compute table.** From existing receipts only, give per solver and dataset:
  - nominal denoiser calls per run;
  - seconds per run, with the GPU state (clean or co-tenant) and the receipt it comes from.
- **X3 (low priority).** For the FFHQ panels, recompute A(x̂) from stored reconstructions. Report, per run:
  - the residual in three radial frequency bands of the 384² canvas;
  - the ℓ1 residual.

**Data rule (A25.4).** Per-run metric tables go to Cheng as a planner ZIP, as in §7, unless Cheng approves committing them (decision D-A25). Commit only receipts and SHA256SUMS.

## 3. Names (round 21, unchanged)

DAPS-4K, DAPS-2K, cohort A/B, ImageNet′, PC-DAPS sample.

## Planner's view

- **Current results.** Both AI reviews are positive (R1: 7.1 on the ICLR scale, accept). Their common asks were:
  - robustness to the GOOD threshold;
  - independence between runs.

  Both are answered with planner checks on committed data. X1 extends the threshold check to every panel.
- **Plan.**
  - XP-K and CP4 first.
  - Then X1 and X2; the planner runs the θ sweep and the M sweep.
  - The supplement freezes Nov 22.
- **Image of the publication.** Unchanged: a preregistered audit with a deployable, ground-truth-free protocol. The supplement now answers the scope and robustness questions a reviewer is likely to raise.
