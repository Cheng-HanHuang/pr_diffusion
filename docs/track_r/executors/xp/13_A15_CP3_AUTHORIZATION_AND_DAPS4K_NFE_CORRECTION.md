# A15: CP3 disposition and correction of DAPS-4K NFE

**Authority:** planner `817f78a88376cce3d67dacd6746b181490ab37ae`, `docs/track_r/executors/xp/11_PLANNER_NOTES_20261031.md`, prereg A15.

CP2 is **accepted** (bit-identical protocol reproduction). Digest must include H1 fails at 32/64 (−0.0179, −0.0344), H3 fails at 32/not evaluable 64, H4 passes (+0.131/+0.136). Registered v4 selection 18/18 versus S2 1/18, with race O_orbit 0/6.

CP3 is **authorized for descriptive XP-I** under A15, subject to sanitized return and planner reexecution. The original A12.4 matched-NFE requirement was infeasible by design: frozen protocol M=2 cannot match one PC-DAPS bag. The original test is withdrawn **as unresolved**, not satisfied; this does not change any frozen G2/G4/G5 results.

## CORRECTION: previous XP memo used DAPS 400×5 erroneously

The `12_CP3_MATCHED_NFE_BLOCKER_20261030.md` note and local A16 NFE ledger refer to **400 annealing × 5 ODE = 2,000** as a DAPS-4K run. **That is wrong.** There are two distinct DAPS methods in this project:

1. **Published phase-retrieval DAPS-4K (E′ G4; D2-8 G5):** `scripts/track_r/xp_t_daps4k_prepare_cpu.py:DAPS4K` locks **400 annealing transitions × 10 Euler ODE denoiser calls = 4,000 nominal NFE per single run**, with **100 Langevin steps per transition**. `scripts/track_r/xp_t_daps4k_worker_gpu.py` copies `settings` to `solver_config` in every individual `XP_T_DAPS4K_ROOT_COMPLETE.json`. This is checked on every G4/G5 root by `xp_a17_cp3_a15_descriptive_cpu.py`, which refuses inconsistent receipts. The frozen M=2 protocol has minimum 8,000 nominal DAPS NFE.
2. **Legacy F8 and historical G6 clean timing DAPS:** `scripts/track_r/xp_a12_g6_daps_gpu.py` explicitly checks **400 × 5 = 2,000 nominal NFE**, and its receipt calls itself `DAPS400x5_native_F8`. Its timings cannot stand for G4/D2 DAPS-4K.

**PC-DAPS own phase-retrieval native bag:** 4 particles × (130 stages×1 Euler + 70 stages×5 Euler) = **1,920 nominal** denoiser calls per bag, plus partially unlogged pruning prior-consistency calls. The prior A16 ledger's **1,928.4 mean lower bound** remains valid as a PC-DAPS bound, but **its DAPS comparator column is obsolete** and MUST NOT be reused.

Therefore a PC-DAPS bag versus DAPS-4K run-0 comparison is **descriptive unequal compute** (approximately 1,920-plus versus 4,000 nominal), not near-matched. Neither the native-quality difference nor GT-free CLIP gain licenses a matched-NFE conclusion.

## A15 work authorized

- **XP-I (a):** paired 180-image bootstrap with 20,000 seeded image-level resamples, E1 minus native mean of four particle metrics; report PSNR, SSIM, VGG-LPIPS and GOOD25 with 95% percentile intervals.
- **XP-I (b):** one bag's E1 vs DAPS-4K run 0 and fixed E1 K=1/2, alongside frozen G4 E′8 M=2 protocol. Label each with nominal NFE and clean, matched-hardware sampling seconds; no matched-compute claim.
- `xp_a17_cp3_clean_timing_cpu.py` extends the clean timing lane to **three same-signed-y FFHQ IDs** on one physical clean GPU; measures native PC bag and genuine DAPS-4K (400×10). It does **not** substitute legacy G6.
- CP3 digest ≤20 lines, A8.6 oracle tables and ImageNet GT/y audit; per-panel strings/CLIP decisions for G2/G4/G5/G3 into authorized `theory_inputs/` mirror.
- **Tail order:** XP-I2 before XP-K, ImageNet XP-I2 only if scored by Nov 5 12:00 ET. **CPU debt by Nov 4 12:00 ET:** FAST FFHQ 7 never-GOOD PNG panels, ImageNet XP-D4 ≤8 of 22 orbit-unsolved selected by image-id SHA256; FAST FFHQ face-CLIP E1 11,520 decisions against v1. No protocol v2. CP4 due Nov15.

**Freeze discipline:** This note supersedes the previous NFE statements, but preserves the earlier records and its error for provenance. No solver settings or preregistered rules changed.
