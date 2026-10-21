# FAST-DIPS mechanics and prospective Noise Picking integration
Date: 2026-10-21. Status: source-backed plan; CPU residual-selector replay code prepared; NO new GPU inference or GPU authorization.

## Current evidence

A1.5 FAST-DIPS at K4 on 8 already-exposed B22 FFHQ images: 31.525134 dB raw RGB8 GT-best-of-four; DAPS 31.517951; NP-LF 30.388019; NP-S2 30.221876; SITCOM 27.438794. FAST sampling four roots 14.851145s/image, four full Python process walls after imports 32.439574s/image. These timing scopes differ from older A1.2 candidate subprocess wall times.

M0 CPU mechanism audit passed for all 32 roots. All 8 fixed signed noisy observations are provably infeasible for exact native global epsilon=0.05 Euclidean measurement ball (mean nonnegative-FFT lower bound 19.085433). The observed per-image minimum over four reconstructed full-observation residuals is approximately 33.07-33.21; K4 best-of-four PSNR ranges 28.32-34.47. Importantly, the minimum-residual reconstruction may be DIFFERENT from the GT oracle PSNR winner; no within-image selection relationship can be asserted from aggregate minima alone.

## Source comparison that governs the experiment

Original source: scripts/pr_external_difffpr_np_benchmark.py pick_noise_oversampled lines 311-404, noise_picking_reconstruct_oversampled lines 407-476; scripts/b22/run_b22_1_np_smoke.py lines 66-74; scripts/track_a/a1_1b_native_one_each.py lines 173-188. Frozen A1.2 NP has 1000 steps, 5 soft candidates early and one late, low-frequency magnitude candidate ranking (NP-LF) or ranking plus previous-image regularizer lambda .01 pre-projection (NP-S2). Late hard projection is low-frequency Gerchberg-Saxton Fourier magnitude replacement.

CRITICAL CONFOUND: A1.2 NP runner CLIPS negative signed observations to zero in memory before ranking/projection; FAST native algorithm uses ORIGINAL signed observations. First proposed NP hybrid must preserve original NP clipped observation for scoring AND new correction. Signed-vs-clipped experiments later require a separate labeled factorial control.

FAST native source: ququlza/FAST-DIPS pinned a9758b38906b7c7950fa04491f37f6634948db26 main.py. One U-Net denoise per diffusion transition, 150 total, outside a short image-space measurement update. K=2 ADMM initialization first x-gradient is exactly zero in exact arithmetic, then first auxiliary projection + dual update makes second x-update nontrivial. Pixel x-update uses autograd VJP and finite-difference forward probe, analytical local step, up to 8 backtracking halvings. The last transition returns x0hat, not x0y. This is already published FAST-DIPS machinery and cannot itself be claimed as ours.

## Explicit effective-correction formula for selected NP image

Given NP-selected denoiser output p, observation y, full magnitude operator A, native rho=200, epsilon=.05:
- r = A(p)-y
- alpha = min(1,epsilon/||r||2)
- v1 = y+alpha*r ; u1 = A(p)-v1
- b2 = v1-u1 = 2*v1-A(p)
- Start at x=p; make just ONE effective image update consisting of S=5 analytical-step/backtracked iterations minimizing
  ||x-p||2^2/(2 gamma)+rho ||A(x)-b2||2^2/2.
- Never call the U-Net inside this correction. Run torch.enable_grad on detached p even if NP outer loop is @torch.no_grad. Preserve originally chosen noise branch eps_prev.

This algebra matches FAST's K=2 initialization in exact arithmetic if the first zero-gradient image-update is skipped. Source equivalence requires direct measured validation before treating it as a software optimization. A plausible VP-to-noise mapping for NP's next timestep is gamma=(1-alpha_bar[t_next])/alpha_bar[t_next]; NOT identical to FAST's EDM noise grid and must be frozen as an explicit hybrid modeling choice.

## Sequential scientific gates

M1.5 CPU-only candidate selector falsification (READY, NO GPU). Use saved M0 per-root CSV, original K1/K2/K4 oracle summaries. Counterfactual compare exact same candidate sets selected by unavailable GT oracle versus lowest full-measurement residual (a GT-free offline selector). Report selected replicate, PSNR regret, Good25 rescue/harm, total fully charged K runtime, residual contrast. This directly tests whether the measured residual can select FAST's correct basin in practice. Does not read GT tensors, PNG, reconstruction floats or GPU.

M1 controlled native FAST mechanism experiment (PROTOCOL ONLY, GPU NOT AUTHORIZED). Choose exposed 69073 rep0 (success), 67174 rep0 (failure), optionally predeclare 66631 rep0 second failure. Compare exact original native replay versus skipping FIRST S=5 x-update per noise level while preserving initial auxiliary projection + dual update; confirm max absolute float discrepancy, PSNR, CUDA timing, forward operator/VJP calls. Capture x0hat and x0y on same final original trajectory by hooking recording, not by changing earlier sampling. Compare raw PSNR and signed measurement residual RMS. K=1 is a PRIOR-ONLY negative control under native K=2 initialization, not a one-effective-correction method. Use new versioned source/output plus CPU tests, Track B priority and an explicit issue41 GPU lease before running.

M2 NP + FAST hybrid (CONDITIONAL ON M1). Add correction AFTER candidate selection returns (x_prev, eps_prev), BEFORE x_prev feeds the next diffusion step; not inside the five-way candidate evaluation. Freeze timestep windows independent of GT. Pre-register four arms under identical seed, checkpoint, clip preprocessing, 1000-step NP scheduler:
  A original NP unchanged;
  B original NP plus single FAST-style effective post-selection correction at fixed late-step locations;
  C original NP plus budget-matched ordinary forward-model residual-gradient correction (negative control for whether projection/dual matters);
  D original NP with the new correction INSTEAD OF original late low-frequency Gerchberg-Saxton hard projection.
Track numerical correction convergence, raw oriented RGB8 PSNR, Good25/Good30/Bad20, residual RMS, 180-degree auxiliary PSNR, U-Net calls, FFTs, VJPs, GPU events, cold subprocess wall and peak memory. Baseline replay must match frozen outputs. Test already-exposed contrasting images as engineering probe, NOT generalization. A large fresh cohort needs planner/data lease; B24 confirmation305 is not authorized.

M3 prospectively compare full candidate quality--compute frontier. FAST-DIPS uses ~150 UNet calls/root vs NP 1000 steps with 5 early candidate evaluations, so even a free small correction is unlikely to close the wall-time gap. A useful NP hybrid paper needs demonstrable complementary effect on correct-basin reliability at equal denoiser and wall budgets, not merely applying published FAST to old NP and gaining PSNR.

## Provenance and preservation

Original A15 GPU execution HEAD d51170f77ee672990c22f1703c5997e949dda9d6 is frozen. Native FAST source a9758b38906b7c7950fa04491f37f6634948db26 frozen. Original M0 CPU execution source a9b5dedb0c5586d55991a7f9cb82f13aea847fff frozen. M0 receipt SHA b7c2e478982c571bf45cacd3eddc505e6752501047aa1467ead75064115bcb74; file under outputs/pr_diffusion/track_a_np_paper/a15_fastdips_exposed_20261020/a15_6_mechanism_cpu_audit_v1. Current source and new M1.5 CPU analyzer on codex/prdiff-track-a-a15-residual-selector-m1-20261021, output exclusive a15_8_fast_residual_selector_cpu_v1, checkout reserved pr_diffusion_track_a_a15_residual_selector. No GPU run reservation is asserted here; preserve all original results and Track B priority.

## Primary sources
FAST-DIPS ICLR 2026 published: https://proceedings.iclr.cc/paper_files/paper/2026/hash/ab378fb084f313a204432f0a1e697bae-Abstract-Conference.html
Official source: https://github.com/ququlza/FAST-DIPS/blob/a9758b38906b7c7950fa04491f37f6634948db26/main.py
Historical native NP source: https://github.com/Cheng-HanHuang/pr_diffusion/blob/f4cd28dcd3ec4860cc9f3990fb24964bb97e97bd/scripts/pr_external_difffpr_np_benchmark.py
Cross-track issue ledger: https://github.com/Cheng-HanHuang/pr_diffusion/issues/41
