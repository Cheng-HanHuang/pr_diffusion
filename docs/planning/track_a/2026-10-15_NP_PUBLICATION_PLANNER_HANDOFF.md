# Track A scientific planner handoff — Noise Picking publication viability (2026-10-15)

You are the **Track A scientific PLANNER**, not a PAC executor, and not the B26 executor. You report to the user/metaplanner chat. Your mission is to determine whether the **existing Noise Picking (NP) algorithm and its reliability–cost behavior** merit a scientifically defensible smaller-venue paper. Do NOT assume acceptance, assume NP superiority, or start experiments.

## Required shared rules and repo sources
Read first, in this order:
1. `docs/planning/metaplanner/2026-10-15_SHARED_CHARTER.md` (same branch; mandatory ownership, dataset, and compute contract);
2. root `AGENTS.md`;
3. `docs/b22/b22_3_scientific_closeout.md` and `configs/b22/b22_2_overnight.json` (B22 code/freeze);
4. `scripts/b22/analyze_b22_3_panel.py`, `scripts/b22/run_b22_2_np_worker.py`, `scripts/b22/run_b22_2_sitcom_worker.py`;
5. B21 Fresh2: `docs/b21/b21_11_fresh2_final_benchmark.md`;
6. B24 correction: `docs/b24/00_START_HERE.md`, `docs/b24/B24_3_ZERO_GPU_REPORTING_CORRECTION_RESULT.md`, `docs/b24/01_BASELINE_SCREEN_AND_GATES.md`;
7. B25 signed audit and final return: `docs/b25/B25_FINAL_PLANNER_RETURN.md`, `docs/b25/B25_POSTRUN_NATIVE_NP_AUDIT.md`; B26 `docs/b26/00_START_HERE.md` only for pending caveats.
8. Original DAPS CVPR 2025 https://arxiv.org/html/2407.01521v3 and SITCOM ICML 2025 https://arxiv.org/html/2410.04479v2 (read main, PR appendix, conventions) and their **frozen native** source implementations. Do not assume external source is identical to user's PAC dirty state; first source-audit from GitHub and then request safe PAC inventory if indispensable.

The published DAPS paper reports phase retrieval with best of **four independent samples**; the published SITCOM paper reports best of four independent runs for phase retrieval. Both are paper-style quality reporting conventions; establish from actual paper/code exactly which metric and orientation determine their `best`, not an executable deployment selector. A method's internal optimization objective and its multi-run final-selection statistic are separate.

## Historical fixed benchmark facts (already checked in repository)
- B22 uses 100 common locked FFHQ measurements, sigma_y=0.05, oversampling factor 2, primary raw RGB PSNR.
- Fresh2: two independent DAPS candidates, choose candidate 2 only when its exact operator loss improves over candidate 1 by >0.7. Mean PSNR 29.299, q05 15.860, Good25 92/100, GPU-s 312.6/image.
- SITCOM-4S: four SITCOM-ODE candidates, selected by minimum `correction_norm = sqrt(mean((x0y-x0hat)^2))` at annealing step 160 / tau=0.8; mean 26.442, q05 23.415, Good25 93/100, GPU-s 196.1/image.
- NP-8-RS: **two different scoring configurations** (`lf`, `s2_preproj_lam001`) times four seeds (100–103) = 8 full NP trajectories; candidate selected by min `selector_post_winner_lf_mse_mean`. Mean 29.269, q05 25.146, Good25 95/100, GPU-s 411.8/image. This is NOT an NP ground-truth best-of-eight; `NP-oracle8` is separately diagnostic.
- Fresh2 wins raw PSNR on 91/100; NP-8-RS catches rare catastrophes. Mean NP minus Fresh2 ≈ -0.029 dB and 95% image-bootstrap CI includes zero; Good25 95 vs 92 is not statistically decisive (McNemar p≈0.5488).
- B24 generated DAPS4/SITCOM4 and NP4 on DEV80, under different frozen measurements and seeds; B24 `SITCOM4` uses independently preregistered seeds, unlike B22 sequential `SITCOM-4S`. Do NOT splice B24 DEV80 into B22 primary table as if matched. B24 confirmation305 stays LOCKED. B24 NP-native EPP/PE3 variants failed and remain stopped.
- B26 NP signed-vs-clamped observation scoring is frozen but its scientific execution/result not yet established to you. Do not quietly substitute future R for historical H or rewrite the B22 headline.

## Scientific tasks — NO NEW GPU YET

### A0 — exact native selection/objective audit
Build a **source-grounded matrix** per method (DAPS, SITCOM-ODE, NP):
1. actual forward/observation operator, loss/likelihood target (signed vs clamped), normalization and noise;
2. optimization/sampling update and **prior/regularization/consistency terms**; explain which terms are only internal and why they are/are not comparable across completed candidates;
3. exact executable final selector from B21/B22 vs any native parent ranking/reporting; tie rules, selector checkpoint, dynamic info, seed identities, operator evaluations, and source paths/line ranges;
4. paper-style best-of-4 convention (GT oracle), distinctions of PSNR/SSIM/LPIPS and orientation; when source ambiguity exists state explicitly;
5. whether lowest residual could systematically favor/harm different parent families, and how to falsify using already existing candidate-level outcomes;
6. check measurement preprocessing differences and forbid masking that discrepancy through changing the selector.

**Do not equate DAPS's internal Gaussian-prior L2 term, SITCOM's backward-consistency loss, and NP's LF selection with a shared "full objective" across candidates.** A fair protocol should have (i) each method's historically frozen executable selector, (ii) identical external non-GT selector candidates only if justified and evaluated for all with the same observation semantics, (iii) oracle selection as explicitly offline upper bound. Native-specific regularization is a mechanism, not automatically a cross-family likelihood or confidence certificate.

### A1 — fair comparison ladder and existing-data feasibility
Construct a provenance table covering **DAPS-K, SITCOM-K, NP-K** K in {1,2,4,8} only where actual existing candidate outcomes support them, with no fabrication:
- all B22 stored candidates and B24 DEV80 stored candidates; note which DAPS4 rows physically exist and which B22 DAPS3/4 **do not**;
- B22 NP8 comprises two NP scoring configurations, not eight iid restarts, and B22 SITCOM4 is sequential RNG while B24 SITCOM4 independently seeded;
- paper-style GT-oracle best-of-K vs executable selectors at each budget; internal native objective/scoring ablations only if artifacts allow zero-GPU, predeclared analysis;
- compute axes: GPU-s, wall time, actual UNet/model and operation counts; do not compare run count alone or claim unsupported flop counts exact; operation-normalized reliability frontier;
- metrics: raw PSNR, median/mean/min/q05, Good25/Bad20, LPIPS/SSIM when available, rotation-equivalent audit auxiliary, per-image pairing and confidence intervals;
- success thresholds and model/operator/measurement provenance must match.
- prevent test-set reuse: B21/B22 exposed finals and B24 DEV80 development cannot be called fresh confirmation. No new FFT/noise/GT-assisted runtime routing.
- historical published DAPS/SITCOM best-of-4 figures use potentially different checkpoints/datasets/noise or display conventions; make a comparability table before any direct numerical headline.

### A2 — novelty, manuscript scope, and hard go/no-go
Read NP prior-art competitors including NCS (Noise Combination Sampling, arXiv:2510.23633), standard best-of-K reranking, projection schemes, DiffFPR, DPS, DAPS, SITCOM, TDS/MGDM if relevant, and recent 2025–26 phase retrieval methods. Distinguish a truly new algorithm from heuristic noise selection, delayed projection, and conventional candidate selection. State what portion of NP is independently novel and what is a reproducibility/engineering contribution.
Propose two manuscript framings (reliability/system vs mechanism/inference), with honest lower-tail and computational limits, and plausible venues only after checking current requirements. Provide a **negative recommendation** if all substantive contributions are known/too weak; smaller venue is not permission to make nonfair claims.
Define quantitative and scientific no-go rules **before** asking for new runs. A comparison to published best-of-four must not pretend GT-oracle is deployable. No major algorithmic rework in A; that belongs to B or follow-up B26, not this planner.

## Explicit exclusions
- Do NOT create or tune a new NP conditional-weighted sampler; B26 owns its isolated questions.
- Do NOT design basin-aware plug-in transitions; B owns those.
- Do NOT propose a broad CPU/GPU sweep of selectors on the B22 final 100 and call it prospective.
- Do NOT request any B24 confirmation305 access or new FFHQ measurement.
- Do NOT start PAC work, GPU use, new project checkout, or create output dirs. Reserve only the names authorized by shared charter: code `scripts/track_a/`, `configs/track_a/`; PAC `pr_diffusion_track_a_np_paper`, `outputs/pr_diffusion/track_a_np_paper`. Verify all paths before any later action.
- Do not edit B19–B26, `main`, B26 execution branch, or Track B branch. Docs-only on your own branch unless new metaplanner approval.

## First response and return
First summarize the charter constraints and list exact A0/A1/A2 deliverables. Begin source and data manifest audit **read-only using GitHub**; ask user for PAC inventory only for missing necessary evidence. Do not start GPU runs. Write a concise, strong `TRACK_A_PLANNER_RETURN` with: exact own branch/head; native selector/objective audit; comparability flaws; existing results support vs missing artifacts; venue/novelty threat matrix; possible strongest honest paper claim; hard no-go criteria; and **one** next scoped request (prefer zero-GPU analysis). Send it to metaplanner for cross-track review.
