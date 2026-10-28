# XP pre-registration v1.1: evaluation-only extensions, aggressive tiered scope

> **Amendment A1 (planner, 2026-10-26, before any XP data or handoff delivery).**
> 1. **XP-E is cancelled.** Cheng reports (2026-10-26 02:56) that DAPS's `test-ffhq` set is **not** disjoint from the locked B24 holdout: there is one shared FFHQ id. The overlapping id is withheld, so excluding it would require membership information that agents must not hold.
>    - **Replacement:** **XP-E′** (Tier C, conditional). The published DAPS phase-retrieval configuration runs on the 180 R0 ids, n = 4, **only if** the R0-N.2 A4.4 record shows that the historical FFHQ DAPS configuration differs from it. Amendment A2 corrects that configuration to DAPS-4K.
>    - The FFHQ bridge to published settings is otherwise already covered by S5 (FAST, published command), XP-H (DPS, published config) and XP-F8.
> 2. **New data rule (§1, §7).** Track R never loads any bundled FFHQ set of DAPS, FAST-DIPS or PC-DAPS: `test-ffhq`, `demo-ffhq`, or any DAPS FFHQ archive. This applies to smoke tests, timing and debugging as well. FFHQ inputs come **only** from the R0 manifests of the 180 ids.
> 3. **New XP-G2** (Tier C), the 512-canvas control: oversample 4.0, i.e. pad 128 per side, so the 256² support is exactly half the canvas per axis.
>    - **Why:** our 384 canvas lies outside the hypothesis of the classical generic-uniqueness theorem (support ≤ half the grid per axis; research pass of 2026-10-26). XP-G2 measures whether the CONSISTENT class shrinks when that hypothesis holds.
> 4. **New exploratory analyses (§6, CPU, optional):**
>    - GT-free reliability diagnostics, including cross-run agreement after G-alignment and a PRTF-style reproducibility score (Chapman et al. 2006);
>    - a "mixed-twin" anatomy of INCONSISTENT failures (Fienup & Wackerman 1986).
> 5. Nothing else changes. Tiers A and B are unchanged.

> **Amendment A2 (planner, 2026-10-26 11:30, before any XP data).** The DAPS configuration for XP-D2 and XP-E′ is corrected.
> - **The problem.** The DAPS paper's phase-retrieval results (Table 1) use **DAPS-4K**. Sec. 4.1 of arXiv 2407.01521v3, verified by the planner: "we use DAPS-1K for all linear tasks and DAPS-4K for all nonlinear tasks", and "DAPS-4K uses 10 NFE and 400 steps", with 100 Langevin steps per denoising iteration. The repository's `commands/pixel_*.sh` run a "DAPS-1K" setting (200 annealing × 5 ODE steps), so they do **not** reproduce the published numbers.
> - **The fix.** XP-D2 and XP-E′ use DAPS-4K at `e7a77d0`:
>   - `+sampler=edm_daps` with `sampler.annealing_scheduler_config.num_steps=400` and `sampler.diffusion_scheduler_config.num_steps=10`;
>   - `+task=phase_retrieval` with `task_group=pixel` (100 Langevin steps, lr 5e-5, τ = 0.01, lr_min_ratio 0.01);
>   - our frozen y and seeds, through the in-repository wrapper.
>
>   Record the resolved configuration. **The published reference values** are pixel DAPS on FFHQ, 30.63 ± 3.13 dB / SSIM 0.851 / LPIPS 0.139, and on ImageNet, 25.78 ± 6.92 / 0.743 / 0.254 (Table 1; best of 4; the code takes per-metric best over runs).
> - **XP-E′ condition** now reads: run only if the historical FFHQ DAPS configuration (A4.4) differs from DAPS-4K.
> - **Cost.** DAPS-4K is about 4× DAPS-1K. XP-T measures it. If XP-D2's projection exceeds 80 GPU-h, apply the existing drop rule: reduce to 25 images by SHA-256(`XPD|<basename>`).

> **Amendment A3 (minor, planner, 2026-10-26).** XP-C also reports, for every GT image (CPU; GT-only evaluation), the fraction of Fourier magnitudes |F_c P z| below σ = 0.05 on the 384² canvas, overall and in the outer half of the frequency radius, plus the ratio of signal energy to m·σ².
> - This supports the paper's setup claim that measurement consistency constrains mainly coarse structure.
> - A planner check on four generic natural images (scikit-image samples, not FFHQ or ImageNet) found 79–93% of magnitudes below σ overall and 92–98% in the outer half.

> **Amendment A4 (planner, 2026-10-26 14:30, before any XP data).** Two lanes, three scope decisions, one analysis.
> 1. **Two lanes replace the tier order for scheduling.** The tiers still decide what is dropped (§3.5).
>    - **CRITICAL lane** (time-aware; feeds the paper):
>      - XP-0, XP-T and the replay gates;
>      - XP-A16, XP-C, XP-C2, XP-D1 (data cut Nov 2);
>      - XP-B16, XP-D2, XP-D3, XP-A64, XP-F8 (data cut Nov 6).
>    - **BACKGROUND lane** (not time-sensitive; fills idle GPU capacity):
>      - XP-B32, XP-D4, XP-G, XP-G2, XP-H, XP-I, the new XP-J and the conditional XP-E′.
>      - Results that arrive by Nov 20 go into the supplement; later ones go to the fallback or journal version.
>    - **Scheduling rule.** A free GPU slot always takes a ready CRITICAL job first, and takes BACKGROUND work only when no CRITICAL job is ready.
>      - BACKGROUND jobs are cut into units of at most about 20 minutes (one or a few runs), so a CRITICAL job never waits longer than that.
>      - Running jobs are never killed.
> 2. **NP is dropped from the paper.** It is an in-house, unpublished variant. Its R0 runs stay in the internal record but are excluded from every paper aggregate. Counts quoted in the paper are recomputed on {DAPS, SITCOM, FAST-DIPS}.
> 3. **SITCOM, resolved from its phase-retrieval repository** (`sjames40/SITCOM_ODE` @ `275ab67`, verified by the planner):
>    - selection is GT-based and per metric (`eval.py` lines 72–73 and 81–83, as in DAPS);
>    - the README phase-retrieval command uses 4 runs (the config default is 10);
>    - the repository has ImageNet model and data configs but no ImageNet-specific phase-retrieval hyperparameters, so **XP-D3 uses the FFHQ phase-retrieval settings with the ImageNet model**, exactly as §1's fallback says.
> 4. **New XP-J** (BACKGROUND, conditional): DiffFPR (Li & Wang, ICML 2024), a published solver built for oversampled Fourier PR, n = 4 on the 180 FFHQ ids.
>    - Its code uses oversample 4.0 (the 512 canvas), so run it **at 512 with the XP-G2 measurements**. Run it also at 384 with the R0 measurements **only if** that is a parameter change plus our y, with no code edits.
>    - Use its published settings and record them. Run selection follows our policies.
>    - XP-0 checks the code at `/egr/research-pac/huang248/external/DiffFPR` (read-only) and whether our y can be fed in; if not, skip it and say so.
> 5. **New analysis** (CPU, CRITICAL analysis, not exploratory): **does a prior score separate GOOD from CONSISTENT runs?**
>    - Among runs with ρ ≤ τ (the GOOD, ORBIT and CONSISTENT runs, with ORBIT taken after orientation correction), report per family and dataset the AUROC of each GT-free score for GOOD vs CONSISTENT: the stored R0 S score (R0 runs), S_mid identity member, 2-forward score, CLIP upright score, UnivFD and ρ itself.
>    - Also report the K-curve of a two-stage selector: "residual threshold τ, then the best prior score among passing runs, then the orientation test".
>    - This is an analysis of what selection *could* gain from a prior term, not a proposed method. It answers whether failure type (ii) is fixable without the ground truth.

> **Amendment A5 (planner, 2026-10-26 23:55 ET).** Written before any XP outcome with run index ≥ 4 had been scored or seen; no XP-K, XP-I or A5.5 data exist. NP's removal from the paper is confirmed by Cheng (2026-10-26 23:01).
> 1. **Clarification, no change.** Tier B = XP-A64, XP-B16, **XP-D2 (DAPS-4K, ImageNet)**, **XP-D3 (SITCOM, ImageNet)** and XP-F8. XP-E′ is Tier C and runs only if A4.4 says so. (The executor log of 2026-10-26 22:34 swaps D2 and D3 and lists XP-E′ in Tier B; this file governs.)
> 2. **XP-F8 gate.** XP-F8 needs, in order:
>    - (a) the R0-N.2 **A4.4** record (annealing steps, ODE steps and NFE per run, from the historical S1/S2 DAPS receipts);
>    - (b) an **8-root exact replay** of historical DAPS candidates (first 2 sorted ids × runs 0–1, per cohort), as passed by FAST and SITCOM, on the source tree that will generate XP-F8. The historical checkout of `e7a77d0` carries default-off B20.11/B21.3 edits in `sampler.py`. Use the pristine sparse checkout if it reproduces the historical candidates exactly with the historical configuration; otherwise use the historical checkout with those features off, and record why;
>    - (c) new seeds that continue each cohort's historical RNG lineage (B22: the sequential stream; DEV80: independent seeds), with the no-collision check.
> 3. **XP-I (PC-DAPS) is promoted to CRITICAL-if-ready**, data cut Nov 6.
>    - Engineering happens now in the BACKGROUND lane. Once its smoke passes, its GPU jobs take CRITICAL priority behind any ready Tier A/B job.
>    - If it misses Nov 6, it reverts to BACKGROUND and the supplement.
>    - Scope as in §2 (FFHQ, 180 ids, one bag of 4 particles, the native FFHQ phase-retrieval command at `59b2f04`: 200 annealing steps, bag size 4, pruning on; our frozen signed y).
>    - **New XP-I2** (BACKGROUND): the same on the 100 ImageNet images with `imagenet256ddpm`.
>    - Analysis per bag: R0 labels and the PC-DAPS `pr_outcomes` labels (A4.1 rule) for every particle; first particle; mean over the bag (PC-DAPS's reporting); E0, E1_CLIP, O_raw and O_orbit over the bag; DAPS at K = 4 on the same ids for comparison.
> 4. **New XP-K, a data-term calibration intervention** (BACKGROUND, parameter-only; pre-specified here).
>    - **Why.** All three solvers weight the measurement far above the Gaussian likelihood at σ = 0.05, and R0-N N1c shows that every GOOD run of DAPS and SITCOM has ρ < σ. In detail:
>      - DAPS's Langevin score uses −∇‖A(x)−y‖²/τ²;
>      - SITCOM-ODE's Langevin loss is ‖A(x)−y‖²/(2τ²);
>      - both use τ = 0.01, i.e. 50× and 25× the weight 1/(2σ²);
>      - FAST-DIPS projects an auxiliary residual onto an L2 ball of radius ε = 0.05, whereas the noise norm is σ√m ≈ 33.3.
>    - **Design.** B22, 100 ids, n = 4 per family, with **exactly one parameter changed** from the frozen settings:
>      - SITCOM-ODE: Langevin τ = σ = 0.05 (the Euler measurement-gradient step is unchanged);
>      - FAST-DIPS: ε = σ√m = 33.26;
>      - DAPS: τ = √2·σ = 0.0707, with the historical FFHQ configuration, after the XP-F8 gate (2).
>    - Seeds follow the XP rule with dataset `ffhq_cal`. Same frozen y. Labels use the frozen native τ thresholds; ρ/σ is also reported.
>    - **Outcomes** (descriptive, vs the same ids' historical native runs, K = 4; DAPS K = 2):
>      - class rates (G/O/C/I);
>      - the median and IQR of ρ/σ for GOOD runs;
>      - the fraction of runs with ρ < ρ★ (A5.5);
>      - first-run Good25;
>      - E1_CLIP and O_orbit at K = 4.
>    - **Directional hypotheses** (exploratory):
>      - H-K1: calibration moves the GOOD-run ρ/σ of DAPS and SITCOM toward 1;
>      - H-K2: the CONSISTENT share of SITCOM's non-GOOD runs falls.
>      - No hypothesis on PSNR.
>    - Not part of v1 and not embargoed. It is reported whatever its outcome. It is **not** proposed as a method.
> 5. **New analysis A5.5: the noise floor** (CPU; evaluation-only; pre-specified now, before any ρ★ value has been computed).
>    - **Per-candidate quantities.** For every scored candidate x̂ of DAPS, SITCOM and FAST-DIPS (R0 runs now; XP stages once their outcomes are allowed), with the exact frozen signed y, in float64:
>      - ρ★ = ‖A(x★) − y‖/√m;
>      - Δ = (‖A(x̂) − y‖² − ‖A(x★) − y‖²)/σ²;
>      - D = ‖A(x̂) − A(x★)‖²/σ²;
>      - C = ⟨y − A(x★), A(x̂) − A(x★)⟩/σ², so that Δ = D − 2C. C > 0 means the run has absorbed part of the noise.
>    - **Report**, per family × cohort × class (G, O_g, O_p, C, I):
>      - n, and the median and IQR of Δ, D and C;
>      - the fraction with ρ < ρ★;
>      - per (family, cohort), whether ρ★ ≤ τ, i.e. whether the ground truth would pass the solver's own consistency threshold.
>    - **Also report** within-image pairwise residual win rates (GOOD vs CONSISTENT, GOOD vs global ORBIT, CONSISTENT vs INCONSISTENT) with image-cluster bootstrap CIs. The planner computed these from the A8 rank strings: 200/205, 50/69 and 92/92; reproduce them.
> 6. **A8b** (CPU; committed; GT-free): per (image, family) row as in A8, the ρ_signed of every historical run in run-index order (6 significant digits), and per (source, family) the frozen τ. The theory executor may read it.
> 7. **A7 becomes required** (target Oct 28), adding the A5.5 columns (`rho_star`, `Delta`, `D`, `C`) to A7.1. It is still delivered only as a ZIP to Cheng, not committed. The UnivFD fix and harm counts are recounted on the three public families from the stored scores.
> 8. **Queue order and drop order, revised.**
>    - **BACKGROUND launch order:** XP-K (SITCOM and FAST parts at once; DAPS after the F8 gate) → XP-I (CRITICAL once ready, item 3) → XP-B32 → XP-G → XP-I2 → XP-H → XP-G2 → XP-J → XP-D4 → XP-E′.
>    - **Drop order:** XP-E′ → XP-J → XP-D4 → XP-G2 → XP-H → XP-I2 → XP-G → XP-K → XP-B32 → XP-I → XP-F8 → XP-D3 → XP-B16 (60 ids) → XP-D2 (25 images) → XP-A64. Tier A is never dropped.
> 9. **Theory freeze contents** (§5.1 amended). The S1–S3 curves stay in v1 as the **exchangeable-ranks reference** (Stroebl-type ceiling). The A8 rank strings reject S2 at K ≤ 4: GOOD beats CONSISTENT in 200/205 within-image pairs, and E0 picks GOOD in 92 of 100 images with a mixed passing set, against 51.9 expected under S2. So v1 should also freeze a **rank-calibrated forecast**: class-dependent residual races fitted to the K ≤ 4 ranks (and to A8b if available), with the tail assumption stated. The monotone-gap shape prediction applies to the S2 reference only. The agreement criterion (§5.6) is unchanged.

> **Amendment A6 (planner, 2026-10-27 12:30 ET).** Written before any outcome of XP-A16, XP-A64, XP-D1 phase 2, XP-B16, XP-F8, XP-K or XP-G had been scored or seen. XP-D1 phase 1 and XP-D2 are generated but not yet scored. TH2 mechanism forecasts v1 are registered (registry, 2026-10-27).
> 1. **A4.4 ruling, pending replay.** The repository records the historical FFHQ DAPS configuration as `edm_daps` at `e7a77d0` (with default-off local edits), 400 annealing steps × 5 ODE steps (2,000 denoiser evaluations), and the pixel task group (100 Langevin steps, lr 5e-5, τ 0.01, lr_min_ratio 0.01). Sources:
>    - `configs/b23/fresh1_frozen.yaml`: the DAPS parent; historical reference `B21_11_fresh2_final_val100_meas5401`; seed rule `seed1 = 22000 + frozen panel row_id`;
>    - `docs/b24/B24_3_ZERO_GPU_CLOSEOUT_AUTHORIZATION.md` l. 51: DEV80 DAPS uses `annealing num_steps=400`, `diffusion num_steps=5`;
>    - `configs/b24/b24_3_dev80_overnight.json`: DAPS4 with "frozen DAPS-1 semantics" and seed domain `B24_DEV80_DAPS_SOLVER_V1`.
>
>    The A5.2 8-root replay **under this configuration** is the confirmation. If it reproduces the historical candidates exactly, this configuration is the A4.4 record and the XP-F8 configuration. If it does not, stop and report.
> 2. **XP-E′ is now eligible and is promoted to CRITICAL-if-ready** (data cut Nov 6). The historical configuration (400 × 5) differs from DAPS-4K (400 × 10). XP-E′ is a separate population with new seeds, so it needs no replay gate. It reuses the replay-verified pristine DAPS-4K worker with the frozen FFHQ y. 720 runs, about 13 GPU-h.
> 3. **Generation is never embargoed.** XP-D1 phase 2 is generated now and stays sealed until v2 is registered, or 48 h after the phase-1 strings are committed.
> 4. **Evaluation panels.** Each K-curve is the exact all-subset U-statistic on a stated panel, ties by lowest run index. Forecasts and observations use the same panel:
>
>    | Family | Primary panel | Secondary panel |
>    |---|---|---|
>    | FAST FFHQ | runs 0–63 (n = 64) | runs 0–15 (n = 16) |
>    | SITCOM FFHQ | runs 0–15 (n = 16) | runs 0–31 (n = 32), if XP-B32 completes before the analysis cut |
>    | DAPS FFHQ | runs 0–7 (n = 8) | — |
>    | FAST ImageNet (v2) | runs 0–15 (n = 16) | runs 0–63, if XP-D4 runs |
>
>    The observed historical runs of each image are part of its panel, so the forecast target is conditional on them (theory note `TH1_CONDITIONAL_PREFIX_TARGET_20261027.md`).
> 5. **Agreement criterion (§5.6), refined.**
>    - If a frozen interval is a predictive interval for the realized panel statistic on these images (it includes the finite-panel U-statistic noise), the forecast is "consistent" iff the observation lies in [lo95, hi95].
>    - SE_comb applies only to intervals that exclude that noise. Each forecast file states which kind it holds.
>    - Also report, per family, policy and forecast variant (S2 reference, race, tail), the number of grid points inside the interval and the mean standardized residual (obs − point)/((hi95 − lo95)/3.92).
>    - The shape test applies to the S2 reference only.
> 6. **Theory inputs directory.** The theory executor may read `docs/track_r/executors/xp/theory_inputs/` on the XP branch, in addition to the r0n return tree. The executor puts there only files cleared for it:
>    - the XP-D1 phase-1 per-image strings (runs 0–3, A8 schema, Good20, with E1_CLIP G1 decisions when available);
>    - after each freeze, nothing beyond what that freeze covers.
>
>    `docs/track_r/executors/xp/returns/` stays forbidden to the theory executor.
> 7. **XP-D2 and XP-D1 phase 1 are scored now** (CPU; both allowed before any freeze). XP-D2 reports the published protocol (DAPS-4K; per-metric oracle over 4 runs) next to the first run, E0, E1_CLIP (G1) and O_orbit.

> **Amendment A7 (planner, 2026-10-27 17:15 ET).** Procedural only. Written before any outcome of XP-A16, XP-A64, XP-D1 phase 2, XP-E′, XP-B16 or XP-F8 had been scored, and before any A5.5 value existed. TH2 mechanism forecasts v1.1 are registered.
> 1. **GPU co-tenancy for generation stages.**
>    - **Default rule.** A generation-only stage (no timing claim) may be admitted on a GPU that is running other users' jobs, under four conditions:
>      - free VRAM is at least the stage's request (about 1.5 × measured peak + 2 GiB);
>      - device memory after admission stays at or below 80%;
>      - at most one Track R worker runs per GPU;
>      - while foreign jobs are present, Track R uses at most 2 GPUs.
>    - Foreign processes are never touched. Co-tenancy is recorded per job.
>    - XP-T timing, and any replay that must be bitwise, still require clean GPUs.
>    - Instead of exiting after 30 min without a free slot, the controller waits and retries, with a hard wall-clock limit and a stop file.
>    - **Opt-out.** Cheng may revoke co-tenancy at any time; the controller then reverts to clean-only admission.
> 2. **XP-B16 lineage gate, by regeneration.** B22 SITCOM candidates came from one post-model RNG stream per image (seed 43 before model load; the post-model state restored per image; runs 0–3 sequential). So, for every B22 image, XP-B16:
>    - regenerates runs 0–3 from that state;
>    - checks them against the historical candidates: bitwise hash equality, or else the §4.1 tolerance of 0.1 dB raw PSNR with an identical label, scored offline;
>    - continues the same stream through runs 4–15.
>
>    This replaces the 2-id replay with a full 100-image replay. DEV80 SITCOM uses independent native seeds with the no-collision check, as historically. Any B22 image whose runs 0–3 fail the check is excluded from the pooled K-curves and reported.
> 3. **XP-F8 seeds.** XP-F8 continues the Fresh2 lineage for B22 runs 2–7 (rule audited from `B21_11_fresh2_final_val100_meas5401`) and the `B24_DEV80_DAPS_SOLVER_V1` domain for DEV80 runs 4–7, with the no-collision check. It follows the 8-root 400 × 5 replay (A6.1).
> 4. **v2 clock.** It started at commit `41d4121` (2026-10-27 12:01:51 ET), so v2 is due by **2026-10-29 12:01 ET**. The CLIP G1 decisions for the D1 phase-1 runs are pending the XP-C worker. If they are not in `theory_inputs/` before the freeze, the v2 E1_CLIP forecast either abstains or states its φ_g and h assumption.

> **Amendment A8 (planner, 2026-10-28 03:00 ET).** Written after TH1 v1 and v2 were frozen and pushed (2026-10-28 06:22–06:33Z) and before any outcome of XP-A16, XP-A64, XP-D1 phase 2, XP-E′, XP-B16 or XP-F8 had been scored. The planner had seen the XP-D2/D3 K = 4 aggregates, which no forecast concerns.
> 1. **Registration.** v1 (`8ef5f94`), its per-image companion (`5782149`) and v2 (`340dc5d`) are registered (registry, this commit).
>    - The planner ran both freezes with the theory executor's production code, at its request. The only code the planner added is packaging, with no model change: the seal's τ comparison (`cfefd00`) and the v2 sealer and launcher (`dd5c5e6`).
>    - **The FFHQ K > 4 embargo is lifted** for the XP analysis.
> 2. **XP-D1 phase 2 stays embargoed** until the earliest of: (a) registration of a v2 E1_CLIP addendum; (b) the theory executor's declaration that there will be none; (c) **2026-10-29 12:01 ET**. The addendum:
>    - uses only the G1/G2 decisions on the 400 phase-1 candidates (run index 0–3), which the executor commits as a new file in `theory_inputs/`;
>    - adds E1_CLIP forecasts (same panel, K and variants) and changes no number in v2;
>    - is written by the theory executor, then run and sealed by the planner, as v2 was.
>
>    XP-D2/D3 outcomes are not relayed to the theory executor until this embargo lifts.
> 3. **E1_CLIP is a per-candidate rule.** The orientation test looks at one candidate. Let g̃(x) = 1 if the orientation that the test chooses for x is GOOD (computed directly: PSNR of the chosen member against GT). Since ρ is orbit-invariant, for a panel of n runs ranked by ρ (ties by run index):
>
>    E1_CLIP(K) = Σ_i g̃(x_(i)) · C(n − i, K − 1) / C(n, K),
>
>    which is E0's formula with g̃ in place of the GOOD indicator. No subset enumeration is needed. The same holds for E1_CLIP_G2, E1_2F and E1_S8.
> 4. **Comparison with registered forecasts.** Before opening any outcome, verify the registry entry and the `git show` SHA. Then apply each file's frozen rule as written:
>    - interval type `predictive_realized_panel` means "inside iff lo95 ≤ observed ≤ hi95";
>    - standardized residual = |observed − point| / max((hi95 − lo95)/3.92, 0.005);
>    - the primary panel's E0 and E1_CLIP coverage counts decide (v2: E0 only), with the residual as tie-break;
>    - secondary panels, race_tau1 and tail are reported but do not decide.
>
>    Report every variant. Commit the comparison before any refit or new model.
> 5. **Lineage-gate exclusions (A7.2).** If B22 images fail the gate, the primary SITCOM B22 and pooled comparisons use the retained images: the companion's per-image points averaged over them, with the frozen interval half-widths scaled by √(n_cohort / n_retained) and flagged approximate. The all-image comparison is reported as a secondary row.
> 6. **XP-D2 comparability (no extra GPU).** Before XP-D2 enters the paper as "the published protocol":
>    - **GT/y identity check.** The scored GT must be the image that produced y. For each of the 100 images:
>      - **exact form:** regenerate n from its recorded seed and check ‖y − σn − A(x★)‖/√m ≤ 1e-4, with x★ the scorer's GT tensor (float32 round-off gives about 1e-6; a GT mismatch at 40 dB PSNR gives about 5e-3);
>      - **statistical form** (if n cannot be regenerated): ρ(x★, y)/σ must lie within 1 ± 0.0043, i.e. 4 SD of ‖n‖/√m, which allows for 100 images.
>
>      Report the worst image and the torch and torchvision versions of the y-generation and scoring environments. If any image fails, rescore D1–D3 with the GT tensor of the y-generation environment.
>    - **LPIPS.** Add DAPS's convention, `piq.LPIPS(replace_pooling=True)` (VGG16) on [0, 1] inputs, as the published-protocol LPIPS. Keep AlexNet LPIPS as secondary.
>    - **Checkpoint.** Record the SHA-256 and size of the ImageNet checkpoint used by D2 and D3, and whether it is the file that DAPS's `download.sh` fetches as `imagenet256.pt`.
>    - **Wrapper check on FFHQ.** From XP-E′ at K ≤ 4 on cohort A, report the DAPS-4K per-metric best of 4 next to DAPS Table 1 (30.63 / 0.851 / 0.139) and PC-DAPS's reproduction of DAPS-4K (29.66 / 0.770 / 0.177).
>    - **Native spot check** with `posterior_sample.py` (8 images × 4 runs): Tier C, only if the checks above leave doubt.
>    - **In the paper,** XP-D2 is reported next to two references: DAPS Table 1 (25.78 dB) and PC-DAPS's reproduction of DAPS-4K (20.98 dB). We do not claim to reproduce DAPS Table 1.
> 7. **GPU order.** XP-E′ continues. The next free GPU takes XP-B16, lineage gate first, then the 8-root DAPS replay and XP-F8. CLIP scoring (ViT-L/14) runs on CPU whenever no GPU is free. The XP-C prior and UnivFD tests on GT take GPU minutes when a GPU is available.

> **Amendment A9 (planner, 2026-10-28 06:00 ET).** Written after the FAST FFHQ comparison with v1 (XP `07d5460`) and the D1 phase-1 CLIP decisions (XP `a254f23`) were published. No outcome of XP-E′, XP-B16, XP-F8 or XP-D1 phase 2 had been scored, and XP-B16 and XP-F8 had not been generated.
> 1. **FAST vs v1 is accepted as the first prospective result.** v1 is not refitted.
>    - The frozen rule (primary-panel E0 coverage; E1_CLIP still pending) favors S2 over the race for FAST: 15 vs 6 of 21 cohort–K points.
>    - By cohort it splits: the race wins B22 (6/7 vs 1/7); S2 wins DEV80 and the pooled panel (7/7 vs 0/7 each).
>    - Both miss the oracles: O_raw coverage is 6/21 for the race and 2/21 for S2.
>    - Report it exactly so, with every variant.
> 2. **Exploratory decomposition.** Labeled post hoc; it uses frozen parameters only and introduces no new model.
>    a. **Exchangeability check:** class rates (G, O_g, O_p, C, I) per cohort in runs 0–3, 4–15 and 16–63, with a two-proportion test per class and cohort. A shift would point to a protocol change rather than to the mixture model.
>    b. **Composition-conditional selection test:** for each image and K, the E0 that each frozen rank law (race, tail, race_tau1, S2) predicts given the realized class counts of the image's 64 runs, compared with the observed E0. The prediction is exact: the average over K-subsets of the Plackett–Luce first-place probability of a GOOD run. Its interval comes from simulating each rank law on the realized labels. This separates selection error from generation error.
>    c. **Case lists:**
>       - the 7 images with no GOOD run in 64;
>       - the 4 images whose E0 pick at K = 64 misses an existing GOOD run.
>
>       For each, give the ID, the prefix string, the 64-run class counts, the winning class, and the winner's residual gap to the best GOOD run in units of σ.
>    d. **E1_CLIP for FAST:** score the FFHQ face set on all new candidates, so that the frozen primary rule can be applied in full.
> 3. **Release to the theory executor.** FAST A16/A64 per-image strings go to `theory_inputs/`, plus XP-E′ strings once scored. The schema is A8's, extended to 64 runs: labels, signed ρ and rank order. Release them only after `v2 E1` is registered or the A8.2 deadline passes, so that the addendum stays blind to the FAST result.
> 4. **v3: an optional second forecast for XP-B16 (SITCOM) and XP-F8 (DAPS).**
>    - **Scope:** the same estimand, panels, policies and comparison rule as v1. It addresses the identified failure, for example by using each image's FAST 64-run outcome as a difficulty covariate.
>    - **Allowed inputs:** anything released under item 3, plus A8 and A8b. Never a SITCOM run index ≥ 4, or a DAPS run index beyond the historical prefix.
>    - **XP-B16 and XP-F8 outcomes stay sealed** until the earliest of: v3 registration; the theory executor declaring that there will be none; **2026-10-31 12:00 ET**. Generation and the A7.2 lineage gate proceed meanwhile.
>    - **v1 stays the primary registered forecast** for B16 and F8; v3 is reported beside it, never instead.
> 5. **The v2 E1 addendum code has an anchor error.**
>    - **The error.** At theory `3a33899`, the forecaster anchors its control variate to the probability that the E0 pick lies in the orbit classes, not to the O_orbit oracle.
>    - **Its size.** On the real inputs this inflates E1 by 0.14–0.34: the race forecast at K = 16 is 0.79 as written and 0.60 corrected.
>    - **The fix** (`TH1_V2_E1_ANCHOR_FIX.patch`) computes the oracle on the same simulated panels, and adds a contract check that its Monte Carlo mean reproduces the registered O_orbit within 0.01.
>    - **Process.** The theory executor applies the patch verbatim; then the planner runs, seals and registers `v2 E1`. A8.2 is unchanged.
> 6. **GPU campaign (authorized).** A6 is complete: 2,320 roots and 28.9 GPU-h, about 68 of 400 GPU-h used in total. Run from a clean integration branch:
>    - the 8-root DAPS 400 × 5 replay on a clean GPU (bitwise, or the §4.1 tolerance), then XP-F8 (920 runs);
>    - XP-B16: the A7.2 lineage gate on all 100 B22 images, then runs 4–15 (2,160 runs; B22 continues its streams, DEV80 uses independent seeds);
>    - two GPUs in parallel when available. A7.1 co-tenancy applies to generation; replays run on clean GPUs only;
>    - then the XP-C prior and UnivFD tests on GT, then the BACKGROUND lane;
>    - **optional (Tier C):** 192 more FAST runs on each of the 7 never-GOOD images (1,344 runs, about 2 GPU-h at A16's 4.3 s per run), to tell a zero success probability from a small one.

> **Amendment A10 (planner, 2026-10-28 15:30 ET).** Written after `v2 E1` was registered (planner `7d1882c`) and before any XP-D1 phase-2 outcome had been scored. XP-B16 is in production (748/2,160 roots at the last checkpoint); B16 and F8 outcomes stay sealed (A9.4).
> 1. **Co-tenancy (replaces the A7.1 two-GPU cap; Cheng, 2026-10-28 15:02).**
>    - **Generation stages** (every GT-free sampling job, including B16, B32, F8 production and CLIP scoring) may run alongside other users' jobs on any of the four GPUs. The other A7.1 conditions stay:
>      - free VRAM is at least the job's request;
>      - device memory stays at or below 80% after admission;
>      - at most one Track R worker runs per GPU;
>      - foreign processes are never touched.
>    - **Timing-measured work** (XP-T, and any run whose wall time is reported) runs on clean GPUs only. When a foreign job appears, no new timing job starts. A timing job that overlapped a foreign job is marked timing-contaminated and rerun.
>    - **Replays and lineage gates** may run under co-tenancy. They pass on bitwise equality, or on the registered tolerance: 0.1 dB raw PSNR with an identical label.
>    - **Telemetry and validity.**
>      - Each receipt records the foreign-process inventory sampled during the job.
>      - Production wall time is never a validity criterion; only smoke groups keep the speed envelope.
>      - On Oct 28, B16 ran on four GPUs while a foreign job was present. That is recorded as a deviation from A7.1, with no effect on reconstruction validity.
> 2. **Campaign window and budget.**
>    - **Window and cap.** The B16 campaign continues past its 12-hour window (Oct 28, 18:57 ET) through **Oct 30, 12:00 ET**. B16, the DAPS replay and F8 share a cap of 70 GPU-h, including the 10.9 GPU-h already spent; total use stays under the 400 GPU-h cap.
>    - **Order:**
>      - B16 to completion;
>      - the DAPS lane as soon as its adapter is certified, in parallel with B16 whenever a GPU is free. The lane is the 8-root 400 × 5 replay with the historical code, configuration, seeds and checkpoint, then F8;
>      - then the XP-C GPU tests;
>      - B32 only after F8.
> 3. **XP-D1 phase 2 is unblinded** (A8.2(a)). Compare it with v2 (E0, O_raw, O_orbit) and with v2 E1 (E1_CLIP from G1 decisions on the phase-2 candidates) by the frozen rules. Add the A9.2 decomposition: the exchangeability check (phase 1 vs phase 2), the composition-conditional selection test, and the never-GOOD case list.
> 4. **The A9.2(b) test for FAST** was computed by the planner from the published strings (`docs/track_r/planner_analysis/A9_2b_FAST_conditional_selection/`). The executor reproduces it independently with its own code.

**Status:** FROZEN v1.1 with amendments A1–A10 (planner, 2026-10-28).
- **GO given.** Cheng, 2026-10-26 00:57: *"GO aggressively, the lab gpus are not actively been used and so even if some runs failed it's no hurts. Try to get as much as possible out."*
- **Supersedes `XP_PREREG_v1.md`.** No XP data existed when v1.1 was frozen: there are no executor commits after `33a34bd`, and no XP or theory branch exists. v1 is kept for the record.
- **This file is self-contained.** The executor does not need v1.

**Purpose.** Strengthen the audit for the CVPR 2027 attempt (`2026-10-25_TOP_VENUE_PLAN.md`) with:
- a second dataset (ImageNet), including a replication of the field's published DAPS protocol;
- K-scaling that **tests theory predictions frozen in advance**, in both regimes the theory distinguishes (FAST: few consistent-wrong runs; SITCOM: many), plus DAPS;
- same-GPU timing for one compute axis;
- orientation tests across domains;
- supplement-grade robustness (noise level, DPS backbone, concurrent PC-DAPS on the same images).

**Constraints.**
- **No new method.** Every result is reported whatever its outcome.
- **Cap: 400 GPU-hours**, counted as wall-hours per occupied GPU, all stages and scoring included. PAC has 4 × RTX PRO 6000 Blackwell (R0.5 record); XP-0 re-inventories them.
- **The real limits are the data cuts:** Tier A by **Nov 2**, Tier B by **Nov 6** (main paper), Tier C by **Nov 20** (supplement, due Nov 23).

## 0. Changes from v1

| # | Change |
|---|---|
| 1 | GO recorded. Cap 45 → 400 GPU-h. Work is organized in tiers A (required), B (main paper) and C (supplement), run as a work-conserving queue (§3) |
| 2 | FAST FFHQ: n = 16 (A), then **n = 64** (B). FAST ImageNet: n = 16 (A, was 8), then n = 64 (C) |
| 3 | SITCOM FFHQ K-scaling on **all 180 ids** (was 60), n = 16 (B), then n = 32 (C) |
| 4 | **New:** DAPS FFHQ K-scaling to n = 8 on all 180 ids (B) |
| 5 | DAPS ImageNet on **all 100 images** (was 25), using the **published phase-retrieval configuration**: DAPS-4K, per amendment A2 (B) |
| 6 | **New (C):** XP-G noise robustness (FAST, σ ∈ {0.02, 0.10}); XP-H DPS backbone; XP-I PC-DAPS on the same 180 ids |
| 7 | **Two-stage prediction freeze:** v1 for FFHQ (by Oct 31), v2 for ImageNet FAST (after its runs 0–3). Plus an **embargo** on K > 4 outcomes until each freeze (§5) |
| 8 | **Replay and exchangeability gates** before adding runs to historical families (§4) |
| 9 | **Exact U-statistics** by order-statistic identities instead of subset enumeration (§6) |
| 10 | Per-image label and decision strings are committed (Cheng approved the A8 strings, 2026-10-26) |
| 11 | **New (A):** XP-C2, the equivariance-defect profile of each checkpoint on GT images. It tests the theory's mechanism for orbit errors (TH1.1 Cor. 4) |

## 1. Fixed objects

| Object | Frozen choice |
|---|---|
| Operator | R0 pinned operator A(x) = \|F_c(pad₆₄((x+1)/2))\|, 384² canvas, channel-wise, σ = 0.05, signed y. This is the DPS/DAPS standard (`oversample: 2.0`, `sigma: 0.05`) |
| FFHQ ids | The 180 ids of B22 (100) and DEV80 (80), exactly as in the R0 manifests. **No other FFHQ id.** XP-E is cancelled (A1). **Never load** the DAPS, FAST-DIPS or PC-DAPS bundled FFHQ sets (`test-ffhq`, `demo-ffhq`, any DAPS FFHQ archive), not even for smoke tests, timing or debugging. Override every repository default that points to them |
| FFHQ measurements | **Reuse the R0 measurement file per id.** Before any re-run of a historical family, verify per id that this y has the same SHA-256 as the y used by that family's historical runs. **On mismatch, stop and report.** Never substitute silently |
| XP-G measurements | The same 180 ids, **paired noise**: n_R0 = (y_R0 − A(x★))/0.05, then y_σ = A(x★) + σ·n_R0 for σ ∈ {0.02, 0.10}. Only σ changes. Save each y_σ and its SHA |
| XP-G2 operator and measurements | The same operator with oversample 4.0 in the DAPS/FAST convention: pad = 128 per side, canvas 512², m = 3·512². **Verify in XP-0** that the code's pad formula gives 512. σ = 0.05; y = A₅₁₂(x★) + σ·n with seed = first 8 bytes of SHA-256(`XPG2MEAS\|ffhq\|<id>`) mod 2⁶³. Save y and its SHA. Solver settings are unchanged apart from the operator; if SITCOM cannot take the oversample setting without code edits, run FAST only and record this |
| ImageNet images | DAPS's 100-image `test-imagenet` set (also used by FAST-DIPS and PC-DAPS), in the DAPS/FAST-DIPS loader order `sorted(Path(root).rglob(ext))`, indices 0–99. Record each file's SHA-256 in the XP-0 manifest |
| ImageNet measurements | New. y = A(x) + σ·n with n ~ N(0, I) and σ = 0.05. Seed = first 8 bytes of SHA-256(`XPMEAS\|imagenet\|<basename>`) as an unsigned integer mod 2⁶³. Save y and its SHA. **The same y serves every family** |
| Checkpoints | FFHQ: `ffhq_10m.pt` (SHA `81d53574…7faa`), also DPS's FFHQ model. ImageNet: `/egr/research-pac/huang248/models/imagenet256.pt` (2,211,383,297 bytes; record SHA-256 in XP-0) |
| FAST-DIPS | R0.5 frozen settings (T = 150, K = 2, S = 5, ρ = 200, ε = 0.05), `FAST_DIPS_A15` @ `a9758b38`, env `fast_dips_a15_blackwell`. These equal the repository's published phase-retrieval command for both FFHQ and ImageNet (`configs_fast_dips_cli.txt`). ImageNet uses the `imagenet256ddpm` preset. **XP-G keeps every setting**, including ε (a fixed constraint tolerance in the code, not a noise parameter) |
| SITCOM | FFHQ: the B22 native settings (`configs/b22/b22_2_overnight.json`, `sitcom` block). ImageNet: SITCOM's own ImageNet phase-retrieval settings if its code or paper defines them; otherwise the FFHQ settings with the ImageNet model. Record which |
| DAPS, FFHQ | The historical configuration of the B22/DEV80 DAPS runs (confirm against the R0-N.2 A4.4 record) |
| DAPS, ImageNet and XP-E′ | **DAPS-4K, the configuration of the DAPS paper's phase-retrieval results** (amendment A2), at `e7a77d0`: annealing 400 steps × diffusion 10 ODE steps; pixel task group with 100 Langevin steps, lr 5e-5, τ = 0.01, lr_min_ratio 0.01. Our frozen y and per-(image, run) seeds are fed in through the in-repository wrapper. Record the resolved configuration and any deviation. (The repository's `commands/pixel_*.sh` use 200 × 5 and are **not** the published PR configuration) |
| DPS (XP-H) | The official DPS repository (`DPS2022/diffusion-posterior-sampling`) at a commit pinned in XP-0, using its phase-retrieval config (record the guidance scale) with `ffhq_10m.pt`. Our y is fed in. If the code is absent, **Cheng** clones it into a new external path |
| PC-DAPS (XP-I) | PC-DAPS @ `59b2f046` with its own FFHQ phase-retrieval command (4 particles). Our y is fed in. Save every particle, plus the method's own returned estimate. If the code is absent, **Cheng** clones it |
| Seeds | **Continue each family's historical rule** for new run indices. Document the rule, and show that no new seed equals an existing one for the same (family, image). For families or datasets with no history (all ImageNet, XP-G, XP-G2, XP-H, XP-I, XP-E′): the first 8 bytes of SHA-256(`XP\|<family>\|<dataset>\|<uid>\|<r>`) mod 2³¹, where XP-G uses dataset `ffhq_s<σ>` |
| Batching | Batching across (image, run) pairs is allowed **only if** each sample draws from its own generator seeded by the rule above, and the replay gate (§4) passes for batched vs single execution. **Timing claims use XP-T batch-1 measurements only** |
| Labels, FFHQ (σ = 0.05) | R0 frozen labels: GOOD = PSNR_raw ≥ 25; ORBIT = raw < 25 and some non-identity member ≥ 25; CONSISTENT/INCONSISTENT by the **frozen R0.7 τ for that (source, family)**. τ is not re-estimated |
| Labels, ImageNet | **Primary threshold 20 dB**: GOOD_IN = PSNR_raw ≥ 20; ORBIT_IN = raw < 20 and orbit-best ≥ 20. τ_IN = the 0.95 quantile of ρ_signed over GOOD_IN runs per family, **computed on run indices 0–3 only and then frozen** for all later runs. **Secondary:** Good@{15, 18, 22, 25}, and the PC-DAPS orientation labels (amendment 2, A4.1 rule) |
| Labels, XP-G and XP-G2 | As FFHQ with GOOD at 25 dB. τ is computed per setting (each σ; the 512 canvas) and per family, on run indices 0–3, and then frozen |
| Labels, new families (DPS, PC-DAPS) | As FFHQ. τ is computed on that family's runs 0–3 (all its runs, for PC-DAPS) and reported as such |
| Orientation tests | **FFHQ:** the frozen N2 CLIP face prompts. **Generic set G1:** "a photo." vs "an upside-down photo.". **Ensemble G2:** upright {"a photo.", "a photo of a scene.", "a photo of an object.", "a good photo.", "a natural photo.", "a photo of something."} vs inverted {"an upside-down photo.", "an upside-down photo of a scene.", "an upside-down photo of an object.", "a photo that is upside down.", "a photo rotated by 180 degrees.", "an inverted photo."}. Embedding, normalization and the rule "choose 111 iff u(R₁₁₁x) > u(x)" exactly as N2. **Prior:** the 2-forward rule (t = 300, J = 1, paired R ε, global2) and S_mid (t ∈ {300, 400, 500, 600}, J = 4; global2 and all8), each with the dataset's own checkpoint. **UnivFD:** as N2 |
| Evaluation metrics | PSNR raw and for all 8 orbit members; SSIM and LPIPS (piq, as DAPS) for raw and member 111 (LPIPS may run on GPU: it is evaluation only); ρ_signed, ρ_clipped |

## 2. Stages and tiers

GPU-hour figures are planner estimates from historical timings (FAST 3.3 s/run was measured on this hardware in R0.5; the others come from older campaigns). **XP-T replaces them.** Each estimate assumes one worker per GPU.

**Tier A: required, data cut Nov 2.**

| Stage | Content | New runs | Est. GPU-h |
|---|---|---|---|
| **XP-0** | Bounded inventory (§4 of the handoff): GPUs; `test-imagenet` present (`test -d`; if absent, **Cheng** downloads via the DAPS README gdown link into a new data path); checkpoint SHAs; environment smoke imports; presence of the DPS and PC-DAPS code; the per-id y-SHA equality check of §1 | — | CPU |
| **XP-T** | Timing and throughput on **one GPU model**. Batch-1 timing: 3 runs per (family, dataset), covering DAPS, SITCOM, NP and FAST on the first 3 B22 ids (sorted), and FAST, SITCOM and DAPS on the first 3 ImageNet basenames; plus DPS and PC-DAPS if their code is present. CUDA-event sampling time and process wall. Per-candidate time for CLIP (2 encodings), the 2-forward prior and S_mid all8. **Throughput:** W ∈ {1, 2, 4} concurrent workers on one GPU, for FAST and SITCOM on FFHQ; choose W\* per family (largest aggregate throughput with memory ≤ 80% and outputs matching W = 1 within 0.05 dB PSNR) | ≈ 60 | ≈ 1 |
| **XP-A16** | FAST on FFHQ, 180 ids × run indices 4–15, giving **n = 16 per id** together with S5's runs 0–3 | 2,160 | ≈ 2 |
| **XP-C** | Orientation tests on clean ground truth: 100 ImageNet and 180 FFHQ GT images, with CLIP (face set for FFHQ only; G1; G2), the prior (2-forward; S_mid global2 and all8) and UnivFD. Metric: identity-win rate (000 preferred over 111; for all8, over all 7 others) | — | < 0.5 |
| **XP-C2** | **Equivariance-defect profile** of each checkpoint (tests THEORY_PROGRAM TH1.1 Cor. 4), on the same GT images. For t ∈ {100, 200, …, 900} (DDPM-1000 index, mapped to each code's own noise parameterization) and R ∈ {R₁₁₁, R₁₀₀}, draw x_t from x★ with 2 noise draws per (image, t), seed SHA-256(`XPC2\|<dataset>\|<basename>\|<t>\|<j>`). Report δ(t) = mean ‖D_t(R x_t) − R D_t(x_t)‖² / (3·256²), and δ̃(t) = δ(t) / mean ‖D_t(x_t) − x★‖² / (3·256²), per dataset with bootstrap CIs | — | < 0.3 |
| **XP-D1** | ImageNet measurements (§1), then FAST n = 16 on all 100. **Phase 1** = runs 0–3; **phase 2** = runs 4–15 | 1,600 | ≈ 3–6 |

**Tier B: main paper, data cut Nov 6.**

| Stage | Content | New runs | Est. GPU-h |
|---|---|---|---|
| **XP-A64** | FAST on FFHQ, run indices 16–63 (n = 64) | 8,640 | ≈ 8 |
| **XP-B16** | SITCOM on FFHQ, **all 180 ids**: B22 runs 4–15 and DEV80 runs 4–15 (n = 16) | 2,160 | ≈ 29–53 |
| **XP-D2** | DAPS on ImageNet, n = 4 on all 100, with **DAPS-4K** (the published PR configuration; A2) | 400 | ≈ 35–80 |
| **XP-D3** | SITCOM on ImageNet, n = 4 on all 100 | 400 | ≈ 11–28 |
| **XP-F8** | DAPS on FFHQ, all 180 ids: B22 runs 2–7 and DEV80 runs 4–7 (n = 8) | 920 | ≈ 40–50 |

**Tier C: supplement, data cut Nov 20.**

| Stage | Content | New runs | Est. GPU-h |
|---|---|---|---|
| **XP-B32** | SITCOM on FFHQ, run indices 16–31 on all 180 ids (n = 32) | 2,880 | ≈ 39–71 |
| **XP-H** | DPS on FFHQ, n = 4 on all 180 ids | 720 | ≈ 12–30 |
| **XP-I** | PC-DAPS on FFHQ, 1 run (4 particles) on all 180 ids. Compare at matched NFE with DAPS + E1_CLIP | 180 | ≈ 16–32 |
| **XP-G** | FAST on FFHQ, σ ∈ {0.02, 0.10}, n = 8 per σ on all 180 ids | 2,880 | ≈ 3 |
| **XP-D4** | FAST on ImageNet, run indices 16–63 (n = 64) | 4,800 | ≈ 9–17 |
| **XP-G2** | 512-canvas control (§1) on all 180 ids: FAST n = 8 and SITCOM n = 4 | 2,160 | ≈ 11–20 |
| **XP-E′** | **Conditional:** DAPS-4K (the published PR configuration), n = 4 on all 180 ids. Runs only if A4.4 shows that the historical FFHQ DAPS configuration differs from DAPS-4K | 720 | ≈ 20–40 |
| **XP-J** | **Conditional (A4):** DiffFPR, n = 4 on all 180 ids, at the 512 canvas (its native setting; XP-G2 measurements), and at 384 only if that needs no code edits | 720–1,440 | ≈ 10–40 |
| ~~XP-E~~ | **Cancelled (A1).** DAPS `test-ffhq` overlaps the locked holdout | — | — |

**Scoring** (CLIP, 2-forward, S_mid, LPIPS) adds about 5–15 GPU-h in total. Estimated grand total: ≈ 250–450 GPU-h. The upper end exceeds the cap, so the drop order (§3.5) will bind if the XP-T projections land high.

## 3. Queue rules

1. **Order of value** (launch priority): XP-T → XP-A16, XP-C, XP-C2, XP-D1 phase 1 → XP-D1 phase 2, XP-A64 → XP-B16 ∥ XP-D2 → XP-D3 ∥ XP-F8 → XP-B32 → XP-I → XP-H → XP-G2 → XP-J → XP-G → XP-D4 → XP-E′ (conditional). Amendment A4 assigns these stages to the CRITICAL and BACKGROUND lanes.
2. **Work-conserving.** No GPU is left idle while any authorized, smoke-passed job is ready. A lower-priority job that is ready may fill a GPU while a higher one is still in engineering. Example: XP-B32 needs no new code, so it may run while XP-H/XP-I are being built.
3. **Concurrency:** W\* workers per GPU, per family, from XP-T. Never exceed 80% GPU memory.
4. **Smoke test per (family, dataset):** 2 images × 2 runs. Runtime must be within ±50% of the XP-T projection, and the configuration must replay the frozen settings. Otherwise stop that family and report.
5. **Drop order** if over budget or late (the reverse of value): XP-E′ → XP-J → XP-D4 → XP-G → XP-G2 → XP-H → XP-I → XP-B32 → XP-F8 → XP-D3 → XP-B16 reduced to the 60 ids of v1 (SHA-256(`XPB|<id>`), first 30 B22 + first 30 DEV80) → XP-D2 reduced to 25 images (SHA-256(`XPD|<basename>`)) → XP-A64. **Tier A is never dropped.**
6. **Budget check.** Before launching each stage, project its GPU-h from XP-T. Stop if the cumulative projection exceeds 400.

## 4. Gates for adding runs to historical families (XP-B, XP-F)

The K-scaling estimators pool new runs with historical runs, so the two must be **exchangeable**: the same distribution of outcomes given y.

1. **Replay gate.** On the current code, environment and GPU, re-run historical run indices 0 and 1 for the first 2 ids (sorted) of each (cohort, family). Pass if, for every replayed run, the raw PSNR is within 0.1 dB of the stored candidate and the R0 label is identical. Bitwise identity is recorded if it holds.
2. **If replay fails:** stop that family and report the differences. The planner then decides between (a) proceeding with new runs only (K-curves from new runs, with historical runs analyzed separately) and (b) a distribution check on 20 ids × 4 new runs.
3. **Seeds.** New seeds come from the historical rule (§1). Report the rule and the no-collision check.
4. FAST S5 (XP-A) was produced in R0.5 on this hardware. Its gate is the same replay of 2 ids × 2 runs.

## 5. Theory predictions, freeze order and embargo

1. **Freeze v1 (FFHQ).** Before any FFHQ outcome with run index ≥ 4 is analyzed or shared, the theory executor commits `docs/track_r/theory/TH1_PREDICTIONS_v1.json` on its theory branch. The fit uses only data with K ≤ 4 (R0 runs; A8 counts). It contains predicted values with 95% intervals:
   - **FAST**, S5 180 ids: K ∈ {8, 12, 16, 24, 32, 48, 64};
   - **SITCOM**, 180 ids, per cohort and pooled: K ∈ {8, 12, 16, 24, 32};
   - **DAPS**, 180 ids, per cohort and pooled: K ∈ {4 (B22), 6, 8};
   - policies E0, E1_CLIP, O_raw and O_orbit; metric Good25;
   - plus the model-implied **shape prediction**: the gaps O_orbit − E1_CLIP and O_raw − E0 are non-decreasing in K (THEORY_PROGRAM v1.1, TH1.2).
2. **Freeze v2 (ImageNet FAST, optional but preferred).** After the XP-D1 phase-1 per-image counts (runs 0–3) are committed, the theory executor may commit `TH1_PREDICTIONS_v2.json` for FAST ImageNet at K ∈ {8, 16} (and 32, 64 if XP-D4 runs), Good20, using runs 0–3 only. If v2 is not committed within 48 h of the phase-1 counts, ImageNet K-scaling is reported as descriptive only.
3. **Registry.** The planner records each freeze in `docs/track_r/theory/PREDICTIONS_REGISTRY.md` on the planner branch: theory branch, commit SHA, file SHA-256, commit time, and the time the planner saw it on GitHub.
4. **Order check.** The XP analysis script verifies, for each prediction file used, that its registry entry exists, that the SHA-256 of `git show <sha>:<path>` matches, and that its commit time precedes the XP commit that first contains the matching outcomes.
5. **Embargo.** Until the matching freeze is registered, the executor commits and reports **no outcome data** (labels, counts, metrics or curves) for:
   - FFHQ runs with index ≥ 4 in XP-A, XP-B or XP-F (and index ≥ 2 for B22 DAPS);
   - ImageNet FAST runs with index ≥ 4.

   **Allowed before the freeze:** completion receipts (run counts and failures only), timing, XP-C and XP-C2 results, XP-D1 phase-1 counts, XP-D2/XP-D3 outcomes (no predictions concern them), and the Tier C stages other than XP-B32 and XP-D4.
6. **Agreement criterion,** per prediction: SE_comb = √(SE_obs² + ((hi95 − lo95)/3.92)²). "Consistent" if |obs − pred| ≤ 2·SE_comb, otherwise "rejected". The shape prediction is "consistent" if no bootstrap-significant decrease (one-sided 95%) occurs between consecutive K in the grid. **Both outcomes are reported.**

## 6. Policies and estimators (analysis on CPU)

**Policies:**
- first run;
- **mean-of-K** (mean metric over the K runs);
- E0 (minimum ρ_signed; ties → lowest run index);
- **E1_CLIP**: E0 followed by the dataset's primary CLIP global decision (FFHQ: face set; ImageNet: G1);
- E1_CLIP_G2; E1_2F (2-forward prior); E1_S8 (the R0-frozen S_mid all8 repair);
- O_raw;
- O_orbit;
- **per-metric oracle**: best PSNR, best SSIM and best LPIPS chosen independently over the K runs (the DAPS/FAST-DIPS code convention);
- **single-pick oracle**: the best-PSNR run, with all of its metrics reported;
- for PC-DAPS only, its own returned estimate (as published) in addition to the above on its particles.

**Estimators.**
- **Primary: exact U-statistics** over all K-subsets of an image's n runs, by order-statistic identities (no enumeration, no random subsampling). For a selector that picks the argmin of a score, with the runs ranked 1…n by ascending score (ties by run index), run j of rank r_j is picked with probability C(n − r_j, K − 1) / C(n, K). For a max-type oracle, the run of ascending rank j is the maximum with probability C(j − 1, K − 1) / C(n, K). Good-type oracles use 1 − C(n − g, K) / C(n, K) (pass@k form).
- **Continuity:** also report the first-K version (runs 0 … K−1).
- **CIs:** image-cluster bootstrap, 10,000 replicates, seed rule `XPBOOT|<stage>`.

**Outputs:**
- K-curves for every policy (Good25 or Good20, mean, q05), with predictions overlaid where frozen;
- a compute frontier: success vs seconds per image, from XP-T batch-1 timings only;
- failure anatomy per dataset and family, and the post-selection decomposition (generation failure vs selection miss vs uncorrected orientation) as a function of K;
- orientation-test accuracy per domain (XP-C on GT; on reconstructions: fix rate and harm rate);
- the inflation decomposition per metric: per-metric oracle vs single-pick vs E1_CLIP vs mean-of-K vs first run;
- XP-D2: the published DAPS protocol (DAPS-4K; per-metric oracle over 4 runs) next to GT-free E1_CLIP and first run, with the published Table 1 values (25.78 ± 6.92 dB / 0.743 / 0.254) as the external reference;
- XP-I: PC-DAPS vs DAPS + E1_CLIP at matched NFE on the same ids;
- XP-G: anatomy and E1_CLIP vs O_orbit at σ ∈ {0.02, 0.05, 0.10}.
- XP-G2: anatomy at the 384 vs 512 canvas, with the CONSISTENT share per family as the primary descriptive quantity.

**Exploratory analyses** (A1; CPU; optional; after the Tier B analysis; labeled EXPLORATORY; no threshold is tuned for claims):
- **GT-free reliability diagnostics** for the E1_CLIP pick. Report AUROC for predicting that the pick is a failure, from:
  - (a) the pick's ρ_signed;
  - (b) cross-run agreement: the median PSNR between the pick and the other K − 1 runs, each first aligned to the pick by its best orbit member (GT-free);
  - (c) a PRTF-style score in the spirit of Chapman et al. (2006): the magnitude of the mean of the G-aligned runs' Fourier transforms over the measured magnitudes, averaged over frequency bands.

  Report per family and per dataset at K = 4 and K = 16.
- **Mixed-twin anatomy** (Fienup & Wackerman 1986) of INCONSISTENT runs. For each frequency, ask whether the run's Fourier phase is closer to the phase of x★ or of its global twin R₁₁₁x★. Report the magnitude-weighted fraction closer to x★, and its spatial-frequency profile.
  - A value near 0.5 with a structured profile indicates a simultaneous-twin stagnation state.
  - This uses GT, so it is evaluation only.

## 7. Outputs and data rules

- **PAC roots:** `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/xp_<stage>_<UTCSTAMP>/`.
- **Committed** (`docs/track_r/executors/xp/returns/XP_<UTCSTAMP>/`): aggregate CSV/JSON, figures, SHA256SUMS, and **per-image strings** (one row per (dataset, cohort, family, stage, id)):
  - counts: `n_runs`, `n_GOOD`, `n_ORBIT_global`, `n_ORBIT_partial`, `n_CONSISTENT`, `n_INCONSISTENT`;
  - `run_labels` in run order, alphabet G (GOOD), O (ORBIT global: member 111 is the best member), P (ORBIT partial), C, I;
  - `e0_rank_order`: run indices sorted by ascending ρ_signed (ties by index), comma-separated;
  - GT-free decisions per run in run order: `dec_clip_face` (FFHQ), `dec_clip_g1`, `dec_clip_g2`, `dec_prior2f` as 0/1 (1 = choose 111), and `dec_smid_all8` as the chosen member index 0–7.
  - **No paths, metric values or images.** Labels are GT-derived; Cheng approved committing such strings on 2026-10-26.
- **Delivered as a planner ZIP, not committed:** per-candidate tables (with metric values) and contact sheets (as in amendment 2 §A7).
- **GT** is used only for evaluation and labels. Selection policies never read GT.
- **Forbidden:**
  - access to confirmation305;
  - FFHQ ids beyond §1;
  - loading any bundled FFHQ set (`test-ffhq`, `demo-ffhq`, DAPS FFHQ archives);
  - merging the data-governance audit branch (`docs/daps-test-ffhq-confirmation305-audit-*`) into any Track R branch: its file path contains a forbidden token;
  - any change to a historical environment, output or checkout.
