# Top-venue attempt — planner plan (2026-10-25; updated 2026-10-27)

> **Fourth update, 2026-10-27, 12:30 ET** (the newest; read it first).
> - **Overnight GPU (executor, issue #42):**
>   - generated and certified: XP-A16 (2,160 runs), XP-A64 (8,640), XP-D1 phase 1 (400) and XP-D2 (400);
>   - about 39 GPU-h used of 400;
>   - all FFHQ K > 4 outcomes stay sealed.
> - **Theory:**
>   - the S2 rejection is independently verified;
>   - the race model has a weighted ceiling and **no monotone gap** (counterexample checked);
>   - the conditional-prefix estimand is adopted;
>   - the sampled noise-floor law is Eρ²/σ² = 1 − (r/m)(1 − 1/κ);
>   - TH2 mechanism forecasts v1 are **registered**.
> - **Planner check of the theory code:** the planner ran the theory scripts on the real A8 file. The audit passes, and the forecaster runs end to end in 95 s; its non-convergence flags are BFGS precision artifacts.
> - **Prereg A6:**
>   - A4.4 ruled from documents (400 × 5), with the replay to confirm;
>   - XP-E′ eligible and CRITICAL-if-ready;
>   - D1 phase 2 generated now;
>   - evaluation panels fixed (FAST n = 64 and 16; SITCOM 16 and 32; DAPS 8; ImageNet 16);
>   - predictive-interval agreement rule;
>   - a `theory_inputs/` directory;
>   - XP-D2 and D1 phase 1 scored now.
> - **Next 48 h:**
>   - B16, D3, D1-P2, replay, F8 and E′ (about 55 GPU-h);
>   - v1 freeze, target Oct 28, 18:00 ET;
>   - v2 within 48 h of the D1-P1 strings;
>   - the FFHQ K-scaling analysis starts as soon as v1 is registered.
>
> **Third update, 2026-10-27, 00:00 ET.**
>
> **Decision recorded:** NP is dropped (Cheng, 2026-10-26 23:01). Every paper count is now on DAPS, SITCOM and FAST-DIPS: 1,960 runs, 540 image–solver pairs.
>
> **Verified facts** (A8 @ `83abf26`, A4.3, B2, N1c; ledger C-A1–C-A7, C-S5–C-S7):
> 1. **GT-free selection.** E0 followed by the CLIP test recovers 92.6–97.5% of the orbit-oracle mean-PSNR gain. It beats the GT raw oracle in 4 of 5 strata, and its Good25 is within 0–2 images of O_orbit.
> 2. **The residual is a strong within-image verifier.**
>    - GOOD beats CONSISTENT in 200/205 pairs and INCONSISTENT in 570/571.
>    - In the 100 mixed-passing pairs, E0 picks GOOD in 92, against 51.9 expected under S2. **S2 is rejected.**
>    - Only 3 selection misses occur in 540 pairs.
> 3. **Generation failure dominates at K ≤ 4:** 66 of the 70 failures left after CLIP.
> 4. **Noise floor.** Every GOOD run of DAPS and SITCOM has ρ < σ (medians 0.946σ and 0.884σ); FAST sits at 0.997σ, while the truth has ρ★ = σ ± 0.1%.
>    - The cause is in the code. DAPS and SITCOM-ODE weight the data term 50× and 25× the likelihood at σ = 0.05 and return a Langevin iterate. FAST constrains ‖v − y‖ ≤ 0.05 (vs σ√m ≈ 33) but returns its denoised estimate.
>    - So τ is solver-relative, and for DAPS and SITCOM it would reject the ground truth.
> 5. **Orientation.**
>    - CLIP fixes 58/58 global orbit errors with 0/1,171 harm.
>    - Rotated runs fit worse than GOOD runs for DAPS (21/24) and FAST (15/15), but not for SITCOM (14/30). The solver's non-equivariance thus leaks into the ranks, while Cor. 1 still holds exactly.
>
> **Actions taken:**
> - Prereg **A5**:
>   - XP-I becomes CRITICAL-if-ready, and XP-I2 is added;
>   - new XP-K, a parameter-only calibration of each data term to σ;
>   - analysis A5.5 (Δ = D − 2C per candidate), A8b, and A7 now required;
>   - the F8 replay gate;
>   - v1 freezes both the S2 reference and a rank-calibrated forecast;
>   - revised queue and drop orders.
> - Theory guidance and executor notes (2026-10-27).
> - Paper revised (Secs. 0–7, ledger).
> - Freeze target moved up to **Oct 29, 18:00 ET** (hard deadline Oct 31).
>
> **Planner's view.**
> - **The story changes, for the better.**
>   - The K ≤ 4 data do not support "an imperfect verifier whose ceiling bites as K grows".
>   - The measured picture is sharper: the residual **cannot** certify orientation (exact symmetry) or absolute correctness (a solver-set noise floor below σ), but it **can** rank runs of the same image almost perfectly.
>   - Today, what limits the protocol is generation, not selection.
>   - The frozen K-scaling test now asks a real question: does the near-perfect paired ranking survive to K = 64 (race), or do rare low-residual wrong runs take over (tail competition; SITCOM is the place to look)? Either outcome is publishable, and the forecasts are written before the data.
> - **Risks:**
>   1. If the race model simply holds, the K-curves confirm "selection is not the bottleneck", which is solid but less dramatic. The paper then rests on the noise floor, the symmetry results and the audit.
>   2. The CONSISTENT class may be mostly near misses rather than other images. The A7 contact sheets decide, and the wording is hedged until then.
>   3. The theory is standard machinery applied carefully. Reviewers will judge the empirical findings.
>   4. PC-DAPS (Oct 22) is concurrent work that reviewers will raise; XP-I is the answer.
>   5. Page budget: about 7.5 pages of text already, with placeholders. The estimators and part of Sec. 4 will move to the supplement.
> - **Publication image:**
>   - Title direction (decide Nov 10): something like "What Can a Measurement Residual Certify? Symmetry, Noise Floors and Best-of-K in Diffusion Phase Retrieval".
>   - Main text:
>     - Fig. 1: the orbit and the defect proxy;
>     - Tab. 1: protocols;
>     - Tab. 2: the FFHQ decomposition (done);
>     - Fig. 2: K-curves with S2 and race bands, plus rank win rates;
>     - Tab. 3: ImageNet, including the published DAPS-4K protocol;
>     - a noise-floor table with XP-K;
>     - a PC-DAPS comparison, if ready.
> - **Odds.** CVPR acceptance is plausible but not likely: roughly 20–30% if Tier A/B and v1 land cleanly. SPL/EUSIPCO remains a strong fallback whatever happens.

> **Update 2026-10-26 (Cheng's decisions, 00:57).**
>
> | # | Decision | Cheng's answer | Effect |
> |---|---|---|---|
> | 1 | GO for XP | **"GO aggressively … Try to get as much as possible out."** | `XP_PREREG_v1_1.md` supersedes v1 (no data yet): tiers A/B/C, cap 400 GPU-h on 4 × RTX PRO 6000 |
> | 2 | CVPR or SPL | **CVPR** | Unchanged: go/no-go Nov 8, enrollment Nov 10, paper Nov 16 |
> | 3 | Start the theory executor | **Yes** | Handoff revision B; `THEORY_PROGRAM_v1.md` is now v1.1 |
> | 4 | Commit the A8 label strings | **Yes** | Amendment 2 revision D.1. A8 also gains `e0_rank_order` and the decision strings |
> | 5 | Title | **No fixed title**; keep A as a candidate, stay flexible | See below |
>
> **XP v1.1 in one line.**
> - **Tier A, by Nov 2:** FAST FFHQ n = 16; XP-C; the XP-C2 equivariance defect; FAST ImageNet n = 16.
> - **Tier B, by Nov 6:** FAST FFHQ n = 64; SITCOM FFHQ n = 16 on all 180 ids; DAPS ImageNet n = 4 on all 100 (the published PR configuration, DAPS-4K, per amendment A2); SITCOM ImageNet n = 4; DAPS FFHQ n = 8.
> - **Tier C, by Nov 20, for the supplement:** SITCOM n = 32; DPS; PC-DAPS on the same ids; noise robustness; FAST ImageNet n = 64. XP-E was later cancelled; see the second update below.
>
> **Theory v1.1, from the full reads** (`2026-10-26_THEORY_LIT_DIGEST.md`):
> - **Our E0 formula is Stroebl et al.'s App. C model.** Argmin equals accept-first in law under S1–S2. We credit them. The novelty is the exact symmetry, the estimators and the frozen out-of-sample predictions.
> - **New Corollary 4.** Measurement guidance is provably orientation-blind, so orientation preference comes only from the network's non-equivariance. This is the inverse-problem analog of Dandapanthula & Boffi's Theorem 6. XP-C2 measures it.
> - **New monotone-gap theorem.** The GT-free gap is non-decreasing in K. It is a free, falsifiable shape prediction.
> - **TDS.** Asymptotic exactness is with respect to the model posterior, which keeps orbit and decoy mass. Exact sampling bounds what selection can achieve but does not remove failures.
>
> **Prediction freeze.**
> - v1 (FFHQ) by **Oct 31**. v2 (ImageNet FAST) is optional.
> - **Embargo** on K > 4 outcomes until each freeze. The registry is `docs/track_r/theory/PREDICTIONS_REGISTRY.md`.
>
> **Go/no-go on Nov 8.** The four conditions of §3 still apply. **XP-B16 (SITCOM, at least partly back) is now strongly desired**, because it is the theory's c ≫ 0 regime.
>
> **Title candidates** (flexible; choose at enrollment, Nov 10):
> - A: "Best-of-K Without Ground Truth: An Audit and a Selection Theory for Diffusion Phase Retrieval"
> - C: "What Does Best-of-K Measure? Symmetry, Selection Ceilings and an Audit of Diffusion Phase Retrieval"
> - D: "Orientation-Blind Guidance and Ground-Truth-Free Selection in Diffusion Phase Retrieval"
>
> **Papers.**
> - The file labeled "TGD" was TDS (Wu et al. 2023). TGD (arXiv 2605.03712) is still missing, and a direct fetch was not permitted.
> - Received and read in full: Stroebl et al.; Dandapanthula & Boffi; TDS.
>
> **Second update, 2026-10-26 (03:00).**
> - **XP-E is cancelled.** Cheng's audit found that DAPS `test-ffhq` shares one id with the locked holdout. A new hard rule bans loading any bundled FFHQ set (prereg amendment A1). The FFHQ bridge to published settings remains through S5 (FAST), XP-H (DPS) and the conditional XP-E′ (the published DAPS-4K configuration on our 180 ids; amendment A2).
> - **TGD is now read in full** (digest §4). Facts for the novelty map:
>   - its App. F.4 remark on DAPS's per-metric oracle is verbatim and verified;
>   - residual-selected DAPS best-of-4 reaches **31.40 ± 2.38 dB at about 17 s/img on FFHQ PR**, beating A-TGD (its Table 4);
>   - A-TGD for phase retrieval *is* an early-pruned, residual-selected best-of-4 (λ ≡ 1, no resampling).
>
>   So "residual selection is a strong GT-free baseline" is **TGD's empirical point, not ours.** We add:
>   - the theory that residual and likelihood selection (including SMC weights and pruning) is **G-blind** (Cor. 1, 1′);
>   - an orientation step;
>   - a cross-solver decomposition;
>   - K-scaling against frozen predictions;
>   - the anatomy.
> - **Research pass** (digest §5):
>   - our canvas violates the classical uniqueness hypothesis, so we assume orbit identifiability and add a 512-canvas control (XP-G2);
>   - **coherent diffraction imaging already selects multi-start runs by Fourier error** (GHIO 2007; Chapman et al. 2006), so E0 has precedents we must cite;
>   - a weak-assumption theorem is now in scope: Cor. 4(b), a TV bound on orientation preference by the network's equivariance defect.
>
> **Novelty map, revised:**
>
> | Point | Already taken | What remains ours |
> |---|---|---|
> | GT best-of-k misleads | PC-DAPS (qualitative); TGD F.4 (per-metric oracle remark) | The measured decomposition across 4–6 solvers × 2 datasets × 3 metrics |
> | Residual-selected best-of-K | TGD (DAPS; matched wall-clock); CDI multi-start practice | G-blindness theory (Cor. 1, 1′); orientation step; K to 64 with frozen predictions |
> | 8-fold orbit | DAPS App. F.2 / Fig. 13 (verified: "eight images … exactly the same measurement"; Fig. 18: more 180° rotations on ImageNet); classical trivial associates; Fienup & Wackerman's equal-probability twins | Prop. 1 and Cor. 4/4(b) (equivariance mechanism, measured defect) |
> | Imperfect-verifier model | Stroebl et al. | Orbit classes, exact estimators, out-of-sample test, monotone gap |

**Trigger.** Cheng (2026-10-25 23:17) asked to:
- aim higher than EUSIPCO (SPL or CVPR), keeping EUSIPCO as the fallback;
- strengthen or redesign the experiments;
- scope a theory direction, with a handoff for a separate executor.

**This note is binding for planning.** It is not GPU authorization.
- **Authorized here:** the R0-N.2 CPU block (amendment 2 revision D) and the theory executor (no PAC).
- **Needs Cheng's explicit GO:** GPU stage XP (`XP_PREREG_v1.md`).

**Companion documents:**
- `XP_PREREG_v1.md` (experiments);
- `THEORY_PROGRAM_v1.md` (theory);
- `docs/track_r/executors/xp/00_XP_EXECUTOR_HANDOFF.md`;
- `docs/track_r/executors/theory/00_THEORY_EXECUTOR_HANDOFF.md`;
- amendment 2 revision D.

---

## 0. Planner's view for Cheng

### How I view the current results

**The FFHQ audit is complete, replayed and planner-verified:** 3,240 runs, four solver families, two cohorts.

**What stands:**
- **GT-free selection.** Signed residual, then a generic uprightness test, recovers **89–97% of the gain to the orbit oracle O_orbit** (mean PSNR) in every family. See §2: the "93–129%" figure was relative to O_raw and should not be used.
- **What remains.** After that, 93 of the 99 still-failing image-strata are generation failures.
- **SITCOM.** Its failures fit y in 57% of cases.
- **Orientation.** A generic test suffices. CLIP fixes 84/84 global errors with 0/2,040 harm.

**What does not stand:** diffusion-prior repair as a method (K2).

**Novelty map**, from three research passes and the planner's full reads:
- **Not found elsewhere:**
  - per-solver consistent-wrong rates;
  - a quantitative per-metric vs single-pick vs GT-free vs mean-of-K decomposition;
  - a seed-vs-image variance split;
  - a probabilistic best-of-K selection model;
  - a formal exact-orbit treatment.
- **Already taken:**
  - the qualitative best-of-k critique (PC-DAPS);
  - the per-metric-oracle remark (TGD App. F.4);
  - residual-selected DAPS at matched compute (TGD);
  - ImageNet orientation prevalence (PC-DAPS).

### My plan

1. **Attempt CVPR 2027** (enrollment Nov 10, paper Nov 16), with a hard **go/no-go on Nov 8**.
2. **Fallbacks:** SPL (rolling) and EUSIPCO 2027 (Feb 6; withdraw from CVPR by Feb 5 if its reviews, due Jan 25, are poor).
3. **Strengthen with three things that are new:**
   - a **theory** section: the orbit proposition, a selection model with pre-registered **out-of-sample predictions**, and a per-metric inflation formula;
   - **one more dataset** (ImageNet);
   - **K-scaling** that tests the theory.
4. **Run three workstreams in parallel:**
   - PAC executor: R0-N.2 CPU block now, then XP GPU blocks on GO;
   - theory executor: TH1 by Nov 6, TH2 memo by Nov 13;
   - planner: drafts the CVPR paper from Oct 27.

### Publication image (CVPR 2027 attempt)

**Working title (pick one):**
- A: "Best-of-K Without Ground Truth: An Audit and a Selection Theory for Diffusion Phase Retrieval"
- B: "Is the Ground Truth Doing the Work? Auditing Best-of-K Evaluation in Diffusion-Based Phase Retrieval"

**Content (8 pages + supplement):**
1. The protocol audit:
   - the literature protocol table;
   - the inflation decomposition, on PSNR/SSIM/LPIPS, FFHQ and ImageNet.
2. Theory:
   - the exact-orbit proposition and its selector corollaries;
   - the (p, o, c) best-of-K selection model with closed forms and ceilings;
   - the per-metric oracle inflation.
3. K-scaling to K = 16 (FAST, and SITCOM on a subset) against **pre-registered model predictions**, plus a compute frontier.
4. Failure anatomy on FFHQ and ImageNet:
   - the post-selection decomposition;
   - instance hardness: seed vs image, and cross-solver agreement.
5. Orientation tests across domains: CLIP vs the solver's prior vs a realness detector.
6. Reporting recommendations.

**Odds (planner judgment):**
- CVPR ≈ 8–15%;
- SPL ≈ 25–35% with the same content condensed;
- EUSIPCO ≈ 45–55% with these additions.

Every path reuses the same work.

---

## 1. What was checked today

- **The attachment.** It is byte-identical (MD5 `fe894c56…`) to the Oct 24 03:46 N2 message, which was already verified against GitHub (comparison JSON SHA `322c3ef0…0233` recomputed) and ruled on in decision §10 (K2 TRIGGERED). Nothing new.
- **The executor branch.** There are **no commits after `33a34bd`**, so R0-N.2 (amendment 2) has not run, or has not been pushed. It now becomes urgent (§7).
- **Missing papers.** Cheng's note says reference papers were to be attached, but none arrived; only the PC-DAPS PDF from Oct 24 is present. The priority list is in §10.
- **Research reports read in full** (all 2026-10-24):
  1. "Prior Art, Scoop Risk and Venue Choice";
  2. "Scoping Extensions for a EUSIPCO 2027 Audit";
  3. "Theory of Diffusion-Prior Inverse Solvers for Fourier Phase Retrieval".
- **CVPR 2027 dates**, from the official call for papers ([cvpr.thecvf.com/Conferences/2027/CallForPapers](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)):
  - paper enrollment **Nov 10, 2026**; full paper **Nov 16, 2026** (AoE, no extension); supplementary Nov 23;
  - **reviews Jan 25, 2027**; rebuttal Jan 25 – Feb 1; decisions Feb 25;
  - the dual-submission window runs Nov 16 → Feb 22;
  - work posted after Sept 15, 2026 counts as contemporaneous (this covers PC-DAPS, posted Oct 22);
  - **reviewing is mandatory for authors** unless exempted.
- **ICML 2027.** About Jan 22 (estimated, not official). It conflicts with the EUSIPCO fallback, so it is not recommended.

## 2. Correction to the headline ratio (prompted by the theory report)

The earlier "GT-free recovers 93–129% of the gain" compared E1 with **O_raw**. O_raw cannot change orientation, whereas E1 can, so the ratio can exceed 100%.

The right oracle for an orientation-correcting protocol is **O_orbit**, and E1 ≤ O_orbit per image by construction. Planner arithmetic from the R0.8 P4 matrix, mean PSNR (Good25 in parentheses):

| Stratum, max K | First run | O_raw | O_orbit | E1 | Recovered vs O_orbit |
|---|---|---|---|---|---|
| B22 DAPS, K=2 | 27.23 (0.80) | 29.42 (0.92) | 30.14 (0.95) | 30.07 (0.95) | 97.4% |
| B22 SITCOM, K=4 | 22.97 (0.71) | 26.66 (0.95) | 26.87 (0.96) | 26.64 (0.94) | 94.3% |
| B22 NP, K=8 | 25.58 (0.75) | 29.40 (0.96) | 29.40 (0.96) | 29.12 (0.94) | 92.7% |
| DEV80 DAPS, K=4 | 23.29 (0.59) | 28.83 (0.85) | 29.03 (0.86) | 28.82 (0.86) | 96.4% |
| DEV80 SITCOM, K=4 | 19.59 (0.43) | 24.12 (0.69) | 25.06 (0.73) | 24.70 (0.73) | 93.4% |
| DEV80 NP, K=4 | 22.32 (0.51) | 26.07 (0.71) | 26.29 (0.71) | 25.86 (0.70) | 89.2% |
| FAST (S5), K=4 | 23.81 (0.57) | 29.00 (0.86) | 29.18 (0.87) | 28.99 (0.86) | 96.3% |

**Headline from now on:**
- GT-free selection recovers **89–97%** of the orbit oracle's gain.
- It is **0–2 images below O_orbit** in Good25.
- It is ≥ O_raw (the field's convention) wherever the GT pick is itself misoriented.

## 3. Venue strategy

| Path | When | Fit | Odds | Fallback mechanics |
|---|---|---|---|---|
| **CVPR 2027 (recommended attempt)** | Enrollment Nov 10; paper Nov 16 | 8 pages: audit + theory + 2 datasets + K-scaling | ≈8–15% | Reviews Jan 25. If poor: withdraw by **Feb 5** → EUSIPCO Feb 6. If rebutted and rejected Feb 25 → SPL (rolling) or ICIP (Mar 31, to be verified) |
| SPL | Any time (target ≈ Dec 10 if CVPR is no-go) | 4 + 1 pages, theory + condensed audit | ≈25–35% | First decision usually in 1–2 months. Withdraw by Feb 5 if still pending and EUSIPCO is wanted |
| EUSIPCO 2027 | Feb 6 | 5 pages incl. references | ≈45–55% | ICIP 2027 |

**Go/no-go rule for CVPR, Nov 8.** GO requires all four:
1. R0-N.2 returned;
2. XP-A and XP-C returned;
3. TH1 propositions written with frozen predictions;
4. XP-D at least partially returned (FAST ImageNet).

If any is missing, switch to SPL, targeting ≈ Dec 10. The same draft is condensed.

**Costs Cheng should weigh:**
- about 20–30 hours over three weeks: PAC blocks plus editing;
- CVPR's mandatory author reviewing in Dec–Jan;
- the Jan 25 – Feb 5 window to decide between rebuttal and withdrawal.

## 4. What is new relative to the closest work

| Point | Closest work | What we add |
|---|---|---|
| GT best-of-k misleads | PC-DAPS (qualitative, plus a DAPS-only mean-of-k table) | Measured decomposition across 4 solvers × 2 datasets × 3 metrics, with each layer separated (GT selection, per-metric oracle, K) |
| Per-metric oracle | TGD App. F.4 (one remark) | Quantified, with a closed-form inflation model |
| Residual-selected best-of-K | TGD (DAPS only, matched wall-clock) | Cross-solver, symmetry-aware, failure-rate frontiers, K up to 16, tested against model predictions |
| Orientation | PC-DAPS (prevalence, in-sampler fix) | Exact-orbit proposition; generic test vs solver prior vs realness detector, across 2 domains |
| Failure types | Burns & Fridovich-Keil (toy); DAPS++ (failure rate) | Per-solver consistent-wrong rates; generation-failure dominance after selection; seed-vs-image split |
| Selection theory | pass@k; Stroebl et al. (LLMs) | First transfer to diffusion inverse solvers, with out-of-sample predictions |

## 5. Experiments (details and frozen choices in `XP_PREREG_v1.md`)

> **Superseded on 2026-10-26** by `XP_PREREG_v1_1.md` (the tiers and the 400 GPU-h cap in the update box). The table below is the v1 scope, kept for the record.

| ID | What | Compute | Priority |
|---|---|---|---|
| R0-N.2 (CPU) | N1h; protocol ladder (mean-of-K, E1_CLIP, vs O_orbit); PC-DAPS-label crosswalk; A7 observation export with SSIM/LPIPS; **A8 per-image label counts** (feeds TH1) | CPU | **Now; return by Oct 29** |
| XP-0/XP-T | GPU and ImageNet inventory; same-GPU timing calibration of all 4 solvers | ≈0.3 GPU-h | GO + 1 day |
| XP-A | FAST K-scaling on the 180 FFHQ ids, K → 16 | ≈2 GPU-h | Required |
| XP-C | Orientation tests on 100 ImageNet GT images: CLIP generic and ensemble prompts vs ImageNet prior vs UnivFD | Minutes | Required |
| XP-D | ImageNet audit (the field's 100-image test set): FAST K=8 (all), SITCOM K=4 (all), DAPS K=4 (25 images) | ≈20–30 GPU-h, parallel over GPUs | FAST required; SITCOM/DAPS best effort by Nov 6 |
| XP-B | SITCOM K-scaling (K → 16) on 60 FFHQ ids: tests the model's predicted SITCOM ceiling | ≈14 GPU-h | Optional, by Nov 6 |
| XP-E | FAST K=4 on the field's `test-ffhq` set, as a bridge to published numbers | ≈0.4 GPU-h | **Cancelled 2026-10-26:** not disjoint from the holdout (one shared id) |

The total cap is 45 GPU-hours. All of it is evaluation only, with no new method.

## 6. Theory (details in `THEORY_PROGRAM_v1.md`; handoff for a separate executor)

- **TH1 (for the paper, due Nov 6):**
  - TH1.1: the exact-orbit proposition, and the exactness of our R under the canvas convention;
  - TH1.2: the (p, o, c, u) selection model, with closed forms for O_raw, O_orbit, E0 and E1, ceilings and heterogeneity;
  - TH1.3: unbiased estimators (pass@k-type U-statistics);
  - TH1.4: per-metric oracle inflation;
  - TH1.5: **predictions frozen before XP-A/XP-B data exist.**
- **TH2 (separate study, feasibility memo by Nov 13).** Inner-functional theory: sample (DAPS) vs mode-in-ball (FAST-DIPS) vs range-MAP (SITCOM). It uses an exactly symmetric toy with a Gaussian-mixture prior plus a near-ambiguous decoy, and connects to speciation time.
  - Not for the CVPR main text, but possibly one supplement figure.
  - **Competition risk:** the PC-DAPS / Burns & Fridovich-Keil group.

## 7. Timeline to CVPR

| Date | PAC executor / Cheng | Theory executor | Planner |
|---|---|---|---|
| Oct 26–27 | Cheng says GO/NO-GO on XP; forwards amendment 2 rev D and the XP handoff | Starts TH1 | Theory and XP docs (this commit); paper skeleton |
| Oct 27–29 | R0-N.2 CPU block; XP-0/XP-T; XP-A/XP-C implementation and smoke | TH1.1, TH1.2 | Drafting §1–3 |
| Oct 30 | — | **TH1.5 predictions frozen** (using A8 counts if available, otherwise the P4 aggregates) | Review TH1 draft |
| Oct 30 – Nov 2 | XP-A, XP-C runs; XP-D implementation and smoke | TH1.3, TH1.4 | Results sections from R0-N.2 |
| Nov 2–5 | XP-D (and XP-B) runs, parallel | TH1 final (Nov 6) | Figures |
| **Nov 8** | — | — | **CVPR go/no-go** |
| Nov 10 | Enrollment: title, abstract, authors | TH2 memo (Nov 13) | Abstract |
| Nov 11–15 | Cheng/coauthor edits | — | Final draft |
| **Nov 16** | **Submit** | — | Supplement by Nov 23 |

## 8. Risks

| Risk | Mitigation |
|---|---|
| ImageNet test images absent on PAC | XP-0 checks; Cheng downloads DAPS's `test-imagenet` (gdown link in the DAPS README) |
| ImageNet runtimes (DAPS ≈ 320 s/run on a 4090) | DAPS limited to 25 images; it is the first item dropped |
| The theory gets called elementary | Frame it as explanatory, with **out-of-sample predictions tested by XP-A/XP-B**; cite the pass@k / imperfect-verifier lineage |
| CVPR reviewers see phase retrieval as niche | Lead with the protocol lesson that generalizes (best-of-K evaluation in generative inverse problems); ImageNet + theory |
| Burnout | Drop order: XP-B → DAPS-ImageNet → SITCOM-ImageNet. The Nov 8 no-go keeps SPL/EUSIPCO intact |

## 9. Decisions for Cheng (default if no reply)

| # | Decision | Default |
|---|---|---|
| 1 | GO for XP GPU stage (≤ 45 GPU-h, evaluation only) | **No GPU until "GO"** |
| 2 | CVPR attempt with the Nov 8 go/no-go, or SPL directly | CVPR attempt |
| 3 | Start the theory executor on TH1 now | Yes |
| 4 | XP-E: confirm that DAPS `test-ffhq` ids are not in confirmation305 (you check; agents do not) | Skip XP-E. **Resolved 2026-10-26: not disjoint, XP-E cancelled** |
| 5 | Title A or B | A |

## 10. Papers to attach, if you can (only abstracts or excerpts read so far; in priority order)

1. Makris, Fearnhead, Nemeth, *Tempered Guided Diffusion*, arXiv 2605.03712: App. F.4 and the matched wall-clock tables.
2. Dandapanthula & Boffi, arXiv 2606.02884.
3. Stroebl, Kapoor, Narayanan, *Inference Scaling fLaws*, arXiv 2411.17501.
4. Porteous et al., arXiv 2605.06538.
5. FAST-DIPS, arXiv 2603.01591: **Appendix A.3** (Propositions 1, 5, 6).
6. SITCOM, arXiv 2410.04479: **Appendix D** (Proposition D.1). You may know it already.
7. *Local MAP Sampling for Diffusion Models*, arXiv 2510.07343.
8. Biroli, Bonnaire, de Bortoli, Mézard, *Dynamical regimes of diffusion models*, Nat. Commun. 2024.
9. Crafts & Villa, IEEE OJSP 2025 (BIPSDA).
10. Burns & Fridovich-Keil, arXiv 2605.30330 (full PDF).

## 11. Sources

- CVPR 2027 call for papers: https://cvpr.thecvf.com/Conferences/2027/CallForPapers
- ICML 2027 estimate (not official): https://www.opencurious.com/ai-conference-deadlines/icml-2027
- Research reports (2026-10-24), as listed in §1.
- R0.8 P4 matrix (ZIP SHA `13c008f3…efab`); N1/N2 returns at `5af47db` / `33a34bd`.
