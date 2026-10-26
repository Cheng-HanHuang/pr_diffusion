# Top-venue attempt — planner plan (2026-10-25)

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

| ID | What | Compute | Priority |
|---|---|---|---|
| R0-N.2 (CPU) | N1h; protocol ladder (mean-of-K, E1_CLIP, vs O_orbit); PC-DAPS-label crosswalk; A7 observation export with SSIM/LPIPS; **A8 per-image label counts** (feeds TH1) | CPU | **Now; return by Oct 29** |
| XP-0/XP-T | GPU and ImageNet inventory; same-GPU timing calibration of all 4 solvers | ≈0.3 GPU-h | GO + 1 day |
| XP-A | FAST K-scaling on the 180 FFHQ ids, K → 16 | ≈2 GPU-h | Required |
| XP-C | Orientation tests on 100 ImageNet GT images: CLIP generic and ensemble prompts vs ImageNet prior vs UnivFD | Minutes | Required |
| XP-D | ImageNet audit (the field's 100-image test set): FAST K=8 (all), SITCOM K=4 (all), DAPS K=4 (25 images) | ≈20–30 GPU-h, parallel over GPUs | FAST required; SITCOM/DAPS best effort by Nov 6 |
| XP-B | SITCOM K-scaling (K → 16) on 60 FFHQ ids: tests the model's predicted SITCOM ceiling | ≈14 GPU-h | Optional, by Nov 6 |
| XP-E | FAST K=4 on the field's `test-ffhq` set, as a bridge to published numbers | ≈0.4 GPU-h | Optional; **only if Cheng confirms these ids are disjoint from confirmation305** (agents must not check) |

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
| 4 | XP-E: confirm that DAPS `test-ffhq` ids are not in confirmation305 (you check; agents do not) | Skip XP-E |
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
