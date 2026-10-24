# Letter plan — failure anatomy and exact-symmetry repair for diffusion phase retrieval

**Status:** PLAN v1.2, 2026-10-24, planner Claude.
- **v1.2** repositions the letter after concurrent work PC-DAPS (arXiv 2610.31592). See [`2026-10-24_PC_DAPS_POSITIONING.md`](2026-10-24_PC_DAPS_POSITIONING.md).
- **v1.1** (2026-10-23) updated §1–§2 after the R0.8 tables.
- Numbers marked ⟨…⟩ come from R0-N.

## 0. Title

The v1.1 titles are retired: "What best-of-K hides…" and "Do we need ground truth…". PC-DAPS is titled "Beyond Best-of-k" and already makes the protocol critique.

**Candidates:**
- **A (default):** "Failure Anatomy and Exact-Symmetry Repair for Diffusion-Prior Phase Retrieval"
- **B:** "Ground-Truth-Free Selection for Diffusion Phase Retrieval: Symmetry Repair and What Remains"
- **C:** "Where Diffusion Phase Retrieval Fails After Ground-Truth-Free Selection"

**Owner of the draft:** planner. Cheng edits; Cheng decides coauthors.

**Format:** IEEE Signal Processing Letters. Four pages of technical content plus one page of references only; no over-length option (SPS information for authors).

**Fallbacks:**
- EUSIPCO 2027 (Darmstadt, 30 Aug – 3 Sep 2027): paper deadline 2027-02-06, per eusipco2027.de as found in the 2026-10-24 research pass.
- ICIP 2027 (Singapore): 2027-03-31, per the IEEE SPS event page. Third-party trackers say early February, so verify.

## 1. Pitch (v1.2)

**Practice and question.** Diffusion-prior Fourier phase retrieval is reported as the best of four runs, chosen with the ground truth (GT). Concurrent work (PC-DAPS) shows that this hides rotated and color-inverted runs, and fixes them inside DAPS. We ask two complementary questions across solver families:
- How much of the GT best-of-K number does a deployable, GT-free, post-hoc procedure recover?
- What fails after that?

**Data.** FFHQ, with DAPS, SITCOM, FAST-DIPS and NP: 3,240 runs on a random cohort and a hard-enriched cohort.

**Findings:**
1. **GT-free parity.** Signed-residual selection plus exact-orbit prior repair matches GT best-of-K in every family.
   - It is never more than 2 images (Good25) or 0.28 dB (mean) below GT best-of-K.
   - It is up to 3 images and 0.65 dB above, where GT best-of-K itself returns misoriented outputs (B22 DAPS and DEV80 SITCOM).
   - Going from the first run to GT best-of-K gains 2.2–5.5 dB.
   - The GT-free protocol recovers 93–129% of that gain (planner arithmetic, see positioning note §3.2; ⟨mean-of-K ladder, N1i⟩).
2. **What remains.** Repairable symmetry errors are a minority of failures: 7.1% pooled, 3.6% for FAST-DIPS. Most failures are generation failures.
   - For DAPS and FAST-DIPS (78–89% of failures) they do not fit y.
   - For SITCOM (57% of failures) they fit y, so no residual test can see them.
   - ⟨Crosswalk to PC-DAPS's orientation-only labels, N1i⟩.
3. **The repair.** It compares only the exact orbit members of one candidate. These share one likelihood under any magnitude-noise model and the same rotation-invariant statistics, so the prior ratio decides and the complexity confound cancels.
   - It fixes 84/85 orbit errors (S_mid 85/85) and harms 0/2,040 good outputs, at ⟨32⟩ forwards.
   - A learned realness detector fixes 62/85 and harms 220.
   - Used across candidates, the same score is only a mediocre failure detector (AUROC 0.83–0.86 on DAPS/FAST), as the complexity effect predicts.

**Recommendations:** report the GT-free number; use the signed residual; repair symmetry with the solver's own prior. Future solver work should target generation failures.

## 2. Contributions (allowed claims, v1.2 order)

1. **Cross-family failure anatomy under GT-free selection** on a random and a hard-enriched cohort, crosswalked to PC-DAPS's outcome labels ⟨N1i⟩. We are not aware of a comparable cross-solver analysis; re-check at writing time.
2. **GT-free parity with GT best-of-K:** the protocol ladder ⟨mean-of-K, E0, E1, E2, O_raw, O_orbit, with E_P exploratory⟩. What remains are generation failures.
3. **Post-hoc, solver-agnostic exact-symmetry repair with harm accounting:**
   - Proposition 1, made explicit: posterior ratio = prior ratio on the exact orbit, for any magnitude-noise model; rotation-invariant statistics are constant on the orbit.
   - The paired-noise estimator ⟨variance factor⟩. We claim its validation, not its invention.
   - A cost curve ⟨N1d: cheapest dev-selected configuration, its forwards, its B22 accuracy⟩. Pre-registered per-t results already show that 32 forwards suffice in every family.
   - Comparisons with UnivFD and ⟨CLIP zero-shot, N2⟩.
4. **Mechanism.** Across candidates, the denoising-loss score tracks image complexity: Spearman 0.43–0.46 with high-frequency energy among good DAPS/FAST outputs. Inside the exact orbit, that confound cancels.
   - The per-t results agree: the lowest noise level fails, consistent with CCLR's finding that DDPM complexity bias concentrates in low-noise ELBO terms.
   - PC-DAPS's mid-noise σ choice (≈ DDPM t 144–396) is consistent with this.
5. **Negative findings:**
   - realness detectors are unreliable for orientation, and inferior to the signed residual as post-hoc detectors on DAPS/FAST;
   - the clipped residual misranks.

**Open validity item:** checkpoint training membership (N1h).
- `ffhq_10m.pt` likely trained on all 70k FFHQ images (ILVR App. C.2; inferred). B22 is validation-category but probably seen in training.
- State this as a limitation. ⟨Optional N4a held-out faces probe, if Cheng approves and the data exist⟩.
- Re-read the complexity literature before citing: Serrà et al. (ICLR 2020), Nalisnick et al. (ICLR 2019), Goodier & Campbell CCLR (arXiv 2310.17432).

## 3. Not allowed

**Unchanged from v1.1:**
- Orbit enumeration is not new: DiffFPR's code enumerates the same 8 patterns, but aligns them with GT.
- GT-free failure detection is not new (FA-FPR).
- No claims about FA-FPR's mid-trajectory restart policy. We test detectors on final outputs only.
- No generalization beyond FFHQ at σ = 0.05 and 2× oversampling.
- No claims about NP as a method.
- No claims about confirmation305 or any untouched cohort. B22 is a fixed random validation-category panel; DEV80 is hard-enriched.

**New in v1.2 (PC-DAPS):**
- No "first to show GT best-of-k hides orientation errors", and no "first to report outcome frequencies or mean-of-k".
- No "first GT-free, prior-based fix of the phase-retrieval orientation ambiguity". PC-DAPS does it on FFHQ, in-sampler.
- No novelty claim for the paired twin test itself. It is in PC-DAPS's public code (global, ImageNet only). We claim its post-hoc, solver-agnostic, harm-accounted validation.
- Do not say "orbit errors are rare" without qualification. Say "repairable orbit errors", i.e. an orbit member ≥ 25 dB.
- No claim that PC-DAPS's cluster score is confounded by complexity. Untested: our complexity evidence is across images.
- No head-to-head numbers against PC-DAPS. Images and configurations differ, so quote theirs only as their numbers.

## 4. Structure (about 4 pages)

1. **Introduction** (0.6 page).
   - GT best-of-K practice (DPS, DAPS, SITCOM, FAST-DIPS) and GT orientation alignment (DiffFPR, DDRM-PR, I2I-PR).
   - FA-FPR's detector; concurrent PC-DAPS (one sentence; draft in the positioning note §7).
   - The question; contributions.
2. **Setup** (0.5 page).
   - Operator: the pinned DAPS/FAST definition, σ = 0.05, signed y.
   - The four families with native settings ⟨DAPS NFE per run, N1i A4.4⟩.
   - Cohorts: B22 random 100, DEV80 hard-enriched.
   - Policies: mean-of-K, E0, E1, O_raw, O_orbit.
   - Labels: GOOD, ORBIT, CONSISTENT, INCONSISTENT, and the τ rule; PC-DAPS labels as a crosswalk. Good25 follows field practice (DAPS++ uses PSNR < 25 as phase-retrieval failure).
3. **Exact-symmetry repair** (0.6 page).
   - Proposition 1 and its one-line proof.
   - The estimator with paired coupling R_s ε.
   - Cost.
   - S_mid and cheap configurations.
4. **Results** (1.9 pages).
   - **Fig. 1:** anatomy stacked bars per family, B22 vs DEV80, with the PC-DAPS-label crosswalk as an inset or a sentence ⟨N1i⟩.
   - **Table 1:** repair success and harm for prior S, S_mid, the cheap configuration, UnivFD and CLIP zero-shot (global only).
   - **Table 2:** protocol ladder per family: mean-of-K, E0, E1, O_raw, O_orbit (Good25 / mean / q05) on B22, with DEV80 in parentheses.
   - **Fig. 2:** repair success vs UNet forwards (log x), with harm marked.
   - **One line plus a short table:** post-hoc detection AUROC (signed ρ vs UnivFD vs prior). SITCOM's 0.68 is explained by the residual distributions.
5. **Discussion and recommendations** (0.4 page).
   - Report deployable best-of-K; use the signed residual; repair orientation with the prior.
   - Remaining failures are generation failures.
   - Limitations: FFHQ only; checkpoint membership; ImageNet, where PC-DAPS reports weaker prior disambiguation ⟨N4b if run⟩.

**Optional visual** (if space allows): four examples showing raw rotated failure, repaired result and GT.

## 5. Evidence status

| Item | Source | Status |
|---|---|---|
| Anatomy (pooled and per source, including per-family consistent counts) | R0.7 / R0.8 ZIP | have (planner-verified) |
| Anatomy by image cohort, with S5 split into B22 and DEV80 | R0-N N1a | pending |
| Repair success/harm: S, S_mid, UnivFD; per-t repair | R0.7 / R0.8 ZIP | have (planner-verified) |
| Deployable vs O_raw / O_orbit for S1–S5 strata | R0.8 P4 matrix | have (planner-verified); FAST split by cohort pending N1g |
| First-run → GT best-of-K → GT-free ladder | R0.8 P4 (planner arithmetic) | have; mean-of-K rung pending N1i |
| PC-DAPS-label crosswalk; E_P prior-only selection | R0-N N1i (amendment 2) | pending (optional, non-blocking) |
| DAPS configuration (NFE per run) from receipts | R0-N N1i A4.4 | pending |
| Complexity correlation (Spearman of S with high-frequency energy) | R0.7 secondary | have |
| Checkpoint training membership of evaluation ids | R0-N N1h | pending |
| Held-out data inventory | R0-N N3 / N3b | pending |
| Clean-image orientation probe (held-out faces, ImageNet) | N4 | **proposed; needs Cheng's go** |
| Cheap configuration (dev→eval) and paired-variance factor | R0-N N1d | pending |
| CLIP zero-shot baseline | R0-N N2 | pending (optional, recommended) |
| Post-selection decomposition with image identities; cohort-split tables | R0-N N1b / N1g | pending |
| Residual distributions | R0-N N1c | pending |
| Costs | R0-N N1e | pending |
| Conditional detection (exploratory) | R0-N N1f | pending |

## 6. Timeline (v1.2, compressed: "submit fast")

| Date | Milestone |
|---|---|
| 2026-10-24 → 10-31 | R0-N executed, including amendments 1–2. Cheng runs 2–3 CPU blocks, plus the optional N2 GPU block |
| 2026-10-26 → 11-06 | Planner drafts the introduction, setup, method and related work; these do not depend on R0-N |
| 2026-11-02 → 11-09 | Planner fills results, figures and tables from R0-N |
| 2026-11-10 → 11-20 | Cheng and coauthor edits |
| **2026-11-24** | **SPL submission target.** Optional same-day arXiv post; coauthors decide |
| 2026-12-11 | K4 time box, unchanged: the hard stop |

**Drop rule:** any optional item (N1i, N2, N4) not returned by 2026-11-06 is dropped from the submission.

## 7. Kill criteria and watch items

- **K1–K4:** as listed in `2026-10-23_R0_PLANNER_DECISION.md` §5. Unchanged.
- **W1 (new, watch):** if a PC-DAPS revision or another paper publishes post-hoc exact-orbit repair with cross-solver evaluation before we submit, the planner re-rules. Repair would then be presented as a replication, and the anatomy and protocol ladder carry the letter. The planner checks arXiv before the draft freeze and again before submission.
