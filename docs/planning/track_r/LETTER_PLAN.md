# Letter plan — "What best-of-K hides in diffusion phase retrieval"

**Status:** PLAN v1, 2026-10-23, planner Claude. Numbers marked ⟨…⟩ come from R0-N.

**Owner of the draft:** planner. Cheng edits; coauthors are decided by Cheng.

**Format:** IEEE Signal Processing Letters. Four pages of technical content, plus one page of references only, with no over-length option (SPS information for authors).

**Fallbacks:**
- EUSIPCO 2027: Darmstadt, 30 Aug – 3 Sep 2027. Submission deadline listed by a third-party calendar as 2027-02-06; verify on the official site.
- ICIP 2027: deadline not yet published.

## 1. One-paragraph pitch

Diffusion-prior Fourier phase retrieval papers report the best of four runs chosen by ground truth (GT), and some also align orientation with GT. We ask what that oracle hides. Across four solver families and 3,240 runs on FFHQ, failures fall into three groups:

1. **Measurement-inconsistent** — the run does not fit y. This is the dominant mode for DAPS and FAST-DIPS. The free signed residual detects these almost perfectly.
2. **Measurement-consistent but wrong** — prominent for SITCOM, where the residual is nearly uninformative.
3. **Exact-symmetry (180°) errors** — invisible to any measurement-based test by construction.

For group 3, the posterior ratio equals the prior ratio, and a paired-noise denoising-loss test using the solver's own diffusion prior fixes 98.8–100% of cases with zero observed harm. A learned realness detector of the kind used for failure detection in FA-FPR fixes 73% and harms 10.8% of good outputs.

With signed-residual selection plus prior repair, deployable Good25 is within 0–2 images of the GT oracle in every stratum. What remains are generation failures. We recommend reporting the deployable number.

## 2. Contributions (allowed claims)

1. A failure anatomy of DAPS, SITCOM, FAST-DIPS and NP under deployable selection, on a random cohort and a hard-enriched cohort. We are not aware of a comparable cross-family analysis; this must be re-checked at writing time.
2. GT-free exact-symmetry repair with the solver's own prior:
   - a proposition (posterior ratio = prior ratio on the orbit, for any magnitude-noise model);
   - a paired-noise estimator, with its variance reduction ⟨factor⟩;
   - a cost–accuracy curve: ⟨cheapest dev-selected configuration, its forwards, its accuracy on the B22 evaluation split⟩.
3. Evidence that the deployable protocol nearly closes the gap to the GT oracle, and that what remains is generation failure.
4. Negative findings:
   - realness detectors are unreliable for orientation and inferior to the signed residual as post-hoc failure detectors on DAPS and FAST;
   - the clipped residual misranks.

## 3. Not allowed

- Orbit enumeration is not new: DiffFPR's code enumerates the same 8 patterns, but aligns them with GT.
- GT-free failure detection is not new (FA-FPR).
- No claims about FA-FPR's mid-trajectory restart policy. We test detectors on final outputs only.
- No generalization beyond FFHQ at σ = 0.05 and 2× oversampling.
- No claims about NP as a method.
- No claims about confirmation305 or any untouched cohort. B22 is a fixed random validation-category panel; DEV80 is hard-enriched.

## 4. Structure (about 4 pages)

1. **Introduction** (0.6 page). GT best-of-K practice (DPS, DAPS, SITCOM, FAST-DIPS); GT orientation alignment (DiffFPR, DDRM-PR, I2I-PR); FA-FPR's detector. The question; contributions.
2. **Setup** (0.5 page):
   - operator (pinned DAPS/FAST definition), σ = 0.05, signed y;
   - the four families with native settings;
   - cohorts: B22 random 100, DEV80 hard-enriched;
   - K-run policies;
   - labels: GOOD, ORBIT, CONSISTENT, INCONSISTENT; the τ rule.
3. **Exact-symmetry repair** (0.7 page):
   - Proposition 1 and its one-line proof;
   - the estimator, with paired coupling R_s ε;
   - cost;
   - the S_mid and cheap configurations.
4. **Results** (1.8 pages):
   - **Fig. 1** — anatomy stacked bars per family, B22 versus DEV80.
   - **Table 1** — repair success and harm: prior S, S_mid, cheap configuration, UnivFD, CLIP zero-shot (global only).
   - **Table 2** — deployable versus oracle per family: O_raw, O_orbit, E0, E1 (Good25 / mean / q05) on B22, with DEV80 in parentheses.
   - **Fig. 2** — repair success versus UNet forwards (log x), with harm marked.
   - **One line plus a short table** — post-hoc detection AUROC (signed ρ vs UnivFD vs prior), with SITCOM's 0.68 explained by the residual distributions.
5. **Discussion and recommendations** (0.4 page):
   - report deployable best-of-K;
   - use the signed residual;
   - repair orientation with the prior;
   - the remaining failures are generation failures, which is where future solver work should go;
   - limitations.

**Optional visual** (if space allows): four examples showing raw rotated failure, repaired result, and GT.

## 5. Evidence status

| Item | Source | Status |
|---|---|---|
| Anatomy (pooled) | R0.7 | have |
| Anatomy by cohort, per-family consistent counts | R0-N N1a | pending |
| Repair success/harm: S, S_mid, UnivFD | R0.7 | have |
| Cheap configuration (dev→eval) and paired-variance factor | R0-N N1d | pending |
| CLIP zero-shot baseline | R0-N N2 | pending (optional, recommended) |
| Deployable vs oracle, including O_raw | R0-N N1b / N1g | pending |
| Residual distributions | R0-N N1c | pending |
| Costs | R0-N N1e | pending |
| Conditional detection (exploratory) | R0-N N1f | pending |

## 6. Timeline

| Date | Milestone |
|---|---|
| 2026-10-24 → 10-31 | R0-N executed (Cheng runs 2–3 CPU blocks, plus the optional GPU block) |
| 2026-11-02 → 11-13 | Planner full draft, figures and tables |
| 2026-11-16 → 12-04 | Cheng and coauthor edits |
| 2026-12-11 | Submission-ready (time box K4) |

## 7. Kill criteria

K1–K4 as listed in `2026-10-23_R0_PLANNER_DECISION.md` §5.
