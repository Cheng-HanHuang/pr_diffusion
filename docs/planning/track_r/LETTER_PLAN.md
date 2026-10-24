# Paper plan — a ground-truth-free audit of diffusion-prior phase retrieval

**Status:** PLAN v1.3, 2026-10-24, planner Claude.
- **v1.3** applies the K2 ruling (decision §10): CLIP zero-shot matches the diffusion prior on global orientation, so prior repair is no longer a contribution. The paper becomes an audit. SPL is replaced by EUSIPCO 2027 as the recommended venue.
- **v1.2** repositioned the plan after PC-DAPS (arXiv 2610.31592); see [`2026-10-24_PC_DAPS_POSITIONING.md`](2026-10-24_PC_DAPS_POSITIONING.md) and its K2 update box.
- **v1.1** (2026-10-23) followed the R0.8 tables.
- Numbers marked ⟨…⟩ come from R0-N.2.

## 0. Title, venue, owner

**Retired titles:**
- "What best-of-K hides…" (PC-DAPS owns that framing);
- any title containing "repair" as a method.

**Title candidates:**
- **A (default):** "A Ground-Truth-Free Audit of Diffusion-Prior Phase Retrieval: Selection, Symmetry, and Generation Failures"
- **B:** "Ground-Truth-Free Selection Nearly Matches Oracle Best-of-K in Diffusion Phase Retrieval"
- **C:** "What Remains After Ground-Truth-Free Selection in Diffusion Phase Retrieval"

**Venue:** Cheng and the coauthors decide. Facts are to be verified by the 2026-10-24 research pass.
- **Recommended:** EUSIPCO 2027 (Darmstadt; paper deadline 2027-02-06 per eusipco2027.de as found earlier).
- **Alternative:** ICIP 2027 (Singapore; 2027-03-31 per the IEEE SPS event page).
- **Not recommended:** IEEE SPL. Odds for an audit letter there are about 10–20% (planner judgment), and its speed advantage no longer matters.

**Format:** keep the content to 4 pages plus references. That fits all three venues.

**Preprint:** post to arXiv when the draft is final, if coauthors agree.

**Owner:** the planner drafts; Cheng edits; Cheng decides coauthors.

**Coauthor note:** the SITCOM finding (57% of failures fit y) is unflattering to a method several coauthors wrote. Show it to them early; the finding stays.

## 1. Pitch (v1.3)

**Practice and question.** Diffusion-prior Fourier phase retrieval is reported as the best of K runs, chosen with the ground truth (GT). Concurrent work (PC-DAPS) argues that this hides rotated and color-inverted runs. We audit the protocol across solver families and ask:
1. How much of the GT best-of-K number survives when selection is GT-free?
2. What fails after that?

**Data.** FFHQ, with DAPS, SITCOM, FAST-DIPS and NP: 3,240 runs on a random cohort (B22) and a hard-enriched cohort (DEV80).

**Findings:**
1. **Almost all of the K-run gain survives without GT.**
   - Going from one run to GT best-of-K gains 2.2–5.5 dB.
   - Selecting by the signed measurement residual, then fixing global orientation with a generic uprightness test, recovers 93–129% of that gain (prior-based E1; ⟨CLIP-based E1 and mean-of-K, N1i⟩).
   - It is never more than 2 images (Good25) or 0.28 dB (mean) below GT best-of-K, and up to 3 images and 0.65 dB above it where the GT pick is itself misoriented.
   - What GT best-of-K hides is per-run unreliability (single-run failure rates of 21–60% by family and cohort) and the K× compute, not GT information.
2. **What remains is generation failure.** After GT-free selection and orientation correction, 93 of the 99 still-failing image-strata have no candidate reaching 25 dB in any orbit member.
3. **Failure modes depend on the solver.**
   - For DAPS and FAST-DIPS, 78–89% of single-run failures do not fit y, and the signed residual detects them (AUROC ≈ 0.98).
   - For SITCOM, 57% of failures fit y as well as good runs do (AUROC 0.68), so residual tests cannot see them.
4. **Symmetry errors are a minority, and a generic test fixes them.**
   - Exact-orbit errors are 7.1% of failures pooled (FAST-DIPS 3.6%).
   - On the exact orbit the likelihood is identical, so the choice is a pure prior comparison (Proposition 1). Any good uprightness prior then works:
     - CLIP zero-shot: 84/84 global errors fixed, 0/2,040 good outputs harmed;
     - the solver's own diffusion prior (S_mid, or 2 forwards at t = 300): 84/84, 0/2,040.
   - A realness detector of the kind FA-FPR uses fixes 71/84 and harms 5.3% of good outputs.

**Recommendations:**
- Report GT-free best-of-K alongside, or instead of, GT best-of-K, and report single-run failure rates.
- Select by the signed residual. Resolve orientation with any uprightness test, not a realness detector.
- Future solver work should target generation failures.

## 2. Contributions (allowed claims, v1.3 order)

1. **Cross-solver failure anatomy under GT-free selection**, on a random and a hard-enriched cohort ⟨crosswalked to PC-DAPS's outcome labels, N1i⟩. Re-check novelty at writing time.
2. **The protocol ladder:**
   - rungs: ⟨mean-of-K⟩, first run, E0, E1 (prior), ⟨E1 (CLIP)⟩, O_raw, O_orbit;
   - the GT-free share of the K-run gain;
   - single-run failure rates.
3. **Post-selection decomposition:** fixes, selection misses, uncorrected orbit errors and generation failures (15 / 5 / 1 / 93 at maximal K).
4. **Residual behavior by solver:** signed versus clipped residual; why it works for DAPS/FAST and not for SITCOM (N1c distributions).
5. **Orientation:**
   - Proposition 1 (posterior ratio = prior ratio on the exact orbit) explains why any uprightness prior suffices.
   - Evaluation of three test types with harm: generic VLM, the solver's prior, a realness detector.
   - Low-noise prior terms hurt (full S 83/84; t = 50 fails), consistent with low-noise complexity bias (CCLR).

**Demoted to supporting detail** (one or two sentences, no contribution claim):
- the paired estimator and its 5× variance factor;
- the 2-forward configuration;
- the complexity correlation (Spearman 0.43–0.46).

**Open validity item:** checkpoint training membership (N1h).
- `ffhq_10m.pt` likely trained on all 70k FFHQ images (inferred from ILVR App. C.2). State this as a limitation that affects all solvers equally.
- It matters less now that no claim depends on the prior recognizing orientation.

## 3. Not allowed

**Carried over:**
- **Novelty:** orbit enumeration and GT-free failure detection are not new (DiffFPR, FA-FPR).
- **FA-FPR:** no claims about its restart policy.
- **Scope:** no generalization beyond FFHQ, σ = 0.05, 2× oversampling.
- **NP:** no claims about it as a method.
- **Cohorts:** no claims about confirmation305 or untouched cohorts.
- **PC-DAPS:**
  - no "first to show best-of-k hides orientation errors";
  - no "first GT-free prior-based orientation fix";
  - no novelty for the paired twin test;
  - no claims about PC-DAPS's cluster score;
  - no head-to-head numbers against it.
- Do not say "orbit errors are rare" without "repairable".

**New in v1.3 (K2):**
- **Prior:** no claim that the diffusion prior is needed for, or better at, orientation correction.
- **Orientation as a method:** no novelty claim for orientation correction. Present it as a generic uprightness test with two instantiations.
- **CLIP:** no claim that CLIP orientation is new technology.
- **Speed:** no speed ordering between CLIP and the 2-forward prior rule. The latter's cost is a projection.
- **Harm wording:** never write "zero harm" unqualified. Write "no harm observed in 2,040 good outputs from 180 images".

## 4. Structure (about 4 pages)

1. **Introduction** (0.6 page).
   - GT best-of-K practice (DPS, DAPS, SITCOM, FAST-DIPS) and GT orientation alignment (DiffFPR, DDRM-PR, I2I-PR).
   - FA-FPR's GT-free detector; concurrent PC-DAPS (one sentence, draft in the positioning note §7, with the repair clause removed).
   - The audit questions; contributions.
2. **Setup** (0.6 page).
   - Operator, σ, signed y.
   - Four solver families with native settings ⟨DAPS NFE per run⟩.
   - The two cohorts.
   - Labels: GOOD (Good25, following DAPS++'s PSNR < 25 failure rule), ORBIT, CONSISTENT, INCONSISTENT, and the τ rule; PC-DAPS labels as a crosswalk.
   - Policies: mean-of-K, E0, E1, O_raw, O_orbit.
   - Proposition 1 (three lines), and the two orientation tests.
3. **Results** (2.3 pages).
   - **Fig. 1:** anatomy stacked bars per family, B22 vs DEV80 ⟨plus crosswalk⟩.
   - **Table 1:** the protocol ladder per family (Good25 / mean / q05), B22 with DEV80 in parentheses, plus the GT-free share of the gain.
   - **Table 2:** orientation tests (CLIP, prior S_mid, prior 2-forward, prior S, UnivFD): global errors fixed, harm, cost note.
   - **Fig. 2:** residual distributions by family and label (from N1c).
   - **Text:** the post-selection decomposition (15 / 5 / 1 / 93); detection AUROC by family.
4. **Discussion, recommendations, limitations** (0.5 page).
   - The recommendations from §1.
   - Limitations: FFHQ only; one operator and noise level; checkpoint membership.
   - Generality: ImageNet orientation is harder for priors (PC-DAPS), and CLIP's behavior there is untested by us.

## 5. Evidence status

R0-N (N0–N3) is complete and accepted (decision §§9–10). R0-N.2 is the remaining CPU block (amendment 2, revision B).

| Item | Source | Status |
|---|---|---|
| Anatomy, pooled and by cohort | R0.7 / R0-N N1a | **have** |
| Ladder: first run, E0, E1 (prior), O_raw, O_orbit; FAST split | R0.8 P4 / R0-N N1g | **have** |
| Mean-of-K rung; E1 with CLIP orientation | R0-N.2 A4 (N1i) | pending (**required**) |
| PC-DAPS-label crosswalk | R0-N.2 A4 (N1i) | pending (recommended) |
| Post-selection decomposition (counts) | R0-N N1b | **have**; image identities pending (A3, optional) |
| Residual distributions; detection AUROC | R0-N N1c / R0.7 P3 | **have** |
| Orientation tests: CLIP, S, S_mid, UnivFD (global-only) | R0-N N2 recovery | **have** (planner-verified, SHA-matched) |
| 2-forward prior rule; paired-variance factor | R0-N N1d | **have** (supporting detail only) |
| Costs | R0-N N1e; N2 receipt | **have** (CLIP measured; prior rules projected) |
| Checkpoint training membership | R0-N.2 B2 (N1h) | pending (**required**) |
| DAPS configuration (NFE per run) | R0-N.2 A4.4 | pending |
| Draft figures | R0-N (Pillow) | drafts; redraw for the paper |
| N3b; N4 | — | **withdrawn** |

## 6. Timeline (v1.3)

| Date | Milestone |
|---|---|
| 2026-10-24 | R0-N complete; K2 ruled |
| by 2026-11-07 | R0-N.2: one CPU block (N1h, N1i with mean-of-K, CLIP-E1 and the crosswalk; optional A3) |
| 2026-10-27 → 11-27 | Planner drafts the full paper |
| **2026-12-11** | **K4: draft submission-ready** (the planner's deadline) |
| 2026-12 → 2027-01 | Cheng and coauthor edits, at their pace |
| **2027-02-06** | **EUSIPCO 2027 deadline (recommended target; verify).** ICIP 2027 (2027-03-31) is the alternative |

Optional arXiv post once the draft is final.

## 7. Kill criteria and watch items

- **K1:** not triggered. **K2:** TRIGGERED (applied above). **K3:** PASS.
- **K4:** reinterpreted as the draft-ready date.
- **W1 (watch):** if a PC-DAPS revision or another paper publishes a cross-solver, GT-free audit of diffusion phase retrieval before we submit, the planner re-rules. The planner checks before the draft freeze and before submission.
- **Stop option:** if Cheng decides not to write, the repository record is the deliverable and the project goes dormant (D4).
