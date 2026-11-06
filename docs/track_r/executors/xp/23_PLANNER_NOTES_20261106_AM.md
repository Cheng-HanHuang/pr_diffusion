# Planner notes to the XP executor, round 23 (2026-11-06, morning)

**Responds to:** no new return. A third AI review (weak reject) asked for one new experiment; triage in `docs/track_r/reviews/AI_REVIEW_3_TRIAGE_20261106.md`.

**Governing amendment:** prereg **A26** (proposed). **Do not start XP-NF until Cheng's GO and the new manifest are committed.**

## 1. XP-NF (A26), once Cheng gives GO

- **Images.** Cheng supplies a manifest of 100 new FFHQ validation ids, disjoint from the 180 R0 ids and from the locked B24 holdout. You never read, list or check the holdout. Load the images through the R0 FFHQ pipeline, never from a bundled set.
- **Measurements:** one per image; same operator, σ = 0.05, signed y; R0 measurement code.
- **Gate.** Run a 2-root identity replay for DAPS-2K (F8 roots) and DAPS-4K (E′ roots), then a 2-root smoke with a GPU-hour projection.
- **Generation:** 8 runs per image per family (1,600 runs), fresh seeds by each family's rule. Co-tenancy per A10.1. Ceiling 50 GPU-h; if the projection exceeds it, DAPS-4K goes first.
- **Scoring:**
  - strings (labels at θ = 25 dB with τ = the new cohort's 95th percentile of GOOD ρ, evaluation only);
  - ρ_signed per run;
  - face CLIP decisions with margins;
  - the frozen protocol v1 (`c598176`) with M = 2, N_MAX = 8 and the τ_S values frozen in A26.4 (DAPS-2K 0.0475318, DAPS-4K 0.0475928);
  - image-bootstrap intervals as in C-E5.
- **Return:** a receipt with NF-H1 to NF-H3 outcomes, the strings and decisions (committed as before), SHA256SUMS, and the GPU-h ledger.
- **Order:** XP-NF before XP-K, unless Cheng decides otherwise (D-A26b). CP4 stays by Nov 15, 12:00 ET.

## 2. CPU tasks

- **X4 (new):** image tiles for a qualitative panel, PNG, 256², in the style of the Fig. 3 unsolved tiles. For each case, give the ground truth, the chosen output and, where relevant, the competing run:
  1. FFHQ, residual pick is GOOD among mixed runs;
  2. FFHQ, a global orbit error fixed by the face test;
  3. ImageNet, a GOOD pick flipped wrongly by the generic test with a large margin;
  4. SITCOM FFHQ, a CONSISTENT wrong run picked because no GOOD run exists.

  Choose the first eligible image in id order and record the selection rule in the receipt. Tiles go to Cheng as a ZIP; only the receipt is committed.
- **X1–X3** from round 22 stay queued after CP4.

## 3. Standing rules

Unchanged:
- Cheng runs every PAC block from pushed, clean commits.
- No access of any kind to the locked B24 holdout.
- No bundled FFHQ sets.
- No merges of `main` or the audit branch.

## Planner's view

- **Current results.** The third review lowered no number. It showed overclaims in the text, which are fixed. Its main ask is a new-image test of the DAPS stopping gain.
- **Plan.**
  - XP-NF: GO and manifest by Nov 7, generation Nov 7–9, scoring by Nov 10.
  - CP4 by Nov 15.
  - XP-K only if budget and time allow.
- **Image of the publication.** An audit with three firm results: selection without ground truth, a prospectively tested selection model, and the coverage limit. The stopping protocol is a conditional result, scoped to its tested setting until XP-NF reports.
