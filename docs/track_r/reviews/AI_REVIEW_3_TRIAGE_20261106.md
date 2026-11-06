# Triage of the third AI review (planner, 2026-11-06)

**Source.** A GPT ("Astra") review run on a friend's account, to avoid fitting Cheng's own account history: [`AI_REVIEW_3_ASTRA_20261106.md`](AI_REVIEW_3_ASTRA_20261106.md).
- **Verdict:** weak reject, 4/10, confidence 3/5. It reviewed the main text only, not the supplement.
- **Barrier it states:** "whether the evidence supports the paper's strongest conclusions."

**Planner's assessment.** This review is the most useful of the three. Most of its points are correct.
- **Real overclaims:** W1 (abstract cherry-picked five of six tests), W3 ("exactly when stopping helps"), W5 (two meanings of s), W6 (noise cancellation presented as an explanation), W12 ("unsolved", "generation is not predictable").
- **Real gaps:** W2 (no new-image test of the positive DAPS result) and W8 (threshold accuracy). W8 turned out to matter: DAPS stopping needs τ within about 1% (C-V4).
- **Partly a misreading:** W4 (H3 was registered only on three panels). It still exposed unclear text.

Categories:
- **W:** writing only.
- **1:** out of this paper's control or scope.
- **2:** more experiments.

"Done + data" means a planner analysis on committed data now backs the text (`planner_analysis/A27_review3_checks/`, ledger C-R8, C-V4, C-V5).

## Weaknesses

| # | Point | Cat. | Response | Where | Status |
|---|---|---|---|---|---|
| W1 | Abstract reports 67/69 from five of six tests; the excluded test is the largest and fails. Separate end-to-end forecasting, conditional modeling and diagnosis. Report widths, errors and interval scores | W + 2 | **Agreed.** All six tests are now reported. Selected run, race vs (S2): coverage 76/111 vs 53/111; mean interval score 30 vs 182; mean absolute error 3.1 vs 8.0. The race loses tests 1 (clearly) and 4 (narrowly). The conditional analysis is relabeled "a retrospective diagnosis, which no forecaster could run". | Abstract; intro; Sec. 5 text; Tab. 2 adds IS columns and an "All" row; supp Tab. 16 | **Done + data (C-R8)** |
| W2 | Positive stopping gains come only from fresh runs of known images; the image-disjoint panels (FAST-DIPS, ImageNet′) are negative | 2 | **Agreed. This is the most important gap.** Text now scopes the claim ("on fresh runs of known images"; limitation added). Experiment proposed: **XP-NF** (prereg A26), the frozen protocol for DAPS-2K/4K on 100 new FFHQ validation images, with τ frozen from existing data. Supporting fact: on cohort A alone (random validation images) the gains are +3.0 (DAPS-2K) and +2.6 to +3.1 (DAPS-4K). | Abstract, intro, Sec. 7; A26 | Text done; **experiment needs Cheng's GO and manifest (D-A26a)** |
| W3 | Prop. 1 is a decomposition, not a comparison theorem; "we show exactly when stopping helps" and "stopping cannot help" are too strong | W | **Agreed.** Now: "gives its cost and success exactly; empirically, stopping helped where failed runs did not fit the measurement"; "it did not help". Sec. 4 states that Prop. 1 is not a comparison with fixed K. | Abstract, intro, Sec. 4, Sec. 7 | **Done** |
| W4 | "H3 fails wherever it was tested" contradicts Tab. 1 rows meeting the criterion; H1 is a point estimate, not noninferiority | W | **Partly a misreading.** H3 was registered on three panels only (FAST-DIPS ImageNet 16 and 32, DAPS-2K FFHQ), and failed on all three (1/3, 5/11, 1/3). The rows the reviewer cites were not registered for H3. Text now says this. H1 and H2 are labeled as point-estimate criteria. Panels with intervals of ±2–3 are called "inconclusive". | Sec. 4 | **Done** |
| W5 | Two meanings of s (pass probability vs class probability) | W + data | **Agreed.** Sec. 5 now uses κ = q + c and states how κ differs from s. SITCOM: 82/1,688 GOOD runs fail τ. DAPS-4K with label-free τ: 139/210 INCONSISTENT runs pass on ImageNet. | Sec. 5; supp §Pass probability | **Done + data (C-V5)** |
| W6 | Noise-energy cancellation does not explain reliable ranking (e_i depends on η); ‖η‖² vs mσ² | W | **Agreed.** Reworded: the identity motivates the comparison but does not guarantee it; the evidence is empirical, also per image (92/100 image–solver pairs, Sec. 5). ‖η‖² corrected. The intro sentence is removed. | Sec. 4, intro | **Done** |
| W7 | Novelty is the audit, not the components; compare more selectors and stopping rules | W + 2 | **Agreed.** Intro now says: "Residual selection, the measurement's blindness to orientation and adaptive stopping all have precedents; we contribute their measurement, together and prospectively." Selector and stopping comparisons are in the supplement: the prior test, the confidence-conditioned stop, M = 1–4 (Tab. 17). | Intro; supp | Done; **P7** compact table optional |
| W8 | A label-free τ within 2% may still change policy behavior; pilot accounting | 2 | **Agreed, and important.** Rerunning the frozen protocol on the held-out panels shows that DAPS stopping needs τ within about 1%. At −2%, no run passes and every image is flagged, which a user sees. At +2%, it keeps +0.7 to +1.9. The labeled, label-free and cross-cohort thresholds lie within 0.4% of each other and keep +2.6 to +5.0. Selection does not use τ. Pilot reuse is now stated: on DAPS-4K it reused the evaluated images' first four runs, a one-time amortized cost. | Sec. 4; supp Tab. 17 | **Done + data (C-V4)** |
| W9 | DAPS reproduction gap (22.3 vs 25.78 dB) unresolved | 1 + W | **Partly answerable.** Verified: identical GT and y, published 400 × 10 configuration from receipts. On our FFHQ images the protocol gives 29.8 vs 30.6 dB in the DAPS paper. The ImageNet gap is unexplained, and the text now says so. Policy comparisons use the same runs, so they are unaffected. | Sec. 6; supp §Reproducing DAPS | **Done** (the cause stays open: 1) |
| W10 | "Beats the oracle" compares different action sets | W | **Agreed.** The lead claim is now recovery of 80% of the orbit-oracle PSNR gain. The raw/per-metric comparison is framed as "legacy reporting misses this cheap step". The Sec. 6 heading is neutralized. | Abstract, intro, Sec. 6 | **Done** |
| W11 | FLAG_O is batch-relative, not calibrated | W + 2 | **Agreed.** Defined as "a batch-level triage rule rather than a calibrated confidence". The supplement notes that a frozen absolute margin is the deployable form. | Sec. 4; supp §Orientation flag | Text done; **P6** risk–coverage curve and frozen-margin transfer optional |
| W12 | "Unsolved" and "generation is not predictable" overstate | W | **Agreed.** Now "no good run within our budget"; "no selection among the runs"; "Selection was forecast well; generation was not"; the moment argument is limited to "class counts alone", and image features "might, which we did not test". | Abstract, intro, Sec. 5 | **Done** |

## Questions

| # | Question | Cat. | Response | Status |
|---|---|---|---|---|
| Q1 | Timeline of freezes and information | W | The supplement registry (Tab. 11) gives every freeze with its hash and UTC time; the main text points to it. A compact "what was known at each freeze" table is optional (P8). | Mostly done |
| Q2 | DAPS gains on new FFHQ images? | 2 | XP-NF (A26). | Awaiting GO |
| Q3 | H3 vs Tab. 1 | W | See W4. | Done |
| Q4 | Does Prop. 1 prove superiority? | W | No. It is a decomposition. The supplement has a counterexample showing heterogeneity alone does not help. The text now says so. | Done |
| Q5 | Relation of the two s; fractions failing τ | W + data | See W5. | Done |
| Q6 | All six tests with error, width and interval score | data | See W1. | Done |
| Q7 | A near-zero-success component in the mixture? | 2 (theory) | Plausible fix for the test-1 miss. Proposed to theory: a post hoc refit with a spike at zero on the test 1/5/6 prefixes, scored like Tab. 16, labeled exploratory (theory round 23). | Queued |
| Q8 | Calibration cost; pilot disjointness | W | See W8. Pilot reused on DAPS-4K; one-time cost, amortized; the cross-cohort pilot works as well. | Done |
| Q9 | Cause of the DAPS gap | 1 | See W9. | Open (1) |
| Q10 | Robustness to threshold changes of the label-free size | data | See W8 (Tab. 17). | Done |
| Q11 | FLAG_O vs residual confidence; frozen-margin transfer | 2 | P6. | Queued |
| Q12 | Qualitative panel in the main text | 2 | X4: the executor exports tiles for four cases (residual selection right, rotation fixed, confident wrong flip, consistent failure). Main text only if space allows; otherwise supplement. | Queued |

## Limitations the review lists

- **Already stated or now added:** scope (one operator, simulated data), image diversity (cohort B), threshold-defined success (supplement θ sweep), CLIP as an auxiliary prior on upright benchmarks, global-only correction, average compute vs latency, single output vs posterior sampling.

## Plan changes

1. **XP-NF (A26):** GPU, about 22–45 GPU-h, ceiling 50. **Needs Cheng:** GO, a manifest of 100 new FFHQ validation ids disjoint from the 180 and the holdout (D-A26a), and the order relative to XP-K (D-A26b; the planner recommends XP-NF first).
2. **CPU, planner:**
   - P6: FLAG_O risk–coverage and frozen-margin transfer;
   - P7: compact selector and stopping table;
   - P8: freeze-information timeline;
   - P1–P3 from A25.
3. **CPU, executor:** X4 qualitative tiles, plus X1–X3 from A25.
4. **Theory:** Q7 refit with a near-zero component (exploratory); review of the new wording.

## Planner's view

- **Current results.** The review lowered no number; it showed where the text claimed more than the numbers. After this pass:
  - every forecasting claim covers all six tests;
  - every stopping claim is scoped to fresh runs of known images;
  - the threshold-accuracy limit is stated with data.
- **Remaining weak point.** No new-image test of the DAPS stopping gain. XP-NF closes it in about three days if Cheng approves by Nov 7.
- **Image of the publication.** An honest audit. Its firm results are:
  - selection without ground truth (near the orbit oracle on FFHQ; 80% of the gap on ImageNet; large PC-DAPS gains);
  - a prospectively tested selection model (interval score 30 vs 182);
  - the coverage limit.

  The stopping protocol is a conditional result:
  - it helps DAPS when its threshold is set to within 1%;
  - it is neutral elsewhere;
  - its new-image test is pending.
