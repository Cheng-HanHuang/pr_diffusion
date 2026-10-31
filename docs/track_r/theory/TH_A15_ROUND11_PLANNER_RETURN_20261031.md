# Track R theory executor — Round 11 / A15, 2026-10-31

**Access and integrity.** Uploaded theory_upload_20261031.zip was tested for ZIP corruption and all **four** SHA256SUMS.txt entries verified before 11_PLANNER_GUIDANCE_20261031.md, SUPP_INTEGRATION_README.md or the two supplied main/supplement PDFs were read. The already accepted planner supplement compiled as ten pages on Oct 31; the present additions have NOT yet been compiled with the planner's current wrapper. No docs/track_r/executors/xp/returns were accessed. Numerical data are the planner-authorized CP2/CP3 aggregate relay and the already released F8 experiment, not any additional held image strings.

## Deliverable 1 — dated S3 addendum DONE, earlier S3 preserved

- Source: docs/track_r/theory/TH_SUPPL_S3_POST_REGISTRATION_ADDENDUM_20261031.tex
- Ready-to-copy complete S3: docs/track_r/theory/TH_SUPPL_S3_COMPLETE_DRAFT_20261031.tex.
- The full S3 begins with the **verbatim unchanged** TH_SUPPL_S3_COMPLETE_DRAFT_20261030.tex prefix, then a visibly dated A15 addendum, without refitting any frozen model or silently changing the previous outcome interpretation.
- Locked selected-run coverage tests 2–5: race **57/59**, S2 **17/59**. Points are correlated; not 59 independent Binomial trials. Oracle forecasts optimistic in four of five tests.
- CP2 third prospective ImageNet v4: race E0/E1 **18/18** versus S2 **1/18**, O_raw **2/6**, O_orbit **0/6**; K32 orbit observed .750 vs .840 predicted, K64 .780 vs .899; 25 images with no orbit success in 16 runs, 22 still without success in 64. No structural-zero spike was in the registered v4.
- F8 v1 race **21/28**, S2 **22/28**; v3 **17/28**. Race E0 7/7 and E1 5/7, S2 E0/E1 14/14; K8 pooled raw/orbit oracles overpredicted.
- **16-moment nonidentifiability:** idealized 16-run per-image Bernoulli counts determine a degree-16 truncated moment sequence, not the degree-64 oracle. Explicitly, Uniform[0,1] and its 9-node left Gauss–Radau rule have identical moments through degree 16, but the latter has a 1/81 atom at p=0 and different O64. Numerically verified with shifted Jacobi roots: max moment discrepancy across degrees 0..16 <9e-15, O64 gap ≈0.000939468, infinite-run gap 1/81. This proves nonidentification but **does not explain the empirical ~0.09–0.12 v4 orbit forecast errors quantitatively**. Finite 100-image noise and class/rank ambiguity worsen extrapolation. Near-zero mass affects low-order moments but is not uniquely identifiable from them, except degenerate cases.

## Deliverable 2 — S1 post-outcome amendments DONE

Appended three dated paragraphs to docs/track_r/theory/TH_SUPPL_S1_PROTOCOL_THEORY_DROPIN_20261029.tex; earlier text preserved:

1. **Large-N H1 failure.** For hard fraction pi0 at s_h ≪1/N and easy fraction 1−pi0, the exact equal-compute gain is
   Δ_N = pi0[S_h(N)−F_h(Kbar_N)] + (1−pi0)[S_e(N)−F_e(Kbar_N)],
   Kbar_N=pi0 E[T_h]+(1−pi0) E[T_e],
   with E[T_h]≈N while E[T_e]≈2/s_e. Under an explicit stylized hard-zero-success case with easy stopped gain saturated at H2+, a crossing F_e(K*)=H2+ yields negative gain when pi0>(K*−t_e)/(N−t_e). This is **conditional**, not universal. Released XP-D4 H1 is −1.8 pp at N32 and −3.4 pp at N64; 4–6 low-pass images absorb ~43% of total runs. Labeled post hoc; no inference that these images have latent s=0.
2. **Noise transfer H-G1.** Normalized signed residual depends on reconstruction displacement d_sigma/sigma, not sigma alone. Fixed FAST-DIPS epsilon=.05 gives epsilon/sigma 2.5 at sigma=.02 and .5 at sigma=.10. Actual tau .02223 vs linear .02003316 (+10.97%) and .09809 vs .1001658 (−2.07%): both violate 1%. Denoiser response and GOOD-class conditioning can shift; this does not falsify the noise-floor identity or establish a monotonic sign.
3. **PC-DAPS.** A bag of four mutually interacting particles is not four iid solver runs (even if their joint law is permutation exchangeable). Choosing residual-minimum/E1-orientation among one bag is GT-free; native particle mean GOOD25 79.0%, E0 82.8%, E1 84.4%, all descriptive. Matched-NFE claims withdrawn. A stopping protocol across independently generated bags may satisfy bag-level iid assumptions after defining bag-level residual, but has no experimental evaluation.

## Integration/QA

- **Keep S2 frozen as-is.** The planner's existing layout-only edits include correct math mode for the FLAG_C title, external M-prefix cross-references, and orbit lemma restatement. Leave its flagged stale empirical sentences unresolved until XP-C2 and other authorized source measurements arrive.
- For S3 replace the older integration copy with TH_SUPPL_S3_COMPLETE_DRAFT_20261031.tex, rather than including the dated standalone addendum twice. Do not replace older individual registered statements.
- For S1 copy the appended version TH_SUPPL_S1_PROTOCOL_THEORY_DROPIN_20261029.tex, while retaining the planner's LaTeX math-title fix in the integration directory. S2 integration unchanged.
- Planner should re-run pdflatex/bibtex/pdflatex/pdflatex in the supplement folder and check cross-refs/length; the theory executor does not claim a new integrated 10-page build or unreviewed layout equivalence.
- No main paper edits or protocol v2 are proposed. The open empirical placeholders belong to planner-controlled S2/S4–S9, not invented theory data.
