# Track R theory — Round 15 / A19 return (2026-11-02)

**Archive integrity:** checked every entry of `SHA256SUMS.txt` in `theory_upload_20261102.zip` **before** reading the guidance. All **four** sha256 checks passed, including the guidance, A19 summary, and main/supplement PDF bytes. Read the two Markdown texts for the post-scoring release. Did not analyze or change the PDFs.

**Access:** XP-N now officially released. For a directly checked new diagnostic only, read the released `docs/track_r/executors/xp/theory_inputs/XP_N_RUNS_0_63/XP_N_CPN64_STRINGS.csv` at XP **2d791c5**. No `xp/returns/` files accessed and no embargoed outcomes opened. Code v5 81003cc and the pushed forecast at XP 55ba86b (2026-11-01 19:53:58Z, prior to run 4+) remain unchanged.

## 1. Dated S3 Test 6 addendum — DONE

Files:
- `docs/track_r/theory/TH_SUPPL_S3_TEST6_XPN_ADDENDUM_20261102.tex` — standalone dated insertion.
- `docs/track_r/theory/TH_SUPPL_S3_COMPLETE_DRAFT_20261102.tex` — **verbatim identical** 2026-10-31 S3 prefix plus the dated Test 6 insertion appended exactly once. Use this OR the standalone insertion, never both.

As scored under the frozen complete-panel rule (five K values, E0/E1 beta_class), image-disjoint ImageNet Test 6 gives E0+E1 coverage:
`race=10/10` (**winner**), `race_tau1=7/10`, `tail=5/10`, `S2=0/10`.
Absolute standardized errors 1.236, 1.329, 2.258, 7.441. Oracle coverage: race O_raw=3/5, O_orbit=0/5 (all variants O_orbit=0/5). At K64, E0 forecast .462 [.378,.544], actual .520; E1 forecast .626 [.534,.711], actual .610; orbit oracle .880 [.821,.931], actual .750. O_raw underforecast at K4/8 (.417/.510 vs .470/.564). Combined tests 2-6: race 67/69 vs S2 17/69; nested K and repeated images make these **correlated** and not a binomial sample of independent points.

**Directly verified raw-class overlap diagnostic from the authorized 100-image 64-run strings:**
- Both GOOD and non-GOOD correct-orbit runs: **67** images.
- GOOD only: **6** images.
- Correct-orbit non-GOOD only: **2** images.
- Neither: **25** images.
- Among 69 images with any non-GOOD orbit candidate, 67 (97.1%) also have GOOD.
- Descriptive Pearson correlation of total 64-run GOOD and non-GOOD orbit candidate counts across images: **+0.36877**.
- Thus the exact `O_orbit−O_raw = Pr(orbit-present AND GOOD-absent)` at K64 is **.020** observed, versus **.164** forecast: the extrapolation expects 16.4 orbit-only images per 100 but observes 2.
- This is evidence of a common image-level easy/hard propensity and **misspecified joint class-rate mixture**, not just a uniformly optimistic oracle. An unconditional single Dirichlet class-prior has negative cross-coordinate covariance; the actual fixed-prefix conditional v5 predictive model is more complex, so this covariance comparison is a descriptive diagnostic rather than a formal statistical rejection test.

**C-T8 relation:** four-run five-class counts theoretically identify polynomial moments only through total degree four of the image-dependent joint rate mixing distribution. `D_K = E[(1−theta_G)^K−(1−theta_G−theta_O)^K]` at K64 is not generically identified. That is a mathematical limit on what short-run data guarantee, **not** a quantitative prediction of the registered v5 miss or an excuse to change the forecast. The systematic orbit-only difference and 67/6/2/25 overlap show that the *specific* Dirichlet extrapolation fails the joint long-run image correlation. There is no structural-unsolvability certificate from finite 64-run failures.

## 2. S1 N-H1a update — DONE

Inserted **one dated sentence** immediately after the existing large-N stopping paragraph in `docs/track_r/theory/TH_SUPPL_S1_PROTOCOL_THEORY_DROPIN_20261029.tex`. The original binary call N-H1a predicted PASS but **FAILED**: observed -1.516 points vs ≥-1 threshold, though bootstrap 95% interval [-3.60,+.47] spans 0 and the margin. One percentage point is below the roughly two-point half-width/resolution of a single 100-image panel; both no meaningful stopping advantage and a moderate-budget loss remain compatible. N-H1b at Nmax64 correctly predicted a negative-gain pass: -3.102 observed with interval [-5.96,-.10]. The S3 addendum reports both as distinct from v5 selection coverage, without changing the prospective note.

## 3. Integration and verification

- Old S3 `TH_SUPPL_S3_COMPLETE_DRAFT_20261031.tex` **unchanged** (checked the byte-for-byte prefix of the 2026-11-02 complete file).
- S2 unchanged. No main paper changes; no v5 code, registration receipt, selection rule, or earlier H2/XP-N pre-score edited.
- Checked balanced TeX braces, unique Test 6 label, no residual known literal `\\n` bug, and no rendered `planner`/`executor` role words in the new insertion. Previous 15-page planner integration is the last confirmed full build; a new integrated LaTeX compile **has not** been claimed.
- Integrate S1 current source and S3 2026-11-02 complete draft (or separate addendum once) into the previously accepted S1–S3 wrapper; leave S2 untouched and retain current anonymized math-mode and cross-reference fixes. Final acceptance milestone: **Nov 6**.

**Interpretive bottom line:** v5's class-sensitive *selector* generalized to new images under its frozen coverage rule, while the independent Dirichlet *generation* model dramatically overpredicted the existence of orbit-only rescued images. Nonidentifiability is a valid warning about extrapolating from 4 runs; model misspecification is the empirical explanation of this *particular* measured prediction failure.
