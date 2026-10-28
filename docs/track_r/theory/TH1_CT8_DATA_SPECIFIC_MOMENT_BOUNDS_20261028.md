# C-T8 data-specific moment bounds — after FAST64 release (2026-10-28)

**Exploratory supplement calculation, NOT a prospective v1 prediction or a claim of a novel moment theorem.** Code: scripts/track_r/theory/check_th1_identification_lp.py. The procedure was designed after inspecting the FAST64 forecast miss, but its numerical constraints use **ONLY** the original permitted A8 FAST four-run GOOD-count histograms, not the new 64-run labels. No XP return directory or withheld B16/F8 result was opened.

## Model and inferential target

Within one fixed FFHQ cohort, assume image-specific independent run success probabilities \(p_i\) are drawn iid from some unknown mixing distribution \(\mu\) on [0,1], with 4 conditionally iid Bernoulli attempts per image. Then the count \(J_i\) has probabilities
\[
 q_j(\mu)=\int_0^1 {4\choose j}p^j(1-p)^{4-j}\,\mu(dp),\qquad j=0,\ldots,4.
\]
The target is the **population** 64-attempt oracle success probability for a new image,
\[
 O_{64}(\mu)=\int_0^1[1-(1-p)^{64}]\,\mu(dp).
\]
This is NOT the registered fixed-image conditional-prefix realized-panel estimand. Moreover, the DEV80 cohort was deliberately enriched, so its finite-sample confidence interpretation requires the additional assumption that the 80 selected images behave as an iid population sample; otherwise report its set as descriptive sensitivity only.

For each of five multinomial count categories, compute a two-sided Clopper–Pearson binomial interval at marginal confidence 99% (\(\alpha_j=0.01\)). **Bonferroni** gives simultaneous coverage at least \(1-5(0.01)=95\%\) for the vector \((q_0,\ldots,q_4)\), under the stated iid image sampling assumption. The constraints \(L_j\le q_j(\mu)\le U_j\) are imposed **jointly**, along with \(\mu\ge0\) and total mass one; no fictitious equality to each noisy empirical frequency is imposed. The impossible exact-empirical-moment LP seen by the planner is therefore avoided without simply declaring inconsistent observed frequencies to be population moments.

## Continuum outer certificate from a finite grid

Use p mesh \(x_i=i/10000\), spacing \(h=10^{-4}\). The derivative identity
\[
 B'_{4,j}(p)=4[B_{3,j-1}(p)-B_{3,j}(p)]
\]
with boundary Bernstein terms zero gives \(\sup|B'_{4,j}|\le4\), while \(\sup|[1-(1-p)^{64}]'|\le64\). Moving any continuum mixing distribution to its **nearest grid points** changes each count probability by at most \(2h\), and the target by at most \(32h\). Thus solve the finite LP with count constraints **expanded** to \([L_j-2h,U_j+2h]\), and report the **conservative continuum outer interval**
\[
 [\max(0,\mathrm{LPmin}-32h),\,\min(1,\mathrm{LPmax}+32h)].
\]
Up to ordinary numerical optimizer tolerances, this contains the target values of **all continuum mixing distributions** compatible with the simultaneous CP bands; this is not merely a grid-restricted lower/upper estimate. This deterministic discretization correction is a useful way to make the numerical illustration defensible.

## Reproducible result

| Cohort | A8 FAST count histogram J=0,1,2,3,4 | Continuum outer 95% set for population O_raw(64) |
|---|---|---|
| **B22, n=100** | (5, 15, 21, 26, 33) | **[0.8696, 1.0000]** |
| **DEV80, n=80** | (20, 12, 14, 21, 13) | **[0.6113, 1.0000]** |

The bands include the 64-run observed fraction with ≥1 GOOD, B22 99/100 and DEV80 74/80, as well as broad ranges of possible forecast outcomes. They **do not** identify a precise structural-zero fraction, even after imposing all five count-category constraints. Their conservatism reflects both sampling noise and the intrinsic four-moment obstacle: narrower claims would require more assumptions or data.

**Interpretive caveat:** The exact Gauss-Radau nonidentifiability pair in TH1_5_RADAU_NONIDENTIFIABILITY_20261026.md is a *population-level mathematical counterexample*. This LP is its data-specific analogue under a conservative joint confidence model, not a distribution-free bound on every fixed image or on the exact K-subset continuation protocol. Cite classical truncated Hausdorff moments, Gauss–Radau, and Clopper–Pearson/Bonferroni; do not claim the LP as new optimization theory.

**Run:** python scripts/track_r/theory/check_th1_identification_lp.py --a8 R0N2_PER_IMAGE_COUNTS.csv --output CT8_BOUND.json --grid-size 10001. The script SHA-checks the A8 source and uses no K>4 data.
