# Track R TH1 A6: exact image-conditional expected K-subset oracle for a fixed prefix

**October 27, 2026. Status: PROVED, synthetic + raw-A8 smoke tested.**
No XP results or post-calibration run outcomes examined.

## Why it is useful

For A6.4 the future panel contains the **observed original n0 runs** and m=nmax-n0 unknown future runs. We condition on the original label counts C_i, not just an image-level prior, and average over all K-subsets of that completed panel. The point forecasts for O_raw and O_orbit should not fluctuate between the S2, rank-race and tail variants merely because we used finite Monte Carlo simulation. They are independent of class-residual-score modeling.

## Closed form (PROVED)

For image i let known original label counts be C=(C_G,C_O,C_P,C_C,C_I), n0=sum C and m=nmax-n0. A fitted cohort Dirichlet hyperparameter vector alpha gives a posterior on per-image run class probabilities
\[
\theta\mid C\sim {\rm Dirichlet}(C+\alpha).
\]

Fix a set of accepted classes \(A\subseteq\{G,O,P,C,I\}\), with \(g_0=\sum_{j\in A}C_j\), posterior total \(T=\sum_j(C_j+\alpha_j)\), and posterior nonaccepted parameter \(B=\sum_{j\notin A}(C_j+\alpha_j)\). The exact conditional expectation of the K-subset oracle on a completed nmax panel is

\[
\boxed{
\mathbb E[\widehat O_A(K)\mid C]
=1-\frac1{\binom{n_{\max}}{K}}\sum_{j=\max(0,K-m)}^{\min(K,n_0-g_0)}
\binom{n_0-g_0}{j}\binom{m}{K-j}
\frac{(B)_{K-j}}{(T)_{K-j}}.
}
\]

The index j is the number of **known original nonaccepted** runs in a subset. The remaining K-j draws must come from the m unknown future runs and also be nonaccepted. Conditional on theta this event has probability (1-theta_A)^(K-j); integrating its beta marginal produces (B)_(K-j)/(T)_(K-j). The combinatorial factor is exactly the fraction of all K-subsets with j old nonaccepted runs. This is a hypergeometric counting identity plus a standard Dirichlet moment; not claimed as a new general theorem.

Choose \(A=\{G\}\) for O_raw, and \(A=\{G,O,P\}\) for O_orbit. This exactly computes the **posterior mean of the realized-panel K-subset oracle**, not a new independent K-sample pass-at-K. For nmax=n0 and K=n0, the formula reduces to the already observed oracle indicator. For nmax>n0 and K=n0, the expected all-subset score is **not** the observed prefix-only score because subsets may include future runs.

## Practical advantages and one limitation

1. Guarantees identical O_raw and O_orbit point forecasts across all three structural variants and the race prior sensitivity arm, while preserving *variant-specific* E0 and E1.
2. Eliminates avoidable simulation noise from the anatomy generation-failure forecast \(1-O_{\rm orbit}\), the selection-loss difference and the crossover estimate.
3. Does not remove actual uncertainty in the **realized future panel**. The intervals must still include finite-panel U-statistic noise; we retain the simulated oracle sample distribution for the predictive spread, but recenter it on the exact analytic conditional mean.
4. Cohort mixing must preserve B22/DEV80 at exactly 100:80. The formula depends on nmax; an 8-of-16 forecast and an 8-of-64 forecast are distinct conditionally on the fixed prefix and must have different panel keys.

## Local independent verification (authorized calibration only)

- The uploaded file /mnt/data/R0N2_PER_IMAGE_COUNTS.csv was SHA-256 checked directly in Python/CaaS: **197d0c8ea2aa47fa17dd379d518f647d7a47d89b3c614449fb106959b7bb45a6**, 79,404 bytes, 760 data rows.
- Retaining permitted solver-source combinations produces 540 rows, 1,960 runs, exactly G1171/O58/P1/C215/I515.
- Independent local Python/NumPy/SciPy smoke reconstruction of Dirichlet fitting, rank-conditional exponential scores and exact K-subset U-statistics completed on DAPS/SITCOM/FAST primary panels across S2, race, tail and race_tau1; ranks and class counts matched the planner's accepted data.
- Independent numeric comparison of the boxed exact oracle to 80 Monte Carlo completed panels per family/cohort at all primary K values had absolute error at most **0.0061** across those checks; this is within the expected finite-sample Monte Carlo range, and there are no data-derived post-K4 outcomes in that calculation.
- A reusable synthetic regression was committed as scripts/track_r/theory/check_th1_exact_prefix_oracles.py. The forecaster was updated to use the exact oracle as its **point**, retaining realized-panel and bootstrap variation for its **interval**, and check_a6_panel_contract.py now enforces cross-variant exact oracle agreement.

**Important QA distinction:** the local independent smoke reimplemented the source specification and ran successfully, but does not certify the full GitHub forecaster's current end-to-end A6 output. Run the current GitHub code on the authenticated original A8 and then execute check_a6_panel_contract.py before freezing.

## Freeze status

This document is a mathematical and input-integrity note only; it is **NOT** TH1_PREDICTIONS_v1.json. A8b gate October 28 noon ET, target v1 freeze October 28 18:00 ET. The planner registry has not been opened/edited by this executor.
