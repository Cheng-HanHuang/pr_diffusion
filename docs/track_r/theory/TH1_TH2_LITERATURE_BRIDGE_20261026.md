# Track R: literature-to-theory synthesis and positive bridge results (2026-10-26)

**Scope and evidence:** This is a theory-executor supplement to TH1_WORKING_PROOFS_20261026.md after reviewing the separately supplied in-depth literature audit. All work uses abstract algebra and synthetic objects. No XP outcomes, GT images, or protected data were opened. The recommendations are **not new reconstruction methods** and are not a claim that these elementary mathematical principles were invented here. The planner retains control of the paper and predictions registry.

## A. Paper-level scientific position: connect, do not progressively shrink

Track R is **not** claiming that Fourier-magnitude ambiguity was first discovered here, that invariant Markov laws are new, or that verifier false positives were first noticed here. The precise, positive question is:

> In a fixed, practically used Fourier phase-retrieval benchmark with learned diffusion priors, how much of multi-run reconstruction failure is attributable to **the intrinsic measurement quotient**, to **the solver's induced distribution over prior-supported orbit representatives and measurement-consistent non-orbit decoys**, and to **a GT-free verifier that cannot see those distinctions**? Can that explanation predict out-of-sample best-of-K behavior across three structurally different published solvers?

The chain is: (i) exact group symmetry identifies information omitted by the measurement; (ii) Bayesian prior and approximate sampling dynamics determine orientation and non-orbit mode occupancy; (iii) the candidate class probabilities and selection-rank distribution determine the benefit and ceiling of K runs; (iv) pre-frozen forecasts on unseen runs test whether the abstraction is predictive. The first parts inherit classical foundations; **the scientific integration, family-by-family failure taxonomy, rank diagnostics, and prospective validation are the differentiating contribution**, conditional on experimental success.

This is a **characterization / evaluation / theory** paper, not a proposal for a new sampler or a claimed universal optimality theorem. Do not erase DAPS's prior eight-image observation, the classical twin ambiguity, Stroebl's verifier ceiling, GeoDiff's invariant-marginal lemma, or TGD's residual best-of-N precedent; place each at its correct link.

### Verified source-to-role map (independently cross-checked on publication/author pages)

- Hayes-type uniqueness and classical ambiguities: contextualize the quotient; the 256/384 canvas is **outside** the half-grid condition, and no claim is made that G exhausts the ambiguity set. Barnett et al., *Geometry of the Phase Retrieval Problem*, arXiv:1808.10747 and Cambridge University Press 2022.
- DAPS, CVPR 2025, shows eight distinct measurement-identical images in its supplement; it analyzes an ideal exact-inner-sampling marginal property, not the actual implementation's posterior exactness. https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Improving_Diffusion_Inverse_Problem_Solving_with_Decoupled_Noise_Annealing_CVPR_2025_paper.html
- SITCOM, ICML 2025, introduces step-wise triple-consistency optimization; not a posterior-exactness theorem. https://proceedings.mlr.press/v267/alkhouri25b.html
- FAST-DIPS, ICLR 2026, proves local step/descent properties and a local Gaussian KL bound; this is distinct from our conditional **trajectory equivariance KL**. https://proceedings.iclr.cc/paper_files/paper/2026/hash/ab378fb084f313a204432f0a1e697bae-Abstract-Conference.html
- Stroebl, Kapoor & Narayanan, ICLR 2026, supplies the imperfect-verifier resampling ceiling. Our S1--S2 **argmin/accept-first equality in selected-label law** establishes the physical-score mapping rather than re-deriving a novel general verifier effect. https://proceedings.iclr.cc/paper_files/paper/2026/hash/95591b7557edea46ae1d90b0e9e825c5-Abstract-Conference.html
- TGD, Makris et al., arXiv:2605.03712, uses likelihood potentials and accelerated likelihood pruning; its ideal SMC consistency is not a practical guarantee. TDS (Wu et al., NeurIPS 2023) adds prior/proposal factors to weights, so **only likelihood factors** are always orbit-blind.
- Beinert & Plonka, *Ambiguities in one-dimensional discrete phase retrieval from Fourier magnitudes*, 2015, classifies 1D zero-flip ambiguities. The exact 1D construction in section C below is an **illustration of established algebra** that makes TH2(e) quantitative, not a novel ambiguity theorem. https://num.math.uni-goettingen.de/plonka/pdfs/Beinert_Plonka.pdf
- GeoDiff/Xu et al. 2022 and Hoogeboom et al. 2022: invariant initial law + equivariant kernels implies invariant output. Use this as standard lemma, then instantiate it for the **actual measurement group and conditional solver structure**.

## B. PROVED: orbit-conditional information decomposition

**Proposition (finite-group quotient, not a novelty claim).** Let G be a finite group acting measurably on image space; let O(X) be the unlabeled orbit of X, and suppose Y has a conditional distribution of the form L(dy | A(X)), where A(gx)=A(x). Let X have any prior distribution pi and use independent observation noise conditional on X. Then

\[
Y\ \perp\ X\mid O(X).
\]

In particular, for a measurable orientation index J(X) obtained by choosing a measurable representative on free orbits,

\[
I(J;Y\mid O)=0,\qquad
P(J=j\mid Y,O)=P(J=j\mid O).
\]

*Proof:* A(X) is constant on the fibers of O, so A(X)=a(O) for a measurable function a. For every measurable observation event V,

\[
P(Y\in V\mid X,O)=L(V\mid a(O))
                  =P(Y\in V\mid O).
\]

Conditional independence follows by the definition of a Markov kernel. J is a function of X, hence also independent of Y conditional on O. The information equality uses a well-defined conditional mutual information (in the finite J case it always exists). On a free orbit with a density and an invariant reference measure,

\[
P(J=j\mid O=o,Y=y)
=\frac{\pi(g_j c(o))}{\sum_{h\in G}\pi(hc(o))}
\]

with the standard disintegration interpretation; for continuous priors the comparison is between disintegrated conditional masses/densities, **not** probabilities of individual points.

**Consequence (conditional Bayes ceiling).** With 0-1 orientation-representative loss and an oracle that reveals O but not J, even a Bayes classifier using Y and O cannot exceed

\[
\max_j P(J=j\mid O)
\]

conditional orientation accuracy. Under a uniform G-invariant prior on a free eight-member orbit, that ceiling is 1/8. Under an upright-favoring learned prior it may be far higher. This makes precise the statement **measurement cannot repair exact orientation, but learned prior can**. It is an information decomposition, not a lower bound on the overall reconstruction quality of arbitrary solver outputs; a solver's non-equivariant procedural choices can contribute in addition to its learned denoiser.

**Why this supports, rather than narrows, Track R.** Theorem 1 and the TV theorem now occupy the dynamic leg between the conditional prior distribution and the practical sampler's selected-orbit frequencies. The predicted/observed frequency split can be interpreted as a quantitative prior-plus-dynamics phenomenon; we do not mistake symmetry of the forward model for uniform orientation under an upright-trained prior. A natural **analysis only**, not a new selection method, is to report orientation distribution conditional on orbit-error images, and compare it with the measured denoiser equivariance-defect profile.

## C. PROVED: exact non-orbit 1-D decoy and finite-noise ranking formula

**Exact zero flip, already classically known.** Let the real, nonnegative length-three vectors

\[
z=(1,5,6)/7,\qquad z'=(2,7,3)/7.
\]

They are in [0,1]^3 and are neither translations nor reversals of one another. They arise from convolution

\[
(1,5,6)=(1,2)*(1,3), \qquad
(2,7,3)=(2,1)*(1,3).
\]

For |w|=1, the factor identity |1+2w|=|2+w| gives

\[
|1+5w+6w^2|=|2+7w+3w^2|,
\]

so their padded discrete Fourier magnitudes agree at **every Fourier frequency and for any pad length M>=3**, including the nonaliased regime M>=5. This is a nontrivial *exact* measurement-consistent decoy, not an element of the reversal orbit. Mapping x=2z-1 and x'=2z'-1 keeps both in [-1,1]^3 and matches the physical centering convention in the Track R operator (in this 1D toy). This is an example, not proof that analogous real-image decoys are frequent in Track R's 2D natural-image benchmark.

**Gaussian discrimination for a fixed candidate pair (exact theorem).** Let a=A(x), b=A(nu), d=b-a, delta=||d||_2, and take the observation

\[
Y=a+\sigma Z,\qquad Z\sim N(0,I_m),\ \sigma>0.
\]

For squared-residual selection between the correct *fixed* candidate x and a fixed decoy nu,

\[
\|Y-b\|^2-\|Y-a\|^2
 =\delta^2-2\sigma Z^\top d.
\]

For delta>0, the probability over **measurement noise** that residual ranking favors x is exactly

\[
P\bigl(\rho(x)<\rho(\nu)\bigr)=
\Phi\!\left(\frac{\delta}{2\sigma}\right).
\]

For delta=0 the residuals are identically tied for **every** observation Y: with randomized fair tie-breaking, the preference probability is 1/2; with deterministic asymmetric ties, preference probability can be 0 or 1. For delta>0 it increases strictly from 1/2 toward 1 as delta/sigma grows. This is the mathematically explicit D12/TH2(e) intuition **in this controlled fixed-candidate, noise-averaged model**, not a theorem that the real solvers' residual AUROC equals this formula.

**K-way qualification:** With m deterministic decoys b_j and one true candidate a, x wins against all decoys exactly on the Gaussian polyhedral event

\[
2\sigma Z^\top d_j<\|d_j\|^2
\quad\hbox{for all }j,\quad d_j=b_j-a.
\]

The Gaussian test variables have covariance matrix \(\sigma^2(d_i^\top d_j)_{ij}\). The all-decoys win probability depends on this *entire geometry*, not only individual pairwise AUROCs. The report's D2 counterexample therefore has a constructive explanation: a pairwise AUROC >1/2 does **not** control a K-way argmin. A safe positive use of AUROC is **diagnostic**, while the actual K-policy behavior is estimated by exact rank-weighted U-statistics and/or a complete score-distribution model.

**Major caveat for Track R:** all runs for a fixed image share one frozen noisy observation Y, and generated candidates themselves depend on Y. Conditional on that Y, neither the probability-over-noise formula nor fixed-candidate independence can be substituted into S2. This result is a tractable **TH2 toy bridge** and a quantitative interpretation, not an input to the v1 prediction fit.

**Sensitivity to approximate symmetry (PROVED).** For a generic pair with A(gx)=A(x)+d under Gaussian measurement noise, the log-likelihood ratio obeys

\[
\log\frac{L_y(gx)}{L_y(x)}
= \frac{\langle y-A(x),d\rangle}{\sigma^2}
-\frac{\|d\|^2}{2\sigma^2}.
\]

Therefore its magnitude is at most
\(\|y-A(x)\|\|d\|/\sigma^2+\|d\|^2/(2\sigma^2)\).
For our exact per-channel half-turns d=0 identically. This quantifies how operator changes or approximate preprocessing symmetry can reopen measurement discrimination. It is not an added noise model for the registered 384-canvas predictions.

## D. Integration with existing planned results

1. **TH1.1 (main text):** Present the exact Fourier symmetry and posterior ratio as the *measurement leg*, GeoDiff kernel propagation as the *sampler leg*, and the new trajectory KL comparison as a quantitative bridge. Disclose that each theorem's general ingredients preexist. Mention the orbit-conditional information decomposition as a supplement proposition or short interpretive sentence, not as a standalone new method/theorem claim.
2. **TH1.2:** Keep Stroebl visibly credited. Tie its false-positive category to our choice of objective: for raw GOOD, both ORBIT and CONSISTENT are false positives; for orbit-recovery, CONSISTENT is a false positive; after global repair, PARTIAL ORBIT remains an uncorrected passer. This explains why the global-only selection gap tends to \((o_p+c)/s\) on images with q>0.
3. **TH1.3/TH1.4:** Preserve the exact U-statistics and metric-wise oracle envelope, but recognize standard pass@K/unbiased subset estimators as inherited, and use them as a rigorous *empirical protocol*. Being able to predict curves that subsequently agree or disagree is more informative than presenting elementary combinatorics as new.
4. **TH2:** The zero-flip decoy and the formula \(\Phi(\delta/(2\sigma))\) give a concrete, falsifiable local mechanism for non-orbit consistency: transition from indistinguishable exact decoys to noisy separable near-decoys. This enriches the work without changing any solver or GT-free selection method.
5. **Prediction freeze:** None of the above results uses XP data. Do not change fit assumptions on the strength of toy-noise results. Fit only DAPS, SITCOM and FAST-DIPS at K<=4; planner alone edits PREDICTIONS_REGISTRY.md.

## E. Required paper wording corrections from the external audit

- Posterior density ratios only where denominators are positive, or use cross-product equality and include normalization in region formulas.
- Uniform 1/8 mass only for distinct, disjoint group-related neighborhoods on a free orbit.
- Conditional solver equivariance must include stochastic Langevin microsteps and optimizer states; do not assert that every implementation ingredient other than the denoiser is automatically equivariant.
- Gaussian TV comparison needs an honest Gaussian-transition representation; deterministic non-equivariant terminal readout and extra random variables require new handling.
- Selection gap at infinite K has factor 1{q>0} (or 1{p>0} for raw); the c/s miss ceiling must include partial orbit mass when appropriate.
- Full TDS weights are **not** generally G-invariant; only likelihood-only potentials are.
- D2 original "AUROC>0.5 implies better K-way argmin" is false; S2 remains an *assumption* to test, not something proven by zero-flip symmetry.

**Status:** Sections B--C are proved elementary statements; the specific prevalence of such modes in 2D solver trajectories remains EMPIRICAL/CONJECTURE. All comparisons with particular practical sampler implementations remain conditional until audited. This file is for the theory program's development, not the predictions freeze.