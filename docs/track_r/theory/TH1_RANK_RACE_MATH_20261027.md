# TH1 rank-calibrated class race: exact results and corrected gap shape

**2026-10-27. Status:** PROVED for iid candidate draws and independent exponential residual surrogates; CODE/CPU calibration from the now-authorized A8 K<=4 rank strings is in progress. **No XP returns, no K>=4 records beyond the permitted A8 calibration, no PAC or GT images**.

## 1. Model definition and connection to actual residuals

Fix one image and its categorical run probabilities \(\theta=(p,o_g,o_p,c,u)\). Each run independently draws class j and an independent nonnegative synthetic *ranking score* \(T\mid j\sim\mathrm{Exponential}(w_j)\), choosing the minimum T. The class weight is a **relative tendency to win the residual race** and is NOT a parameter or scale of the actual Fourier residual measured in physical units. A larger \(w_j\) favors lower scores. Set \(w_G=1\); the main S1 reference sets \(w_I=0\) (score \(+\infty\)) and \((w_O,w_P,w_C)>0\). For the single observed partial-ORBIT run, \(w_P=w_O\) by necessity; its independent race strength is not identifiable. S1 has one observed G/I violation, so this assumption is an approximation and an explicit model risk.

**Exact class winning integral (PROVED):**
\[
W_j(K\mid\theta,w)=
K\theta_j\int_0^\infty
w_je^{-w_j t}\left(\sum_\ell\theta_\ell e^{-w_\ell t}\right)^{K-1}\,dt,\quad w_j>0.
\]
For \(w_I=0\), \(W_I(K)=u^K\), and the other classes' probabilities sum to \(1-u^K\). The integral follows by designating the winning run; its score has class j density \(\theta_jw_je^{-w_jt}\) and all other K-1 scores exceed t independently with probability \(\sum_\ell\theta_\ell e^{-w_\ell t}\). This is a special case of ordinary competing-risk/order-statistic formulas and does NOT constitute a new general theorem.

For fixed counts \(N_j\) within a K-pool, conditional on those labels,
\[
\Pr(\text{winner is in class }j\mid N)=
\frac{N_jw_j}{\sum_\ell N_\ell w_\ell}
\]
when at least one positive-rate candidate is present. If all candidates have weight zero, the winner is in I (with a fixed tie rule). This is the Plackett--Luce/exponential-race identity.

Set
\[
E_0^{race}=W_G,\qquad
E_1^{race}=(1-h)W_G+\tau_g W_O+\tau_p W_P.
\]
The registered reference retains S1--S3 \(E_0^{S2}=[1-u^K]p/s\),
\(E_1^{S2}=[1-u^K]a/s\),
\(O_{raw}=1-(1-p)^K\),
\(O_{orbit}=1-(1-q)^K\)
with \(q=p+o_g+o_p\), \(s=q+c\), \(a=(1-h)p+\tau_go_g+\tau_po_p\).
The hard G/O/P/C label distribution is **identical** across the reference and class-race models; only the within-pool selection law changes. Observed source family and cohort strata remain separate.

## 2. Limiting race ceiling is weighted, not exchangeable (PROVED)

If \(\sum_{\ell\ne I}\theta_\ell w_\ell>0\), the strong law yields \(N_j/K\to\theta_j\) almost surely. By bounded convergence, for every positive-weight class j,
\[
\lim_{K\to\infty}W_j(K)=
\frac{\theta_jw_j}{\sum_{\ell\ne I}\theta_\ell w_\ell}.
\]
If all positive-rate class probabilities vanish, the winner stays in I. Thus
\[
E_0^{race}(\infty)=\frac{p w_G}{p w_G+o_gw_O+o_pw_P+cw_C},
\]
for positive denominator, and zero otherwise. This is NOT \(p/s\).
The global-only repaired ceiling is
\[
E_1^{race}(\infty)=\frac{p w_G+o_g w_O}
{p w_G+o_gw_O+o_pw_P+cw_C}
\]
when orientation fixing is perfect and the denominator is positive.
An exact orbit pair g.x and x has identical *physical* Fourier residual, but separately generated G and O reconstructions may have different non-orbit perturbations. Therefore distinct effective race weights do not violate exact Fourier invariance.

## 3. A key correction: monotone oracle gap does NOT survive the class race

The S1--S3 **exchangeable-rank reference** has a nondecreasing oracle--selector gap. **This monotonic-gap conclusion is false in general even under an exponential Plackett--Luce race with GOOD favored over every bad class.**

An explicit two-class example has independent GOOD probability \(p=1/2\), BAD probability \(1/2\), rates \(w_G=1,\ w_B=0.03\). For K runs, the oracle-minus-race-selector success gap is exactly
\[
D_K=\sum_{g=1}^{K-1}
\binom Kg 2^{-K}\frac{(K-g)0.03}{g+(K-g)0.03}.
\]
A direct exact finite binomial sum gives
\[
D_6=0.0386195271,\qquad
D_7=0.0385396532,
\]
so it **decreases** between K=6 and K=7. Its limit is
\[
D_\infty=\frac{0.5(0.03)}{0.5(1)+0.5(0.03)}
       =0.0291262136.
\]
The race E0 itself can still increase, while the *oracle-vs-race difference* decreases. This happens because an additional GOOD can dislodge an earlier selected BAD, making the selection-loss event nonmonotone as a set.

**Consequences:** The planner's requested 'monotonicity of the gap under the race' must be stated as a **negative result**, not a supposed theorem. Do not attach the S2 shape constraint to the rank-calibrated forecast or heavy-tail alternative. Report the full forecasted shape, allowing unimodality/nonmonotonicity, and reserve the monotone-gap theorem for the exchangeable-ranks reference.

Under the mild condition that all successful classes' positive rates are >= all unsuccessful classes' rates, the *success probability* of minimum-score selection is nondecreasing in K. This does not imply its difference from a better oracle is monotone. The argument is the monotone likelihood ratio of the candidate score in the winning class and the stochastic decrease of the sample minimum; alternatively use the exponential-race coupling in two-class cases. We do not need a stronger universal result about families with nonordered weights.

## 4. Within-A8 rank calibration (K<=4 only)

From **760 authorized A8 rows**, retain exactly **540** image--solver records:
- DAPS B22 n=2, DEV80 n=4;
- SITCOM B22/DEV80 n=4;
- FAST-DIPS S5 B22/DEV80 n=4.
These represent **1,960 runs**, exactly G=1171, O=58, P=1, C=215, I=515, reproducing planner totals.
Pairwise in-image audit confirms G outranks C **200/205**, G outranks O **50/69**, and G outranks I **570/571**. For the 100 image--solver records with GOOD and at least one other passing class, E0 selects G in **92** versus **51.9 expected under S2**. These statistics establish empirical departure from S2 at calibration depth; same-image rank comparisons are correlated and should not be treated as independent Bernoulli trials for final uncertainty.

A first **regularized composite Bradley--Terry pair-rank fit** pools class-pair outcomes within each family, takes \(w_G=1\) and \((\log w_O,\log w_C)\sim N(0,2.5^2 I)\) as a stabilizing penalty. P is tied to O and I is set to last under S1. Numerical preliminary weights:
\[
\begin{array}{c|cc}
  &w_O&w_C\\ \hline
 DAPS&0.17040&0.02011\\
 SITCOM&0.95358&0.04538\\
 FAST&0.03658&0.02360
\end{array}
\]
Within the 180 calibration images of each family, this in-sample fit yields expected G wins ~157.6/180 (DAPS), 141.8/180 (SITCOM), 154.7/180 (FAST), compared with observed 158, 144, 154. These are **not out-of-sample accuracy claims**. PL full-rank likelihood and the pairwise composite are distinct estimators; use the named composite method, block-bootstrap by image for any parameter uncertainty, and avoid calling the fit exact PL MLE.

**Physical comparison:** G versus global O preference is strongly family-dependent: DAPS 21/24, SITCOM 14/30, FAST 15/15, while both O and C are small classes. The family-pooled weights ignore possible cohort-specific race differences and the one FAST S1 violation. Retain those as explicitly modeled simplifications and sensitivity checks.

## 5. Heavy-tail sensitivity model and identification limits

Five of 205 within-image G/C comparisons have C below G. A **preregistered stress alternative** can make each C run an independent, rare extreme low-score decoy with probability \(\epsilon=5/205\), beating all G/O/P runs when present; ordinary C always loses to G, but its ordering against O/P must be specified in the code. This creates nonmonotone best-of-K behavior at sufficiently large K, analogous to the known Yano--Ishibashi--Oyamada tail-competition regimes (arXiv:2610.19362). The 5/205 statistic is a **composite pair frequency with dependence**, not an independently estimated per-run extreme-tail probability. Thus setting epsilon=5/205 is a sensitivity calibration, not an identified empirical tail law.

The more mechanistic alternative is a C score mixture whose lower-support component falls below the G support and whose common component lies above it. Any selection outcome conditional on observed finite class counts can then be computed through exact race/tie rules. The dataset is insufficient to identify tail shape from only four ranks; if A8b supplies actual per-run residual gaps before the freeze, prefer fitting their lower-tail behavior.

## 6. Correct scientific decomposition and error bars

The **audit-defined generation failure** is no run GOOD or ORBIT in K attempts,
\[
F_{\rm gen}(K)=(1-q)^K.
\]
The overall policy failure decomposes exactly as
\[
1-E_1(K)=
\underbrace{(1-q)^K}_{\rm no\ orbit\ candidate}
+
\underbrace{\bigl(O_{\rm orbit}(K)-E_1(K)\bigr)}_{\rm selection\ loss}.
\]
This holds for every selector that outputs one candidate and does not create new orbit-good candidates; no S1/S2 needed. It is NOT the same as the S2 paper's earlier 'no passer' convention \(u^K=(1-s)^K\). Define the crossover in the first decomposition and average **per-image** probabilities over the known 100:80 image mixture.

For prospective empirical uncertainty with n=16,32,64 repeated draws on the same images, distinguish (a) posterior model-conditional extrapolation, (b) uncertainty estimating the race weights and latent image class probabilities, and (c) finite-K **U-statistic sampling noise** of the empirical curves. Bootstrap images as clusters and, where needed, use within-image run resampling or analytic first-order U-statistic variance. The already proved 2/4-run moment obstruction means none of these is a distribution-free identification band for K=64.

## 7. Pre-freeze discipline

The primary frozen forecast must have the rank-race policy as E0/E1, S1/S2 as an explicit reference with its own monotone-gap prediction, and at least one predeclared rare-C stress alternative. Predictions, code version, A8 data checksum and A8b-data inclusion status must be committed and registered before any XP K>4 outcomes are opened. This document is *not* that freeze. The governing dates are target **2026-10-29 18:00 America/Detroit** and hard limit 2026-10-31 23:59.
