# Track R A11: exact stop-at-M theory under the class-race model

**29 October 2026. PROVED within the specified iid conditional-on-image mark model.**
This is a proposed new theory result for the FROZEN protocol in
`GTFREE_PROTOCOL_v1.md` (SHA-256 `7e9a5fd8b817a97351d07ae371b947688c8430cd5b9268fc99b954a493223919`),
not a claim that the model is exactly the native diffusion solver. No later
ImageNet D1-P2, SITCOM B16 or DAPS F8 strings or `xp/returns` have been read.
The protocol's decision is made from **the physical residual** rho, threshold
rho <= tau, and an independent zero-shot orientation score on the selected
candidate. The stopping flag FLAG_C means **zero passing candidates**, not
fewer than M passing candidates.

## 1. Correct joint model and exact truncated stopping law

Fix a measurement, image and solver. Each run is an iid *marked* draw
`(J,R,A)` conditional on the image: `J` is G/O_g/O_p/C/I,
`R=rho` is its nonnegative signed-measurement residual,
`V=1{R<=tau}` is its threshold indicator, and `A` denotes whether the
zero-shot orientation decision on that candidate would yield GOOD.
Write `theta_j=P(J=j)` and **per-class passing rates**
`v_j=P(V=1|J=j)`, so

\[
 s=P(V=1)=\sum_j\theta_jv_j,\quad r=1-s,
 \qquad \eta_j^+=\theta_jv_j/s,\quad
 \eta_j^-=\theta_j(1-v_j)/r.
\]

Crucial: since tau is the 95th percentile of **GOOD** historical residuals,
`v_G` is approximately 0.95 on a matching calibration distribution, **not
1**. C and I are defined using their own frozen classification threshold;
when the deployment tau is transferred from another source (ImageNet D1),
`v_C=1,v_I=0` also need not hold exactly. The fitted PL class weights are
*residual ranking surrogates* and do not by themselves determine v_j or the
joint conditional residual distribution. Do not treat threshold pass as an
independent Bernoulli attached to a score without acknowledging this extra
approximation.

Let N=N_max and 1<=M<=N. Define the stopping time
`T=min{ inf{t: sum_{i<=t}V_i=M}, N }`. Then for M<=t<N,

\[
 P(T=t)=\binom{t-1}{M-1}s^M r^{t-M},\qquad
 P(T=N)=P\{\operatorname{Bin}(N-1,s)<M\}.
\]

The expected number of runs, with truncation handled exactly, is

\[
 E T=\sum_{t=0}^{N-1}P\{\operatorname{Bin}(t,s)<M\}
 =\sum_{t=0}^{N-1}\sum_{j=0}^{\min(M-1,t)}{t\choose j}s^j r^{t-j}.
\]

For the frozen M=2 and s>0 this reduces to

\[
 \boxed{E T=\frac{2-Nr^{N-1}+(N-2)r^N}{s}},
 \quad E T|_{s=0}=N,\quad E T|_{s=1}=2.
\]

The protocol raises FLAG_C iff no candidate passed in all N runs:

\[
 \boxed{P(\mathrm{FLAG}_C)=r^N.}
\]

This is **not** P{Bin(N,s)<M}; images ending with exactly one passing
candidate are not FLAG_C, even though the protocol hit its run limit.

## 2. Exact success functional, including rejected GOOD runs

Let `H_j^+` denote the probability that the smallest-residual run among
`j>=1` iid draws conditional on V=1 yields GOOD **after the frozen orientation
step**, and `H_j^-` the corresponding probability among j draws conditional
on V=0. Since every passing residual is <=tau and every nonpassing residual
is >tau, the selected run comes from the passing group whenever it is
nonempty. The first M passing marks are iid with the conditional passing
law, **independent of the waiting time**. Hence, with
`b_{N,j}(s)={N choose j}s^j r^{N-j}`,

\[
\boxed{
 P(\mathrm{E1\ success})=
 \Big(\sum_{j=M}^{N} b_{N,j}(s)\Big)H_M^+
 +\sum_{j=1}^{M-1}b_{N,j}(s)H_j^+
 +r^N H_N^- .
}
\]

In particular, for M=2,

\[
 \boxed{S_{2,N}=(1-r^N-Nsr^{N-1})H_2^+
       +Nsr^{N-1}H_1^++r^NH_N^-.}
\]

This is **exact** under the iid marked-run model and does not require S1,
S2, exponential class races or a perfect orientation scorer. The last term
matters if a GOOD reconstruction misses tau (the approximately 5% GOOD tail)
or if the protocol's threshold differs from the classification threshold.
For E0 rather than E1, replace the candidate success mark A by `1{J=G}`.

For any fixed integer k, the same physical residual selection without
stopping has success

\[
 F(k)=\sum_{j=1}^{k}b_{k,j}(s)H_j^+ +r^k H_k^-.
\]

For a global cohort with multiple images i, compare `mean_i S_i` with the
**frozen linearly interpolated** fixed policy curve at
`K_bar=mean_i E[T_i]`: if `k=floor(K_bar)` and `a=K_bar-k`, the fixed
reference is `(1-a)mean_i F_i(k)+a mean_i F_i(k+1)`. Using the real-valued
power r^{K_bar} instead would **not** match the frozen evaluator.

**What is required to evaluate H_j:** class probabilities and the
threshold-pass probabilities alone do not determine residual ranking
within passing or nonpassing candidates. Let f_j(t) be each class's
residual density and a_j(t)=P(A=1|J=j,R=t), which incorporates a possible
harmful GOOD flip h and successful global/partial orbit correction phi.
For 0<s<1 and continuous residuals,

\[
 H_m^+=\frac{m}{s^m}\sum_j\int_0^\tau
    a_j(t)\,\theta_j f_j(t)
    \Big(\sum_{\ell}\theta_\ell[F_\ell(\tau)-F_\ell(t)]\Big)^{m-1}dt,
\]

\[
 H_n^-=\frac{n}{r^n}\sum_j\int_\tau^\infty
    a_j(t)\,\theta_j f_j(t)
    \Big(\sum_{\ell}\theta_\ell[1-F_\ell(t)]\Big)^{n-1}dt.
\]

These are the exact minimum-score competing-risk integrals. At zero
probability groups use their continuous limits. Alternatively, a **separate
conditional-PL approximation** replaces within-pass winner probabilities
by `m eta_j^+ int w_j^+ exp(-w_j^+t) (sum eta_l^+exp(-w_l^+t))^{m-1}dt`;
the frozen A8 rank weights w_j are not automatically equal to these
pass-conditioned rates. This is the principal assumption obstacle for
an honest v3P forecast from existing four-run data.

**FLAG_C identifying truly unsolved panel images.** Define
`q=P(J in {G,O_g,O_p})` and
`t=P(V=0, J not in {G,O_g,O_p})`. With an
independently generated full N-run panel,

\[
 P(\mathrm{FLAG}_C,\mathrm{no\ orbit-good\ in\ panel})=t^N,
\quad P(\mathrm{no\ orbit-good})=(1-q)^N.
\]

Thus per-image FLAG_C precision and recall under full-panel
orbit-oracle labels are `(t/r)^N` and `(t/(1-q))^N` when denominators
are positive. Across heterogeneous images, use **ratios of summed
expected counts** and not averages of per-image precision. This is
not a claim that FLAG_C certifies failure or reveals the unseen p_i.

## 3. When stopping beats equal-compute fixed K: valid statements

For any one image, T(s) decreases stochastically in s: couple all
Bernoulli pass indicators by common Uniform[0,1] draws. Consequently,
`E[T(s)]` is nonincreasing, and for a nondegenerate population of
consistency rates, `Cov(s,E[T(s)])<=0`. Stopping **does redistribute
compute** toward low-s images. But this alone is not an accuracy theorem:
low-s images may be nearly impossible, and a strong class-race selector
may benefit from seeing >M passing candidates on solvable images.

**A transparent sufficient condition.** Assume no GOOD can be selected
when no threshold-passing candidate exists, and let
`H_min,i=min_{1<=j<=M} H^+_{i,j}` and
`H_max,i=max_{1<=j<=ceil(K_bar)} H^+_{i,j}`.
Then `S_i >= H_min,i (1-r_i^N)` and
`F_i^interp(K_bar) <= H_max,i (1-tilde{r_i^{K_bar}})`, where
`tilde{r^K}=(1-a)r^k +a r^{k+1}` for k=floor K.
Therefore **if**

\[
 \sum_i H_{\min,i}(1-r_i^N)\ \ge\
 \sum_i H_{\max,i}(1-\widetilde{r_i^{\bar K}}),
\]

then the stopping protocol does at least as well as the frozen
fixed-at-mean-K comparator. This is conservative but testable. A more
informative exact criterion simply compares the two preceding
finite sums with all per-image `H_j` terms included.

**Counterexample to a universal heterogeneity theorem.** Two equally
weighted image types, N=16, M=2. Image 1 has pass probability
`s_1=0.02` but its passing runs are all C, never GOOD; all other runs
are I. Image 2 has `s_2=0.90`, split equally among passing GOOD and
passing C, with I otherwise. Within the passing group, take class-race
weights `(w_G,w_C)=(1,0.1)`, and zero success after nonpassing candidates.
The exact truncated calculation gives

\[
 E[T_1]=15.8031,\ E[T_2]=2.2222,\ \bar K=9.01267;
\quad S_{stop}=0.35227 < F_{fixed}(\bar K)=0.44263.
\]

Stopping **loses by 9.04 percentage points** despite extreme
consistency-rate heterogeneity: it wastes the long budget on an unsolved
image while limiting the solvable image to two passing candidates.
This counterexample does not contradict the positive FAST FFHQ
in-sample result; it proves that an unqualified general theorem is false.

## 4. Why a high CONSISTENT share can erase the gain

A concrete reduced model isolates the source of failures. Let each run
be GOOD G with probability p, CONSISTENT-but-wrong C with
`c=(1-p)alpha`, and INCONSISTENT I with `u=(1-p)(1-alpha)`.
Assume all G/C pass tau and all I fail; orientation is irrelevant.
Then `alpha=c/(c+u)` is precisely the fraction of failed runs that
are CONSISTENT, `s=p+c=1-u`, and `q=p/s` is the GOOD fraction of
passing runs. If passing residual ranks are exchangeable, selection
among any positive number of passes is GOOD with probability q:

\[
 S_{2,N}=q(1-u^N),\quad
 F(k)=q(1-u^k),\quad
 \boxed{\mathrm{Gain}(\alpha)
  =\frac{p}{p+(1-p)\alpha}\left[
       \widetilde{u^{\bar K(\alpha)}}-u^N\right]},
\]

where `K_bar(alpha)=E[T(s)]` is the formula in Sec. 1 and the tilde
means the frozen linear interpolation. This model yields nonnegative
gain, tends to **zero as alpha->1** because s->1, T->M and
K_bar->M, and demonstrates explicitly that residual-consistent *wrong*
candidates make early stopping less informative. For p=0.3,N=16,M=2,
our numerical regression gives gain +9.43 points at alpha=0,
+1.88 points at alpha=.50, +0.035 points at alpha=.95, and zero in
the limit. These figures are **synthetic**, not Track R predictions.

Under the two-class passing **race**, replace q by

\[
 H_j^+=\sum_{g=1}^j\binom jg q^g(1-q)^{j-g}
  \frac{gw_G}{gw_G+(j-g)w_C}.
\]

The stopping/fixed finite sums above then remain valid and the
alpha->1 gain still converges to zero (T=M, both policies see M runs),
but the gain can have either sign for intermediate alpha. With the
actual threshold rule `v_G` is closer to 0.95 than 1; the limit
`alpha->1` **alone** need not make `s->1` if rejected GOOD runs
remain. The clean vanishing theorem thus needs the all-G-pass
assumption (or an explicit limiting condition `s->1`), and should
not be asserted from the C/(C+I) ratio alone.

**Connection to development (descriptive, not a model fit).** DAPS
DEV80 has roughly 12% of failed runs in C and an equal-compute +4.9
point gain; SITCOM DEV80 has around 60% C and a -0.1 point gain.
This comparison is consistent with the model's mechanism, but does
not establish causality or prove the gain is monotonically decreasing
in C share in a general class-race model.

## 5. Claim and protocol boundaries

1. The closed forms are PROVED for iid marked runs **conditional on an
   image**. The actual finite-panel random-order estimator shuffles a
   *fixed realized panel*, and is not identical to an unconditional
   negative-binomial prediction when prefixes are observed. Conditional
   forecast registration requires conditioning on original tau-pass
   indicators, relative physical residual order and original G1/face
   orientation success labels, not only the A8 four-run class counts.
2. FLAG_O is transductive: its threshold is the 30th percentile of
   margins among outputs that do not raise FLAG_C. Without a model of
   the margin distribution *and cohort*, the class-race model does not
   yield a legitimate closed form for FLAG_O or H4 coverage. This does
   not affect the FLAG_C formula or E1 success definition.
3. These formulas explain *possibility and limits* of gain but not the
   numerical outcome of H1--H4 on the three frozen held-out panels.
   Whether predictions based only on historical runs are calibrated
   remains a prospective experiment; no B16/F8/D1-P2 phase-2
   individual strings have been used in this note.
4. Classical components: truncated negative-binomial waiting time,
   competing-risk integrals, sample-minimum order statistics, and
   binomial moment problems. The contribution lies in joining them to
   the pre-frozen, physically interpretable GT-free phase-retrieval
   protocol with measured class/solver behavior and held-out tests.
