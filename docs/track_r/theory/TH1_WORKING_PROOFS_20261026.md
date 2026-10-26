# TH1 working proofs and scope audit (26 October 2026)

**Status:** PROVED for abstract statements below; CONDITIONAL for their applicability to actual solver implementations. This is a theory-executor working file, **not** the frozen predictions registry. Branch: \`theory/prdiff-track-r-th-20261026\`. No XP returns, ground-truth image files or restricted holdout materials were consulted. Notation follows \`THEORY_PROGRAM_v1.md\` v1.2.

## 0. Corrections needed before paper integration

1. **The monotone-gap limit requires a positive oracle success probability.** If \`q=0\` and \`c>0\`, then \`O_orbit-E1=0\` for all K, not \`(c+o_p)/s=1\`. The limit \`(c+o_p)/s\` is valid only if \`q>0\`. For the raw gap \`O_raw-E0\`, the analogous formula needs \`p>0\`. Mixtures with hard images must keep the indicator.
2. **Imperfect orientation tests do not spoil monotonicity under S1--S3.** The existing program conjectures that \`h>0\` may break monotonicity for small K; that is false when the stated error rates are K-independent. In fact, for any \`0<=h,tau_g,tau_p<=1\` the same monotonic theorem holds because \`b=(1-h)p+tau_g o_g+tau_p o_p<=q<=s\`. K-dependent orientation-test error rates can break it, but are a *different* model.
3. **Particle weights:** Only weights that are functions solely of the measurement likelihood, or a residual, are automatically G-invariant. The full incremental weights in TDS include reverse-prior transition and proposal-density terms and need *not* be G-invariant for a non-equivariant denoiser. For TGD, likelihood incremental potentials are invariant; that does not make prior-dependent proposals equivariant. A-TGD's *residual pruning index* is invariant. The corollary should be phrased at the likelihood factor / residual-pruning level, not as a blanket claim about all SMC weights.
4. **Gaussian TV theorem scope:** Each non-equivariant transition that contributes to the output must be followed by an additive Gaussian with strictly positive, state-independent variance, or separately controlled. A terminal deterministic non-equivariant projection/denoising map cannot simply be "folded into" a prior Gaussian mean: it changes the transition family. This is a conditional theorem, not yet certified for the native DAPS / FAST-DIPS / SITCOM implementations.
5. **What is generation failure?** "No measurement-consistent run" has probability \`u^K\`. "No orbit-correct run" has probability \`(1-q)^K\`, generally larger and more directly related to \`O_orbit\`. The paper needs to name which is intended.
6. **Orbit cardinality:** Eight rotations form a group of order eight, but the image orbit itself can have fewer than eight *distinct* members when one or more channels are 180-degree symmetric. Uniformity over eight labels requires a trivial stabilizer, or else use uniformity over distinct members.

## 1. Exact forward symmetry: Lemma and Proposition 1 (PROVED)

Put \`z=(x+1)/2\` and \`f=Pz\`, padded centrally from length 256 to 384 along both axes; image coordinates are \`j=64,...,319\`. On a single channel, 180-degree reversal is \`(Rz)[u,v]=z[255-u,255-v]\`. For any \`j,k\` on the 384 torus,
\`P(Rz)[j,k]=f[383-j,383-k]=f[-j-1,-k-1]\` (indices mod 384); this also holds off the support. Thus \`P(Rz)\` is a reversed, translated copy of \`f\`. Its discrete Fourier transform differs from \`conj(Ff)\` by a phase factor of unit modulus. Therefore \`|F P(Rz)|=|F Pz|\` *pointwise* at every grid frequency; FFT centering and orthonormal scaling do not change this. Acting independently on the three channels yields \`A(gx)=A(x)\` for every \`g in G=(Z/2Z)^3\`. This holds for every image and every signed measurement realization y, without a uniqueness assumption.

The map \`z -> -z\` is another Fourier-magnitude symmetry in ambient real signal space, but except for the zero signal leaves the physically admissible cube \`z in [0,1]^d\`, so it is not an extra within-domain orbit operation.

**Identifiability caveat.** This proves \`G x*\` lies in the ambiguity set. It does **not** prove that the ambiguity set equals \`G x*\`. The 256-wide support occupies 2/3 of the 384 grid; the standard half-grid hypothesis for generic nonaliased autocorrelation is not met. Any statement identifying all measurement-consistent signals with these eight orbit members must separately assume *orbit identifiability* (or be made relative to the full ambiguity set). Source lineage: Hayes (1982); Barnett et al., *Geometry of the Phase Retrieval Problem*; DAPS, arXiv:2407.01521 App. F.2 (the eight visually distinct outputs are already observed there).

Let \`g\` act measurably, invertibly and with unit absolute Jacobian on the domain, let \`L_y(x)=ell(y,A(x))\`, and let prior density \`pi\`. Invariance gives \`L_y(gx)=L_y(x)\`. Hence, wherever both densities are meaningful and \`pi(x)>0\`,
\`p(gx|y)/p(x|y)=[L_y(gx) pi(gx)]/[L_y(x)pi(x)]=pi(gx)/pi(x)\`.
No quotient is asserted at zero prior density. For measurable S, change variables \`v=gu\` gives
\`P(gS|y)=Z_y^{-1} integral_S L_y(u)pi(gu)du\`.
This is Proposition 1 and the region form.

## 2. Corollaries 1, 1-prime, 2 and 3 (PROVED with explicit scopes)

**Cor. 1 (selector invariance).** Suppose a selector's chosen *index* is a function only of \`(A(x_1),...,A(x_K),y)\` plus fixed run-index tie breaks. Replacing any \`x_i\` by \`g_i x_i\` preserves its input tuple and the index; the algorithm receives no measurement signal revealing which within-orbit orientation is correct. This is unconditional on the prior or generator. It does *not* imply that applying a learned prior in a later step is impossible.

**Cor. 1' (likelihood factor / pruning).** Under likelihood \`L_y(x)=ell(y,A(x))\`, the potential \`w(x)=f(L_y(x))\` is invariant: \`w(gx)=w(x)\`. This covers likelihood-only importance potentials and choosing the minimum-residual particle at an intermediate or terminal time. In SMC with \`W_i\` containing proposal and reverse-model density ratios, \`W_i\` itself is invariant only if *those additional factors* transform equivariantly. TDS [Wu et al., NeurIPS 2023] and TGD [Makris et al., arXiv:2605.03712] have such additional mechanics; do not assert they make the whole particle distribution G-invariant. Their measurement factors alone carry no orientation information. PC-DAPS's prior-consistency pruning is a *different* factor and may discriminate orbit members.

**Cor. 2 (Bayes within orbit).** For finite distinct orbit \`Gx\` with positive posterior masses and identical likelihood, conditional probabilities are proportional to \`pi(gx)\`, and the MAP *orbit representative* is \`argmax_g pi(gx)\`. For a continuous prior, this should be understood as an orbit-representative density comparison, or as equal-small-neighborhood limits; a single point has probability zero. A proxy selecting the same maximizer is Bayes-optimal *for 0-1 orbit-representative loss*, but not necessarily globally Bayes-optimal for PSNR or other continuous losses. CLIP's or denoising-score's calibration to this ordering is **EMPIRICAL, NOT PROVED**.

**Cor. 3 (invariant nuisance cancels).** If \`log q(x)=log pi(x)+h(t(x))+e(x)\` with \`t(gx)=t(x)\`, then \`log q(gx)-log q(x)=log pi(gx)-log pi(x)+e(gx)-e(x)\`. Thus exact same-orbit ordering follows if \`e(gx)\` is also invariant (or if the approximation errors are too small to change the pairwise prior margins). The power spectrum, histogram and 180-degree invariant total variation are examples of invariant statistics, not automatically of complete prior proxies.

## 3. Cor. 4 and Cor. 4(b) (PROVED for abstract chains; applicability conditional)

**Cor. 4 (kernel version).** Let the initialization law \`P_T\` be invariant under a finite group of orthogonal linear maps \`G\`. Suppose each transition kernel satisfies \`K_i(gx,gB)=K_i(x,B)\` for measurable B. Then all marginals are G-invariant. Induction: \`P_{i-1}(gB)=int K_i(x,gB)P_i(dx)=int K_i(g^-1 x,B)P_i(dx)=P_{i-1}(B)\`. For deterministic maps \`F_i\`, this condition follows from \`F_i(gx)=gF_i(x)\`; additive isotropic Gaussian noise preserves it. If the orbit of the target has eight distinct points and eight disjoint G-translated neighborhoods, conditional probabilities of these neighborhoods given their union are each 1/8. This result uses no phase-retrieval uniqueness assumption. Precedent: Xu et al. 2022 (GeoDiff), Hoogeboom et al. 2022 (equivariant diffusion).

For any differentiable \`f_y(x)=phi(A(x),y)\`, \`f_y(gx)=f_y(x)\` and, wherever the gradient exists, \`nabla f_y(gx)=g nabla f_y(x)\`, since g is orthogonal. For nonsmooth points, an equivariant subgradient selection is needed. Fixed-step gradient descent, backtracking based on invariant scalar criteria, Euclidean projections *when uniquely defined* and their canonical minimum-norm choices, coordinatewise clipping and Adam with invariant coordinate state commute with pixel permutations. Ties in nonunique projections, arbitrary optimizer restarts, non-equivariant stopping rules, or non-invariant tie breaks can violate the hypothesis.

**Family audit, conditional (not code-verified):**
- **DAPS:** the measurement-likelihood Langevin gradient, isotropic noise and Gaussian quadratic tether commute with G *provided* the denoiser and all initializations do. Verify whether its final denoise is deterministic and whether its actual state-dependent step schedules are invariant.
- **FAST-DIPS:** measurement projection / Gauss--Newton and scalar line search commute under orthogonal conjugation *if* linear systems are solved by a G-compatible rule (e.g. unique regularized solution or Moore--Penrose pseudoinverse) and all stopping/tie logic is invariant. A deterministic final projection is not automatically covered by the Gaussian TV bound below.
- **SITCOM:** optimization in latent \`v\` needs the denoiser equivariant **at every inner iterate**, matching transforms of initialization, optimizer state, and selection among local minima. Its differentiation through \`D\` also makes the Jacobian's equivariance relevant. Its output must not be assumed equivariant solely because the residual is.

**Cor. 4(b) (Gaussian-transition TV theorem).** Suppose \`x_T~P_T\` is G-invariant and independent innovations satisfy \`x_{i-1}|x_i=x ~ N(m_i(x),s_i^2 I)\` with \`s_i>0\` constant in x and invertible orthogonal g. Let \`P\` be the joint forward-in-reverse-time path law of \`(x_T,...,x_0)\`; let \`Q=g#P\` (rotate *every* state). Under Q the initial law is the same and the transition mean is \`m_i^g(x)=g m_i(g^-1 x)\`, with covariance \`s_i^2 I\`. Write \`Delta_i^g(x)=m_i(x)-m_i^g(x)\`. The chain rule gives
\`KL(P||Q)=sum_i E_P ||Delta_i^g(x_i)||^2/(2s_i^2)\`,
assuming the sum is finite; otherwise the bound is vacuous. Projection onto the final-coordinate marginal is a data-processing map, so
\`TV(L,g#L) <= TV(P,Q) <= sqrt(KL(P||Q)/2) = [sum_i E_P||Delta_i^g(x_i)||^2/(4s_i^2)]^(1/2)\`
by Pinsker. For any measurable B, \`|L(B)-L(gB)|<=TV(L,g#L)\` (replace g by inverse, the same in our order-two group). This proof needs neither score exactness, log-concavity nor identifiability.

**Scope guard:** A state-dependent covariance gives extra KL terms; a terminal map \`x_0=H(x_1)\` that is both deterministic and non-equivariant is *not* Gaussian and cannot be inserted into this argument without an additional theorem. The abstract theorem does not certify all concrete solver steps. An inequality in terms of *denoiser defect* also needs an explicitly proved stability estimate such as \`||Delta_i^g(x)|| <= C_i delta_i(x)+C'_i eta_i(x)\`, with \`eta\` a Jacobian defect if differentiated guidance uses D. The displayed TV theorem bounds **transition-mean defects** unconditionally within its assumptions, not the raw relative defect XP-C2 measures. XP-C2 on noised ground-truth images is a descriptive proxy, not the transition-law expectation in this inequality.

**Cor. 5 (exact posterior sampler).** If the chain outputs the model posterior \`p_theta(x|y)\` exactly, then Prop. 1 gives relative density weights for *corresponding orbit representatives*. For distinct neighborhoods whose likelihoods and volume match by G, the conditional probability of orbit member g is proportional to its *integrated* prior mass, not necessarily the point-density \`pi_theta(gx*)\` unless a local-small-ball approximation is invoked. A residual-only selector cannot distinguish exact orbit transforms, but can change relative probabilities of *distinct* posterior samples if their residuals differ; it is too strong to claim a residual selector never changes overall decoy rates.

## 4. TH1.2: closed forms, correspondence and monotone gap (PROVED)

Fix one image. Independent runs have class probabilities \`(p,o_g,o_p,c,u)\` summing to one. Define \`q=p+o_g+o_p\`, \`s=q+c=1-u\`, \`b=(1-h)p+tau_g o_g+tau_p o_p\`; assume \`0<=h,tau_g,tau_p<=1\`. Assume:
- **S1:** every INCONSISTENT residual exceeds every non-INCONSISTENT residual in the K-sample set;
- **S2:** conditional on the non-INCONSISTENT class labels, residual ranks of these entries are uniformly exchangeable (continuous scores or equivariant tie-breaking);
- **S3:** the subsequent orientation classifier's class-conditional probabilities are the K-independent numbers in b, independent of residual ranking conditional on the selected class.

Then for K>=1, with the convention \`E0=E1=0\` when \`s=0\`,
\`O_raw(K)=1-(1-p)^K\`;
\`O_orbit(K)=1-(1-q)^K\`;
\`E0(K)=[1-u^K]p/s\`;
\`E1(K)=[1-u^K]b/s\`.
Indeed the number of non-INCONSISTENT runs is Binomial(K,s); conditional on it being positive the minimum is label-uniform by S2, so success probabilities are respectively p/s and b/s. The two oracles succeed iff at least one respective type occurs.

**Stroebl et al. correspondence:** Define verifier "accept" iff a run is in {GOOD, ORBIT-global, ORBIT-partial, CONSISTENT}; within the non-INCONSISTENT part, accept-first in an independent K-sample sequence has the same **marginal returned-label distribution** as argmin residual under S1--S2. Proof: the first acceptable label has distribution of a single run conditional on acceptable (p/s,og/s,op/s,c/s), and acceptance occurs with probability \`1-u^K\`. One should not claim the *same sample index* or a coupled sample-level equivalence. Stroebl, Kapoor & Narayanan, ICLR 2026, App. C study this verifier-acceptance model with completeness / soundness and image/task heterogeneity; our symmetry subclass does not newly discover the imperfect-verifier ceiling.

**Ceilings and gaps:** When \`s>0\`, \`E0(infty)=p/s\`, \`E1(infty)=b/s\`. The corrected gap limits are
\`lim (O_orbit-E1) = 1{q>0} - b/s\`,
\`lim (O_raw-E0) = 1{p>0} - p/s\`.
If \`q>0\`, global-only perfect repair (\`h=0,tau_g=1,tau_p=0\`) gives orbit gap limit \`(o_p+c)/s\`; otherwise the gap is zero. The same formula for the raw gap reduces to \`(o_g+o_p+c)/s\` only if \`p>0\`.

**Corrected monotone-gap theorem (stronger than requested):** For *any* constant admissible h, tau_g, tau_p as above,
\`[O_orbit(K+1)-E1(K+1)]-[O_orbit(K)-E1(K)] = q(1-q)^K - b(1-s)^K >= 0\`,
because \`0<=b<=q<=s<=1\`. Likewise the raw gap increment is \`p[(1-p)^K-(1-s)^K]>=0\`. At K=1 the orbit gap is \`q-b = hp+(1-tau_g)og+(1-tau_p)op\` (equals \`o_p\` in the perfect global-only case). The raw gap is 0. Integration over any fixed population distribution of image types preserves non-decreasing gaps. This does *not* assert observed finite-sample point estimates must be monotone or that S3 holds after selection.

**Two different failure decompositions:** Relative to E1, define \`F_gen^cons(K)=u^K\` (no measurement-consistent sample) and \`F_sel^cons(K)=(1-u^K)(1-b/s)\` (at least one consistent sample, selected one not GOOD after repair). Exactly \`1-E1=F_gen^cons+F_sel^cons\`. These cross at the smallest integer K solving \`u^K <= (1-u^K)(s-b)/s\`. Relative to O_orbit, unavoidable generation failure is instead \`F_gen^orbit=(1-q)^K\`; it is *not* equal to \`u^K\` when c>0. For the global-only case b=p+og, \`1-b/s=(op+c)/s\`, not merely c/s; partial orbit errors also count as selection misses. For a population, average the *two terms* over image types before computing K*; avoid averaging per-image K*.

**Inflation decomposition:** The identity \`O_raw-first = (E1-first)+(O_raw-E1)\` is arithmetic, but \`O_raw-E1\` need not be nonnegative because orientation repair can outperform the raw oracle (e.g. p=0, og>0). The term is not generically "the GT-only contribution"; for a nonnegative GT selection advantage compare *matched objectives*, e.g. \`O_orbit-E1>=0\`. The paper must not describe signed \`O_raw-E1\` as purely oracle information.

**S1/S2 diagnostics and relaxations:** An empirical label-stratified inversion audit of residual rankings tests S1. A permutation/randomization test of the GOOD, ORBIT, CONSISTENT residual-rank distribution conditional on a fixed image tests S2, though n<=4 limits power. If any pair is inverted relative to S1 with probability <= epsilon, a union bound implies the S1-only failure event in a K set is <= binom(K,2) epsilon. AUROC(GOOD,CONSISTENT)>1/2 is *not sufficient* to guarantee that K-way residual argmin beats accept-first for every K: pairwise rank dominance does not determine all order-statistic tails. Sufficient stronger condition: GOOD residual distribution stochastically dominates CONSISTENT in the lower-is-better direction (with the other classes' effects modeled).

## 5. Exact estimators (PROVED)

Given n>=K i.i.d. runs for one fixed image and a policy \`T_K\` returning a scalar score from an *unordered* K-subset with index-stable deterministic tie breaking, the full-subset statistic \`U_K = binom(n,K)^(-1) sum_{|I|=K} T_K((X_j)_{j in I})\` is unbiased for \`E[T_K(X_1,...,X_K)]\` by exchangeability of the n independent runs. This is a degree-K U-statistic; no model S1--S3 is needed.

For argmin with global rank r=1 (best/lowest) to n and per-run outcome \`z_r\`, the selected run is r precisely when the subset contains r and its K-1 other members lie in ranks r+1,...,n. Thus
\`U_argmin = [sum_{r=1}^n binom(n-r,K-1) z_r]/binom(n,K)\`.
For argmax with global increasing-score rank j, all other selected ranks must lie below j, so
\`U_argmax = [sum_{j=1}^n binom(j-1,K-1) z_j]/binom(n,K)\`.
Ties must be ordered by the same deterministic global tie key that the subset-level policy uses. For binary oracle success with g successful runs, \`U_pass=1-binom(n-g,K)/binom(n,K)\`, Chen et al., *Evaluating Large Language Models Trained on Code*, 2021, §2.1. A binary orbit oracle uses g+o successful-orbit runs. For a metric-specific oracle, sort by *that same metric*; never accidentally mix independently maximized metrics as a single returned candidate.

Uncertainty from fixed-images bootstrap must resample **images as clusters**; when using parametric forecasts, also resample fit parameters. Estimators for K<=n are finite-sample unbiased per image, but cannot directly estimate K>n: extrapolation needs explicitly stated modeling assumptions.

## 6. Per-metric oracle inflation (PROVED for Gaussian setting)

If (M1,M2) on each iid run is bivariate normal with \`M2=mu2+sigma2[rho Z1+sqrt(1-rho^2)Z2]\`, \`Z1,Z2\` independent standard normals, selecting the maximum M1 has expected M2 \`mu2+rho sigma2 c_K\`, whereas selecting maximum M2 has expected M2 \`mu2+sigma2 c_K\`, with \`c_K=E max_{1..K}N(0,1)\`. Therefore \`Delta_K=(1-rho)sigma2 c_K>=0\`. For LPIPS reverse the sign to a "higher is better" metric. This formula is specific to the bivariate-Gaussian assumption and not guaranteed for PSNR/SSIM/LPIPS empirical distributions. Without normality, nonnegativity follows pointwise because \`max M2 >= M2\` at the M1-maximizer. A crude range bound \`0<=Delta_K<=sup M2-inf M2\` applies if M2 is bounded.

## 7. Next work and provenance

- **No frozen predictions yet.** A8 per-image strings are not assumed available on 26 October. Data path on the planner branch was not found; the allowed R0 P4/analysis files were verified on \`codex/prdiff-track-r-r0n-20261023\`. If A8 is still uncommitted Oct 29 noon ET, fit solely on P4 at K<=4 and label the unavoidable aggregate-identifiability limitation. Only DAPS / SITCOM / FAST-DIPS, no NP.
- **Not yet code-verified:** family-specific complete transition equivariance; prior-regularity assumptions behind the suggested non-transversality/Hölder claims; actual prediction fit; TH2 zero-flip constructions.
- **Primary source pointers:** Hayes 1982; Barnett et al. 2020/2022; Fienup & Wackerman 1986; DAPS arXiv:2407.01521 App. F.2; Stroebl et al. ICLR 2026 App. C; Xu et al. GeoDiff ICLR 2022; Hoogeboom et al. EDM 2022; Wu et al. TDS NeurIPS 2023; Makris et al. TGD arXiv:2605.03712; Dandapanthula & Boffi arXiv:2606.02884; Chen et al. 2021. These references are lineage and not a claim that all full texts were independently checked.
