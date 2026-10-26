# TH1 text for the CVPR draft (working, 2026-10-26)

**Integration constraint:** These are ready-to-adapt drafts for the *existing* sections \`sec/4_symmetry.tex\` and \`sec/5_selection.tex\`, not direct edits. All theorems below are conditional on their explicitly stated hypotheses. This file must be checked against final XP counts and citations; no XP outcomes have been viewed.

## Half-page version

**The measurement cannot choose orientation.** Write \(\mathcal A(x)=|\mathcal F_c(P((x+1)/2))|\), where \(P\) centrally embeds each \(256^2\) channel in a \(384^2\) canvas. Reversing either image channel by \(180^\circ\) reverses its padded array up to a one-pixel translation, leaving its Fourier magnitude unchanged. Consequently \(\mathcal A(gx)=\mathcal A(x)\) for all \(g\in G\cong(\mathbb Z_2)^3\). For a likelihood depending only on \(\mathcal A(x)\), the posterior density ratio within an orbit is \(p(gx\mid y)/p(x\mid y)=\pi(gx)/\pi(x)\), wherever the quotient is defined. No selector based solely on measurement predictions and \(y\) can prefer one orbit representative to another. This extends to *likelihood-only* particle weights and residual-based pruning, but not automatically to the entire weights of prior-dependent sequential Monte Carlo. The eightfold ambiguity was already observed in DAPS; the formal result characterizes the information unavailable to all such selectors (Hayes 1982; DAPS App. F.2; TDS; TGD).

**Why learned priors matter.** If a reverse sampler starts in a \(G\)-invariant distribution and every transition kernel is \(G\)-equivariant, its output is also invariant (Xu et al. 2022; Hoogeboom et al. 2022). When eight target-neighborhoods are distinct and disjoint, each has equal conditional mass. More quantitatively, for transitions \(x_{i-1}\mid x_i\sim N(m_i(x_i),s_i^2I)\), \(s_i>0\) constant, define \(\Delta_i^g(x)=m_i(x)-g m_i(g^{-1}x)\). Pinsker and the Gaussian-chain KL identity yield
\[
\big|P(x_0\in B)-P(x_0\in gB)\big|\le
\sqrt{\sum_i E\|\Delta_i^g(x_i)\|^2/(4s_i^2)}.
\]
This bound applies to *stochastic Gaussian transitions*; deterministic, non-equivariant terminal steps require separate control. Thus only non-equivariant components of the sampler can generate orientation preferences in this model.

**An imperfect-selector ceiling.** Per image, let \(p,o_g,o_p,c,u\) be probabilities of GOOD, global orbit, partial orbit, consistent-wrong, and inconsistent, with \(q=p+o_g+o_p\), \(s=q+c\), and \(b=(1-h)p+\tau_g o_g+\tau_p o_p\). Under independent runs, residual separation (S1), exchangeable residual ranks among measurement-consistent types (S2), and a K-independent orientation test (S3), the four success curves are \(O_{\mathrm{raw}}=1-(1-p)^K\), \(O_{\mathrm{orbit}}=1-(1-q)^K\), \(E_0=(1-(1-s)^K)p/s\), and \(E_1=(1-(1-s)^K)b/s\), with \(E_0=E_1=0\) if \(s=0\). The selector has the *same label law* as Stroebl et al.'s accept-first imperfect verifier (ICLR 2026, App. C). Because \(b\le q\le s\), both \(O_{\mathrm{orbit}}-E_1\) and \(O_{\mathrm{raw}}-E_0\) are non-decreasing in K, including when \(h>0\) is constant. The orbit gap tends to \((o_p+c)/s\) **if \(q>0\)** and global-only repair is perfect. When \(q=0\), the gap is identically zero.

## One-page version

**Exact symmetry, not uniqueness.** Let \(R_s\) independently rotate each channel of an image by \(180^\circ\), giving an eight-element group \(G\). In an even-sized \(384\times384\) padded array, the transformation of a channel is \(f[j,k]\mapsto f[-j-1,-k-1]\), a reversal followed by a translation. The Fourier reversal and shift identities imply \(|\mathcal F_cf|=|\mathcal F_cR_sf|\) pointwise. Hence the signed-noise measurement operator \(\mathcal A(x)=|\mathcal F_cP((x+1)/2)|\) is exactly \(G\)-invariant. This proves that the ambiguity set contains \(Gx^\star\) but does *not* assert that it equals it: the 256/384 support ratio violates the half-grid hypothesis used by standard generic-identifiability results. DAPS already illustrated eight measurement-indistinguishable images, and Fienup--Wackerman described twin stagnation in classical iterative reconstruction.

Suppose \(p(x\mid y)\propto \ell(y,\mathcal A(x))\pi(x)\) and \(G\) acts by measure-preserving maps. Pointwise, wherever defined,
\[
\frac{p(gx\mid y)}{p(x\mid y)}=\frac{\pi(gx)}{\pi(x)}.
\]
Thus the Bayes orbit-representative rule is the prior-density maximizer. Any selector whose inputs are solely \(\mathcal A(x_i)\) and \(y\) returns the same index after replacing candidates \(x_i\) by arbitrary orbit members. Likelihood-only SMC potentials and minimum-residual pruning satisfy the same invariance. However, *complete* TDS importance weights contain learned-prior/proposal transition terms, and their invariance must not be inferred from that of the likelihood factor. This distinction isolates where a learned prior can enter.

The same principle holds during reconstruction. If initialization is \(G\)-invariant and all transition kernels are equivariant, induction over the kernels shows that the final distribution is invariant (GeoDiff; equivariant diffusion). Conditional on landing in eight disjoint transformed target neighborhoods, their probabilities are \(1/8\) apiece; the assumption of a free orbit is needed. Measurement-gradient guidance, orthogonally invariant projections, and coordinate-permutation-compatible optimizer steps commute with \(G\), provided all tie breaks and step rules preserve it. In actual solvers this is a conditional implementation statement: a non-equivariant denoiser, its Jacobian where differentiated, or non-equivariant inner solver choices can violate the hypothesis. For Gaussian transitions \(x_{i-1}=m_i(x_i)+s_i\xi_i\) with \(\xi_i\sim N(0,I)\), \(s_i>0\) fixed, and invariant \(x_T\), the \(g\)-conjugate chain has transition mean \(g m_i(g^{-1}x)\). The chain rule gives \(\mathrm{KL}(P\|g_\#P)=\sum_i E_P\|\Delta_i^g(x_i)\|^2/(2s_i^2)\); Pinsker and data processing imply the total-variation bound displayed above. It does not need an exact score, log-concavity or identifiability, but it does need the Gaussian transition model *through the output*.

For quantitative K-scaling, model independent runs of a fixed image by five classes with probabilities \((p,o_g,o_p,c,u)\). Write \(q=p+o_g+o_p\), \(s=q+c\). Suppose (S1) inconsistent outputs are always worse in residual than the other types, (S2) residual ranks among the latter are exchangeable, and (S3) a post-selection orientation test fixes a global error with probability \(\tau_g\), fixes a partial error with \(\tau_p\), and harms a GOOD output with \(h\), independently of K after conditioning on class. Then
\[
\begin{aligned}
O_{\rm raw}(K)&=1-(1-p)^K,& O_{\rm orbit}(K)&=1-(1-q)^K,\\
E_0(K)&=[1-(1-s)^K]\frac p s,&
E_1(K)&=[1-(1-s)^K]\frac{(1-h)p+\tau_go_g+\tau_po_p}{s}.
\end{aligned}
\]
The ratio terms are set to zero for \(s=0\). This is exactly the *returned-class distribution* of a verifier that accepts GOOD, ORBIT and CONSISTENT entries and returns the first acceptable entry. We therefore credit Stroebl, Kapoor and Narayanan's imperfect-verifier model and its false-positive ceiling; our specialization makes the orbit classes explicit and tests the prediction prospectively. Writing \(b=(1-h)p+\tau_go_g+\tau_po_p\), the one-step increment of the orbit oracle-selection gap is
\[
q(1-q)^K-b(1-s)^K\ge0,
\]
since \(0\le b\le q\le s\). The raw oracle-selection gap also increases. Under perfect global-only repair (\(h=0,\tau_g=1,\tau_p=0\)), the orbit gap starts at \(o_p\) and tends to \((o_p+c)/s\) **for images with \(q>0\)**; it remains zero on structurally impossible images \(q=0\). Integrating over image types preserves monotonicity, not the naive expression without the indicator.

For a fixed image with n independent runs, each policy's average over all size-K subsets is an unbiased U-statistic, not an extrapolation. For an argmin residual rule, a run of ascending rank r is chosen in \(\binom{n-r}{K-1}/\binom nK\) subsets; for an argmax oracle, ascending rank j wins in \(\binom{j-1}{K-1}/\binom nK\). For binary oracle success this reduces to the unbiased pass@K formula \(1-\binom{n-g}{K}/\binom nK\) (Chen et al. 2021). Extrapolation to K>n instead needs a heterogeneous population model and confidence intervals, frozen before additional runs are inspected.

**Integration checklist.** Replace the paper's unconditional claim about all SMC weights; explicitly include positive transition variances in the TV theorem; state that orbit neighborhoods are disjoint and distinct; correct the gap ceiling at q=0; separate no-consistent generation failure \((1-s)^K\) from no-orbit-success generation failure \((1-q)^K\); and avoid describing \(O_{\rm raw}-E_1\) as always positive, since orbit repair can beat the *raw* oracle.

**Citation lineage:** Stroebl et al. ICLR 2026 App. C; Chen et al. 2021 §2.1; Hayes 1982; Fienup & Wackerman 1986; DAPS arXiv:2407.01521 App. F.2; Xu et al. GeoDiff ICLR 2022; Hoogeboom et al. equivariant diffusion 2022; Wu et al. TDS NeurIPS 2023; Makris et al. TGD arXiv:2605.03712; Dandapanthula & Boffi arXiv:2606.02884.
