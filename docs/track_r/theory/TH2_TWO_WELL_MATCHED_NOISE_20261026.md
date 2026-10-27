# Track R TH2: exact nonlinear two-well mechanics under matched noise budgets

**2026-10-26, evening.** **PROVED** for the two mathematical toy models below; synthetic CPU Monte Carlo checked; **NOT PROVED for native DAPS, SITCOM-ODE or FAST-DIPS**. Never used PAC, XP results, GT images or confirmation305. No new solver or method proposed. Read together with TH2_FINITE_STEP_BASIN_DYNAMICS_20261026.md: that earlier Gaussian *single-well* negative result is preserved, not overwritten.

## A. A mathematically exact two-well reaction-coordinate model

Take a phase-retrieval measurement fiber containing the distinct non-reversal zero-flip signals
\[
z=(1,5,6)/7,\qquad v=(2,7,3)/7,\qquad
|\mathcal F z|=|\mathcal F v|.
\]
Represent an anchor's position along their joining segment by a signed scalar x. Define a *toy* two-well potential
\[
U(x)=\frac{\kappa}{2}(|x|-b)^2,\qquad
b=\|z-v\|/2>0.
\]
Its two minima at \(\pm b\) represent the z and v basins, while the cusp x=0 represents a basin boundary. This is a deliberately simplified *prior/algorithmic basin landscape*, not the exact Fourier data-fidelity objective, which is constant on the two target signals but need not be piecewise quadratic on the connecting segment.

For x not zero, a deterministic Euler drift step with \(\eta\kappa=1-a\), \(0<a<1\), is
\[
F(x)=a x+(1-a)b\,{\rm sgn}(x).
\]
The resulting Langevin update is \(X_{j+1}=F(X_j)+\sigma\xi_j\), with independent standard normal \(\xi_j\). For x>0 the deterministic map stays positive, as already proved in the local-well note; for x<0 it stays negative. The stochastic chain can switch sign.

### Proposition 1: an exact positive correction to the affine Gaussian approximation (PROVED)

Start with \(X_0=b\), apply two Langevin updates of innovation scale \(\sigma>0\), and write \(c=b/\sigma>0\). The first state is \(X_1=b+\sigma Z_1\). Define the *affine single-well continuation*
\[
Y_2=b+a\sigma Z_1+\sigma Z_2.
\]
The true two-well second update obeys the **pathwise exact identity**
\[
X_2=Y_2-2(1-a)b\,\mathbf1\{Z_1<-c\}.
\]
In particular, \(X_2\le Y_2\) pointwise, so
\[
\boxed{\Pr(X_2<0)>\Pr(Y_2<0)=
\Phi\!\left(-\frac{c}{\sqrt{1+a^2}}\right)}
\]
for every b>0, sigma>0, 0<a<1, with strict inequality.

**Proof:** When X1>=0 the two drift maps coincide. When X1<0, the two-well drift uses \(-(1-a)b\) where the affine positive-well drift would use \(+(1-a)b\), hence the exact reduction by \(2(1-a)b\). A positive-measure set with Y2 between zero and that reduction, conditional on Z1<-c, shows strictness.

The exact endpoint sign probability is a one-dimensional integral:
\[
\begin{split}
\Pr(X_2<0)&=
\Phi\!\left(-\frac{c}{\sqrt{1+a^2}}\right)+D(a,c),\\
D(a,c)&=\int_{-\infty}^{-c}\phi(z)\left[
\Phi(-(2a-1)c-az)-\Phi(-c-az)
\right]\,dz>0.
\end{split}
\]
The integrand is strictly positive for z<-c because \((2a-1)c<c\). This is a finite-step endpoint calculation, NOT Kramers' continuous-time first-passage asymptotics.

The first two moments can also be written exactly. Set \(p=\Phi(-c)\), \(h=2(1-a)b\), and let \(\phi\) be the standard normal density:
\[
E[X_2]=b-hp,\qquad
\mathrm{Var}(X_2)=\sigma^2(1+a^2)+h^2p(1-p)+2h a\sigma\phi(c).
\]
Thus replacing the nonlinear endpoint law by a **moment-matched Gaussian** does not generally preserve its sign probability. For a=0.2, c=1, the exact sign error is 0.25414235; the affine Gaussian's is 0.16339978; and a Gaussian matching the nonlinear endpoint mean and variance gives 0.27360874. Monte Carlo agrees with the exact integral (CPU receipt below). None of the three is a measured failure rate of a published solver.

### Proposition 2: equal TOTAL innovation variance, opposite qualitative outcomes (PROVED)

The previous comparison keeps per-step noise fixed, so more microsteps inject more total noise. To remove that confound, consider a *hard-well reset* limit. Let \(\Sigma>0\) be the fixed total standard-deviation budget. Starting from positive sign S0=+1, perform J>=2 repeated steps
\[
X_j=b\,{\rm sgn}(X_{j-1})+
     \frac{\Sigma}{\sqrt J}\xi_j,
\quad S_j={\rm sgn}(X_j),\quad X_0=b.
\]
The **sum of the J innovation variances is \(\Sigma^2\)**. The comparator performs one positive-well reset followed by one Gaussian innovation:
\[
\widetilde X=b+\Sigma\xi,\quad
\Pr(\widetilde X>0)=\Phi(r),\quad r=b/\Sigma.
\]
Each inner sign flips with probability
\[
\epsilon_J=\Phi(-r\sqrt J).
\]
The exactly induced symmetric two-state Markov transition matrix is
\[
M_J=\begin{pmatrix}1-\epsilon_J&\epsilon_J\\
\epsilon_J&1-\epsilon_J\end{pmatrix}.
\]
Diagonalizing it gives the exact final positive-basin probability
\[
\boxed{
P_{\rm inner}(S_J=+1)=
\frac{1+[1-2\Phi(-r\sqrt J)]^J}{2}.
}
\]
No first-passage approximation is involved: this is the exact discrete sign process of the stated hard-reset kernel.

**Proof:** At either well center the probability that the next Gaussian innovation crosses zero is epsilon_J. Thus the sign process is a Markov chain with eigenvalues 1 and \(1-2\epsilon_J\), starting from +1. Apply J powers.

**Ordering changes with measurement-scaled noise.** For each fixed J>=2:
- As r decreases to zero, \(P_{\rm inner}=1/2+O(r^J)\), while \(P_{\rm outer}=\Phi(r)=1/2+r/\sqrt{2\pi}+O(r^3)\). Thus \(P_{\rm inner}<P_{\rm outer}\) at sufficiently low SNR.
- As r tends to infinity, the inner failure is at most \(J\Phi(-r\sqrt J)\), which is asymptotically smaller than the outer failure \(\Phi(-r)\). Thus \(P_{\rm inner}>P_{\rm outer}\) at sufficiently high SNR.
By continuity, at least one positive crossover exists. **There is no universal ordering of inner Langevin-style basin updates versus a single outer re-noise even with the SAME sum of innovation variances.** This is a statement only about the hard-reset idealization; the actual solvers do not implement this kernel.

At J=4:
\[
\begin{array}{c|cc}
 r & P_{\rm inner}(S_4=+) & P_{\rm outer}(X>0)\\
 \hline
 0.3&0.52077672&0.61791142\\
 1.0&0.91502404&0.84134475
\end{array}
\]
The crossing reversal is much larger than CPU Monte Carlo uncertainty; simulated 300,000 trajectories in each regime yield 0.51872 and 0.91520, respectively.

## B. Structural connection to three existing solvers (CONDITIONAL)

The native implementation audit showed:
1. DAPS uses inner Langevin correction and Gaussian re-noising;
2. the pinned SITCOM-ODE benchmark ALSO uses stochastic inner SGD/Langevin correction, rather than a purely deterministic SITCOM inner solver;
3. FAST-DIPS pixel uses deterministic inner ADMM correction given its denoiser anchor and Gaussian re-noising at nonterminal steps, with a special terminal denoiser readout.

The two toys prove **possibility**, not a solver ranking. In particular, the sign-reset idealization contains an exact hard mode-selection step; no claim that any native solver performs this exact map. To connect it to DAPS versus SITCOM versus FAST, one must estimate (a) effective basin boundary geometry, (b) denoiser-induced anchor distribution, (c) measurement and prior drift near boundaries, and (d) accumulated innovation variance and schedule, without violating the A8/XP embargo. These quantities are not identified from simple K<=4 class counts.

The sign-reset process starts from a deliberately positive-biased anchor; it does not contradict the earlier G-invariant-initialization theorem. If an exact group-equivariant sampler is started from an invariant mixture of positive/negative anchors, the final distribution remains invariant.

## C. Specific prior art and publication judgment

Two-well Langevin metastability and Kramers escape theory are classical, including Hänggi--Talkner--Borkovec's *Reaction-rate theory: fifty years after Kramers* (Rev. Mod. Phys. 1990, https://doi.org/10.1103/RevModPhys.62.251) and Lelièvre--Le Peutrec--Nectoux (2025, https://doi.org/10.5802/jep.303). These elementary discrete kernels do **not** constitute a newly discovered general theory.

Especially close in our domain, **Kocabas et al., PC-DAPS, arXiv:2610.31592 (22 Oct 2026)** already give an explicit two-atom-prior/quadratic-nonlinear-measurement toy of prior-inconsistent measurement modes and premature mode selection (main text Fig. 2, Appendix A.2); they propose particle clustering and prior consistency pruning. https://arxiv.org/html/2610.31592v1 . They already establish the phenomenon and should be a prominent precedent, not a paper we work around by artificially narrowing our goals. Their goal is a *corrective sampling method*. Track R's more defensible role is an independent, controlled **characterization with three solver families, exact Fourier G orbits, measured failure taxonomy, operational GT-free selection and prospectively preregistered K predictions**. The present mechanics supplement can help interpret such results but is not a standalone novel TH2 theorem.

**Go/no-go:** continue only if a native-update-faithful comparison or a quantitative empirical predictive bridge can be established. Otherwise include this as technical intuition or a supplementary negative/positive mechanistic check, not a headline theorem.

## D. Reproducibility

CPU-only test: scripts/track_r/theory/check_th2_two_well.py, seed 20261026; runs 300,000 hard-reset trajectories for each r and 500,000 two-step paths, plus analytic Gaussian integrals. Outputs:

- J=4,r=0.3: exact 0.52077672, simulated 0.51872, outer 0.61791142;
- J=4,r=1.0: exact 0.91502404, simulated 0.91520, outer 0.84134475;
- a=0.2,c=1: true two-well crossing 0.25414235, simulated 0.254198, affine 0.16339978, moment-matched Gaussian 0.27360874.

The script has been executed on a local CPU independently of PAC and passed. No empirical solver outcome is inferred.
