# Theory program v1 — selection theory (TH1) and inner-functional theory (TH2)

**Status:** v1 (planner, 2026-10-25). Executed by a **separate theory executor** with no PAC access; see `docs/track_r/executors/theory/00_THEORY_EXECUTOR_HANDOFF.md`.
- **TH1** feeds the paper (due Nov 6, CVPR attempt).
- **TH2** is a separate study (feasibility memo due Nov 13).

**Literature basis:** the three research reports of 2026-10-24 ("Prior Art…", "Scoping Extensions…", "Theory of Diffusion-Prior Inverse Solvers…") and the planner's full reads (PC-DAPS, DPS PDF, DAPS/FAST-DIPS/PC-DAPS code). What exists:
- exactness and bias results under idealizations: DPS Jensen gap; DAPS time-marginal invariance under exact inner sampling; Porteous et al.'s Feynman–Kac bias; Burns & Fridovich-Keil's finite-sample posteriors; PnP-DM/DPnP; TDS;
- local optimization facts: FAST-DIPS Props 1, 5, 6; SITCOM App. D Prop D.1;
- mode-commitment theory: Biroli et al.'s speciation; Raya & Ambrogioni;
- best-of-n with imperfect verifiers: pass@k; Stroebl et al.; Dandapanthula & Boffi.

**Not found anywhere:**
- a formal exact-orbit treatment for diffusion sampling;
- a probabilistic model of GT-free best-of-K for inverse solvers;
- a comparison of inner functionals (sample vs mode vs range-MAP) for mode selection.

## Notation (shared)

- **Image and operator.** x ∈ [−1, 1]^{3×256×256}. A(x) = |F_c(P((x+1)/2))| per channel, where P pads centrally into a 384² canvas and F_c = fftshift∘FFT(ortho)∘ifftshift. Measurements y = A(x★) + σn with σ = 0.05; y is signed.
- **Residual.** ρ(x) = ‖A(x) − y‖₂ / √m, with m = 3·384².
- **Orbit group.** G = {R_s : s ∈ {0,1}³}, where R_s rotates channel c by 180° iff s_c = 1, so |G| = 8.
- **Run labels:**
  - GOOD: PSNR_raw ≥ 25;
  - ORBIT: PSNR_raw < 25 and max_s PSNR(R_s x) ≥ 25;
  - CONSISTENT: a non-orbit failure with ρ ≤ τ_{source,family};
  - INCONSISTENT: the remaining failures.
- **Policies** (K runs):
  - first run;
  - mean-of-K;
  - E0 = argmin ρ;
  - E1 = E0 followed by an orientation test;
  - O_raw = GT best raw;
  - O_orbit = GT best over runs and orbit members;
  - per-metric oracle;
  - single-pick oracle.
- **Data available** (committed, path-free):
  - the R0.8 P4 matrix (policy × K ∈ {1, 2, 4, 8} × stratum);
  - R0-N N1a–N1g;
  - the N2 comparison;
  - soon, **A8 per-image label counts** (R0-N.2, expected by Oct 29).

## TH1 — selection theory (for the paper)

**TH1.1 Exact orbit.**
- **(a) Lemma.** For the canvas convention above, A(R_s x) = A(x) for every s.
  - Proof route: the 180° rotation inside the 256² frame equals index negation on the 384² canvas composed with a one-pixel shift (the canvas is even). The shift theorem then gives a pure phase factor.
  - Also record the out-of-domain exact symmetry z ↦ −z, where z = (x+1)/2, and why the prior gives it zero mass.
- **(b) Proposition 1.** Let G act by measure-preserving maps with A∘g = A, let the likelihood have the form ℓ(y, A(x)), and let π be the prior. Then p(gx | y) / p(x | y) = π(gx) / π(x).
- **Corollaries:**
  1. Any selector depending on (A(x), y) alone is G-invariant, so it cannot resolve orbit errors.
  2. The Bayes rule within an orbit is argmax_g π(gx). Any proxy with the same within-orbit ordering is Bayes-consistent. CLIP uprightness and the solver's denoising loss are two such proxies; their consistency is an empirical question (N2, XP-C).
  3. For a proxy log q = log π + h(s(x)) + e with s G-invariant (per-channel power spectrum; pixel histogram; total variation under 180° rotation), h cancels within the orbit.
- **(c) Region form.** P(gS | y) = ∫_S ℓ(y, A(u)) π(gu) du.

**TH1.2 The (p, o, c, u) selection model.**

*Setup.* Per image, K i.i.d. runs, each falling in class GOOD, ORBIT, CONSISTENT or INCONSISTENT with probabilities (p, o, c, u). Write s = p + o + c.

*Assumptions:*
- **S1** (separation): INCONSISTENT runs have a larger ρ than every run of the other classes.
- **S2** (exchangeability): within {GOOD, ORBIT, CONSISTENT}, the residual ranks are exchangeable.
- **S3** (orientation test): it turns ORBIT into GOOD with probability τ and turns GOOD into a failure with probability h.

*Results to derive and prove:*

| Quantity | Formula |
|---|---|
| O_raw(K) | 1 − (1−p)^K |
| O_orbit(K) | 1 − (1−p−o)^K |
| E0(K) | [1 − (1−s)^K] · p/s |
| E1(K) | [1 − (1−s)^K] · ((1−h)p + τo)/s |

*Consequences:*
- **Ceilings** (K → ∞): E1 → ((1−h)p + τo)/s. With c > 0, this is the false-positive ceiling (cf. Stroebl et al.). The asymptotic gap O_orbit − E1 is c/s when τ = 1 and h = 0.
- **Population model:** heterogeneity over images. Use a finite mixture of image types, or Dirichlet. Fit by maximum likelihood on per-image class counts (or on the P4 K-curves if A8 is late).
- **Relaxations.**
  - Imperfect separation: pairwise violation rate ε.
  - Informative residual within consistent classes: AUROC a_{GC}.

  Give first-order corrections, and show how to **test S1/S2** on the per-candidate table (A7).
- **Inflation identity.** O_raw(K) − first run = [E1(K) − first run] + [O_raw(K) − E1(K)].
  - The second term is the only part attributable to GT information.
  - Bound it with c/s and the orbit terms.

**TH1.3 Estimators.**
- **Unbiased U-statistics** for K ≤ n from per-image counts (g, o, c, u) among n runs.
  - O_raw: 1 − C(n−g, K)/C(n, K) (pass@k).
  - O_orbit: the analogous form with g + o.
  - E1 under S1–S3: a hypergeometric sum.
- Prove unbiasedness.
- **Model-free comparison.** The empirical selector averaged over all K-subsets is model-free. The gap between the model-free and model-based estimates is the test of S1/S2.
- **Uncertainty:** image-cluster bootstrap.

**TH1.4 Per-metric oracle inflation.**
- **Definition.** For metrics (M₁, M₂) per run, Δ_K = E[max_k M₂] − E[M₂ at argmax_k M₁] ≥ 0.
- **Gaussian copula** (correlation ρ, std σ₂): Δ_K = (1 − ρ) σ₂ c_K, where c_K = E[max of K i.i.d. N(0,1)]. Values: c₂ = 0.564, c₄ ≈ 1.029, c₈ ≈ 1.424, c₁₆ ≈ 1.766.
- **Success/failure mixture:** extend the formula, or bound it.
- **Test:** predict Δ_K from within-image correlations of PSNR, SSIM and LPIPS, and check against the X1 data (A7).

**TH1.5 Frozen predictions — the out-of-sample test.**
- **Fit:** the TH1.2 population model on **K ≤ 4 data only**:
  - FAST: S5, 180 images × 4 runs;
  - SITCOM: the 60 XP-B ids, 4 runs each.
- **Predict:** E1_CLIP, O_raw and O_orbit at K ∈ {8, 12, 16}, with intervals from a parametric bootstrap.
- **Freeze:** commit `docs/track_r/theory/TH1_PREDICTIONS_v1.json` **before** the XP-A and XP-B returns.
- **Expected direction** (a hypothesis, not a result):
  - FAST: c ≈ 0, so E1 tracks O_orbit;
  - SITCOM: c ≫ 0, so E1 saturates below O_orbit, and selection misses grow with K.

**TH1 deliverables (Nov 6).**
- `docs/track_r/theory/TH1_NOTE.tex` (and PDF): statements, full proofs, a one-paragraph lay summary per result.
- `scripts/track_r/theory/` (CPU, numpy/scipy, seeded): fits, estimators, figures.
- `TH1_PREDICTIONS_v1.json`, frozen as above.
- A half-page and a one-page draft of the paper-ready text.

## TH2 — inner-functional theory (separate study; memo Nov 13)

**Question.** How do the ways the measurement and the prior enter the inner step determine the failure type? The types are orbit error, consistent-but-wrong, and inconsistent.

**Unified update.** x_{t_{i−1}} = T_i(x_{t_i}; y) + σ_{i−1}ξ. The inner target is q_i(x₀) ∝ ℓ(y, A(x₀)) N(x₀; D(x_{t_i}), r_i² I). Three functionals:

| Functional | Definition | Real-solver analogue |
|---|---|---|
| T^S | A sample from q_i: exact, or N Langevin steps | DAPS |
| T^M | The mode of N(x₀; D(x_t), r²I) on the ball {‖A(x₀) − y‖ ≤ ε}, i.e. the projection of D(x_t) onto the measurement set | The FAST-DIPS limit |
| T^R | x₀ = D(v*), with v* = argmin_v ‖A(D(v)) − y‖² + λ‖v − x_t‖² | SITCOM (range-restricted MAP) |

**Toys** (exact denoisers in closed form, so the algorithm is isolated from score error):
- **(P-a) Orbit prior.** w N(μ, s²I) + (1−w) N(gμ, s²I), with an exactly g-invariant A.
  - Instance 1: g = −I with A(x) = |Bx|.
  - Instance 2 (preferred): **1D Fourier phase retrieval**, x ∈ ℝ^d with d = 8–64, A(x) = |DFT(pad x)|, and g = reversal (exactly invariant).
- **(P-b) Near-ambiguous decoy.** Add a decoy component ν with A(ν) ≈ A(μ) within the noise level. This models ill-conditioned consistent-wrong modes, cf. Barnett et al.'s Hölder ill-conditioning.

**Targets:**
- **(a) T^S exact.** The orbit weight equals 1 − w, the prior ratio. This combines the DAPS exactness argument with Proposition 1.
- **(b) T^S with the Gaussian surrogate.** Orbit-weight distortion as a function of r_i and the schedule. Relate it to the speciation time t_S (Biroli et al.: t_S ≈ ½ log d for mean-separated mixtures).
- **(c) T^M.** It is deterministic given x_t, so the orbit choice is set at the first decisive level. Give closed forms in 1D/2D, or d → ∞ asymptotics.
- **(d) T^R.** Basins of the nonconvex objective, and the probability of decoy capture as a function of λ and the decoy geometry.
- **(e) Predictions for real solvers,** to be tested later under a separate GPU pre-registration:
  - later measurement enforcement → fewer orbit errors, more inconsistent failures;
  - stronger measurement weight → more consistent-but-wrong outputs.

**Memo (Nov 13).**
- What is tractable.
- First numerics on the 1D Fourier toy.
- Candidate theorems with proof sketches.
- Overlap check against PC-DAPS, Burns & Fridovich-Keil, *Local MAP Sampling* (arXiv 2510.07343) and arXiv 2606.24516.
- A go/no-go recommendation for a standalone paper.

## Rules for both

- **Claims:** proven, or explicitly labeled CONJECTURE / NUMERICAL. Assumptions are stated at every use.
- **Numerics:** seeded, CPU only, reproducible from committed scripts.
- **Data:** no PAC access, no GT images, no confirmation305. Only the committed aggregates and per-image counts are used.
- **Scope:** no new method proposals. Any real-solver prediction goes to the planner, who pre-registers it.
