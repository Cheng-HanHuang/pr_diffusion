# Theory program v1.2: selection theory (TH1) and inner-functional theory (TH2)

**Status:** v1.2 (planner, 2026-10-26). Executed by a **separate theory executor** with no PAC access; see `docs/track_r/executors/theory/00_THEORY_EXECUTOR_HANDOFF.md`.

**v1.2 changes (2026-10-26, later the same day).** These follow the full TGD read and the phase-retrieval-theory research pass (digest §§4–5). No theory work had started.
- **TH1.1:**
  - an identifiability caveat: the classical uniqueness theorem does not cover our 384 canvas;
  - new **Cor. 1′**: likelihood-weighted SMC and residual pruning are G-blind;
  - **Cor. 4(b)** is now a theorem target with a proof route: a total-variation bound on orientation preference by the sampler's equivariance defect (KL chain rule + Pinsker), with weak assumptions.
- **TH1.2:** where the CONSISTENT class comes from (non-transversality; noise slack at high frequencies), and the 512-canvas control (XP-G2).
- **TH2:** exact zero-flip decoys (δ = 0) and ready-made exactly solvable toys.
- **TH1** feeds the paper (note due Nov 6; predictions frozen by **Oct 31**). CVPR attempt.
- **TH2** is a separate study (feasibility memo due Nov 13).

**v1.1 changes (2026-10-26).** No theory work had started.
- **Literature.** Full reads of Stroebl et al., Dandapanthula & Boffi and TDS are digested in `docs/planning/track_r/2026-10-26_THEORY_LIT_DIGEST.md` (items D1–D12). Read it first.
- **TH1.1:** new Corollary 4 (equivariant samplers) and Corollary 5 (the exact-sampler limit), with a measurable equivariance defect (XP-C2).
- **TH1.2:** the exact correspondence with Stroebl's model; a **monotone-gap theorem**; the generation-vs-selection bottleneck shift; the orbit split into global and partial.
- **TH1.3:** exact order-statistic U-statistics for argmin- and max-type policies.
- **TH1.5:** the prediction set grows with XP v1.1: FAST to K = 64, SITCOM on all 180 ids to K = 32, DAPS to K = 8. An optional v2 freeze for ImageNet FAST. A registry and an embargo.
- **TH2:** the within-mode over-concentration lens (D8) and S2 as an expectation statement (D12).

**Literature basis.**
- The three research reports of 2026-10-24 ("Prior Art…", "Scoping Extensions…", "Theory of Diffusion-Prior Inverse Solvers…").
- The digest above.
- The planner's full reads: PC-DAPS, the DPS PDF, and the DAPS/FAST-DIPS/PC-DAPS code.
- A research pass on phase-retrieval theory approaches (launched 2026-10-26). The planner forwards the relevant parts when it returns.

**What exists:**
- exactness and bias results under idealizations: DPS Jensen gap; DAPS time-marginal invariance under exact inner sampling; Porteous et al.'s Feynman–Kac bias; Burns & Fridovich-Keil's finite-sample posteriors; PnP-DM/DPnP; TDS (D10–D11);
- local optimization facts: FAST-DIPS Props 1, 5, 6; SITCOM App. D Prop D.1;
- mode-commitment theory: Biroli et al.'s speciation; Raya & Ambrogioni; Dandapanthula & Boffi's Theorems 5–6 (D6–D9);
- best-of-n with imperfect verifiers: pass@k; Stroebl et al. (D1–D5).

**Not found anywhere:**
- a formal exact-orbit treatment for diffusion sampling in inverse problems;
- a probabilistic model of GT-free best-of-K for inverse solvers, tested out of sample;
- a comparison of inner functionals (sample vs mode vs range-MAP) for mode selection.

## Notation (shared)

- **Image and operator.** x ∈ [−1, 1]^{3×256×256}. A(x) = |F_c(P((x+1)/2))| per channel, where P pads centrally into a 384² canvas and F_c = fftshift∘FFT(ortho)∘ifftshift. Measurements y = A(x★) + σn with σ = 0.05; y is signed.
- **Residual.** ρ(x) = ‖A(x) − y‖₂ / √m, with m = 3·384².
- **Orbit group.** G = {R_s : s ∈ {0,1}³}, where R_s rotates channel c by 180° within the 256² frame iff s_c = 1, so |G| = 8. Each R_s is a permutation of pixel coordinates.
- **Run labels:**
  - GOOD: PSNR_raw ≥ 25;
  - ORBIT: PSNR_raw < 25 and max_s PSNR(R_s x) ≥ 25. Global if the best member is 111, partial otherwise;
  - CONSISTENT: a non-orbit failure with ρ ≤ τ_{source,family};
  - INCONSISTENT: the remaining failures.
- **Policies** (K runs):
  - first run;
  - mean-of-K;
  - E0 = argmin ρ;
  - E1 = E0 followed by an orientation test (E1_CLIP: global only);
  - O_raw = GT best raw;
  - O_orbit = GT best over runs and orbit members;
  - per-metric oracle;
  - single-pick oracle.
- **Data available** (committed, path-free):
  - the R0.8 P4 matrix (policy × K ∈ {1, 2, 4, 8} × stratum);
  - R0-N N1a–N1g;
  - the N2 comparison;
  - **A8 per-image strings** (R0-N.2, expected Oct 28–29): counts, run labels, the E0 rank order, the orbit's best member, and the stored CLIP, 2F and S decisions (amendment 2 revision D.1).

## TH1 — selection theory (for the paper)

### TH1.1 Exact orbit

- **(a) Lemma.** For the canvas convention above, A(R_s x) = A(x) for every s.
  - Proof route: the 180° rotation inside the 256² frame equals index negation on the 384² canvas composed with a one-pixel shift (the canvas is even). The shift theorem then gives a pure phase factor.
  - Also record the out-of-domain exact symmetry z ↦ −z, where z = (x+1)/2, and why the prior gives it zero mass.
  - **Precedents.** These are the classical trivial associates (Hayes 1982; Barnett et al. 2020) and the twin image (Fienup & Wackerman 1986). DAPS already shows "eight images with disparate perceptual features but with exactly the same measurement" (App. F.2, Fig. 13 of arXiv v1; verified), and notes more 180° rotations on ImageNet (Fig. 18). Fienup & Wackerman (verified) state that a symmetric iterative algorithm reconstructs the object or its twin "with equal probability": the classical counterpart of Cor. 4. Our contribution is the formal proposition, its corollaries and the measurements, not the observation.
  - **Identifiability caveat (digest §5).** Hayes-type generic uniqueness needs the support to be at most half the grid per axis. Ours is 256/384 = 2/3, so the autocorrelation aliases. **Do not cite Hayes as covering our operator.** Wherever identifiability is needed, either assume it explicitly ("x★ is identifiable up to G") or work modulo an ambiguity set Amb(x★) ⊇ G·x★. The 512-canvas run (XP-G2) is the empirical control.
- **(b) Proposition 1.** Let G act by measure-preserving maps with A∘g = A, let the likelihood have the form ℓ(y, A(x)), and let π be the prior. Then p(gx | y) / p(x | y) = π(gx) / π(x).
- **(c) Region form.** P(gS | y) = ∫_S ℓ(y, A(u)) π(gu) du.
- **Corollary 1.** Any selector depending on (A(x), y) alone is G-invariant, so it cannot resolve orbit errors. This statement is **assumption-free**.
- **Corollary 1′ (new; digest D13).** The same holds for likelihood-weighted particle methods:
  - SMC weights built from p(y | x₀) (TDS, TGD) are G-invariant;
  - residual pruning (A-TGD) is G-invariant.

  So reweighting, resampling and pruning cannot correct orientation; only the proposals (the prior) can. This is why prior-consistency pruning (PC-DAPS) is the mechanism that works. State it for a generic SMC step.
- **Corollary 2.** The Bayes rule within an orbit is argmax_g π(gx). Any proxy with the same within-orbit ordering is Bayes-consistent. CLIP uprightness and the solver's denoising loss are two such proxies; their consistency is an empirical question (N2, XP-C).
- **Corollary 3.** For a proxy log q = log π + h(s(x)) + e with s G-invariant (per-channel power spectrum; pixel histogram; total variation under 180° rotation), h cancels within the orbit.
- **Corollary 4 (equivariant samplers; new).** Model each solver as a chain x_{i−1} = Φ_i(x_i, ξ_i; y), where:
  - x_T and the ξ_i have G-invariant laws (isotropic Gaussians are invariant under pixel permutations);
  - each Φ_i is assembled from the network's denoiser D_i, the map A, y, linear combinations, coordinatewise maps (clipping; Adam-type updates), gradients of functions f(A(·), y), and Euclidean projections onto G-invariant sets.

  **Claim.** If every D_i is G-equivariant (D_i(gx) = gD_i(x)), the output law is G-invariant. Hence the orbit member is uniform given the orbit, and the orbit-error share among outputs landing in G·x★ is 7/8.
  - **To verify per family:** DAPS (Langevin), FAST-DIPS (projection / Gauss–Newton with backtracking), SITCOM (optimization over v), DPS (gradient guidance) and NP. State which steps need care, such as ties in projections or data-dependent step sizes.
  - **Contrapositive** (the paper's statement): **any orientation preference comes only from the learned network's non-equivariance.** Measurement guidance is orientation-blind. This is the inverse-problem analog of Dandapanthula & Boffi's Theorem 6(ii) (digest D6). Cite the invariance-propagation lemma from equivariant generative modeling (Hoogeboom et al. 2022; Xu et al. 2022).
  - **Corollary 4(b): a quantitative version (theorem target; weak assumptions).**
    - **Assumptions.** The chain has stochastic steps x_{i−1} = m_i(x_i) + s_i ξ_i, with ξ_i ~ N(0, I) and state-independent s_i > 0, and x_T has a G-invariant law. Deterministic sub-steps (ODE steps, inner optimizers, projections) are folded into the mean maps m_i. Let Δ_i(x) = m_i(x) − g m_i(g⁻¹x).
    - **Claim.** TV(Law(x₀), g#Law(x₀)) ≤ √( Σ_i E_P‖Δ_i(x_i)‖² / (4 s_i²) ). Hence, for every set B (for example the GOOD neighborhood of x★), |P(x₀ ∈ B) − P(x₀ ∈ gB)| obeys the same bound.
    - **Proof route:**
      - the g-conjugated chain, with means g m_i(g⁻¹·), has output law g#Law(x₀);
      - data processing on path measures;
      - Pinsker;
      - the KL chain rule for Gaussian-transition chains with equal covariances.
    - **Scope.** DAPS (Langevin steps), FAST-DIPS and SITCOM (deterministic inner steps followed by Gaussian re-noising) and DPS (ancestral steps) fit this form. Purely deterministic samplers (s_i = 0) need a coupling/Lipschitz argument instead.
    - **Reduction.** Every component except the network is G-equivariant, so bound Δ_i by Lipschitz-weighted denoiser defects δ_i(x) = ‖D_i(gx) − gD_i(x)‖.
      - For steps that differentiate through D (DPS guidance; SITCOM's optimization over v), the defect of the Jacobian J_D also enters. Bound it separately.
      - DAPS and FAST-DIPS act on x₀ given D(x_t), so there the defect enters only through D(x_t).
    - This is a probabilistic result needing **no** exact score, log-concavity or uniqueness. It turns the observed orientation preference (GOOD rate minus ORBIT_g rate) into a **lower bound** on the network's equivariance defect along the trajectory.
  - **Measurement.** XP-C2 measures δ(t) on noised GT images for both checkpoints, a proxy for E_P‖Δ_i‖² along trajectories. Predictions:
    - the domain with the smaller defect at mid-trajectory times has the higher orbit-error share (ImageNet vs FFHQ);
    - the Cor. 4(b) inequality, evaluated with the proxy, is not violated.
- **Corollary 5 (exact-sampler limit; new).** If a sampler is exact for the model posterior pθ(x₀ | y) (TDS in the infinite-particle limit, digest D10):
  - its orbit-member probabilities are πθ(gx)/Σ_{g'} πθ(g'x) (Prop. 1);
  - its consistent-decoy rate is the posterior mass of decoys;
  - no residual-based selector changes either.

  Exactness therefore bounds what selection can achieve but does not remove failures.

### TH1.2 The selection model

*Setup.* Per image, K i.i.d. runs (S0; digest D9), each in class GOOD, ORBIT-global, ORBIT-partial, CONSISTENT or INCONSISTENT with probabilities (p, o_g, o_p, c, u). Write o = o_g + o_p, q = p + o and s = q + c.

*Assumptions:*
- **S1** (separation): INCONSISTENT runs have a larger ρ than every run of the other classes.
- **S2** (exchangeability): within {GOOD, ORBIT, CONSISTENT}, the residual ranks are exchangeable.
- **S3** (orientation test): it turns ORBIT-global into GOOD with probability τ_g (CLIP: τ_g ≈ 1; partial: τ_p = 0) and turns GOOD into a failure with probability h.

*Results to derive and prove:*

| Quantity | Formula |
|---|---|
| O_raw(K) | 1 − (1 − p)^K |
| O_orbit(K) | 1 − (1 − q)^K |
| E0(K) | [1 − (1 − s)^K] · p/s |
| E1(K) | [1 − (1 − s)^K] · ((1 − h)p + τ_g o_g + τ_p o_p)/s |

*Consequences to prove:*
1. **Correspondence with Stroebl et al. App. C** (digest D1). Under S1–S2, argmin selection equals accept-first in law. The formulas are their single-type model with completeness 1, and the population mixture is their easy/hard mixture. **Credit them explicitly.** Our additions are the orbit classes, the symmetry-aware step, the estimators and the out-of-sample test.
2. **Ceilings** (K → ∞): E1 → ((1 − h)p + τ_g o_g + τ_p o_p)/s. With c > 0, this is the false-positive ceiling.
3. **Monotone-gap theorem** (digest D3). For h = 0, τ_g = 1 and τ_p = 0, per image:
   - O_orbit − E1 equals o_p at K = 1, is non-decreasing in K, and tends to (o_p + c)/s;
   - O_raw − E0 is non-decreasing likewise;
   - both statements survive averaging over images.

   Also give the version for h > 0 (where monotonicity may fail at small K; state the exact condition).
4. **Bottleneck shift** (digest D4). Generation failures E[(1 − s)^K] fall while selection misses rise toward E[(c + o_p)/s · 1(q > 0)]. Derive the crossover K per family from the fitted population.
5. **Population model.** Use heterogeneity over images: a finite mixture of image types (2–4 types, chosen by held-out likelihood), or Dirichlet. Fit by maximum likelihood on per-image class counts, or on the P4 K-curves if A8 is late. Report the fitted mixture and its identifiability.
6. **Relaxations:**
   - imperfect separation, with pairwise violation rate ε;
   - informative residual within the consistent classes, with AUROC a_GC (argmin then beats accept-first; digest D2).
   - **Avoid this over-claim** (digest §5). The research report calls the large-K orbit share of residual selection "assumption-free". It needs S2 restricted to {GOOD, ORBIT}: the same within-mode residual law in both classes. That holds exactly for equivariant samplers (D12) and only approximately otherwise.
7. **Where the CONSISTENT class comes from** (a remark, not a rate):
   - **Non-transversality.** At non-transversal points the local inverse is only Hölder (Barnett et al. 2020), so wrong images can exist at distance ~ε^α with residual ≤ ε. Smooth images are prone to this.
   - **Noise slack.** With the ortho FFT, σ = 0.05 is comparable to or larger than the signal at most high frequencies (digest §5), so images that differ mainly at high frequencies can stay within the noise.
   - **Directional hypothesis for XP-G2** (stated before the data; descriptive, not a frozen prediction): at the 512 canvas, the CONSISTENT share falls for SITCOM.

   Give first-order corrections, and show how to **test S1/S2** from `e0_rank_order` with the labels (A8).
8. **Inflation identity.** O_raw(K) − first run = [E1(K) − first run] + [O_raw(K) − E1(K)].
   - The second term is the only part attributable to GT information.
   - Bound it with (c + o_p)/s and the orbit terms.

### TH1.3 Estimators

- **pass@k form.** Unbiased U-statistics for K ≤ n from per-image counts. O_raw: 1 − C(n − g, K)/C(n, K) (Chen et al. 2021). O_orbit: the same with g + o.
- **Order-statistic identity** (exact, O(n log n)). For an argmin selector, with runs ranked 1…n by ascending score (ties by run index), run j of rank r_j is picked with probability C(n − r_j, K − 1)/C(n, K). For a max-type oracle, the run of ascending rank j is the maximum with probability C(j − 1, K − 1)/C(n, K). **Prove** that these give the exact average over all K-subsets, and that this average is unbiased for the K-run policy value when runs are i.i.d.
- **Model-free comparison.** The empirical selector averaged over all K-subsets is model-free. The gap between the model-free and model-based estimates is the test of S1/S2.
- **Uncertainty:** image-cluster bootstrap. Note the correlation between fit and test when they share images.

### TH1.4 Per-metric oracle inflation

- **Definition.** For metrics (M₁, M₂) per run, Δ_K = E[max_k M₂] − E[M₂ at argmax_k M₁] ≥ 0.
- **Gaussian copula** (correlation ρ, std σ₂): Δ_K = (1 − ρ) σ₂ c_K, where c_K = E[max of K i.i.d. N(0,1)]. Values: c₂ = 0.564, c₄ ≈ 1.029, c₈ ≈ 1.424, c₁₆ ≈ 1.766.
- **Success/failure mixture:** extend the formula, or bound it.
- **Test:** predict Δ_K from within-image correlations of PSNR, SSIM and LPIPS. Check against the A7 data (K ≤ 4), then against XP (K ≤ 64) after the freeze.

### TH1.5 Frozen predictions: the out-of-sample test

- **Fit** on **K ≤ 4 data only** (A8 per-image strings; the P4 aggregates at K ≤ 4 if A8 is late):
  - FAST: S5, 180 images × 4 runs;
  - SITCOM: B22 (100 × 4) and DEV80 (80 × 4);
  - DAPS: B22 (100 × 2, so K ≤ 2) and DEV80 (80 × 4).

  **Do not fit on P4's K = 8 NP column.** It may be reported as a retrospective check, labeled "not out-of-sample".
- **Predict,** with 95% intervals from a parametric bootstrap (state how model and sampling uncertainty are combined):
  - FAST: K ∈ {8, 12, 16, 24, 32, 48, 64};
  - SITCOM: K ∈ {8, 12, 16, 24, 32}, per cohort and pooled;
  - DAPS: K ∈ {4 (B22), 6, 8}, per cohort and pooled;
  - policies E0, E1_CLIP, O_raw, O_orbit; metric Good25;
  - plus the shape statement of TH1.2 item 3.
- **Freeze v1:** commit `docs/track_r/theory/TH1_PREDICTIONS_v1.json` by **Oct 31** (hard), and tell the planner. The planner registers it (`docs/track_r/theory/PREDICTIONS_REGISTRY.md`).
- **Optional v2 freeze (ImageNet FAST).** After the planner forwards the XP-D1 phase-1 counts (runs 0–3, Good20), commit `TH1_PREDICTIONS_v2.json` within 48 h for K ∈ {8, 16, 32, 64}.
- **Embargo for the theory executor.** Read **no XP return** before v1 is registered. Between v1 and v2, read only XP-D1 phase-1 counts.
- **Expected direction** (a hypothesis, not a result):
  - FAST: c ≈ 0, so E1 tracks O_orbit;
  - SITCOM: c ≫ 0, so E1 saturates below O_orbit, and selection misses grow with K;
  - DAPS: in between.

### TH1 deliverables (Nov 6)

- `docs/track_r/theory/TH1_NOTE.tex` (and PDF): statements, full proofs, a one-paragraph lay summary per result.
- `scripts/track_r/theory/` (CPU, numpy/scipy, seeded): fits, estimators, figures.
- `TH1_PREDICTIONS_v1.json` (and v2 if made), frozen as above.
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
  - The exact denoiser of this mixture is **not** g-equivariant when w ≠ ½. This makes Corollary 4's mechanism explicit: the orbit preference comes from w.
- **(P-b) Near-ambiguous decoy.** Add a decoy component ν with ‖A(ν) − A(μ)‖ = δ, at or below the noise level. This models ill-conditioned consistent-wrong modes, cf. Barnett et al.'s Hölder ill-conditioning.
  - **δ = 0 exactly:** in 1-D, take ν as a **zero-flip** of μ's z-transform (Beinert & Plonka 2015; Edidin 2019). It is measurement-identical and distinguishable only by the prior.
  - **δ > 0:** perturb the zero-flip.
- **Ready-made, exactly solvable testbeds** for external validity (digest §§2, 4, 5):
  - TGD's 2-D |x| toy (5-component GMM prior, τ = 0.005, noise 0.01, exact posterior by separability);
  - Dandapanthula & Boffi's symmetric GMM;
  - Burns & Fridovich-Keil's finite-sample prior as an exact-posterior oracle.

**Targets:**
- **(a) T^S exact.** The orbit weight equals 1 − w, the prior ratio. This combines the DAPS exactness argument with Proposition 1.
- **(b) T^S with the Gaussian surrogate or plug-in guidance.** The orbit-weight distortion as a function of r_i and the schedule. Relate it to the speciation time t_S (Biroli et al.: t_S ≈ ½ log d for mean-separated mixtures). Then the within-mode over-concentration of plug-in guidance (digest D8): when does T^S behave like T^M?
- **(c) T^M.** It is deterministic given x_t, so the orbit choice is set at the first decisive level. Give closed forms in 1D/2D, or d → ∞ asymptotics.
- **(d) T^R.** Basins of the nonconvex objective, and the probability of decoy capture as a function of λ and the decoy geometry.
- **(e) S2 as an expectation statement** (digest D12):
  - exact between GOOD and ORBIT for equivariant samplers;
  - for exact-preimage decoys (δ = 0), exchangeable on average over the measurement noise;
  - for δ > 0, a_GC as an explicit increasing function of δ/σ.

  This is the most paper-relevant TH2 target, because it justifies S2 and gives its failure direction.
- **(f) Predictions for real solvers,** to be tested later under a separate GPU pre-registration:
  - later measurement enforcement → fewer orbit errors, more inconsistent failures;
  - stronger measurement weight → more consistent-but-wrong outputs;
  - higher σ → more consistent-but-wrong outputs (XP-G measures this descriptively).

**Memo (Nov 13).**
- What is tractable.
- First numerics on the 1D Fourier toy.
- Candidate theorems with proof sketches.
- Overlap check against PC-DAPS, Burns & Fridovich-Keil, Dandapanthula & Boffi, TGD, *Local MAP Sampling* (arXiv 2510.07343) and arXiv 2606.24516.
- A go/no-go recommendation for a standalone paper.

## Rules for both

- **Claims:** proven, or explicitly labeled CONJECTURE / NUMERICAL. Assumptions are stated at every use.
- **Numerics:** seeded, CPU only, reproducible from committed scripts.
- **Data:** no PAC access, no GT images, no confirmation305. Only the committed aggregates and per-image strings are used, subject to the TH1.5 embargo.
- **Scope:** no new method proposals. Any real-solver prediction goes to the planner, who pre-registers it.
