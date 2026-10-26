# Theory literature digest (planner, 2026-10-26)

**For:** the theory executor, who may not have the PDFs. Every statement under "Verified" was read by the planner in the paper's **main text or the named appendix**. Anything else is labeled.

**Sources read in full on 2026-10-26** (PDFs attached by Cheng):
1. Stroebl, Kapoor, Narayanan, *The Limits of Inference Scaling Through Resampling*, arXiv 2411.17501v3 (26 Mar 2026): §§1–6 and App. C.
2. Dandapanthula & Boffi, *Are we really tilting? The mechanics of reward guidance in flow and diffusion models*, arXiv 2606.02884v1 (1 Jun 2026): §§1–6 and App. C.10.
3. Wu, Trippe, Naesseth, Blei, Cunningham, *Practical and Asymptotically Exact Conditional Sampling in Diffusion Models* (TDS), NeurIPS 2023: §§1–4 and App. A.5.

**Added later on 2026-10-26:**
- **§4:** TGD (Makris, Fearnhead, Nemeth, *Tempered Guided Diffusion*, arXiv 2605.03712v1, 5 May 2026), read in full by the planner (main text and Apps. A–G), from the PDF Cheng attached.
- **§5:** the condensed findings of the research pass "Phase-retrieval theory by assumption strength". These are **not** planner-verified unless marked; verify before citing.

---

## 1. Stroebl et al.: resampling with an imperfect verifier

### Verified

- **Abstract.** "Resampling cannot decrease this probability [of false positives], so it imposes an upper bound to the accuracy of resampling-based inference scaling, regardless of compute budget."
- **Weak models cannot catch up.** If a strong model's single-sample accuracy exceeds a weak model's accuracy conditional on passing the verifier, then no amount of resampling lets the weak model match it (§§1, 3).
- **Optimal K is finite and low.** With a cost on false positives, the optimal number of samples is finite and often ≤ 5–10, even at zero compute cost (§4, Fig. 4).
- **Heterogeneity is the cause.** Per-task single-sample accuracy is strongly bimodal. Easy tasks are solved in a few attempts. The remaining hard tasks produce false positives at a higher rate, so the false-positive share of returned answers rises with K (§4).
- **Methodology.** 200 samples per task and model (1,000 for one model). Curves are averaged over 1,000 random permutations of the same sample set (§§3–4).
- **App. C model:**
  - **Generator.** Tasks are easy (T₁) or hard (T₂), with priors p₁, p₂ and per-sample success r₁ > r₂.
  - **Verifier.** Completeness c = P(accept | correct); soundness s = P(reject | incorrect).
  - **Rejection probability** of a sample: βᵢ = (1 − c)rᵢ + s(1 − rᵢ).
  - **Belief after k − 1 rejections:** p_{Tᵢ}^{(k)} = βᵢ^{k−1}pᵢ / (β₁^{k−1}p₁ + β₂^{k−1}p₂).
  - **Per-attempt probabilities:** P_TP^{(k)} = Σᵢ p_{Tᵢ}^{(k)} c rᵢ and P_FP^{(k)} = Σᵢ p_{Tᵢ}^{(k)} (1 − rᵢ)(1 − s).
  - **Value:** EV_k = [V_TP P_TP^{(k)} + V_FP P_FP^{(k)}] · (β₁^{k−1}p₁ + β₂^{k−1}p₂), and Reward(K) = Σ_{k ≤ K} EV_k.
  - **Fit to Llama-3.1-8B on HumanEval+:** r₁ = 0.87, r₂ = 0.13, c = 1, s = 0.75, p₁ = 0.58, p₂ = 0.42 (easy means pass@1 ≥ 0.5).
- **Protocol.** Accept-first: return the first sample that passes.

### Transfer to TH1.2 (planner derivations; the theory executor proves or refutes)

- **(D1) Argmin is accept-first in law.** Under S1 (inconsistent runs have larger ρ than every other run) and S2 (residual ranks exchangeable within {GOOD, ORBIT, CONSISTENT}), argmin-ρ over K i.i.d. runs returns a uniformly random member of the consistent set, whenever that set is non-empty. The first passing sample of an accept-first loop has the same law. So **E0(K) = [1 − (1 − s)^K]·p/s is exactly Stroebl's single-type model**, with completeness 1, "rejection" = INCONSISTENT (probability u = 1 − s), and false positives = CONSISTENT. Their two-type mixture is our population heterogeneity.
- **(D2) When S2 fails, argmin does at least as well.** If ρ ranks GOOD below CONSISTENT more often than chance, argmin beats accept-first. The gain is a function of the within-consistent AUROC a_GC.
- **(D3) Monotone gap.** Per image, with q = p + o and global-only orientation repair (orbit split o = o_g + o_p, with τ_g = 1 and τ_p = 0):
  - O_orbit(K+1) − O_orbit(K) = q(1 − q)^K;
  - E1(K+1) − E1(K) = (p + o_g)(1 − s)^K;
  - so the gap increment is ≥ (p + o_g)[(1 − q)^K − (1 − s)^K] ≥ 0, because q ≤ s.

  The gap is o_p at K = 1, is non-decreasing in K, and tends to 1 − (p + o_g)/s. The same algebra gives O_raw − E0 non-decreasing. Population averages keep the monotonicity. **A bootstrap-significant decrease in the observed gap falsifies S1–S3.**
- **(D4) Bottleneck shift.** As K grows, generation failures (no consistent run) shrink like E[(1 − s)^K], while selection misses grow toward E[c/s]. The model predicts the K at which selection misses overtake generation failures per family: early for SITCOM (c ≫ 0), late or never for FAST (c ≈ 0).
- **(D5) Second verifier.** Stroebl §5 finds false positives are lower quality on an independent axis (code style). Our analog: consistent-wrong outputs may score worse on a prior or realness test. R0's E_P and A4.3 measure this; it is not a TH1 claim.

## 2. Dandapanthula & Boffi: what guidance can and cannot select

### Verified

- **Setting.** Stochastic interpolants. Reward tilt ρ̃₁ ∝ ρ₁e^{λr}. The exact Doob h-transform uses h_t(x) = E[e^{λr(X₁)} | X_t = x] and needs the memoryless schedule σ_t = √(2(1 − t)/t). Practice uses a k-particle plug-in estimate of h_t.
- **Proposition 1** (informal): as the step size → 0, the Euler trajectory converges uniformly in probability to the plug-in flow driven by E[∇log ĥ^{(k)}].
- **Theorem 2.** Gaussian target N(μ, Σ), reward −‖x − a‖², k = 1, memoryless schedule. The plug-in terminal law is N(μ^{(1)}, Σ^{(1)}) with Σ^{(1)} = Σe^{−2λΣ}. Along an eigendirection with x = λσ:
  - variance σe^{−2x} vs the exact tilt σ/(1 + 2x): **exponential over-concentration**;
  - mean-pull eigenvalue √(πx)e^{−x}erfi(√x) vs the exact 2x/(1 + 2x): **overshoot** near x ≈ 0.854.
- **Theorem 3.** W∞(ρ̃₁^{(k)}, ρ̃₁^{(1)}) ≲ √(log k) with a dimension-free constant: more particles help only logarithmically.
- **Proposition 4.** For an isotropic Gaussian, the damping schedule λ_t = λ/(1 + 2λσ²_{1|t}) recovers exact guidance.
- **Theorem 5.** For Gaussian mixtures, the k = 1 plug-in guidance is −2λE[∇X̄₁(x)ᵀ(X̄₁(x) − a)]. The reward enters only through a local posterior sample, so distant high-reward modes contribute negligibly. The exact h-transform instead weights components by πᵢρᵢ_t hᵢ_t, which gives long-range attraction.
- **Theorem 6** (proof in App. C.10). Target ½N(μ, σ²) + ½N(−μ, σ²) with reward −R·1{x < 0}:
  - (i) the exact tilt selects the correct mode with probability (1 + e^{−λR})^{−1};
  - (ii) **any finite-k plug-in flow selects it with probability exactly ½**: the reward gradient is zero almost everywhere, so the guided flow equals the unguided flow, whose drift is odd;
  - (iii) best-of-n by reward gives 1 − 2^{−n}.
- **Experiments.** Best-of-n with a VLM reward performs the mode selection that guidance cannot (FLUX.1, checkerboard). "The importance of the initial seed for mode selection."

### Transfer

- **(D6) Equivariant samplers** (TH1.1, new Corollary 4). Our likelihood is exactly G-invariant, so its gradient is G-equivariant and carries **no** information about which orbit member a trajectory is near. This is the analog of D&B (ii). If the denoiser were G-equivariant, every family's output law would be G-invariant (TH1.1 Cor. 4), and the orbit member would be uniform. **Orientation preference can only come from the learned network's non-equivariance.**
- **(D7) Best-of-K needs a selector that sees the mode.** D&B's (iii) works because their reward distinguishes the modes. Our residual is G-invariant, so residual best-of-K cannot fix orbit errors (TH1.1 Cor. 1). A prior or semantic test (CLIP, the solver's prior) can, and that is E1.
- **(D8) Within-mode over-concentration** (TH2). Plug-in measurement guidance (DPS-type, the 1-particle case) over-concentrates within a mode. Nominal "posterior samplers" therefore behave closer to mode-seekers (T^M) than exact samplers (T^S).
- **(D9) The i.i.d. run model** (S0 of TH1.2) is supported. If guidance cannot move trajectories between well-separated modes, the mode is fixed by the seed early in the trajectory. Runs with independent seeds are then i.i.d. draws of a mode, with probabilities given by basin measures.

## 3. TDS: what "asymptotically exact" means

### Verified

- **Method.** SMC over the reverse chain. Twisting function p̃(y | x_t) = p(y | x̂θ(x_t)); proposals use the score plus ∇log p̃. Weights are w_t = pθ(x_t | x_{t+1}) p̃(y | x_t) / [p̃(y | x_{t+1}) p̃θ(x_t | x_{t+1}, y)].
- **One particle.** TDS reduces to standard gradient guidance (§4: "This choice again recovers TDS with one particle").
- **Theorem 2** (App. A.5). Assume:
  - (a) the final twisting function equals the likelihood;
  - (b) the first twisting function and the ratios p̃(y | x_t)/p̃(y | x_{t+1}) are positive and bounded;
  - (c) each log p̃(y | x_t) is continuous with bounded gradients for t > 0 (the authors call this the strongest assumption);
  - (d) the proposal variances exceed the model variances.

  Then the weighted particle measure converges setwise to **pθ(x₀ | y)** almost surely as the number of particles → ∞ (via Chopin & Papaspiliopoulos, Prop. 11.4).
- **Naive importance sampling** needs a number of particles exponential in KL(pθ(x_{0:T} | y) ‖ pθ(x_{0:T})) (§3.1).

### Transfer

- **(D10) Exact does not mean reliable.** The limit is the **model** posterior. By TH1.1 Prop. 1 it keeps the orbit (mass ratio πθ(gx)/πθ(x)) and gives consistent decoys mass in proportion to prior × likelihood. Even an exact sampler therefore has a per-run failure rate equal to the posterior mass outside GOOD. Residual selection cannot remove decoy mass; only prior information can act, and only within the orbit.
- **(D11) Why practical samplers are far from exact here.** At σ = 0.05 and m = 3·384², the likelihood is extremely concentrated. Conditions (b) and (c) hold only with astronomically large bounds, and the KL in the particle requirement is huge. This is context for the paper (exact SMC is not a practical fix), not a theorem we prove.
- **(D12) S2 as an expectation statement** (TH2 P-b). Suppose decoys are exact preimages, A(ν) = A(x★), each mode is locally Gaussian, and the likelihood dominates the measurement-sensitive directions. Then a posterior sample's residual has the same law in every mode **on average over the measurement noise**, while for a fixed y each mode gets a mode-specific offset. Between GOOD and ORBIT, S2 holds **exactly** for an equivariant sampler, because ρ is G-invariant. Decoys that are only approximately consistent (‖A(ν) − A(x★)‖ ≈ δ) break S2 in a known direction, toward GOOD, with a_GC increasing in δ/σ.

## 4. TGD: annealed SMC over clean-space posteriors

### Verified (main text and appendices)

- **Targets.** π_r(x₀) ∝ p(x₀) p(y | x₀)^{λ_r} with 0 ≤ λ_R ≤ … ≤ λ₀ = 1. Noisy diffusion states are auxiliary variables only.
- **Each stage** reconstructs (DPS-, MPGD- or DAPS-style module), reweights by p(y | x₀)^{λ_{r−1}−λ_r}, optionally resamples, and re-noises (Alg. 1).
- **Theorem 3.1 / B.1** (consistency of ideal TGD). Assume:
  - λ_R = 0, with initial particles i.i.d. from the noisy prior;
  - **exact** stagewise reconstruction kernels;
  - bounded incremental potentials G_r = p(y | x₀)^{λ_{r−1}−λ_r} with π_r(G_r) > 0;
  - consistent resampling.

  Then the weighted particles converge weakly in probability to p(x₀ | y) as N → ∞. The theorem does not cover practical modules or the A-TGD pruning.
- **A-TGD** (Alg. 2). Run N particles for a fraction ρ of the outer stages, then keep i* = argmax_i p(y | x̂₀^{(i)}) (F.6: equivalently the smallest ‖A(x̂₀) − y‖²) and finish that single trajectory.
- **Phase-retrieval configuration** (Table 2, F.9): DAPS module, **λ_r ≡ 1 (no annealing), never resample**, N = 4, ρ = 0.5 (FFHQ) and 0.8 (ImageNet). With λ_r ≡ 1 the weights are constant (App. D), so **A-TGD for PR is "run four DAPS trajectories partway, keep the lowest-residual one, finish it"**: an early-pruned, residual-selected best-of-4.
- **App. D.** With λ_r ≡ 1 and no resampling, TGD reduces exactly to DAPS. N > 1 independent trajectories is "best-of-N baselines, which run several independent trajectories and select the final reconstruction using a data-consistency score".
- **Operator (F.3).** y ~ N(|FP(0.5x₀ + 0.5)|, σ²I), oversampling k = 2 and n = 8 (pad 64), σ = 0.05. **The same as ours.**
- **F.4, verbatim:** "All image inverse-problem generation and evaluation protocols follow DAPS [55], except that DAPS reports oracle best-of-N scores using the trajectory with the best LPIPS, PSNR, or SSIM against the ground truth, whereas we select the trajectory with the lowest measurement error. Since the ground-truth image is unavailable in practical inverse problems, oracle LPIPS/PSNR/SSIM selection is not possible at test time, while measurement-error selection depends only on the observation."
- **F.8.** Hyperparameters were tuned on 5 images disjoint from the 100-image validation set.
- **Matched wall-clock** (Tables 1 and 3). Batched on one H200; time per image = total time / 100. Mean ± s.d. over 100 images:
  - **FFHQ PR, ≈ 4.3 s/img:** A-TGD 29.43 ± 4.07 dB; DAPS N = 1 26.98 ± 8.35; DAPS N = 4 (residual-selected) 27.81 ± 5.12; DPS 12.65 ± 2.42.
  - **ImageNet PR, ≈ 345–357 s/img:** A-TGD 22.18 ± 8.75; DAPS N = 1 19.42 ± 9.31; DAPS N = 4 22.41 ± 9.26; DPS 12.58 ± 2.48.
- **Table 4, FFHQ PR sweep.** At ≈ 17 s/img:
  - **residual-selected DAPS N = 4: 31.40 ± 2.38 dB, LPIPS 0.135**;
  - A-TGD 30.63 ± 4.60, LPIPS 0.152;
  - DAPS N = 1 27.52 ± 8.68, LPIPS 0.225.
- **Table 5** (A-TGD particle ablation, PR, MPGD module): N = 1 gives 22.26 ± 7.61, N = 4 26.38 ± 5.63, N = 16 27.24 ± 4.85. Returns diminish.
- **§6:** pruning "can discard a good trajectory and increase variance, especially in challenging phase retrieval settings".
- **Fig. 9 (planner's visual reading only; the authors do not say this):** several ImageNet PR outputs at 7–9 dB look like orientation errors.

### Transfer

- **(D13) Likelihood weights and residual pruning are G-blind.** SMC weights built from p(y | x₀) are G-invariant, and so is residual-based pruning (TDS, TGD, A-TGD). They cannot correct orientation; only the proposals, i.e. the prior, can. This is TH1.1 Cor. 1′. It also explains why PC-DAPS prunes by *prior* consistency.
- **(D14) A-TGD is selection on an intermediate residual:** a cheaper, noisier verifier. In TH1.2 terms, the orbit class stays verifier-indistinguishable at every ρ.
- **(D15) External support** (not ours to claim):
  - residual-selected best-of-4 is a strong GT-free baseline on FFHQ PR (s.d. 8.68 → 2.38 dB at matched 17 s/img);
  - on ImageNet the s.d. stays ≈ 9 dB after residual selection. This is consistent with frequent G-blind (orientation) failures, which XP-D tests directly.

## 5. Research pass of 2026-10-26: phase-retrieval theory by assumption strength (condensed)

Labels: **[V]** = verified by the planner; **[R]** = from the research report only, so verify before citing.

- **Identifiability gap [R; arithmetic checked by the planner].**
  - Hayes-type generic uniqueness (Hayes 1982; Bruck & Sodin 1979) assumes the support is at most half the DFT grid per axis. Ours is 256 of 384 = 2/3, and the autocorrelation width 511 exceeds 384, so it aliases. **The classical theorem does not cover our operator.**
  - Per channel there are about 384²/2 ≈ 73.7k independent magnitudes (Hermitian symmetry) for 65.5k unknowns, a ratio of ≈ 1.125.
  - The Miao–Sayre–Chapman oversampling heuristic (total ratio 2.25 ≥ 2) is a heuristic, not a theorem.
  - **Consequence for TH1:** state orbit identifiability as an assumption, or work modulo an ambiguity set Amb(x★) ⊇ G·x★. XP-G2 (512 canvas, which satisfies the hypothesis) is the empirical control.
- **Noise scale [V].** The R0 operator uses `fftn(norm="ortho")` (R0 spec line 59). By Parseval, per-channel signal energy is ≈ 65.5k × E[z²] ≈ 1.6×10⁴, against noise energy ≈ 147k × σ² ≈ 3.7×10², with the signal concentrated at low frequencies. So **σ = 0.05 is comparable to or larger than the signal at most high frequencies.** The CONSISTENT class can be noise-sized slack as well as near-ambiguity.
- **Barnett, Epstein, Greengard, Magland 2020** (*Inverse Problems* 36; arXiv 1808.10747) [R]:
  - the local inverse is Lipschitz iff the magnitude torus meets the support subspace transversally; otherwise it is at best Hölder of order α < 1, "often α ≤ 1/2";
  - images convolved with inversion-symmetric kernels inherit non-transversality.
  - **Use:** a deterministic explanation of why consistent-but-wrong images exist at distance ~ε^α. It is an explanation, not a rate; their hypotheses do not match our canvas exactly.
- **Twin stagnation.** Fienup & Wackerman 1986 (*JOSA A* 3:1897; verified, see below) and Guizar-Sicairos & Fienup 2012 (*JOSA A* 29:2367; [R]): iterative algorithms stagnate in simultaneous-twin states. This is the source of XP's exploratory mixed-twin anatomy.
- **Exact 1-D decoys [R].** In 1-D, the non-trivial ambiguities are exactly the zero-flips of the z-transform, up to 2^{N−1} signals with identical magnitudes (Beinert & Plonka 2015; Edidin 2019). These give **exact (δ = 0) measurement-identical decoys** for TH2 (P-b).
- **Ready-made toys:**
  - TGD's 2-D |x| toy [V; App. F.1]: a 5-component GMM prior, τ = 0.005, y = |x₀| + ε with noise 0.01, exact posterior samples via separability;
  - Dandapanthula & Boffi's symmetric GMM [V];
  - Burns & Fridovich-Keil's finite-sample (empirical) prior as an exact-posterior oracle [R].
- **Equivariance propagation:** see the EDM entry below. It is verified, and the lemma is attributed to GeoDiff.
- **Precedents for E0 in coherent diffraction imaging. Must cite:**
  - GHIO (Chen, Miao, Wang, Lee 2007, *PRB* 76:064113; the reference is verified, the rule [R]): from 16 random starts, select the image with the smallest Fourier error as the template;
  - Raines et al. (ankylography; [R]): 50 random starts, best by minimum R_Ewald;
  - Chapman et al. 2006 (*JOSA A* 23:1179; [V]): averaging > 300 trials, and the PRTF.
- **DAPS [V, planner, 2026-10-26 11:30; arXiv v1 HTML].** App. F.2, "More Discussion on Phase Retrieval": "There are multiple disjoint modes with exactly the same measurement for phase retrieval." "We show in Fig. 13 eight images with disparate perceptual features but with exactly the same measurement in phase retrieval." App. F.4, Fig. 18 caption: "The success rate of ImageNet is less than FFHQ and contains more samples with 180 degree rotation." (The research report cited these as App. I.2 / Fig. 18, a later version's numbering.)
  - **So DAPS already observed both the 8-fold orbit and the higher ImageNet rotation rate.** Our contribution is the formal proposition, its corollaries and the measurements, not the observation.
  - **DAPS evaluation [V].** Sec. 4.1: "reporting the best result out of four independent samples" (no criterion stated); DAPS-4K for nonlinear tasks. The code at `e7a77d0` reports the per-metric best over runs (`eval.py` lines 66–84: max for PSNR/SSIM, min for LPIPS, chosen independently). FAST-DIPS at `a9758b38` does the same (`misc.py` lines 665–668).
- **Fienup & Wackerman 1986 [V; author's PDF].** "there is an equal probability that it will reconstruct either of these two objects" (the object and its twin), and "This mode of stagnation often occurs if the support of the object is centrosymmetric." This is the classical, iterative-algorithm counterpart of Cor. 4: a symmetric algorithm picks each twin with equal probability. Cite it.
- **Chapman et al. 2006 [V; arXiv physics/0509066, Sec. 5A].**
  - "All 2D and 3D images displayed in this paper are averages of more than 300 independent phase retrieval trials." Each trial's global phase is aligned first.
  - 2-D solutions from random starts "may differ by a phase vortex"; the authors sorted them into classes and kept the most frequent class (60%).
  - The PRTF is the modulus of the Fourier transform of the averaged image over √(measured intensity), averaged over shells.
- **EDM [V; Sec. 2.2].** Hoogeboom et al. attribute the lemma to **Xu et al. 2022 (GeoDiff)**: "if x∼p(x) is invariant to a group and the transition probabilities of a Markov chain y∼p(y|x) are equivariant, then the marginal distribution of y at any time step is invariant to group transformations as well." Cite GeoDiff for the lemma and EDM for the diffusion application. The research report's wording was a paraphrase.
- **Barnett et al. [partly V; ar5iv §1.1].** Hayes's condition is verified: the support lies in a rectangle with sides at most half of J's, and the solution set "is finite, and generically consists of trivial associates of a single point". So is the count "2(2N)^d trivial associates". The transversality/Hölder theorems remain [R].
- **GHIO [V for the reference only].** Chen, Miao, Wang, Lee, *Phys. Rev. B* 76, 064113 (2007). The selection rule (the smallest-error run as template) remains [R].
- **An over-claim to avoid.** The report calls the large-K orbit share of residual selection "assumption-free". It is not. Residual G-blindness (Cor. 1) is assumption-free. The ceiling also needs the GOOD and ORBIT outputs to have the same within-mode residual law (S2 restricted to {GOOD, ORBIT}), which holds exactly only for equivariant samplers (D12).
- **Lines to use only as contrast [R]:**
  - random-sensing theory: PhaseLift, Wirtinger flow, landscape results, minimax rates;
  - AMP and statistical physics (Barbier et al. 2019; Maillard et al. 2020). Dudeja–Lu–Sen universality needs a random sign diagonal, which is close to a random mask and does not cover a plain oversampled DFT;
  - random-weight generative-prior PR (Hand, Leong, Voroninski 2018).
- **Other [R]:**
  - orbit-recovery / MRA (Bandeira et al.): report errors modulo G (we already report O_orbit and all 8 members);
  - Luby–Sinclair–Zuckerman restarts assume a Las Vegas verifier, which fails here, so they are motivation only.

## 6. Lineage to cite (the executor checks each before citing)

| Lineage | Items |
|---|---|
| Unbiased pass@k | Chen et al. 2021, §2.1 |
| Imperfect-verifier ceilings | Stroebl et al. (above) |
| Mode selection vs guidance | Dandapanthula & Boffi (above) |
| Asymptotic exactness | TDS (above); DAPS time-marginal argument (research report, not planner-verified) |
| Invariance propagation for equivariant diffusion | Hoogeboom et al. 2022 (EDM, molecules); Xu et al. 2022 (GeoDiff). The lemma "an invariant initial law plus an equivariant kernel gives an invariant law" is standard there; our use is for inverse problems with an exactly symmetric forward operator |
| Speciation / early mode commitment | Biroli, Bonnaire, de Bortoli, Mézard 2024 (*Nat. Commun.*; research report, not planner-verified) |
| Residual selection in diffusion PR; the per-metric-oracle remark | TGD (§4, verified): App. F.4 and Tables 1, 3, 4 |
| Residual selection of multi-start runs in CDI | GHIO (Chen et al. 2007); Raines et al. (ankylography); Chapman et al. 2006 (PRTF) (§5, [R]) |
| Twin ambiguity and stagnation | Hayes 1982; Bruck & Sodin 1979; Fienup & Wackerman 1986; Guizar-Sicairos & Fienup 2012; DAPS App. I.2 (§5, [R]) |
| Conditioning of Fourier PR | Barnett, Epstein, Greengard, Magland 2020 (§5, [R]) |
| 1-D exact ambiguities | Beinert & Plonka 2015; Edidin 2019 (§5, [R]) |
