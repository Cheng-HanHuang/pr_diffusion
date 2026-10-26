# Theory literature digest (planner, 2026-10-26)

**For:** the theory executor, who may not have the PDFs. Every statement under "Verified" was read by the planner in the paper's **main text or the named appendix**. Anything else is labeled.

**Sources read in full on 2026-10-26** (PDFs attached by Cheng):
1. Stroebl, Kapoor, Narayanan, *The Limits of Inference Scaling Through Resampling*, arXiv 2411.17501v3 (26 Mar 2026): §§1–6 and App. C.
2. Dandapanthula & Boffi, *Are we really tilting? The mechanics of reward guidance in flow and diffusion models*, arXiv 2606.02884v1 (1 Jun 2026): §§1–6 and App. C.10.
3. Wu, Trippe, Naesseth, Blei, Cunningham, *Practical and Asymptotically Exact Conditional Sampling in Diffusion Models* (TDS), NeurIPS 2023: §§1–4 and App. A.5.

**Still missing:** Makris, Fearnhead, Nemeth, *Tempered Guided Diffusion* (TGD), arXiv 2605.03712. The file Cheng labeled "TGD" was TDS. What we know of TGD (App. F.4 notes the per-metric oracle; residual-selected DAPS at matched wall-clock) comes from earlier excerpts, not a full read.

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

## 4. Lineage to cite (the executor checks each before citing)

| Lineage | Items |
|---|---|
| Unbiased pass@k | Chen et al. 2021, §2.1 |
| Imperfect-verifier ceilings | Stroebl et al. (above) |
| Mode selection vs guidance | Dandapanthula & Boffi (above) |
| Asymptotic exactness | TDS (above); DAPS time-marginal argument (research report, not planner-verified) |
| Invariance propagation for equivariant diffusion | Hoogeboom et al. 2022 (EDM, molecules); Xu et al. 2022 (GeoDiff). The lemma "an invariant initial law plus an equivariant kernel gives an invariant law" is standard there; our use is for inverse problems with an exactly symmetric forward operator |
| Speciation / early mode commitment | Biroli, Bonnaire, de Bortoli, Mézard 2024 (*Nat. Commun.*; research report, not planner-verified) |
