# Track R literature basis (planner record, 2026-10-21)

**Purpose:** record what prior work already covers, so Track R claims stay inside the gap.

**Verification labels:**
- **[FULL]** read in full text.
- **[CODE]** checked in official code.
- **[MAIN]** main-text passages read.
- **[ABS]** abstract or metadata only.
- **[REPO]** from this repository's own audited records.

Refresh this list before writing any paper text. Work published after October 2026 is not covered.

## 1. Directly overlapping work

### FA-FPR — Xue, Dai, Zheng, Peng, Zou, Xiong, "Fourier Phase Retrieval With Diffusion Priors Guided by Failure Detection," ISCAS 2025, DOI 10.1109/ISCAS56072.2025.11042933 [FULL]

**Engine (DAPS-style; T=200 annealing levels):**
- **Stage 1 (T_f = 150 levels).** Each level runs:
  - a 5-step probability-flow ODE to x̂₀;
  - 100 gradient steps on ½[‖x−x̂₀‖²/σ_t² + ‖y−|F(Px)|‖²/β_y²] with β_y = 0.01;
  - DAPS re-noising.
- **Detection.** The coarse x̂₀ goes to UnivFD (Ojha et al., CVPR 2023: CLIP ViT-L/14 features plus a linear head).
  - If p_fake > γ, Stage 1 restarts from fresh noise, up to N times.
  - If no attempt passes, the attempt with the lowest measurement residual is kept.
- **Stage 2 (last 50 levels).** x_{t+1} = x̂ + σ_{t+1}(λz + (1−λ)(x_t − x̂₀)/σ_t), with λ = 0.25 (noiseless) or 0.3 (noisy). λ = 1 if nothing passed detection.

**Setup:**
- m = n(1+k/4), so oversampling 2 is our 384² canvas.
- σ_y = 0.05; 100 FFHQ images; DPS checkpoint; 1000 NFE per run.
- Baselines: WF, HIO, DPS, DiffFPR, DAPS.
- 4 runs per method; the run best matching y is selected (GT-free); explicitly no GT orientation alignment.
- Means only (PSNR/SSIM/LPIPS). No tails, no wall time; cost reported as "Re-Ratio", the fraction of N attempts used.

**Results (noisy, oversampling 2):**

| Method | PSNR (dB) | Stage-1 attempts |
|---|---|---|
| DAPS (4 runs) | 29.79 | 4 full runs |
| FA-FPR-4-0.3 | 29.71 | Re-Ratio 39.44% (≈1.6) |
| FA-FPR-4-0.1 | 29.98 | Re-Ratio 45.81% |
| FA-FPR-8-0.1 | 30.45 | 61.56% of 8 ≈ 4.9 — more than DAPS uses |

- Clear wins only at oversampling 4 (30.65 vs 29.90) and on noiseless data.
- Planner arithmetic, assuming only Stage 1 repeats: FA-FPR-4-0.3 costs about 1.58 × 750 + 250 ≈ 1,430 NFE, versus 4,000 for DAPS with 4 runs. The paper's "all methods are run four times" sentence makes this slightly ambiguous.
- Fig. 1(a): its "trivial ambiguity" examples are per-channel 180° rotations, i.e. chromatic overlays.

**Consequence for Track R:**
- We cannot claim GT-free failure detection for diffusion phase retrieval as new.
- Open gaps relative to FA-FPR:
  - it detects and restarts, and never repairs;
  - its detector has no reason to flag a clean but globally rotated face (to be tested);
  - no tail metrics;
  - no fast engine;
  - no comparison with SITCOM or FAST.
- FA-FPR is the **direct competitor** on reliability per compute.

### DiffFPR — Li & Wang, ICML 2024 (PMLR 235:28673–28687); code github.com/Chilie/DiffFPR [CODE] [ABS]

- `main_pr.py` defaults:
  - oversampling 4 (pad 128 → 512² canvas), σ = 0.05;
  - 20 runs per image saved;
  - 1000 DDPM steps.
- Each step mixes the RAAR iterate into the DDPM sample with weight λ_t = √ᾱ_t. The prior dominates early (λ ≈ 0.006 at t = 999) and RAAR late. **"Prior first, measurement late" is DiffFPR's design.**
- `align(data, ref)` enumerates all 8 per-channel 180° patterns (2³) and returns the one with maximal grayscale SSIM to `ref`, a reference image — presumably GT.
  - An older commented-out variant aligned channels to channel 0 by edge maps: GT-free, but not prior-based.
  - `align` is not called in `main()`, and the evaluation script is not released.
  - FA-FPR states DiffFPR's reported numbers "consider possible 180 degree rotation ambiguity". FA-FPR's GT-free rerun of DiffFPR gives 16.54 dB (oversampling 2, noisy) and 19.53 dB (oversampling 4, noisy).
- **Consequence:**
  - Orbit enumeration is not new.
  - Scoring the orbit with the diffusion prior, as a deployable GT-free step, appears to be new.

### TReg — Kim, Park, Chung, Ye, "Regularization by Texts for Latent Diffusion Inverse Solvers" (arXiv 2311.15658; ICLR 2025) [MAIN]

- Introduction: DPS "still cannot fully overcome the intrinsic symmetry" in Fourier phase retrieval, because the diffusion prior is based on "image statistics, which alone may be insufficient to break the symmetry". A text prompt (e.g. "photography of face") is shown to break it.
- Evidence is qualitative only (one figure).
- **Consequence:** this concerns DPS *sampling dynamics*, not whether the prior's *density* ranks x above its rotation. It is a risk for weak priors (ImageNet, latent text-to-image), not a refutation of orbit repair. R0/R2 must report orientation accuracy separately for FFHQ and, later, ImageNet.

### FAST-DIPS — Kim, Shin, Lim, ICLR 2026 (arXiv 2603.01591); code github.com/ququlza/FAST-DIPS [FULL Table 1 via ICLR proceedings PDF] [CODE]

**Per noise level:**
- one denoiser call, no UNet backprop;
- ADMM with v projected onto the ε-ball around y (ε = 0.05);
- x-updates by steepest descent with an analytic step from a finite-difference probe, plus backtracking;
- DAPS-style re-noising;
- the last step returns x̂₀.

**Phase-retrieval settings:** T=150, K=2, S=5, ρ=200. About 149 denoiser calls per run, versus about 2,200 for NP.

**Table 1, best of 4 (time per run, s):**

| Dataset | DAPS | SITCOM | FAST |
|---|---|---|---|
| FFHQ | 30.253 dB (122.100) | 28.512 dB (37.425) | 29.253 dB (10.354) |
| ImageNet | 22.354 dB (320.926) | 18.704 dB (99.103) | 19.738 dB (16.629) |

- Best of 4 is justified only as "common practice".
- Our noise norm is about 0.05·√(3·384²) ≈ 33.3, so the ε-ball is effectively the point y. Finite ADMM iterations act as early stopping. The repo's A1.5 audit noticed this; it is not a bug claim.

### SCG — Huang et al., ICML 2024 (arXiv 2402.14285) [MAIN]

- Per reverse step: sample n candidates, compute each x̂₀, keep the argmin of the loss. Gradient-free.
- This is NP's selection rule.

### Constrained Particle Seeking (CPS) — AAAI 2026 (arXiv 2603.01837) [MAIN]

- Reports plain SCG at about 6–7 dB on FFHQ restoration tasks.

## 2. Theory adjacent to T1 (hard per-step selection)

- **Evolution-strategy progress-rate theory** (Rechenberg, Schwefel, Beyer). For a (1,λ)-ES on a locally linear function, progress per generation ≈ σ·c_{1,λ}, where c_{1,λ} = E[max of λ standard normals] (e.g. c_{1,5} ≈ 1.163), normalized in the gradient direction. **The order-statistics lemma behind T1 is classical; T1's novelty can only be the diffusion consequence.**
- **Dandapanthula & Boffi, "Are we really tilting?"** (arXiv 2606.02884, June 2026) [ABS; finer statements from the research report, not re-read]. Finite-particle plug-in reward guidance reward-hacks within a mode and cannot select between modes; best-of-n compensates. Soft-limit contrast to T1.
- **Porteous et al., "Diffusion-Based Posterior Sampling: A Feynman-Kac Analysis of Bias and Stability"** (arXiv 2605.06538) [ABS]. DPS's effective guidance strength scales inversely with step size. Gradient-guidance analogue of T1.
- **Uehara et al., inference-time alignment tutorial** (arXiv 2501.09685) [MAIN]. Covers SVDD, SMC and beam search (beam search = SVDD with α → 0). No step-size scaling analysis.
- **Repo B25** [REPO]. Controlled iid study: hard best-of-K selection fits h^0.5 drift, likelihood weighting fits h^1.

## 3. Failure-mode and evaluation context

- **Burns & Fridovich-Keil** (arXiv 2605.30330) [ABS]. Posterior samplers mis-weight modes and hallucinate prior modes and likelihood modes. An upside-down face is a likelihood mode not supported by an upright face prior.
- **Score-KSD posterior-fidelity evaluation** (arXiv 2602.04189) [ABS + secondary page]. Not phase-retrieval specific.
- **DiME evidence estimation** (arXiv 2602.20549) [ABS]. Prior or model selection, not orbit tie-breaking.
- **DDRM-PR** (arXiv 2501.03030) and **I2I-PR** (arXiv 2507.09609) resolve conjugate inversion **with GT** in evaluation [MAIN per research report].
- **DAPS appendix:** "multiple disjoint modes with exactly the same measurement"; ImageNet has more 180° samples; oversampling sweep at 2.0 / 1.5 / 1.0 gives 30.26 / 29.17 / 24.87 dB [MAIN-excerpt per research report].
- **Hu et al.** (arXiv 2610.17064, October 2026) [ABS + excerpt]. Image priors enable single-shot coded-diffraction phase retrieval; Fourier-only phase retrieval of complex objects fails.

## 4. Large-box inpainting (possible second target)

- **ReGuidance** (Karan, Shah, Chen; NeurIPS 2026; arXiv 2506.10955) [ABS]. Probability-flow-ODE inversion of a baseline output, then deterministic steering. Covers large-box inpainting, heavy super-resolution and high-noise deblurring, with theory on multimodal targets.
- **RPU, hallucination-aware sampling** (arXiv 2606.02331) [ABS + page]. Robust prior update inside DPS; box inpainting with human faithfulness studies.

## 5. What Track R may claim if the data support it

1. Exact-orbit repair by prior ratio: GT-free, solver-agnostic, at the cost of tens to hundreds of UNet forwards.
2. Prior-consistent failure detection compared head-to-head with FA-FPR's UnivFD.
3. Reliability per compute with tail metrics and K-charged cost across DAPS, SITCOM, NP, FAST and FA-FPR.
4. Failure anatomy under GT-free selection, labeled retrospective where it uses exposed data.

## 6. Must not claim

- First GT-free failure detection.
- Orbit enumeration as new.
- "Prior first" scheduling as new.
- Per-step selection as new.
- The ES order-statistics lemma as new.
