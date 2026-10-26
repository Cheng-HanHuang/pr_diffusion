# Track R: native solver equivariance and terminal-step audit (2026-10-26)

**Evidence standard:** CODE-VERIFIED means the named public source line at the exact pinned commit was directly inspected; CONDITIONAL means the mathematical consequence additionally assumes equivariance of the denoiser, operators, initial laws, solver state, and floating-point idealization; UNCHECKED means source wrappers and PAC numerical behavior were not inspected. No GT images, PAC files, XP returns, or confirmation305 were used. This is not a proof of posterior correctness for any implementation.

## Shared sufficient conditions (PROVED)

Let `G` be the per-channel 180-degree pixel-rotation group, acting as a permutation of coordinates. Track R's padded Fourier magnitude obeys `A(gx)=A(x)`. With a differentiable scalar objective `F_y(x)=phi(A(x),y)`, `F_y(gx)=F_y(x)`; the chain rule gives `grad F_y(gx)=g grad F_y(x)`. Thus:

- Gradient descent using global scalar step sizes, identical scalar stopping thresholds, and invariant scalar losses is equivariant.
- A gradient through an equivariant denoiser remains equivariant if the full differentiable map is equivariant. Nondifferentiability, nonunique subgradients, and non-equivariant tie-breaking are caveats.
- Coordinatewise Adam and SGD are equivariant under a pixel permutation when their entire optimizer states are permuted, learning rates are shared, and the initialization is compatible.
- Independent isotropic Gaussian innovations have invariant laws. Nonrandom deterministic maps preserve equivariance iff they commute with G.
- An ADMM measurement auxiliary `v=A(x)` and dual `u` live in measurement coordinates, where G acts trivially. Pixel updates must transform by G; measurement projections, if deterministic functions of `A(x)`, `v`, `u` and `y`, commute on the measurement side.
- For the Gaussian KL/TV theorem, it is *not* enough to certify algorithmic equivariance. Each relevant non-equivariant output transition must be represented in a valid positive-variance Gaussian chain, or separately controlled. Final deterministic denoising/clamping and special NaN recovery are not Gaussian transitions.

## A. DAPS native implementation

**Pinned source (CODE-VERIFIED):** [DAPS/sampler.py at e7a77d0 lines 57--100](https://github.com/zhangbingliang2019/DAPS/blob/e7a77d0/sampler.py#L57-L100), [DAPS/cores/mcmc.py lines 123--162](https://github.com/zhangbingliang2019/DAPS/blob/e7a77d0/cores/mcmc.py#L123-L162).

- The outer method alternates deterministic reverse diffusion to a denoised anchor, MCMC correction, and stochastic forward re-noising. The MCMC branch computes a measurement gradient and adds isotropic Langevin-type noise. The no-NaN update is structurally equivariant when the model and gradient are.
- **CODE-VERIFIED caveat:** NaN early stopping returns `torch.zeros_like(x)`. This is a group-equivariant deterministic fallback, but the complete transition is not a globally fixed Gaussian kernel on a branch where this activates; a Gaussian path-KL formula must exclude, bound, or condition on this event.
- **CODE-VERIFIED noteworthy detail:** the outer loop iterates `range(num_steps-1)`, while the test `if step != num_steps-1` is inside it, making that test always true. Consequently the return state, as written at this commit, includes a final re-noise with `sigma_steps[num_steps-1]`, unless its configured scale is zero. Do **not** infer a zero terminal variance merely from the intended schedule. A native wrapper or different commit may alter this.
- **TH1 conclusion:** exact kernel-equivariance is conditional on the network and exact implementation assumptions. A pathwise Gaussian TV bound can be applied to individual Langevin and re-noising microsteps when positive Gaussian scales are verified, but deterministic reverse-diffusion segments and exceptional NaN branches must be incorporated in the correct Markov factorization. No posterior-exactness claim.

## B. SITCOM-ODE native phase-retrieval repository

**Pinned source (CODE-VERIFIED):** [SITCOM_ODE/sampler.py at 275ab67 lines 125--158](https://github.com/sjames40/SITCOM_ODE/blob/275ab67/sampler.py#L125-L158), [lines 253--286](https://github.com/sjames40/SITCOM_ODE/blob/275ab67/sampler.py#L253-L286), [lines 339--378](https://github.com/sjames40/SITCOM_ODE/blob/275ab67/sampler.py#L339-L378).

- **Critical nomenclature:** this public `SITCOM_ODE` adaptation is explicitly based on DAPS, and the audited code uses ODE steps with a data-gradient correction, then SGD/Langevin for each annealing stage. It is not equivalent by inspection to the separate lightweight `prdiffusion/algorithms/sitcom.py` implementation, which uses Tweedie optimization through an inner Adam/SGD loop and resampling. Also do not describe `SITCOM_ODE` as automatically realizing every optimization detail in the original ICML SITCOM paper.
- In `DiffusionSampler._euler`, a score step is followed by a residual-gradient update `x -= scale*norm_grad`. If the score model is equivariant, the measurement norm is G-invariant and the scalar scale is constant, the deterministic ODE correction commutes with G.
- The annealing algorithm uses SGD on a measurement loss plus a quadratic tether, then independent Gaussian Langevin noise. The optimizer state and random draws must transform appropriately.
- A NaN outcome triggers deterministic all-zeros return. This fallback is equivariant but again prevents a blanket globally Gaussian conditional kernel statement.
- Final `xt` is obtained by re-noising `x0y` with the next scheduled scale; verify the last scale before asserting the final output's Gaussian-versus-deterministic status. The frozen experiment is identified by the planner as SITCOM using `sjames40/SITCOM_ODE` commit `275ab67`.
- **TH1 conclusion:** conditional kernel-equivariance applies to the explicitly audited components. The Gaussian-TV theorem must be stated on an augmented/composed chain with the correct per-microstep transitions, not as a theorem inherited from the SITCOM paper.

## C. FAST-DIPS pixel native implementation

**Pinned source (CODE-VERIFIED):** [FAST-DIPS/main.py at a9758b38 lines 34--120](https://github.com/ququlza/FAST-DIPS/blob/a9758b38/main.py#L34-L120), [lines 229--311](https://github.com/ququlza/FAST-DIPS/blob/a9758b38/main.py#L229-L311).

- The scalar proximal correction uses a VJP and either JVP or a finite-difference probe `A(x+eta*g)`; both transform correctly if G is a coordinate permutation, `A(Gx)=A(x)`, the inner search direction transforms, and `eta` is scalar. All dot products/norms and samplewise backtracking tests are invariant. The exact measurement projection `v = y + clip_epsilon(A(x)+u-y)` is an invariant measurement-space operation; it need not choose an orientation.
- The x-update, projection, and ADMM dual step are therefore structurally G-equivariant conditional on an equivariant denoiser anchor and finite precision/tie assumptions.
- **CODE-VERIFIED important terminal issue:** when `is_last_step`, the pixel implementation sets `xt=x0hat`, **not** the computed corrected `x0y`; otherwise it uses `xt=x0y+sigma_next*randn_like(x0y)`. The returned object is `xt.clamp(-1,1)`. Thus the *last* likelihood correction is computed/evaluated but bypassed in the returned pixel output at this exact commit. Do not claim the final corrected pixel state is returned by default. This is a matter for the planner/executor to confirm against the actual pinned wrapper and intended benchmark—not a proposed method alteration.
- Because the terminal path is deterministic denoising plus clamping, the abstract Gaussian-chain TV bound **does not** automatically cover final FAST outputs. An equivariant terminal map would preserve the law-invariance conclusion if the input law were invariant; a non-equivariant denoiser can violate it.

## D. Priority interpretation for the paper

1. State abstract kernel invariance and the Gaussian path-TV inequality as proved under explicit assumptions. Describe their mapping to all three solvers as **conditional and implementation-specific**, not as established posterior exactness.
2. Do not claim the network is the only possible symmetry-breaking source: transformed random states, nonunique choices, special fallbacks, and floating-point path differences can also matter. In the idealized pixel setting with equivariant solver plumbing, the denoiser is the modeled symmetry-breaking source.
3. A real distinction between methods can be studied through *where* the denoiser enters, whether correction is exact/approximate, and whether the final returned state reflects the last measurement correction. Avoid deducing a universal ordering of GOOD/CONSISTENT probabilities from update formulas alone.
4. For predictive modeling, use only the preregistered class counts and residual ranks at K<=4. These source-level findings must not be used to view or revise conclusions after embargoed XP returns.
5. **Action for Cheng/planner, not the theory executor:** confirm that FAST-DIPS's pinned pixel run truly returns the post-denoiser `x0hat` on its last step, and check whether wrappers alter that behavior. Treat as a reproducibility/interpretation clarification, not an experiment request or an unsolicited new method.

**Sources:** Pinned native code above; published solver papers DAPS CVPR 2025, SITCOM ICML 2025, FAST-DIPS ICLR 2026. The auxiliary lightweight local SITCOM implementation was inspected on `theory/prdiff-track-r-th-20261026` at `prdiffusion/algorithms/sitcom.py`; its model is not silently substituted for the native SITCOM_ODE benchmark.
