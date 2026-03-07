# SITCOM implementation review notes (provisional)

Date: 2026-03-07

## Access status

I could inspect `prdiffusion/algorithms/sitcom.py` locally, but external access to:
- GitHub (`https://github.com/Cheng-HanHuang/pr_diffusion`)
- arXiv (`https://arxiv.org/abs/2410.04479`)

was blocked in this execution environment by proxy `403 Forbidden`, so this review is based on code-level checks and canonical SITCOM-style assumptions only.

## Potential mismatches to verify against the paper

1. **Inner optimizer default is Adam, not fixed gradient descent step**
   - Config default is `inner_optim="adam"` with optional SGD.
   - If the paper specifies plain gradient descent with a fixed step `\gamma`, this is a methodological mismatch unless switched to SGD with matching hyperparameters.

2. **Extra early-stopping criterion in the inner loop**
   - `stop_meas_mse` introduces an implementation-specific stopping rule not obviously part of a fixed-`K` algorithm.

3. **Optional low-frequency-only measurement objective**
   - `meas_radius` routes the loss to low-frequency magnitude residual only.
   - If the paper optimizes full-spectrum measurement consistency, this option is an extension.

4. **Resampling uses an explicit tunable `eta_scale`**
   - `eta_scale` allows reducing or amplifying re-noise.
   - If the paper defines a fixed forward-noise level, this knob is an ablation extension.

5. **Initialization uses tunable `init_scale`**
   - `x_t` initialization scales standard Gaussian by `init_scale`.
   - If the paper starts exactly from standard normal, non-1.0 values are off-protocol.

6. **UNet gradient flow is configurable**
   - `backprop_unet=True` by default (paper-faithful per repo comments), but can be disabled.
   - Disabling it would deviate from paper if the paper requires full backprop through denoiser.

7. **Final step computes `x0_final` from `x_t` at last timestep without a final S1 optimization**
   - Outer loop iterates `for i in range(len(timesteps)-1)` then performs one final Tweedie map at `t_last`.
   - Verify whether paper performs an optimization at every timestep including the last index.

## No obvious mismatches from code comments alone

- The closeness regularizer `\lambda ||v_t - x_t||^2` is present.
- The core S1/S2/S3 structure is present and clearly implemented.
- Tweedie mapping formula and resampling equation align with standard DDPM parameterization.
