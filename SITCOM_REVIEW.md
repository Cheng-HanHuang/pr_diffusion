# SITCOM implementation review notes

Date: 2026-03-07

## Access status

I can review the local checkout at `/workspace/pr_diffusion`, including:
- `prdiffusion/algorithms/sitcom.py`
- the local paper file `SITCOM.pdf`

I **cannot access the remote GitHub repository** from this environment. A direct connectivity check to `https://github.com/Cheng-HanHuang/pr_diffusion` fails with proxy `403 Forbidden` (`CONNECT tunnel failed`).

---

## What I compared

I compared the SITCOM implementation in `prdiffusion/algorithms/sitcom.py` against the algorithm text and descriptions extracted from `SITCOM.pdf` (Algorithm 1 / Sec. 3 discussion), focusing on:
- S1/S2/S3 structure
- Tweedie mapping form
- inner-loop objective and regularization
- stopping criterion semantics
- resampling semantics

---

## Confirmed as paper-consistent

1. **S1/S2/S3 structure is present and in the right order**
   - Code performs: inner optimization over `v` (S1) → Tweedie map to `x0_hat` (S2) → forward resampling to next step (S3).

2. **Tweedie mapping form matches the DDPM form used in SITCOM**
   - `x0 = (v - sqrt(1-alpha_bar_t)*eps_theta(v,t))/sqrt(alpha_bar_t)` is implemented in `tweedie_x0_from_v` and used in SITCOM.

3. **Closeness regularization term is implemented**
   - Paper objective includes a closeness penalty (`lambda ||x_i - v_i||^2` style).
   - Code includes `loss_reg = cfg.lam * mean((v - x_t)^2)`.

4. **Resampling step uses the same structure as Algorithm 1 line 9**
   - Paper: `x_{i-1} = sqrt(alpha_bar_{i-1}) * x0_hat + sqrt(1-alpha_bar_{i-1}) * noise`.
   - Code follows this structure in `resample_x_t`.

---

## Confirmed deviations / ablation knobs (not necessarily bugs)

These are implementation extensions that are reasonable for experimentation but are not strict paper defaults unless configured accordingly.

1. **`meas_radius` low-frequency-only measurement loss**
   - Paper objective is written for the forward operator/data term generally.
   - Code can restrict to low-frequency magnitude residual only (`lowfreq_mag_l2`), which is an extension.

2. **`eta_scale` resampling-noise scale**
   - Allows reducing/amplifying re-noise magnitude.
   - `eta_scale=1.0` is closest to nominal SITCOM behavior.

3. **`init_scale` latent initialization scale**
   - Paper initializes from standard Gaussian.
   - `init_scale != 1.0` is an ablation.

4. **`backprop_unet` toggle**
   - Paper optimization expression differentiates through the denoiser term.
   - `backprop_unet=True` is faithful; disabling it is an explicit deviation.

---

## One important concern that should be corrected (semantic mismatch)

### Stopping criterion units: paper uses an **L2 threshold** (`||...||_2 < δ`), code uses an **MSE threshold**

From Algorithm 1 and Sec. 3 discussion in the paper:
- stopping rule is written as a norm-based condition (`||A(f(v))-y||_2 < δ`),
- and text states `δ` is set relative to measurement noise level (e.g., `δ > σ_y * sqrt(m)`) to avoid strict data fitting / noise overfitting.

Current code:
- checks `loss_meas <= stop_meas_mse`,
- where `loss_meas` is an MSE (`mag_mse`) in the default branch.

This is **not the same quantity** as the paper’s L2 threshold and can miscalibrate stopping by a factor depending on image size/channels.

### Recommended fix

Either:
1. rename current option to make it explicitly MSE-based (already done in name), and document that it is a variant, **or**
2. add a paper-faithful threshold option, e.g. `stop_meas_l2`, and compare against an L2 residual (`torch.norm(...)`) in the same measurement space as the chosen data term.

If strict paper faithfulness is required, option (2) is preferable.

---

## Nuanced note: final-timestep handling

Current code loops to second-to-last scheduler index and then applies a final Tweedie map at the last timestep without a separate S1 optimization there.

This is typically acceptable in practice and often equivalent when the last step corresponds to `t=0`, but Algorithm 1 is written as a full loop over all indices. If exact line-by-line reproduction is desired, you may also run S1 at the final listed timestep.

---

## Bottom line

- Core SITCOM mechanism in `sitcom.py` is correctly implemented.
- Most differences are intentional ablation knobs.
- The **main paper-vs-code semantic mismatch** worth addressing is the stopping criterion quantity (`L2 δ` in paper vs `MSE` threshold in code).
