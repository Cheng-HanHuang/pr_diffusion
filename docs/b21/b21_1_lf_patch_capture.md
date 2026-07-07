# B21.1 LF patch capture

Status: complete.  
Generated from PAC output pasted on 2026-07-08.  
Helper: `scripts/b21/capture_lf_patch.py`.

## Patch identity

```text
Repo root:         /egr/research-pac/huang248/pr_diffusion_b19_solver
DAPS root:         /egr/research-pac/huang248/pr_diffusion_b19_solver/external/daps
Upstream ref:      e7a77d0
Diff paths:        sampler.py, posterior_sample.py
Patch file:        docs/b21/patches/daps_b20_lf_guidance.patch
Clean apply check: pass
```

The captured patch applied cleanly to a clean clone/checkout of upstream ref `e7a77d0` using `git apply --check`.

## Default-off guard

The LF guidance branch is disabled unless:

```python
os.environ.get("B20_LF_ENABLE", "0").strip() == "1"
```

Therefore baseline DAPS behavior is unchanged when `B20_LF_ENABLE` is unset, set to `0`, or set to any value other than the exact stripped string `1`.

## Environment variables

| env var | default when unset | effect |
|---|---:|---|
| `B20_LF_ENABLE` | `0` | Enables LF guidance only when equal to exact stripped string `1`. |
| `B20_LF_ALPHA` | `0.25` | Initial low-frequency magnitude-blend strength. |
| `B20_LF_FRAC` | `0.35` | Fraction of annealing trajectory during which LF guidance is active. |
| `B20_LF_RADIUS_FRAC` | `0.12` | Low-frequency radius threshold in FFT frequency units. |
| `B20_LF_VERBOSE` | `0` | If `1`, prints a warning when LF projection raises an exception and is skipped. |

## Exact intervention formula

Inputs:

- `x0y`: the post-measurement-update sample produced inside DAPS at the current annealing step.
- `measurement`: saved phase-retrieval magnitude measurement tensor.
- `step`: current annealing step index.
- `num_steps`: total annealing steps.
- `alpha0 = float(B20_LF_ALPHA)`, default `0.25`.
- `guide_frac = float(B20_LF_FRAC)`, default `0.35`.
- `radius_frac = float(B20_LF_RADIUS_FRAC)`, default `0.12`.

The wrapper computes:

```text
progress = step / num_steps
```

If `B20_LF_ENABLE != 1`, or `guide_frac <= 0`, or `progress > guide_frac`, the update is identity:

```text
x0y_new = x0y
```

Otherwise:

```text
alpha(step) = alpha0 * max(0, 1 - progress / guide_frac)
```

Let `x = x0y` with shape `(B, C, H, W)`. Let the measurement magnitude grid have shape `(Hm, Wm)`, with `Hm >= H` and `Wm >= W`. The code pads `x` symmetrically/croppably to the measurement grid:

```text
xpad = pad(x, left, right, top, bottom)
```

Then it computes the orthonormal 2D FFT:

```text
X = FFT2(xpad, norm="ortho")
phase = X / max(|X|, 1e-12)
```

The target magnitude is the absolute measurement tensor, broadcast to match `(B, C, Hm, Wm)`:

```text
target_mag = |measurement|
X_target = target_mag * phase
```

The low-frequency mask is defined in FFT frequency coordinates:

```text
fy = fftfreq(Hm)
fx = fftfreq(Wm)
rr = sqrt(fx^2 + fy^2)
M = { rr <= radius_frac }
```

The guided Fourier-domain update is:

```text
X_new[k] = (1 - alpha(step)) * X[k] + alpha(step) * X_target[k],  for k in M
X_new[k] = X[k],                                                   for k not in M
```

Finally:

```text
xnew_pad = real(IFFT2(X_new, norm="ortho"))
x0y_new = crop(xnew_pad, original H x W).astype(x0y.dtype)
```

So LF-v1 is a **measurement-domain low-frequency Fourier-magnitude blend**. It is not a pixel-space low-pass blend. It keeps the current `x0y` Fourier phase, blends only low-frequency magnitudes toward the measured magnitudes, and decays linearly to zero over the first `B20_LF_FRAC` fraction of the annealing trajectory.

## Registry update

`LF_v1` remains `fresh-validated` from B20.11/B20.12A evidence, with its reproducibility spec now anchored by:

```text
docs/b21/patches/daps_b20_lf_guidance.patch
docs/b21/b21_1_lf_patch_capture.md
```

## Planner input for B21.4

The harms observed in B20.11/B20.12A should be interpreted against this exact mechanism: LF-v1 applies an unconditional early Fourier-magnitude blend with linearly decaying alpha. B21.4 variants should change one factor at a time where possible:

1. anneal or gate `alpha` more conservatively;
2. use a phase-consistent/MM-style update rather than direct magnitude blending if specified;
3. gate LF guidance based on clean-free trajectory or measurement-health statistics;
4. preserve default-off env-var behavior for every new external-DAPS patch.
