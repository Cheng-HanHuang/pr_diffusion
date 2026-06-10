# Progress report: NP-SITCOM two-branch phase-retrieval update

Updated: 2026-06-10

This report records the current project state after the June 2026 NP-SITCOM experiments.  The previous active progress report, centered on May 23 multi-lambda NP selector validation, is archived as `docs/historical/progress_report_archived_20260610_before_npsitcom_two_branch_update.md`.

## Executive summary

The project has moved from pure Noise Picking (NP) tuning toward a hybrid reliability question:

```text
Can we combine NP's conservative failure-avoidance behavior with SITCOM-ODE's higher successful-reconstruction ceiling to obtain a more reliable phase-retrieval solver?
```

Two branches were created:

1. **Branch A: engineering-light NP-SITCOM candidate selection.**
   Run NP and SITCOM separately, standardize their run-level candidates, and study whether a no-ground-truth selector can choose the reliable candidate.  The current result is strongly positive at the oracle/complementarity level, but executable selection is not solved yet because SITCOM candidates still lack comparable measurement-side residual features.

2. **Branch B: true solver hybrid via sigma-space NP-to-SITCOM handoff.**
   Export NP reconstructions as SITCOM/DAPS-compatible sigma states and continue them with a patched SITCOM-ODE runner.  The smoke and 25-image runs execute successfully, but the current naive handoff is not competitive in quality.  This branch is useful diagnostically but is currently lower priority than Branch A/per-step controller design.

The main conceptual update is that a final-output fallback is probably too coarse.  The stronger hypothesis is that NP should act as a **per-step reliability/manifold controller** inside or alongside the SITCOM trajectory: detect when the reconstruction is drifting into a low-training-density or unstable region, then intervene before the defect becomes irreversible.

## Active benchmark and paths

Primary benchmark:

```text
Dataset: FFHQ 25-image split
Resolution: 256
Measurement: oversampled Fourier magnitude / phase retrieval
Default measurement noise: sigma_y = 0.05

PAC repo:
  /egr/research-pac/huang248/pr_diffusion_repo

FFHQ image root:
  /egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024

Current NP-SITCOM output root:
  /egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610

Original external SITCOM-ODE checkout:
  /egr/research-pac/huang248/external/SITCOM_ODE

Patched SITCOM handoff checkout:
  /egr/research-pac/huang248/external/SITCOM_ODE_npsitcom

External DiffFPR / guided diffusion utilities:
  /egr/research-pac/huang248/external/DiffFPR

FFHQ guided diffusion checkpoint:
  /egr/research-pac/huang248/models/ffhq_10m.pt
```

Environment convention:

```text
prdiff_ffhq:
  use for pr_diffusion_repo, NP, DiffFPR/guided-model code, and NP handoff export.

sitcom_ode_bw:
  use for SITCOM_ODE, SITCOM_ODE_npsitcom, official SITCOM baselines, and patched SITCOM continuation.
```

## Background: why we moved beyond pure NP tuning

The May 2026 NP campaign established a practical NP baseline:

```text
score_radius = 0.6
proj_radius  = 0.2
proj_start   = 300
soft_k       = 5
hard_k       = 1
oversample   = 2
```

Multi-lambda LF/S2 selection improved over fixed settings and the selector was often near-oracle over the available NP candidate pool.  However, later validation showed that some seed pairs still had candidate-availability failures: even an oracle over the available NP candidates could fail on recurring hard images such as `00005`, `00013`, or `00028`.

This motivated the current hybrid direction.  SITCOM-ODE often has higher reconstruction quality when it succeeds, while NP has complementary failure behavior and can rescue some SITCOM failures.  The goal is no longer only to tune a best-of-k NP selector; the goal is to understand how to combine the two solvers into a reliable method with controlled failure modes.

## Branch A: NP-SITCOM candidate selection

### Implementation

Branch A lives under:

```text
scripts/npsitcom/
```

Important scripts:

```text
scripts/npsitcom/run_sitcom_official_ffhq_one_gpu.sh
  Runs official SITCOM-ODE on the FFHQ split using Hydra overrides.

scripts/npsitcom/make_sitcom_image_folder.py
  Creates a SITCOM-compatible image folder from the FFHQ split.

scripts/npsitcom/parse_sitcom_metrics.py
  Converts SITCOM metrics.json into run-level candidate CSV format.

scripts/npsitcom/mix_select_candidates.py
  Mixes NP and SITCOM candidate CSVs and computes oracle/executable selection summaries.
```

Current Branch A output folders:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610/sitcom_official
/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610/branchA_mix
```

The mixer now normalizes both image IDs and noise labels before grouping.  This is important because NP image names may look like `images1024x1024/00000/00005.png`, while SITCOM rows may use `00005`, and noise labels may differ as `0.05`, `0.050000`, or similar.

### Current all-noise candidate pool

The all-noise Branch A pool contains:

```text
NP candidates:
  5 noises × 25 images × 2 configs × 4 seeds = 1000

SITCOM candidates:
  5 noises × 25 images × 4 runs = 500

Total:
  1500 candidates
```

### Main results

After normalizing image IDs and noise values, the oracle result shows strong complementarity:

| Noise | NP best mean | SITCOM best mean | Oracle NP+SITCOM mean | Oracle min | Interpretation |
|---:|---:|---:|---:|---:|---|
| 0 | ~33.83 | ~32.91 | **~35.43** | ~31.54 | both sources contribute |
| 0.01 | ~32.38 | ~32.54 | **~33.43** | ~29.83 | both sources contribute |
| 0.05 | ~29.43 | ~30.00 | **~30.75** | ~29.38 | SITCOM usually wins; NP rescues failure |
| 0.08 | ~27.88 | ~29.01 | **~29.46** | ~27.73 | SITCOM usually stronger |
| 0.10 | ~27.19 | ~28.54 | **~28.84** | ~26.57 | SITCOM usually stronger |

At `sigma_y=0.05`, SITCOM wins most images but has at least one catastrophic failure case (`00005`) where NP remains good.  This confirms that the solvers are not redundant: SITCOM provides a higher ceiling, while NP provides conservative reliability in some failure cases.

### Current limitation

The executable selectors are not solved.  Current non-oracle selectors mostly choose NP because SITCOM rows do not yet have comparable measurement-side diagnostics:

```text
selector_post_winner_lf_mse_mean
noisy_lowfreq_mag_l2
noisy_mag_l2
```

These are available or meaningful for NP but are currently missing or `nan` for SITCOM.  Therefore Branch A currently proves **oracle complementarity**, not yet an executable no-ground-truth hybrid selector.

### Scientific interpretation

Branch A is now the most promising direction.  It should not stop at final-output fallback.  The stronger direction is a per-step defect/OOD controller:

```text
During SITCOM/DAPS trajectory:
  monitor measurement consistency, x0hat/x0y disagreement, correction norm, step jump, and branch disagreement;
  if risk becomes high, invoke NP-style correction/resampling/manifold restoration;
  otherwise continue SITCOM normally.
```

This better matches the hypothesis that NP should engage when the trajectory is leaving the high-density training-data region, rather than only after the final output is already bad.

## Branch B: NP-to-SITCOM sigma handoff

### Implementation

Branch B uses a patched copy of SITCOM-ODE rather than editing the upstream external checkout in place:

```text
/egr/research-pac/huang248/external/SITCOM_ODE
  original public SITCOM-ODE checkout

/egr/research-pac/huang248/external/SITCOM_ODE_npsitcom
  patched working copy with npsitcom_handoff_sample.py
```

Generated states/results remain under the output root:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610/branchB_handoff
/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610/branchB_sitcom_handoff
```

Key scripts:

```text
scripts/npsitcom/run_branchB_export_np_handoff_with_measurement_ffhq_one_gpu.sh
  Runs NP and exports handoff states.  Use prdiff_ffhq.

scripts/npsitcom/export_np_handoff_states_with_measurement.py
  Saves x_sigma, x0_np, sigma, image identity, and the exact NP noisy measurement.

scripts/npsitcom/sitcom_patch/npsitcom_handoff_sample.py
  Patched SITCOM runner copied into SITCOM_ODE_npsitcom.  Use sitcom_ode_bw.

scripts/npsitcom/run_branchB_sitcom_handoff_one_gpu.sh
  Wrapper for patched SITCOM continuation when available locally.
```

SITCOM-ODE uses EDM/DAPS-style sigma states rather than DDPM alpha-bar timesteps, so the handoff state is:

```text
x_sigma = x_NP + sigma * eps
```

not a DDPM-style `sqrt(alpha_bar_t) x + sqrt(1-alpha_bar_t) eps` state.

### Completed run

The first larger Branch B run completed structurally:

```text
handoff_25img_s100_101_sig20_10_5_2
25 images × 2 seeds × 2 NP configs × 4 handoff sigmas = 400 continuations
```

The run produced 400 rows, so the patched handoff pipeline is technically working.

### Main result

The current Branch B result is scientifically negative in its naive form.  The best settings are around `handoff_sigma ≈ 2`, and `s2_preproj` is generally better than plain LF handoff, but the average quality remains far below standalone NP, standalone SITCOM, and Branch A oracle selection at `sigma_y=0.05`.

Interpretation:

```text
The code path works, but naive final-NP-state + sigma noise is not a competitive SITCOM initialization.
```

Likely causes:

1. **State-distribution mismatch.**  SITCOM's internal `xt` is produced by repeated reverse-diffusion, measurement-correction, and forward-noising transitions.  Adding Gaussian noise to a final NP image may not match this trajectory distribution.
2. **Measurement/scaling mismatch risk.**  The patched exporter saves the NP measurement, but any remaining operator or scaling mismatch can hurt continuation.
3. **Wrong intervention granularity.**  A single handoff is likely too coarse if NP's real role is per-step trajectory correction.

### Scientific interpretation

Branch B should continue as a diagnostic/secondary branch, not as the main bet.  It is useful for understanding whether NP states can be made compatible with SITCOM dynamics, but current evidence suggests the more promising solver-hybrid idea is per-step intervention rather than one-shot initialization.

## Current priority ranking

```text
Priority 1:
  Branch A as a path to a per-step SITCOM risk detector and NP intervention controller.

Priority 2:
  Branch B diagnostic sweeps around the best handoff regime and measurement/scaling checks.

Priority 3:
  Branch B as a standalone final solver, only if narrow handoff sweeps or state-matching changes improve substantially.
```

## Recommended split into future chats

The two branches now deserve separate focused workstreams:

```text
New chat: NP-SITCOM Branch A — candidate selection and per-step defect detection
  Focus: no-ground-truth risk features, SITCOM trajectory diagnostics, per-step NP controller.

New chat: NP-SITCOM Branch B — sigma handoff and continuation diagnostics
  Focus: handoff quality, sigma sweeps, measurement scaling, state-distribution mismatch.
```

## What should not be forgotten

1. Branch A's oracle result is strong and should be treated as evidence of real complementarity.
2. Branch A's current executable selectors are not valid final methods because SITCOM rows lack comparable residual features.
3. Branch B's technical success is important: the patched SITCOM continuation pipeline runs.  But current quality is not competitive.
4. The most plausible final algorithm is not a final-output fallback.  It is a per-step reliability/manifold controller that uses NP-style intervention when SITCOM trajectories become risky.
