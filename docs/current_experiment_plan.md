# Current experiment plan: NP-SITCOM Branch A and Branch B

Updated: 2026-06-10

This is the active roadmap after the June 2026 NP-SITCOM two-branch update.  The previous adaptive-reliability NP plan is archived as `docs/historical/current_experiment_plan_archived_20260610_before_npsitcom_two_branch_update.md`.

## Current objective

The project objective is to develop a reliable diffusion-prior phase-retrieval solver for FFHQ-25.  Reliability means not only high average PSNR, but also controlled failure modes: no catastrophic per-image failures, good minimum PSNR, and a method that can be explained as a solver rather than as an offline oracle over many unrelated methods.

The current hypothesis is:

```text
SITCOM-ODE has a stronger successful-reconstruction ceiling.
Noise Picking has more conservative failure behavior and can rescue some SITCOM defects.
A useful hybrid should use NP not merely as a final fallback, but as a reliability/controller mechanism.
```

## Active paths and environments

Use the following path convention on PAC:

```text
Repo:
  /egr/research-pac/huang248/pr_diffusion_repo

FFHQ image root:
  /egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024

Current output root:
  /egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610

Original SITCOM-ODE:
  /egr/research-pac/huang248/external/SITCOM_ODE

Patched SITCOM-ODE for handoff:
  /egr/research-pac/huang248/external/SITCOM_ODE_npsitcom

External DiffFPR utilities:
  /egr/research-pac/huang248/external/DiffFPR

Guided FFHQ checkpoint:
  /egr/research-pac/huang248/models/ffhq_10m.pt
```

Environment convention:

```text
prdiff_ffhq:
  run NP, DiffFPR/guided-model code, CSV parsing/mixing, and NP handoff export.

sitcom_ode_bw:
  run official SITCOM-ODE and patched SITCOM handoff continuation.
```

Do not mix these casually.  Many errors are environment-specific rather than algorithmic.

## Branch A: candidate selection and per-step controller

### Current result

The all-noise Branch A pool contains:

```text
NP:     1000 candidates = 5 noises × 25 images × 2 configs × 4 seeds
SITCOM:  500 candidates = 5 noises × 25 images × 4 runs
Total:  1500 candidates
```

After normalizing image IDs and noise labels, oracle NP+SITCOM selection beats both individual source oracles across all tested noise levels.  This confirms real complementarity.

Approximate corrected oracle summary:

| Noise | NP best mean | SITCOM best mean | Oracle NP+SITCOM mean | Oracle min |
|---:|---:|---:|---:|---:|
| 0 | ~33.83 | ~32.91 | **~35.43** | ~31.54 |
| 0.01 | ~32.38 | ~32.54 | **~33.43** | ~29.83 |
| 0.05 | ~29.43 | ~30.00 | **~30.75** | ~29.38 |
| 0.08 | ~27.88 | ~29.01 | **~29.46** | ~27.73 |
| 0.10 | ~27.19 | ~28.54 | **~28.84** | ~26.57 |

### Current limitation

Branch A is not yet an executable hybrid solver.  Current SITCOM rows lack comparable measurement-side residual features, so the existing non-oracle selectors mostly choose NP and do not fairly evaluate SITCOM candidates.

### Immediate experiments

#### A1. Rerun all-noise mix after normalized-noise patch

```bash
cd /egr/research-pac/huang248/pr_diffusion_repo
export PATH=/egr/research-pac/huang248/conda-envs/prdiff_ffhq/bin:$PATH

git pull --ff-only origin main
python -m py_compile scripts/npsitcom/mix_select_candidates.py

OUT=/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610

NP_CSV=/path/to/np/run_level.csv \
SITCOM_CSV=$OUT/branchA_mix/sitcom_all_noises_run_level.csv \
TAG=branchA_np_sitcom_all_noises_fixed_noiseid \
bash scripts/npsitcom/run_branchA_mix_template.sh
```

Expected structural check:

```text
Each selection method should have 25 selected rows per normalized noise value.
```

#### A2. Compute SITCOM measurement residuals

Goal: make SITCOM rows comparable to NP rows by adding:

```text
noisy_mag_l2
noisy_lowfreq_mag_l2
possibly normalized variants by measurement norm
```

This requires loading SITCOM samples and recomputing the same phase-retrieval measurement residual against the noisy observation.  For strict fairness, use the measurement generated/saved by SITCOM or regenerate deterministically with the same seed/operator.  The preferred future implementation is to patch SITCOM/Branch A parser to save and reuse the exact measurement.

#### A3. Failure-focused analysis

For each noise level and image, compute:

```text
best NP PSNR
best SITCOM PSNR
oracle source winner
source gap
whether SITCOM catastrophically failed
whether NP rescued the image
```

Known important case:

```text
sigma_y=0.05, image 00005:
  NP is normal/good while SITCOM has a catastrophic failure.
```

#### A4. Move from final fallback to per-step detector

The final-output Branch A oracle should motivate a per-step controller, not become the final method.  Instrument SITCOM/DAPS trajectories and record per-step:

```text
sigma
x0hat metric residual
x0y metric residual
low-frequency residual
||x0y - x0hat||
||x0y_t - x0y_{t-1}||
Langevin correction norm
score/update norm
multi-run or branch disagreement
```

Then test whether these features predict final failures before the trajectory collapses.

### Branch A success criteria

A meaningful next Branch A milestone is:

```text
A clean-free selector/controller that detects SITCOM failures early enough to choose or produce a safe NP/SITCOM hybrid state.
```

Metrics:

```text
mean PSNR close to oracle NP+SITCOM
minimum PSNR above NP-only baseline
zero catastrophic failures below 20/25 dB
selector decisions interpretable by residual/risk features
```

## Branch B: sigma handoff and continuation diagnostics

### Current result

The first larger Branch B run completed:

```text
handoff_25img_s100_101_sig20_10_5_2
25 images × 2 seeds × 2 NP configs × 4 sigmas = 400 continuations
```

The pipeline works technically, but current quality is not competitive.  Best behavior appears near `handoff_sigma=2`, and S2-preprojection handoff generally appears better than LF handoff, but the results remain well below standalone NP/SITCOM and Branch A oracle at `sigma_y=0.05`.

### Interpretation

Current Branch B is a negative result for naive one-shot handoff:

```text
x_sigma = x_NP + sigma * eps
```

is not enough to produce a strong SITCOM continuation.  Likely reasons:

1. SITCOM's internal state distribution is not simply noisy final reconstructions.
2. Measurement/operator scaling may still be mismatched.
3. A single handoff is too coarse; per-step correction may be the right granularity.

### Immediate experiments

#### B1. Narrow sigma sweep around the best region

Use fewer images first, focusing on `sigma` near 1--5:

```text
handoff_sigmas = 0,0.25,0.5,1,1.5,2,3,4,5
images = all 25 or focused subset
seeds = 100,101
configs = LF and s2_preproj
```

Purpose: determine whether sigma=2 is a broad optimum or an artifact of coarse grid.

#### B2. Measurement-scaling diagnostic

For each handoff state and final continuation:

```text
measurement residual of x_np
measurement residual of x_sigma denoised/continued output
measurement residual under NP operator vs SITCOM operator
```

If the same image/sample has very different residuals under NP and SITCOM code paths, fix operator scaling before further Branch B sweeps.

#### B3. Hard-image-only handoff

Focus on images where Branch A shows complementarity:

```text
00005 and other SITCOM failure / NP rescue cases
```

Purpose: see whether handoff helps exactly where it should, rather than spending compute on easy images.

#### B4. Per-step handoff variant

If B1/B2 do not improve substantially, move Branch B toward the Branch A controller idea: handoff/intervene inside the SITCOM trajectory rather than only at initialization.

### Branch B success criteria

Branch B is worth promoting only if it can approach or exceed the standalone baselines:

```text
At sigma_y=0.05, mean PSNR should approach 29--30 dB and avoid catastrophic failures.
```

If narrow sweeps remain in the low/mid 20s, Branch B should be treated as evidence against naive one-shot handoff and as motivation for per-step intervention.

## Recommended workstream split

The project should now be split into two focused chats/workstreams:

```text
Branch A chat:
  NP-SITCOM candidate selection and per-step defect detection

Branch B chat:
  NP-SITCOM sigma handoff and continuation diagnostics
```

Use this file plus `docs/progress_report.md` as the handoff context for both chats.
