# PR Diffusion: reliable diffusion-prior phase retrieval

This repository studies diffusion-prior solvers for phase retrieval, with the current active focus on **FFHQ 256×256 phase retrieval** under oversampled Fourier magnitude measurements.

The central goal is not merely to report a best-of-many benchmark number.  The goal is to develop a **reliable phase-retrieval solver**: high average quality, no catastrophic per-image failures, controlled failure modes, and a method that can be explained as a coherent algorithm rather than as a post-hoc oracle over many unrelated runs.

## Current project objective

The project has moved from pure Noise Picking (NP) tuning to an NP-SITCOM hybrid investigation.

The current hypothesis is:

```text
SITCOM-ODE has a higher successful-reconstruction ceiling.
Noise Picking is more conservative and can rescue some SITCOM failure cases.
A useful hybrid should use NP not merely as a final fallback, but as a reliability/controller mechanism.
```

The key question is:

```text
Can we combine NP's conservative failure-avoidance behavior with SITCOM-ODE's stronger successful reconstructions to obtain a reliable phase-retrieval solver?
```

The current evidence says yes at the oracle/complementarity level, but the executable solver design is still open.

## Why the project moved in this direction

Earlier FFHQ NP work established a practical NP baseline and showed that multi-lambda LF/S2 selection is useful.  The current practical NP setting is:

```text
score_radius = 0.6
proj_radius  = 0.2
proj_start   = 300
soft_k       = 5
hard_k       = 1
oversample   = 2
```

Multi-lambda NP selection improved robustness and often selected near-oracle candidates over the available NP pool.  However, later validation showed that some seed pairs still fail because no good NP candidate is generated for recurring hard images.  This means the bottleneck is not only final selector error; it is also candidate generation and trajectory reliability.

SITCOM-ODE provides a different failure profile.  It often produces stronger reconstructions when it succeeds, but can still have catastrophic failures.  This creates a natural complementarity:

```text
NP:       conservative, stable, lower ceiling in many cases, useful rescue behavior
SITCOM:   stronger successful quality, but occasional severe failures
Hybrid:   aim for SITCOM's ceiling with NP's reliability
```

This led to two active branches.

## Current active branches

### Branch A: NP-SITCOM candidate selection and per-step controller

Branch A is the engineering-light hybrid.  It runs NP and SITCOM separately, standardizes their candidates into a common CSV format, and studies how much improvement is possible by selecting between them.

Current all-noise Branch A candidate pool:

```text
NP candidates:
  5 noises × 25 images × 2 configs × 4 seeds = 1000

SITCOM candidates:
  5 noises × 25 images × 4 runs = 500

Total:
  1500 candidates
```

After normalizing image IDs and noise labels, the oracle NP+SITCOM pool beats both individual source oracles across all tested noise levels.  Approximate corrected summary:

| Noise | NP best mean | SITCOM best mean | Oracle NP+SITCOM mean | Oracle min |
|---:|---:|---:|---:|---:|
| 0 | ~33.83 | ~32.91 | **~35.43** | ~31.54 |
| 0.01 | ~32.38 | ~32.54 | **~33.43** | ~29.83 |
| 0.05 | ~29.43 | ~30.00 | **~30.75** | ~29.38 |
| 0.08 | ~27.88 | ~29.01 | **~29.46** | ~27.73 |
| 0.10 | ~27.19 | ~28.54 | **~28.84** | ~26.57 |

This is strong evidence that NP and SITCOM are complementary.  However, current executable selectors are not solved because SITCOM candidates do not yet have comparable measurement-side residual/risk features.  Therefore Branch A is currently best understood as evidence for a more ambitious direction: a **per-step risk detector/controller**.

The intended long-term Branch A algorithm is not simply:

```text
run NP;
run SITCOM;
choose final output.
```

The stronger direction is:

```text
During the SITCOM/DAPS trajectory:
  monitor risk / out-of-distribution / defect signals;
  if the trajectory becomes risky, invoke NP-style correction or resampling;
  otherwise continue the high-quality SITCOM update.
```

### Branch B: NP-to-SITCOM sigma handoff

Branch B is the true-solver handoff experiment.  It exports NP reconstructions as SITCOM/DAPS-compatible sigma states and continues them using a patched SITCOM-ODE runner.

SITCOM-ODE uses EDM/DAPS-style sigma states, so the handoff state is:

```text
x_sigma = x_NP + sigma * eps
```

not a DDPM alpha-bar timestep state.

The first larger Branch B run completed successfully:

```text
handoff_25img_s100_101_sig20_10_5_2
25 images × 2 seeds × 2 NP configs × 4 sigmas = 400 continuations
```

The pipeline works technically, but the current naive one-shot handoff is not competitive in quality.  The best current behavior is near `handoff_sigma ≈ 2`, with S2-preprojection handoff generally better than LF handoff, but the results remain below standalone NP/SITCOM and far below Branch A oracle selection.

Current interpretation:

```text
Branch B is useful diagnostically, but naive final-NP-state + sigma noise is not enough.
The mismatch may be state-distribution mismatch, measurement/operator scaling, or the fact that one-shot handoff is the wrong granularity.
```

Branch B should continue with narrow diagnostics, but the highest-priority solver idea is now the Branch A/per-step controller direction.

## PAC paths and non-repo dependencies

Most active work is run on PAC.  The repo does not contain datasets, external solver checkouts, or output artifacts.  Important absolute paths are:

```text
Repository:
  /egr/research-pac/huang248/pr_diffusion_repo

FFHQ image root:
  /egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024

Current NP-SITCOM output root:
  /egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610

Original external SITCOM-ODE checkout:
  /egr/research-pac/huang248/external/SITCOM_ODE

Patched SITCOM checkout for Branch B:
  /egr/research-pac/huang248/external/SITCOM_ODE_npsitcom

External DiffFPR utilities:
  /egr/research-pac/huang248/external/DiffFPR

FFHQ guided diffusion checkpoint:
  /egr/research-pac/huang248/models/ffhq_10m.pt
```

Environment convention:

```text
prdiff_ffhq:
  Use for this repo, NP, DiffFPR/guided model code, CSV parsing/mixing, and NP handoff export.

sitcom_ode_bw:
  Use for official SITCOM-ODE and patched SITCOM handoff continuation.
```

This distinction matters.  Some failures are environment errors rather than algorithmic errors.

## Main code locations

Current NP-SITCOM scripts:

```text
scripts/npsitcom/
```

Branch A scripts:

```text
scripts/npsitcom/run_sitcom_official_ffhq_one_gpu.sh
scripts/npsitcom/make_sitcom_image_folder.py
scripts/npsitcom/parse_sitcom_metrics.py
scripts/npsitcom/mix_select_candidates.py
```

Branch B scripts:

```text
scripts/npsitcom/run_branchB_export_np_handoff_with_measurement_ffhq_one_gpu.sh
scripts/npsitcom/export_np_handoff_states_with_measurement.py
scripts/npsitcom/sitcom_patch/npsitcom_handoff_sample.py
scripts/npsitcom/run_branchB_sitcom_handoff_one_gpu.sh
```

Historical NP/DiffFPR scripts remain in `scripts/`.  Many older `slurm_*`, `phase_retrieval_*`, and one-off analysis scripts correspond to earlier project phases and should be treated as historical unless explicitly reused.

## Current output organization

Current output root:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610
```

Important subfolders:

```text
sitcom_official/
  Official SITCOM-ODE baseline runs.

branchA_mix/
  NP/SITCOM standardized candidate CSVs and selection summaries.

branchB_handoff/
  NP-exported sigma handoff states and manifests.

branchB_sitcom_handoff/
  Patched SITCOM continuation outputs from NP handoff states.

sitcom_images_ffhq25/
  SITCOM-compatible symlinked FFHQ-25 image folder.

logs/
  nohup logs for current experiments.
```

## Active docs

Start here:

```text
docs/README.md
  Navigation page for active docs.

docs/progress_report.md
  Detailed June 10 NP-SITCOM progress report and current interpretation.

docs/current_experiment_plan.md
  Recommended next experiments for Branch A and Branch B.

docs/historical/
  Archived older plans, previous progress reports, PAC migration notes, and earlier NP-only context.
```

## Evaluation convention

Always specify whether a result is:

```text
all-run mean over every image/seed;
image-level best-of-k;
source oracle;
NP+SITCOM oracle;
raw alignment;
rot180 alignment;
resolve alignment;
noise level;
number of seeds/configs/runs;
whether measurement residuals are comparable across sources.
```

Do not treat a method-level oracle as an executable algorithm.  Branch A oracle results are currently evidence of complementarity, not yet a clean-free selector.

## Current priority ranking

```text
Priority 1:
  Branch A as a path to per-step SITCOM risk detection and NP intervention.

Priority 2:
  Branch B narrow diagnostics around the best handoff sigma and measurement/operator scaling.

Priority 3:
  Branch B as a standalone solver, only if narrow handoff diagnostics improve substantially.
```

Recommended future chat split:

```text
NP-SITCOM Branch A: candidate selection and per-step defect detection
NP-SITCOM Branch B: sigma handoff and continuation diagnostics
```

## Repo hygiene notes

The repository contains historical backup scripts and old experiment plans.  Before final release or paper artifact packaging, review and remove accidental backup files such as:

```text
*.bak*
*_patched.py
```

unless they are intentionally archived.
