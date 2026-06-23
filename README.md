# PR Diffusion: reliable diffusion-prior phase retrieval

Updated: 2026-06-23

This repository studies diffusion-prior solvers for phase retrieval, with the current active focus on **FFHQ 256x256 phase retrieval** under oversampled Fourier magnitude measurements.

The central goal is not merely to report a best-of-many benchmark number.  The goal is to develop a **reliable phase-retrieval solver**: high average quality, controlled failure modes, good minimum PSNR, and a method that can be explained as a coherent algorithm rather than as a post-hoc oracle over many unrelated runs.

## Current project objective

The project moved from pure Noise Picking (NP) tuning to an NP-SITCOM reliability investigation.

The current evidence is more nuanced than the original hybrid hypothesis:

```text
SITCOM-ODE has a high successful-reconstruction ceiling, but occasional catastrophic failures.
NP can rescue some SITCOM failures, but current one-shot NP also has severe image-specific failures.
Naive NP-to-SITCOM sigma handoff is technically useful but not competitive as a solver.
A fair solver must respect a fixed compute / candidate budget, not rely on unbounded fallback retries.
```

Therefore the current practical objective is:

```text
Find clean-free, fixed-budget reliability mechanisms for diffusion-prior phase retrieval.
```

## Current active branches

### Branch A: clean-free controller / selector path

Branch A studies whether failed SITCOM trajectories can be detected without ground-truth images and selectively replaced or controlled.

Current state:

- A14 prospectively validated frozen conservative and aggressive controllers on a fresh SITCOM population.
- A16 replicated the frozen A14 policies; the conservative policy was brittle, but the aggressive residual+consensus OR policy replicated strongly.
- A17--A18.8 show useful anytime and population/candidate-set diagnostics, but no new frozen population policy is ready for prospective A19.

The strongest current Branch-A controller is the aggressive residual+consensus OR policy, but it still does not eliminate the catastrophic floor case.

Relevant docs:

```text
docs/progress_report.md
docs/branch_A_clean_free_certificates.md
docs/branch_A_future_controller_directions.md
```

### Branch B: fixed-budget SITCOM population selection

Branch B began as the true-solver handoff experiment: export NP reconstructions as SITCOM/DAPS-compatible sigma states and continue them using a patched SITCOM-ODE runner.

That handoff pipeline works technically, but B3--B8 showed that forcing NP states through SITCOM continuation is not currently competitive.  The useful Branch-B result is instead a fixed-budget SITCOM population selector:

```text
For one measurement:
  run 4 independent SITCOM-ODE trajectories;
  at tau = 0.8, read correction_norm for each run;
  select the run with lowest correction_norm;
  return its final reconstruction.
```

Pooled over B11, B12, and B16-stage1:

| method | n source-image cases | mean selected PSNR | min selected PSNR | bad25 | bad20 |
|---|---:|---:|---:|---:|---:|
| 4S SITCOM, tau0.8 correction selector | 75 | 30.565 | 5.087 | 2 | 1 |

The two failures split into one selector failure (`B11/image 00027`) and one SITCOM population-generation failure (`B12/image 00017`).

Extra 4-to-8 fallback candidates and same-budget 3S+1NP candidate sets are useful diagnostics, but they are not final methods yet:

- 4-to-8 replacement increased compute and degraded already-good selected outputs in B16A.
- 3S+1NP has an oracle-complementary candidate set, but the executable health-to-NP rule was worse than 4S.
- NP rescues `00017`, but fails badly on `00013`, `00028`, `00034`, `00018`, and `00027` under the tested seed/config.

Relevant doc:

```text
docs/branch_B_fixed_budget_population_selector.md
```

## PAC paths and non-repo dependencies

Most active work is run on PAC.  The repo does not contain datasets, external solver checkouts, model checkpoints, or output artifacts.  Important absolute paths are:

```text
Repository:
  /egr/research-pac/huang248/pr_diffusion_repo

FFHQ image root:
  /egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024

Current phase-retrieval output root:
  /egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045

Earlier NP-SITCOM output root:
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
  Use for official SITCOM-ODE trajectory generation and patched SITCOM handoff continuation.
```

This distinction matters.  Some failures are environment or path errors rather than algorithmic errors.

## Main code locations

Current NP-SITCOM scripts:

```text
scripts/npsitcom/
```

Branch A / SITCOM trajectory scripts:

```text
scripts/npsitcom/run_branchA_sitcom_trajectory_hard.py
scripts/npsitcom/run_sitcom_official_ffhq_one_gpu.sh
scripts/npsitcom/make_sitcom_image_folder.py
scripts/npsitcom/parse_sitcom_metrics.py
scripts/npsitcom/mix_select_candidates.py
configs/branch_A/a14/
```

Branch B handoff scripts:

```text
scripts/npsitcom/run_branchB_export_np_handoff_with_measurement_ffhq_one_gpu.sh
scripts/npsitcom/export_np_handoff_states_with_measurement.py
scripts/npsitcom/sitcom_patch/npsitcom_handoff_sample.py
scripts/npsitcom/run_branchB_sitcom_handoff_one_gpu.sh
```

Historical NP/DiffFPR scripts remain in `scripts/`.  Many older `slurm_*`, `phase_retrieval_*`, and one-off analysis scripts correspond to earlier project phases and should be treated as historical unless explicitly reused.

## Documentation map

Start here:

```text
docs/README.md
docs/progress_report.md
docs/current_experiment_plan.md
docs/branch_B_fixed_budget_population_selector.md
docs/branch_A_clean_free_certificates.md
docs/branch_A_future_controller_directions.md
```

Archived older plans, previous progress reports, PAC migration notes, and NeurIPS phased experiment docs live under:

```text
docs/historical/
```
