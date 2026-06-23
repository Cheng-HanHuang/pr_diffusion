# Progress report: NP-SITCOM phase-retrieval reliability update

Updated: 2026-06-23

This report records the current project state after the June 2026 NP-SITCOM experiments.  The previous NP-only line is archived under `docs/historical/`.

## Executive summary

The project objective is to develop a reliable diffusion-prior phase-retrieval solver for FFHQ-25.  Reliability means high average PSNR, good minimum PSNR, controlled failure modes, and a method that can be explained as a solver rather than as an offline oracle over many unrelated runs.

The project now has two active but different conclusions:

```text
Branch A:
  frozen clean-free controllers reduce many SITCOM failures;
  the aggressive residual+consensus OR policy is the best validated controller;
  however, a catastrophic floor remains and the population/controller story is not final.

Branch B:
  naive NP-to-SITCOM handoff is not competitive;
  the defensible result is fixed-budget 4S SITCOM population selection;
  extra-candidate fallback and NP hybrids are diagnostic, not final methods.
```

## Branch A through A18.8

Branch A is now best understood as a clean-free reliability-controller experiment rather than a simple final-output selector.

### A14 and A16: prospective frozen-policy evidence

A14 prospectively validated two predeclared frozen policies on fresh SITCOM trajectories:

| policy | run mean | run min | bad25 | bad20 | replacements |
|---|---:|---:|---:|---:|---:|
| SITCOM only | 27.184 | 5.084 | 20 | 19 | 0 |
| conservative `consensus_lowfreq_nn` | 30.231 | 5.084 | 2 | 2 | 19 |
| aggressive `residual_or_lowfreq_nn` | 30.428 | 5.084 | 1 | 1 | 21 |
| oracle-risk NP fallback | 30.689 | 26.766 | 0 | 0 | 20 |

A16 replayed the same frozen policies on a fresh SITCOM run:

| policy | run mean | run min | bad25 | bad20 | replacements |
|---|---:|---:|---:|---:|---:|
| SITCOM only | 27.236 | 5.084 | 21 | 16 | 0 |
| conservative `consensus_lowfreq_nn` | 29.646 | 5.084 | 7 | 4 | 15 |
| aggressive `residual_or_lowfreq_nn` | 30.337 | 5.084 | 1 | 1 | 24 |
| oracle-risk NP fallback | 30.605 | 25.005 | 0 | 0 | 21 |

Interpretation:

```text
The conservative policy is brittle.
The aggressive residual+consensus OR policy replicated strongly.
Neither policy eliminates the catastrophic floor.
Do not retune the frozen A14/A16 policies using their validation outputs.
```

### A17 through A18.8: diagnostics after the frozen policies

A17 showed broad anytime visibility: under union diagnostics, all `96/96` bad25 and `86/86` bad20 runs become visible before `50%`.  This was diagnostic evidence only.

A17.5 applied strict cross-fit freeze budgets to the strongest anytime candidates.  The selected thresholds collapsed toward do-nothing and no useful budget-feasible frozen rule survived.

A18 through A18.8 studied population / candidate-set control:

- A18 showed that many image groups retain at least one good SITCOM survivor and that some whole-population failures need external fallback.
- A18.5 found a saturation bug in the first population score; crude top-k rules mostly degenerated into run-index tie-breaking.
- A18.6 corrected the score and made `top3_weighted` / `top2_weighted` useful as diagnostics.
- A18.7 found that candidate-set oracle signal is real, but within-set executable selection remains the bottleneck.
- A18.8 regenerated selective population fallback after split-wiring fixes, but the remaining canonical A18.7 misses were still not fixed.

Current Branch-A conclusion:

```text
A14/A16 provide real prospective controller evidence.
A17--A18.8 provide useful diagnostics.
A new prospective A19 population policy is not ready.
The next Branch-A work should improve population-health / fallback certificates using existing trajectories first.
```

Relevant notes:

```text
docs/branch_A_clean_free_certificates.md
docs/branch_A_future_controller_directions.md
```

## Branch B through B18D

Branch B began as an NP-to-SITCOM sigma-handoff line.  B3--B8 showed that the technical pipeline works, but direct continuation of NP states through SITCOM is not competitive as a solver.  The useful Branch-B result became fixed-budget SITCOM population selection.

### Fixed-budget 4S SITCOM selector

Current Branch-B method:

```text
For one measurement:
  run 4 independent SITCOM-ODE trajectories;
  at tau = 0.8, select the run with lowest correction_norm;
  return its final reconstruction.
```

Pooled over B11, B12, and B16-stage1:

| source | n images | mean selected PSNR | min selected PSNR | bad25 | bad20 |
|---|---:|---:|---:|---:|---:|
| B11 | 25 | 30.741 | 23.774 | 1 | 0 |
| B12 | 25 | 29.826 | 5.087 | 1 | 1 |
| B16-stage1 | 25 | 31.127 | 29.548 | 0 | 0 |
| pooled | 75 | 30.565 | 5.087 | 2 | 1 |

Failure anatomy:

| case | selected PSNR | oracle4 PSNR | interpretation |
|---|---:|---:|---|
| B11 / image `00027` | 23.774 | 29.936 | selector failure: a good SITCOM candidate existed |
| B12 / image `00017` | 5.087 | 5.828 | generation failure: all four SITCOM candidates failed |

### Population-health warning

B14 found a useful warning statistic:

```text
tau = 0.8 correction_norm spread across the 4 runs
trigger if spread >= 0.003186
```

In the pooled B11/B12/B16-stage1 audit:

```text
health-triggered cases: 24 / 75 = 32%
triggered selected bad25: 2
triggered oracle4 bad25: 1
```

This catches the two observed selected failures, but it also fires on many good selected outputs.  It is therefore a diagnostic warning flag, not an automatic replacement policy.

### 4-to-8 escalation is diagnostic only

B15 showed that extra SITCOM candidates can rescue failures in an oracle sense.  However, B16 and B16A showed that replacing triggered Stage-1 selections by lower-residual pooled candidates can degrade already-good outputs:

| policy | mean PSNR | min PSNR | bad25 | bad20 | compute |
|---|---:|---:|---:|---:|---:|
| B16 stage1 4S selector | 31.127 | 29.548 | 0 | 0 | 4x SITCOM |
| B16 replace-if-triggered 4-to-8 | 30.659 | 29.100 | 0 | 0 | 5.44x SITCOM |

Every triggered replacement in B16A hurt PSNR.  This reinforces the phase-retrieval point that a lower Fourier-magnitude residual is not a clean certificate of better reconstruction.

### Same-budget NP hybrids are not final yet

B18 tested fixed-candidate-budget NP/SITCOM combinations.  The oracle 3S+1NP candidate set was promising, but executable NP replacement was not.

Pooled B18B:

| policy | mean PSNR | min PSNR | bad25 | bad20 |
|---|---:|---:|---:|---:|
| 4S baseline | 30.565 | 5.087 | 2 | 1 |
| 3S+1NP oracle candidate set | 31.067 | 25.209 | 0 | 0 |
| 3S health-to-NP | 29.179 | 5.827 | 6 | 5 |
| NP only | 26.398 | 10.434 | 15 | 15 |

B18D unique-image complementarity:

```text
25 unique images:
  NP bad25: 5 / 25
  S4 selected any bad25: 2 / 25
  S4 oracle any bad25: 1 / 25
  NP rescues an S4 selected failure: 1 image
  NP hurts all-good S4 selected populations: 4 images
  NP hurts at least one good S4 source: 5 images
```

The only clean NP rescue was `image 00017`, where NP scored `31.286` while the bad S4 population had oracle4 `5.828`.  But NP failed badly on `00013`, `00028`, `00034`, `00018`, and `00027`, where SITCOM usually succeeded.

Current Branch-B conclusion:

```text
Report fixed-budget 4S SITCOM population selection as the defensible Branch-B method.
Treat 4-to-8 escalation, 3S+1NP, and NP fallback as diagnostics until a clean-free certificate distinguishes NP-rescue from NP-failure regimes.
```

Full Branch-B note:

```text
docs/branch_B_fixed_budget_population_selector.md
```

## Current next steps

1. If writing now, present Branch B around the fixed-budget 4S selector and its failure anatomy.
2. Do not claim 4-to-8 or 3S+1NP as final algorithms.
3. For Branch A, do not launch A19 until the population-health / fallback certificate is improved on existing trajectories.
4. For Branch B diagnostics, study why SITCOM fails population-wise on `00017` while NP succeeds, and why NP fails on `00013`, `00028`, `00034`, `00018`, and `00027` while SITCOM succeeds.
5. Look for clean-free prior-consistency or cross-candidate certificates that can distinguish those regimes without using ground-truth PSNR.

## Active paths and environments

```text
Repo:
  /egr/research-pac/huang248/pr_diffusion_repo

FFHQ image root:
  /egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024

Current phase-retrieval output root:
  /egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045

Earlier NP-SITCOM output root:
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
  repo scripts, NP export, CSV analysis, DiffFPR / guided model utilities

sitcom_ode_bw:
  official SITCOM-ODE trajectory generation and patched handoff continuation
```
