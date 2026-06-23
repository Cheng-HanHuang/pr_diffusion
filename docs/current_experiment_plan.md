# Current experiment plan: NP-SITCOM Branch A and Branch B

Updated: 2026-06-23

This plan records the current active direction after Branch A through A18.8 and Branch B through B18D.  The main change is that Branch B should now be treated as a fixed-budget SITCOM population-selection result plus diagnostics, not as an open-ended fallback or NP-to-SITCOM handoff method.

## Current objective

Develop a reliable diffusion-prior phase-retrieval solver for FFHQ-25.  Reliability means:

```text
high average PSNR;
good minimum PSNR;
controlled bad25 / bad20 counts;
fixed, reportable compute budget;
clean-free selection or control rules;
no post-hoc oracle over unbounded candidate pools.
```

## Branch A status

Branch A is a clean-free controller / selector line.

Current state:

- A14 prospectively validated predeclared frozen conservative and aggressive policies on fresh SITCOM trajectories.
- A16 replicated the same frozen policies on another fresh SITCOM population.  The conservative consensus-only policy weakened, but the aggressive residual+consensus OR policy replicated strongly.
- A17 showed broad anytime visibility under union diagnostics, but this was not an executable rule.
- A17.5 showed that strict cross-fit anytime thresholds collapsed toward do-nothing; no stable budget-feasible anytime rule survived.
- A18--A18.8 made the population / candidate-set direction look real, but the frozen population story is not yet ready for prospective A19.

Best validated Branch-A controller:

```text
aggressive residual_or_lowfreq_nn
= first80 residual-rank arm OR low-frequency nearest-neighbor consensus arm
```

A14 result:

| policy | run mean | run min | bad25 | bad20 | replacements |
|---|---:|---:|---:|---:|---:|
| SITCOM only | 27.184 | 5.084 | 20 | 19 | 0 |
| conservative consensus_lowfreq_nn | 30.231 | 5.084 | 2 | 2 | 19 |
| aggressive residual_or_lowfreq_nn | 30.428 | 5.084 | 1 | 1 | 21 |

A16 result:

| policy | run mean | run min | bad25 | bad20 | replacements |
|---|---:|---:|---:|---:|---:|
| SITCOM only | 27.236 | 5.084 | 21 | 16 | 0 |
| conservative consensus_lowfreq_nn | 29.646 | 5.084 | 7 | 4 | 15 |
| aggressive residual_or_lowfreq_nn | 30.337 | 5.084 | 1 | 1 | 24 |

Branch-A recommendation:

```text
Do not launch A19 yet.
Do not retune the frozen A14/A16 policies based on validation outputs.
If Branch A continues, use existing trajectories to design better population-health / fallback certificates first.
```

Relevant docs:

```text
docs/progress_report.md
docs/branch_A_clean_free_certificates.md
docs/branch_A_future_controller_directions.md
```

## Branch B status

Branch B began as NP-to-SITCOM sigma handoff, but B3--B8 showed that direct NP-state continuation is not a competitive solver.  The useful Branch-B result is now fixed-budget SITCOM population selection.

Current fixed-budget Branch-B method:

```text
4S SITCOM population selector:
  run 4 independent SITCOM-ODE trajectories;
  at tau = 0.8 choose the run with lowest correction_norm;
  return that final reconstruction.
```

Pooled fixed-budget audit over B11, B12, and B16-stage1:

| source | n images | mean selected PSNR | min selected PSNR | bad25 | bad20 |
|---|---:|---:|---:|---:|---:|
| B11 | 25 | 30.741 | 23.774 | 1 | 0 |
| B12 | 25 | 29.826 | 5.087 | 1 | 1 |
| B16-stage1 | 25 | 31.127 | 29.548 | 0 | 0 |
| pooled | 75 | 30.565 | 5.087 | 2 | 1 |

Failure anatomy:

```text
B11 image 00027:
  selected = 23.774
  oracle4  = 29.936
  selector failure; a good SITCOM candidate existed.

B12 image 00017:
  selected = 5.087
  oracle4  = 5.828
  generation failure; all four SITCOM candidates failed.
```

Population-health warning:

```text
tau = 0.8 correction_norm spread
trigger if spread >= 0.003186
```

This catches both selected failures in the pooled audit, but it also fires on many good selected cases.  It should be used as a diagnostic warning flag, not as an automatic replacement rule.

## What not to use as final Branch-B methods

### 1. Naive NP-to-SITCOM handoff

The handoff path remains useful diagnostically, but current evidence says direct continuation of NP states through SITCOM is not competitive.

### 2. 4-to-8 fallback / escalation

B15 showed oracle-style 4-to-8 escalation can repair failures, but B16 and B16A showed that unconditional replacement after escalation hurts already-good Stage-1 selections.

| policy | mean PSNR | min PSNR | bad25 | bad20 | compute |
|---|---:|---:|---:|---:|---:|
| B16 stage1 4S selector | 31.127 | 29.548 | 0 | 0 | 4x SITCOM |
| B16 replace-if-triggered 4-to-8 | 30.659 | 29.100 | 0 | 0 | 5.44x SITCOM |

Therefore 4-to-8 is an oracle/diagnostic curve, not a fair final algorithm.

### 3. Same-budget 3S+1NP / 2S+2NP hybrids

B18 showed that NP has one strong rescue case but several catastrophic failures.  The candidate-set oracle is interesting, but the executable health-to-NP rule is worse than 4S.

Pooled B18B:

| policy | mean PSNR | min PSNR | bad25 | bad20 |
|---|---:|---:|---:|---:|
| 4S baseline | 30.565 | 5.087 | 2 | 1 |
| 3S+1NP oracle candidate set | 31.067 | 25.209 | 0 | 0 |
| 3S health-to-NP | 29.179 | 5.827 | 6 | 5 |
| NP only | 26.398 | 10.434 | 15 | 15 |

B18D unique-image complementarity:

```text
NP bad25: 5 / 25
S4 selected any bad25: 2 / 25
S4 oracle any bad25: 1 / 25
NP rescues S4 selected failure: 1 image
NP hurts all-good S4 selected populations: 4 images
NP hurts at least one good S4 source: 5 images
```

Thus same-budget NP hybrids should not be claimed as final algorithms unless a clean-free certificate is found that distinguishes the NP-rescue regime from the NP-failure regime.

## Recommended next work

No new GPU-heavy Branch-B policy loop is recommended immediately.

Recommended no-GPU / low-compute next steps:

1. Write the Branch-B section around the fixed-budget 4S selector and failure anatomy.
2. Analyze why B12 image `00017` is a SITCOM population-generation failure while NP succeeds.
3. Analyze why NP fails catastrophically on `00013`, `00028`, `00034`, `00018`, and `00027` while SITCOM usually succeeds.
4. Search for clean-free prior-consistency or cross-candidate certificates that distinguish these regimes.
5. For selector failures like B11 image `00027`, test whether better within-4S clean-free ranking can improve selection without increasing compute.

Current final-method candidate:

```text
Fixed-budget 4S SITCOM population selector.
```

Current diagnostic candidates only:

```text
4-to-8 escalation;
3S+1NP candidate-set oracle;
NP rescue analysis;
population-spread warning flag.
```

## Active paths and environments

Use the following path convention on PAC:

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
