# Current experiment plan: NP-SITCOM Branch A and Branch B

Updated: 2026-06-20

## June 20 Branch A status update

Branch A is no longer a final-output selector project.

Current state:

- A0-A4: final-output residual selection was a useful scaffold, but not a viable final solver direction.
- A5-A6: hard-image trajectory logging found early risk signal, with inter-run relative features more promising than absolute residuals.
- A7-A10: held-out retrospective controller analysis found a practical first50 residual-rank AND policy on A8.
- A11: prospectively validated the frozen first50 controller on a fresh trajectory run, but left `7` bad25 misses and `9` false-positive replacements.
- A12: showed many A11 misses were late-developing rather than fundamentally invisible.
- A13: showed first80 residual-rank policies transfer better across A8 / A11 than first50 policies.
- A13.5: showed consensus / nearest-neighbor outlier features catch several residual-rank invisible failures.
- A14 candidate fit: two predeclared development-only frozen policies have now been written before any future A14 trajectory evaluation.

## Frozen A14 policy definitions

Frozen config paths:

- `/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A/A14_frozen_policy_config/frozen_policy.json`
- `/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260616_220045/branch_A/A14_frozen_policy_config/frozen_policy_aggressive.json`
- `/egr/research-pac/huang248/pr_diffusion_repo/configs/branch_A/a14/frozen_policy_conservative.json`
- `/egr/research-pac/huang248/pr_diffusion_repo/configs/branch_A/a14/frozen_policy_aggressive.json`

Primary conservative A14 policy:

```text
policy_name: consensus_lowfreq_nn
policy_family: consensus_single
feature: lowfreq_dist_to_nearest_neighbor
direction: high_is_risky
threshold: 27.855274200439453
fallback: NP selected fallback from the existing FFHQ-25 NP selector CSV
```

Combined A8+A11 development metrics:

- remaining bad25: `16`
- remaining bad20: `13`
- false-positive replacements: `1`
- total replacements: `40`
- run mean PSNR: `29.210`
- image best-of-4 min PSNR: `28.661`

Secondary aggressive A14 policy:

```text
policy_name: residual_or_lowfreq_nn
policy_family: residual_or_consensus
residual arm:
  x0y_full_residual_normed__interrun_rank__first80pct__slope >= 0.8767080745341616
  AND
  x0y_full_residual_normed__interrun_rank__first80pct__last_in_window >= 3.0
consensus arm:
  lowfreq_dist_to_nearest_neighbor >= 27.855274200439453
rule:
  residual_first80 OR lowfreq nearest-neighbor outlier
```

Combined A8+A11 development metrics:

- remaining bad25: `6`
- remaining bad20: `6`
- false-positive replacements: `4`
- total replacements: `53`
- run mean PSNR: `29.944`
- image best-of-4 min PSNR: `28.661`

Selection note:

- the conservative policy remains the primary A14 policy;
- the aggressive policy is predeclared from A8+A11 only and is intentionally marked as higher-replacement-budget;
- the repo-side copies under `configs/branch_A/a14/` are the in-tree references for A14 evaluation;
- no A14 results may be used to change thresholds or features.

## Immediate next experiment

### A14 prospective frozen-policy validation on a fresh trajectory run

The next Branch A experiment should be a true prospective validation of the frozen A14 policy on a fresh SITCOM trajectory run at `sigma_y = 0.05`.

Required rules:

- use `frozen_policy.json` exactly as written for the primary conservative policy;
- also evaluate `frozen_policy_aggressive.json` exactly as written for the secondary aggressive policy;
- do not retune thresholds or features using A14 results;
- evaluate SITCOM-only baseline, both frozen A14 policies, and oracle-risk diagnostics separately;
- report replacement cost, bad25 / bad20 counts, and image-level best-of-4 preservation.

Main questions:

```text
Does the primary conservative A14 policy hold up prospectively on fresh trajectories?
How much extra bad-run reduction does the secondary aggressive policy buy, and at what replacement cost?
```

## Current objective

The project objective is still to develop a reliable diffusion-prior phase-retrieval solver for FFHQ-25. Reliability means not only high average PSNR, but also controlled failure modes: no catastrophic per-image failures, good minimum PSNR, and a method that can be explained as a solver rather than as an offline oracle over many unrelated runs.

## Active paths and environments

Use the following path convention on PAC:

```text
Repo:
  /egr/research-pac/huang248/pr_diffusion_repo

FFHQ image root:
  /egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024

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
  run NP, DiffFPR/guided-model code, CSV parsing/mixing, and offline controller analysis.

sitcom_ode_bw:
  run official SITCOM-ODE and patched SITCOM handoff continuation.
```

Branch B remains available as a secondary diagnostic path, but Branch A prospective controller validation is currently the highest-priority solver experiment.
