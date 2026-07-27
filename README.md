# PR Diffusion: reliable diffusion-prior phase retrieval

Updated: 2026-07-27

This repository studies reliability and failure modes of diffusion-prior solvers for FFHQ phase retrieval under oversampled Fourier-magnitude measurements.

## Current status

B21 method development is complete and frozen for the current FFHQ, `sigma_y=0.05` setting.

The adopted policy, **Fresh2**, is:

1. run two independent full DAPS trajectories;
2. use `ann400`, `diff5`, LF disabled, HIO disabled;
3. begin with trajectory 1;
4. accept trajectory 2 only when its exact operator loss improves by more than `0.7`;
5. return the selected reconstruction.

Prospective result on 100 untouched official-validation images with one locked measurement per image:

| policy | good25 | bad25 |
|---|---:|---:|
| Fresh1 | 80/100 | 20/100 |
| Fresh2 selected | 92/100 | 8/100 |
| two-candidate oracle-any-good | 92/100 | 8/100 |

Fresh2 produced 12 rescues, zero harms, and zero selected-oracle gap at 2.0 full-run equivalents.

The B21.12 atlas further showed that three of the eight official failures are clean reconstructions under the known 180-degree ambiguity, while five remain bad under both candidates and both orientations. The five persistent failures include chromatic/illumination errors, structured twin/ghost mixtures, and one high-complexity collapse.

## Start here

The current authoritative checkpoint is:

```text
docs/checkpoints/2026-07-27_b21_to_b22/00_START_HERE.md
```

Read in order:

```text
docs/checkpoints/2026-07-27_b21_to_b22/00_START_HERE.md
docs/checkpoints/2026-07-27_b21_to_b22/01_PROJECT_CHECKPOINT.md
docs/checkpoints/2026-07-27_b21_to_b22/02_PAC_INVENTORY_AND_FREEZE.md
docs/checkpoints/2026-07-27_b21_to_b22/03_B22_NEW_CHAT_HANDOVER.md
```

Primary completed reports:

```text
docs/b21/b21_11_fresh2_final_benchmark.md
docs/b21/b21_12_failure_atlas_decision.md
docs/b21/b21_12_visual_failure_interpretation.md
docs/b21/b21_registry.md
```

## Phase boundary

### B21: complete

B21 studied fixed-budget reliability mechanisms, including independent restarts, LF guidance, HIO warm starts, continuation branching, restart-budget scaling, and clean-free failure detectors.

The final decision is to retain the simple fixed Fresh2 policy. Do not tune its threshold, restart count, detector, LF/HIO fallback, or other policy component on the final B21.11 panel.

### B22: next

B22 is a new fixed-baseline evaluation stage. Its first task is to inventory and preregister SITCOM and NP configurations before any GPU launch.

The key unresolved question is whether frozen Fresh2 beats or meaningfully complements established baselines in catastrophic-failure rate, PSNR distribution, and reliability-versus-compute tradeoff.

B22 is not authorization to resume Fresh2 method invention.

## PAC paths

Current integration checkout:

```text
/egr/research-pac/huang248/pr_diffusion_b19_solver
```

B21 output root:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver
```

Final B21.11 output:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/
B21_11_fresh2_final_val100_meas5401
```

FFHQ image root:

```text
/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024
```

FFHQ model checkpoint:

```text
/egr/research-pac/huang248/models/ffhq_10m.pt
```

DAPS Python:

```text
/egr/research-pac/huang248/conda-envs/daps/bin/python
```

External SITCOM/NP paths and environments must be re-inventoried through the checkpoint collector before B22. Do not rely on historical README paths without verification.

## Repository workflow

The B21 work is currently represented by a stacked draft-PR chain:

```text
#30 continuation interface
  -> #31 B21.5–B21.10 development
     -> #32 B21.11 final benchmark
        -> #33 B21.12 atlas
```

The checkpoint branch is:

```text
codex/project-checkpoint-b21-to-b22
```

Never directly edit, reset, rebase, merge into, or force-push `b19_solver_integration` from an assistant workflow. Use a dedicated `codex/*` or `llm-agent/*` branch and preserve dirty PAC state.

## Historical work

Earlier NP/SITCOM branch investigations and historical plans remain useful background but are no longer the current project entry point. See:

```text
docs/historical/
docs/progress_report.md
docs/branch_A_clean_free_certificates.md
docs/branch_B_fixed_budget_population_selector.md
scripts/npsitcom/
```

When a historical file conflicts with the 2026-07-27 checkpoint, the checkpoint is authoritative unless a concrete implementation or artifact error is demonstrated.
