# B22 new-chat handover

Use this file to start a new chat after the repository and PAC checkpoint are complete.

## Role of the next chat

The next chat is the execution lead for **B22: fixed-baseline evaluation of the frozen Fresh2 policy**.

It is not a continuation of B21 method invention. It must not tune Fresh2, weaken prior decision gates, or mine the B21.11 final panel for a new detector/fallback.

## Authoritative source order

1. `docs/checkpoints/2026-07-27_b21_to_b22/00_START_HERE.md`
2. `docs/checkpoints/2026-07-27_b21_to_b22/01_PROJECT_CHECKPOINT.md`
3. the completed PAC checkpoint directory produced by `02_PAC_INVENTORY_AND_FREEZE.md`
4. `docs/b21/b21_11_fresh2_final_benchmark.md`
5. `docs/b21/b21_12_failure_atlas_decision.md`
6. `docs/b21/b21_12_visual_failure_interpretation.md`
7. `docs/b21/b21_registry.md`
8. relevant historical SITCOM/NP reports and scripts, only after inventory

If later historical files conflict with the checkpoint, the checkpoint wins unless a concrete bug or artifact mismatch is demonstrated.

## Frozen Fresh2 method

For each locked measurement:

- two independent full DAPS trajectories;
- `ann400`, `diff5`;
- LF and HIO disabled;
- begin with trajectory 1;
- accept trajectory 2 iff `loss2 < loss1 - 0.7`;
- no third restart or conditional fallback.

B21.11 result:

- Fresh1 `80/100` good25;
- Fresh2 selected/oracle `92/100` good25;
- 12 rescues, zero harms, zero two-candidate oracle gap;
- 2.0 full-run equivalents.

B21.12 result:

- 8 official raw-PSNR failures;
- 3 selected outputs become good25 after offline 180-degree alignment;
- 5 remain bad under both candidates and both orientations;
- the five persistent cases comprise chromatic/illumination failures, structured twin/ghost mixtures, and one high-complexity collapse.

## B22 scientific question

Determine whether the frozen Fresh2 policy:

1. beats established project baselines in catastrophic-failure rate;
2. is competitive in mean/median/quantile PSNR;
3. offers a defensible reliability-versus-compute tradeoff;
4. is strong enough for a standalone paper, or should remain a component of a broader project.

## Required baselines

At minimum inventory and evaluate:

- DAPS Fresh1 — already complete on the B21.11 panel;
- DAPS Fresh2 — already complete and frozen;
- official/frozen SITCOM configuration;
- official/frozen NP configuration.

Optional methods must not be added merely because they are easy to run. Each added baseline needs a clear scientific role and a fixed configuration established without tuning on B21.11 outputs.

## Evaluation rules

Report for every method:

- raw PSNR;
- offline ambiguity-aware PSNR: `max(raw, rot180)`;
- good25/bad25 under both conventions;
- mean, median, minimum, and relevant lower quantiles;
- wall time and transparent compute-equivalent accounting;
- failure overlap with Fresh2;
- per-image paired comparisons on identical locked measurements.

Raw PSNR remains the primary comparison for continuity. Ambiguity-aware PSNR is auxiliary and must be labeled ground-truth-assisted/offline.

## First authorized checkpoint: B22.0 inventory only

Before any GPU launch, inspect and report:

1. exact SITCOM/NP repositories and commit hashes;
2. conda environments and package versions;
3. checkpoint/model paths;
4. measurement-generation and measurement-loading interfaces;
5. whether each baseline can consume the exact B21.11 locked measurements;
6. historical configurations previously treated as official/frozen;
7. seed control and output naming;
8. existing parsers and PSNR conventions;
9. estimated per-image runtime and GPU memory;
10. any incompatibility that would invalidate paired comparison.

Do not silently regenerate measurements for a baseline that cannot load the B21.11 tensors. Stop and document the mismatch.

## B22.1 smoke gate

After B22.0 is signed off:

- freeze one SITCOM configuration and one NP configuration;
- select one B21.11 image without looking at method outcome;
- run each baseline on the exact locked measurement;
- verify finite output, raw and rot180 PSNR, runtime, seed/config recording, and path integrity;
- only then implement the multi-GPU full-panel runner.

## Repository rules

- Do not directly modify `b19_solver_integration`.
- Work on a new branch such as `codex/b22-fixed-baseline-comparison`.
- Preserve the dirty PAC checkout and DAPS local diff.
- Do not merge or rewrite the B21 stacked PRs automatically.
- Do not re-derive settled B21 conclusions.
- Stop on measurement incompatibility, metric mismatch, missing checkpoint, or irreproducible historical configuration.

## PAC paths to verify, not blindly assume

```text
repo:
/egr/research-pac/huang248/pr_diffusion_b19_solver

B21.11 output:
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/
B21_11_fresh2_final_val100_meas5401

FFHQ images:
/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024

FFHQ model:
/egr/research-pac/huang248/models/ffhq_10m.pt

historical SITCOM checkouts:
/egr/research-pac/huang248/external/SITCOM_ODE
/egr/research-pac/huang248/external/SITCOM_ODE_npsitcom
```

The PAC checkpoint inventory is the source of truth for their actual existence and current state.

## Copyable opening prompt for the new chat

```text
You are the execution lead for B22: fixed-baseline evaluation of the frozen Fresh2 phase-retrieval policy.

Read the repository checkpoint in this order:
1. docs/checkpoints/2026-07-27_b21_to_b22/00_START_HERE.md
2. docs/checkpoints/2026-07-27_b21_to_b22/01_PROJECT_CHECKPOINT.md
3. docs/checkpoints/2026-07-27_b21_to_b22/03_B22_NEW_CHAT_HANDOVER.md
4. docs/b21/b21_11_fresh2_final_benchmark.md
5. docs/b21/b21_12_failure_atlas_decision.md
6. docs/b21/b21_12_visual_failure_interpretation.md
7. docs/b21/b21_registry.md

I will also provide the PAC checkpoint output produced by scripts/checkpoints/collect_b21_project_checkpoint.sh.

B21 method development is closed. Fresh2 is frozen as two ann400/diff5 DAPS trajectories with LF/HIO disabled and exact-loss margin 0.7. Do not tune it, add Fresh3, fit a detector, or use the final 100-image panel for method search.

Your first authorized task is B22.0 inventory only: inspect the existing SITCOM and NP code/configurations, environments, checkpoints, measurement interfaces, seed control, output parsers, and runtime. Determine whether they can consume the exact B21.11 locked measurements. Do not launch GPU baselines before I review and sign off on the inventory and frozen comparison plan.

Preserve the dirty PAC checkout. Never directly edit or merge into b19_solver_integration. Use a new codex/* branch and stop on any measurement/metric/configuration mismatch instead of silently repairing it.
```

## Handover completion

The old chat may be considered closed after:

- the checkpoint branch is pulled on PAC;
- the collector completes with verified checksums;
- the human sign-off file is filled;
- the new chat receives the checkpoint directory or its key text outputs;
- B22 begins with inventory rather than immediate experiments.
