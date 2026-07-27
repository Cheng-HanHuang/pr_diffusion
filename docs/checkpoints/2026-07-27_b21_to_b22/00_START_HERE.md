# B21 → B22 project checkpoint: start here

Checkpoint date: 2026-07-27

This directory freezes the state of the FFHQ diffusion phase-retrieval project after completion of B21 method development and before starting B22 fixed-baseline evaluation.

## Authoritative reading order

1. `00_START_HERE.md` — this file.
2. `01_PROJECT_CHECKPOINT.md` — scientific conclusions, adopted/rejected methods, repository state, and artifact map.
3. `02_PAC_INVENTORY_AND_FREEZE.md` — PAC-side inventory and preservation procedure.
4. `03_B22_NEW_CHAT_HANDOVER.md` — exact B22 scope and execution constraints.
5. `04_DIRTY_SOURCE_SNAPSHOT_AMENDMENT.md` — preservation of untracked research source and dirty DAPS files.
6. `05_NEXT_CHAT_LAUNCH_PACKET.md` — what is on GitHub, what remains PAC-side, minimum attachments, and the final copyable new-chat opening message.
7. `docs/b21/b21_11_fresh2_final_benchmark.md` — final prospective Fresh2 benchmark.
8. `docs/b21/b21_12_failure_atlas_decision.md` and `docs/b21/b21_12_visual_failure_interpretation.md` — failure decomposition.
9. `docs/b21/b21_registry.md` — full method/policy registry.

Later files must not silently override an earlier file in this order. If a contradiction is found, stop and record it in the checkpoint rather than guessing.

## Phase boundary

### B21 is complete and frozen

The adopted fixed policy for the current FFHQ, `sigma_y=0.05` setting is:

1. run two independent full DAPS trajectories;
2. use `ann400`, `diff5`, LF disabled, HIO disabled;
3. begin with trajectory 1;
4. accept trajectory 2 only when its exact operator loss improves by more than `0.7`;
5. return the selected reconstruction.

The prospective official-validation result is Fresh1 `80/100` and Fresh2 selected/oracle `92/100`, with 12 rescues, zero harms, and zero selected-oracle gap at 2.0 full-run equivalents.

Do not tune the loss margin, restart count, detector, LF/HIO fallback, or other solver-policy component on the B21.11 final 100-image panel.

### B22 is a new evaluation stage

The next stage is a preregistered, fixed-method comparison against established project baselines, primarily SITCOM and NP, on the same locked measurements when technically compatible.

B22 is not authorization to resume Fresh2 method search. Before any GPU launch it must:

1. inventory the exact baseline code, environments, checkpoints, measurement interfaces, and historical frozen configurations;
2. decide which configurations are genuinely fixed before viewing B21.11 baseline outputs;
3. define raw and 180-degree-ambiguity-aware evaluation consistently across methods;
4. pass a one-image locked-measurement smoke;
5. freeze seeds, costs, output paths, and analysis before the full comparison.

## Repository protection rules

- Never directly edit, reset, rebase, merge into, or force-push `b19_solver_integration` from an assistant workflow.
- Use a dedicated `codex/*` or `llm-agent/*` branch.
- Preserve dirty PAC state; do not use `git clean`, destructive resets, or forced submodule updates.
- Do not treat PAC output artifacts as committed merely because a report names them.
- Do not merge the stacked B21 PRs automatically. Review and integrate them in dependency order.

## Current checkpoint branch

```text
codex/project-checkpoint-b21-to-b22
```

This branch is based on the completed B21.12 branch and adds documentation/inventory only. It must not contain new solver behavior.

## Completion condition for this checkpoint

The checkpoint is complete when:

- the repository-side checkpoint documents are committed;
- the PAC inventory and companion source-snapshot collectors have both run and their output directories are preserved;
- the manually reviewed B21.12 label CSV is included in the PAC checkpoint artifacts;
- the branch/PR integration decision is recorded;
- the next chat starts from `05_NEXT_CHAT_LAUNCH_PACKET.md` and the committed checkpoint rather than reconstructing the project from conversation history.
