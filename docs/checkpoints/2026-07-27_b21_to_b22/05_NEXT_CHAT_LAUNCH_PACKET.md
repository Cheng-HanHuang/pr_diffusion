# B22 next-chat launch packet

This file is the final transition packet from the completed B21 execution chat to a new B22 chat.

## What is preserved on GitHub

The following knowledge is committed on branch `codex/project-checkpoint-b21-to-b22` and is therefore available to a future chat through the connected GitHub repository when the branch and paths are named explicitly:

- the frozen Fresh2 policy and B21 phase boundary;
- the complete B21 scientific checkpoint and publication-status assessment;
- the B21.11 final prospective benchmark plan, implementation, result, and decision;
- the B21.12 failure atlas plan, implementation, quantitative decomposition, and visual taxonomy;
- the method/policy registry, including adopted, rejected, superseded, and diagnostic paths;
- the B21-to-B22 PAC checkpoint and dirty-source snapshot procedures;
- the B22 scope, evaluation rules, no-go constraints, and first authorized inventory checkpoint.

The checkpoint branch is stacked on the completed B21 branches. Open draft PR #34 documents the checkpoint but is not merged into the protected branch.

## What is not stored on GitHub

The following remain PAC-side artifacts and cannot be assumed accessible to a new chat merely because their paths are documented:

- 100 locked B21.11 measurement tensors;
- 200 DAPS reconstruction PNGs and 200 metric CSVs;
- B21.12 generated atlas images and locally reviewed CSV;
- the full PAC environment state;
- the exact dirty outer-repository state;
- the exact locally modified DAPS source files and generated YAML configurations;
- the checkpoint and source-snapshot archives themselves.

These are preserved in the PAC checkpoint directories and archives. A new chat cannot read the PAC filesystem directly. The user must provide small inventory files, paste command output, or run requested PAC commands.

## What a new chat does and does not remember

A new chat must not be assumed to inherit the full previous conversation transcript. It may receive limited user-level context, but that is not a substitute for the repository checkpoint. The new chat must ground its work in the committed checkpoint documents and the PAC inventory supplied by the user.

The handoff is therefore artifact-based, not memory-based.

## Repository entry point

```text
repository: Cheng-HanHuang/pr_diffusion
branch: codex/project-checkpoint-b21-to-b22
checkpoint directory: docs/checkpoints/2026-07-27_b21_to_b22
checkpoint PR: #34
```

Authoritative reading order:

1. `00_START_HERE.md`
2. `01_PROJECT_CHECKPOINT.md`
3. `03_B22_NEW_CHAT_HANDOVER.md`
4. `04_DIRTY_SOURCE_SNAPSHOT_AMENDMENT.md`
5. `05_NEXT_CHAT_LAUNCH_PACKET.md`
6. `docs/b21/b21_11_fresh2_final_benchmark.md`
7. `docs/b21/b21_12_failure_atlas_decision.md`
8. `docs/b21/b21_12_visual_failure_interpretation.md`
9. `docs/b21/b21_registry.md`

## PAC checkpoint paths to give the new chat

Provide the actual completed values from the terminal rather than placeholders:

```text
scientific checkpoint:
/egr/research-pac/huang248/outputs/pr_diffusion/checkpoints/B21_to_B22_20260727_033521

companion source snapshot:
/egr/research-pac/huang248/outputs/pr_diffusion/checkpoints/B21_source_snapshot_<ACTUAL_STAMP>
```

The corresponding `.tar.gz` archives remain preservation copies. Do not upload the large archives unless specifically needed.

## Minimum small files to attach to the new chat

From the scientific checkpoint:

- `checkpoint_metadata.txt`
- `repo/key_refs.tsv`
- `repo/status_short_branch.txt`
- `repo/submodule_status.txt`
- `repo/daps_status.txt`
- `repo/daps_diff_stat.txt`
- `artifacts/artifact_inventory.txt`
- `HUMAN_SIGNOFF.md`

From the companion source snapshot:

- `source_snapshot_metadata.txt`
- `source_snapshot_inventory.txt`
- `repo/remote_key_refs.tsv`
- `repo/modified_tracked_files.txt`
- `repo/untracked_source_files.txt`
- `daps/diff_stat.txt`
- `daps/modified_tracked_files.txt`

The new chat does not need the large archive to begin B22.0 inventory.

## Settled B21 conclusions

The frozen Fresh2 policy for FFHQ phase retrieval at `sigma_y=0.05` is:

1. run two independent full DAPS trajectories;
2. use `ann400`, `diff5`, LF disabled, HIO disabled;
3. begin with trajectory 1;
4. accept trajectory 2 only when `loss2 < loss1 - 0.7`;
5. return the selected reconstruction.

Final prospective result on 100 untouched official-validation images:

- Fresh1 good25: `80/100`;
- Fresh2 selected good25: `92/100`;
- Fresh2 two-candidate oracle good25: `92/100`;
- rescues: `12`;
- harms: `0`;
- selected-oracle gap: `0`;
- cost: `2.0` full-run equivalents.

Failure interpretation:

- official raw-PSNR failures: `8`;
- selected candidate good25 after offline 180-degree alignment: `3`;
- persistent under both candidates and both orientations: `5`;
- persistent classes: two chromatic/illumination failures, two structured twin/ghost mixtures, and one high-complexity prior collapse.

These conclusions must not be re-derived or retuned on the B21.11 final panel.

## Closed or rejected B21 directions

Do not restart these merely because the new chat lacks conversational history:

- universal LF second arm lost to equal-cost Fresh2;
- HIO warm-start replacement and HIO auxiliary gate were rejected;
- Fresh3 was rejected as the default fixed budget after disjoint validation;
- continuation branching did not beat fresh restarts;
- raw-loss, normalized-residual, and within-measurement disagreement triggers did not support a selective fallback policy;
- no additional detector, fallback, loss-margin, or restart-count tuning is authorized on the final panel.

## B22 question

B22 determines whether frozen Fresh2 is a standalone publishable reliability contribution or a strong component of a broader phase-retrieval project by comparing it against fixed established baselines, primarily SITCOM and NP.

The first authorized task is inventory only. No baseline GPU run is allowed until exact code/configuration, environment, checkpoint, measurement compatibility, seed control, metrics, runtime, and cost accounting are frozen and reviewed.

## Copyable opening message

```text
You are the execution lead for B22: fixed-baseline evaluation of the frozen Fresh2 phase-retrieval policy in the private repository Cheng-HanHuang/pr_diffusion.

Use branch codex/project-checkpoint-b21-to-b22. Read every checkpoint file before proposing implementation, in this order:
1. docs/checkpoints/2026-07-27_b21_to_b22/00_START_HERE.md
2. docs/checkpoints/2026-07-27_b21_to_b22/01_PROJECT_CHECKPOINT.md
3. docs/checkpoints/2026-07-27_b21_to_b22/03_B22_NEW_CHAT_HANDOVER.md
4. docs/checkpoints/2026-07-27_b21_to_b22/04_DIRTY_SOURCE_SNAPSHOT_AMENDMENT.md
5. docs/checkpoints/2026-07-27_b21_to_b22/05_NEXT_CHAT_LAUNCH_PACKET.md
6. docs/b21/b21_11_fresh2_final_benchmark.md
7. docs/b21/b21_12_failure_atlas_decision.md
8. docs/b21/b21_12_visual_failure_interpretation.md
9. docs/b21/b21_registry.md

I am attaching the small PAC checkpoint and companion source-snapshot inventory files. The large outputs remain on PAC at the paths recorded in those files.

B21 method development is closed. Fresh2 is frozen as two independent ann400/diff5 DAPS trajectories, LF/HIO disabled, with trajectory 2 accepted only when its exact operator loss improves over trajectory 1 by more than 0.7. The final prospective result is Fresh1 80/100 and Fresh2 selected/oracle 92/100, with 12 rescues, zero harms, zero oracle gap, and cost 2.0 full-run equivalents. Do not tune Fresh2, add Fresh3, fit another detector, or use the final 100-image panel for method search.

Your first authorized task is B22.0 inventory only. Inspect the exact existing SITCOM and NP repositories, commits, environments, checkpoints, historical frozen configurations, measurement generation/loading interfaces, seed control, output parsers, PSNR conventions, runtime, GPU memory, and whether each baseline can consume the exact B21.11 locked measurement tensors. Do not launch a GPU baseline before I review and sign off on the inventory and frozen comparison plan. Stop on any measurement incompatibility, metric mismatch, missing checkpoint, or irreproducible historical configuration instead of silently repairing it.

Repository protection rules: never directly edit, reset, rebase, merge into, or force-push b19_solver_integration. Preserve the dirty PAC checkout and local DAPS modifications. Use a new codex/* branch for B22 implementation. Do not automatically merge or rewrite the stacked B21 PRs.
```

## Handoff completion criterion

The handoff is complete when the user opens a new chat with the copyable message, names both PAC checkpoint paths, attaches the listed small inventory files, and asks the new chat to begin B22.0 inventory only.
