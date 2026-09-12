# B24 start here

## Current status

B24 remains isolated from B23 and descends from the signed-off B23.1 base `27505e6328157ac9296c95dc5e611cbeef80de98`. PR #37 remains untouched. B23 cross-family H0 failed; no NP/SITCOM cross-family adapter qualified, and B24 does not reinterpret that result.

The baseline-first phase is complete. Compact freeze evidence is published at `docs/b24/evidence/B24_2_7424_FREEZE_CLOSEOUT.json`.

Final cumulative screen:

- 7424 deterministic screen rows;
- A = 6925;
- B = 107;
- C = 307;
- D = 85;
- final manifest file SHA-256 `b516c8154cbbb790d8a3592b86736bb0d4bd47d0833d85ecf3d6a9d710e950ba`;
- realized universe CSV SHA-256 `4c6eeabe7d73f948ff0820a7f8c87ed091ff94100fc5def32586c5581c449f25`.

The primary balanced `ABC300` cohort is frozen as first 100 A/B/C cases under `B24_CLASS_RANK_V1`, CSV SHA-256 `4599c2a8c1f4a5922640e0c26d2c1efce7f1996d75dcabbff2e9a1c4b427cbce`.

A secondary C1 severity cohort is also frozen: the 100 class-C cases with lowest pinned SITCOM-4 best raw-RGB PSNR, CSV SHA-256 `9c04994dbd91f6a5bb04280736e4dea346c337508a89f30ae8553a42867576b6`. C1 is outcome-selected and diagnostic only; it does not replace the primary hash-ranked C100.

## Next scientific question

Stop baseline image collection. The next question is:

> Does changing how native NP retains and allocates proposals improve recovery beyond independent NP populations at comparable compute?

The machine-readable method specification is `configs/b24/b24_3_method_dev_spec.json`.

Two new candidates are frozen for bounded development:

1. **NP_EPP — early population pruning.** Start four independent native NP roots, prune 4→2 at transition 72 and 2→1 at transition 148 using a measurement-only trailing-32 low-frequency MSE score, and reallocate proposal count 5→10→20 so aggregate pre-projection proposal evaluations remain compute-matched to four independent NP-1 runs. At projection start, fork the selected native state into four independent hard-phase descendants.
2. **NP_DPS — delayed proposal selection.** Run one greedy NP root through transition 71. At transitions 72, 148, and 224, retain all five native NP proposals and advance them as five complete branches through a 76-transition delayed-selection window before pruning to one by the same measurement-only trailing-32 score. At projection start, fork the selected native state into four independent hard-phase descendants.

Both methods preserve complete branch state and named RNG identity. Ground truth may be used offline for terminal-oracle diagnostics only; it may not control pruning, allocation, routing, stopping, or clean-free terminal selection.

## Compute matching

Frozen NP-1 parent: 1000 steps, projection start 300, soft/hard candidate counts 5/1, LF score radius 0.6, projection radius 0.2.

- NP-1: 2199 proposal UNet evaluations + 1 initial model evaluation = 2200 approximate model evaluations.
- four independent NP-1 runs: 8796 proposal evaluations + 4 initial evaluations = 8800.
- NP_EPP: exactly 8796 proposal evaluations + 4 initial evaluations = 8800.
- NP_DPS: exactly 8796 proposal evaluations + 1 initial evaluation = 8797. The three-evaluation difference from NP4 comes only from NP4's additional independent initial roots.

Actual Work-FRE, GPU-active time, wall time, and memory must still be measured; proposal counts do not substitute for calibrated execution cost.

Historical NP-8-RS must retain its actual identity: two scoring configurations (`lf` and `s2_preproj_lam001`) × four seeds, selected by `selector_post_winner_lf_mse_mean`. It is not eight identical NP runs.

## Image roles

Before any project-method result is observed, freeze the new method-stage roles with:

`bash scripts/b24/freeze_b24_method_roles.sh`

The role policy is:

- ABC300 A100: 20 development / 80 confirmation;
- ABC300 B100: 20 development / 80 confirmation;
- ABC300 C100: 20 development / 80 confirmation;
- all D85: 20 development / 65 confirmation.

A second fixed hash selects four pilot images per stratum from the 80 development images, producing a 16-image pilot. Development images receive one new locked measurement. Confirmation images receive two new locked measurements only after the main method is selected and frozen.

Original A/B/C/D labels remain screening strata. New-measurement baseline outcomes must be recomputed and class transitions reported; images are never removed because they become easy.

C1 is secondary only. C1 cases overlapping the primary C100 inherit the primary development/confirmation role. C1 cases outside primary C100 are locked from method development until the main method is frozen.

## Pilot scope and authorization boundary

The next executable scientific stage is the **16-image development pilot**, four images per original A/B/C/D stratum, after the role manifest is frozen.

Pilot arms specified for engineering/scientific validation:

- NP-1;
- four independent NP-1 runs;
- NP_EPP;
- NP_EPP random-pruning ablation;
- NP_EPP no-reallocation ablation;
- NP_DPS.

The pilot must verify native continuation, branch/RNG identity, proposal accounting, checkpoint-score behavior, and memory. DAPS-4 and pinned SITCOM-4 are mandatory on the subsequent 80-image development comparison; they may be omitted from the first engineering smoke if needed.

The global B24 hard process/group ceiling remains **52,452 MiB**. The baseline-specific 10,240-MiB admission gate is **not automatically valid for NP branching**. NP memory/admission must be calibrated on a one-image smoke before parallel pilot execution.

No confirmation measurement may be materialized and no confirmation method may run before one main method is selected and frozen from development.

## Repository / PAC identities

- branch: `codex/b24-bestof4-failure-sweep`;
- draft PR: #38;
- PR base: `codex/b23-execution`;
- PAC worktree: `/egr/research-pac/huang248/pr_diffusion_b24`;
- PAC output root: `/egr/research-pac/huang248/outputs/pr_diffusion/b24`.

Never modify PR #37. Never rebase, squash, force-push, retarget, or rewrite B24 history.
