# Track R XP-C3 / A20 R1 zero-output code recovery (2026-11-03)

**Authoritative science:** prereg A20 (planner a6ea8cc). R1 does not change its test, its candidates, or its deadline.

## What happened

- The original A20 SHA/source freeze was **accepted and pushed** at `25152a3d6ffcc4611f45d12bf24d891331976d91` before any XP-C3 GPU scoring.
- The original 13,600-source manifest SHA-256 is `298b88cdd8f4880fe35fd727c6d7c874dabcd9d09da55cd40a469788d5d19b27`; the frozen scorer SHA-256 is `a6d37d9c4bd319a91cbe4704d8e3f4ac468517f0a56070c83bd6c5c565c980ce`. Both original file bytes remain preserved.
- The first new GPU worker in `xp_a25_xpc3_prod_20261103T002834Z` failed before calling `source(a)` with Python `UnboundLocalError: cannot access local variable 'source' where it is not associated with a value`. A later assignment to local `source = torch.load(...)` made the earlier function name an uninitialized local.
- Original attempt: physical GPU 0, XP-N one-image smoke, process return code 8, approximately 4.00225 seconds, exactly **0.0011117359841915054 GPU-h charged**, **zero fully scored candidates**, zero completed images/panels, final status `STOP_A20_XPC3_UNVERIFIED_OR_BUDGET`, project ledger ~232.0794567 GPU-h. The actual failed final receipt SHA is to be frozen by PAC in the R1 registry.
- XP-I2 completed 100/100 bags, charged 3.0515335384 GPU-h, project total 232.07834495911806 before the failed XP-C3 attempt.

## Minimal allowed change

1. Rename the local `source = torch.load` to `saved_reconstruction`; call `source(a)` still resolves to the original input validator. No denoiser loss, noise generation, schedule, UID, threshold, image, source, checkpoint, or orbit definition changes.
2. **Hard exclusion of physical GPU 0**, in both the CPU scheduler and GPU worker. Only physical GPUs 1, 2, 3 are eligible; the original 80%-VRAM cap and foreign-job protections continue to apply.
3. Full retrospective stage and global budget charge: new run starts with the 0.00111173598 GPU-h consumed. Total stage allowance remains **8 GPU-h**, total project cap **400 GPU-h**, not 8 + old attempt.
4. Original frozen source and code freeze immutable. R1 creates **an additional recovery registry**, `docs/track_r/executors/xp/XP_C3_A20_RECOVERY_R1.json`, that records original and new code hashes, original failure log/final receipt SHA, budget, and GPU mask. It must be committed/pushed before any R1 scorer is admitted. Worker checks original `25152a3` scorer bytes plus both freeze hashes and the new repair SHA at runtime.
5. Run in a **new detached A27 worktree**; never change original A26 (`25152a3`) or existing A24 XP-I2 checkout. Use only `scripts/track_r/xp_a26_xpc3_r1_freeze_and_launch.sh` once. It runs the original A20 tests and new behavioral scope/GPU masking tests before registering R1, then performs a read-only ledger preflight before detaching.
6. A full XP-N image is still the registered smoke; projection precedes full panels in XP-N → G5 → XP-D4 order. Partial panels are never published. New GPU smoke or outcome status **not claimed** by this document.

A13.4 applies if original source or code SHA conflicts, source files drift, old receipt mismatches, project budget fails, or further scientific changes would be needed. No protocol v2, new selector, two-forward secondary test, unregistered code modifications, or matched-compute claim.
