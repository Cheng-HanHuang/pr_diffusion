# A20 XP-C3 R2: native DAPS import boundary correction (2026-11-03)

## Registered observations
- Original science manifest 13,600 candidates: SHA-256 `298b88cdd8f4880fe35fd727c6d7c874dabcd9d09da55cd40a469788d5d19b27`, original frozen A20 commit `25152a3d6ffcc4611f45d12bf24d891331976d91`.
- First original smoke failed before candidate scoring: `UnboundLocalError`, 0.0011117359841915054 GPU-hours charged, 0 complete outputs. Source local naming fix registered by R1.
- R1 recovery registered/pushed `eac8b4e044f04e513d136081d5c398da480bd550`; CPU preflight passed but the first native worker again exited nonzero with 0 completed panels. R1 supervisor log reports terminal `STOP_A20_XPC3_UNVERIFIED_OR_BUDGET` and approximately 0.00222 cumulative GPU-hours. Exact R1 worker cause must be checked directly and **must** match expected ImportError before any R2 registration.
- Independent source inspection found second certain defect: `xp_a25_xpc3_mid_gpu.py` imported `NATIVE` from `xp_t_daps4k_prepare_cpu`, which exports the native checkout as `DAPS`; the actual `NATIVE` binding is in `xp_t_daps4k_native_cpu`. R2 moves this import to its authoritative module; `IMAGENET_MODEL_SHA` remains imported from prepare. This is strictly an import repair, not a change to the A20 scoring method.

## R2 procedure and invariants

1. Preserve A26 and A27 *byte for byte* and use fresh A28 worktree.
2. Execute `scripts/track_r/xp_a28_xpc3_r2_freeze_and_launch.sh`, pinned to its committed source SHA. It runs all original and R1/R2 CPU tests and a separate **native DAPS Python environment import preflight** before registering the recovery.
3. Re-read BOTH failed GPU logs and final status receipts; require actual R1 worker traceback contains `ImportError` and `cannot import name 'NATIVE'`. If the traceback is different, **STOP** for manual diagnosis (do not guess).
4. Require 0 complete image outputs in either failed run, their SHA-verified final ledgers, both old frozen source/scorer hashes and their registered Git revisions, no other science changes, and physical GPU 0 exclusion.
5. Create/commit/push `docs/track_r/executors/xp/XP_C3_A20_RECOVERY_R2.json` containing source/original freeze/R1 registry/original + R1 log + final receipts, R1 code hashes, corrected worker/controller hashes, cumulative charged GPU-hours, and GPU mask **before candidate scoring**.
6. Execute controller preflight under shared ledger lock. Only then launch one detached GPU worker at a time on **physical GPUs 1, 2, 3** with original 80% cap. Carry **all prior charges**, not a new 8-hour allowance. Project total cap remains 400 GPU-hours.
7. XP-N (6,400) → G5 (800) → XP-D4 (6,400), each complete or omitted if projected > remaining 8 GPU-hours. Only completed GT-free panels undergo offline GT evaluation and publication.

Do not claim R2 GPU success until a PAC model-loading / one-image smoke log actually confirms it. No v2 protocol, 2-forward secondary, new selection rule, or matched-compute claim.
