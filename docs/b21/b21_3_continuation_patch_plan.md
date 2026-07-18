# B21.3 continuation patch and smoke plan

Status: implementation branch; GPU smoke not yet executed.

## Purpose

Add a default-off continuation mechanism to the locally B20-patched pixel DAPS code without disturbing fixed-measurement loading or LF guidance.

Controls:

```text
B21_CONT_ENABLE=1
B21_CONT_STATE_PATH=<payload.pt>
B21_CONT_NOISE_SEED=<int>
B21_SAVE_STATE_STEPS=0,200
B21_START_NOISE_SEED=<int>   # smoke-only matched step-0 source
```

A saved payload contains `x0y`, the next global annealing `step`, and the scheduler `sigma` at that index. Continuation starts from

```text
xt = x0y + sigma[step] * eps(B21_CONT_NOISE_SEED)
```

The global step index is preserved, so the existing B20 LF schedule is not restarted during continuation.

## Files

- `scripts/b21/apply_b21_3_continuation_patch.py`: marker/anchor-based source transformation.
- `scripts/b21/apply_b21_3_continuation_patch.sh`: safe wrapper, syntax check, and exact incremental diff capture.
- `scripts/b21/run_b21_3_continuation_smoke.sh`: image `00046`, measurement `5001`, ann400/diff5 smoke.
- `scripts/b21/check_b21_3_continuation_smoke.py`: default-off, step-0 reproduction, payload, and branch-diversity gates.
- `docs/b21/patches/daps_b21_continuation.patch`: generated on PAC from the exact local B20-patched DAPS state.

## Smoke cases

1. `default_full`: all B21 controls unset.
2. `source_full`: explicit start-noise seed; save step 0 and step 200 states.
3. `cont0_same_seed`: load step 0 with the same noise seed; must reproduce `source_full`.
4. `branch_seed7900`, `branch_seed7901`, `branch_seed7902`: load the same step-200 state with independent continuation noise.

## Pass gates

- Default-off full run completes.
- Step-0 and step-200 payloads exist.
- Source and continuation-from-0 PNGs are identical, or differ by at most one uint8 level with exact-loss and PSNR differences at most `1e-5`.
- At least two unique hashes among the three step-200 branches.

No larger branch-vs-fresh pilot is launched by these scripts.
