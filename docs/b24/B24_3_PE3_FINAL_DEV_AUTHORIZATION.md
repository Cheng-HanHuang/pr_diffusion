# B24.3 final DEV-only protected-explorer authorization

Planner/user authorization (2026-09-13):

> Authorize B24 DEV-only final protected-explorer refinement plus cross-family FLOP audit. Implement and run exactly PE3_SCORE and PE3_RANDOM on the existing DEV80 measurements at 8,800 NP UNet evaluations each; perform compute accounting using development data only. No confirmation exposure. If neither PE3 arm passes the prospectively frozen NP4 gate, stop B24 method refinement.

## Scientific motivation

DEV80 falsified `NP_EPP_321` as the main method at the same 8,800-UNet work as `NP4_INDEPENDENT`: its rare large rescues did not compensate for frequent large harms. The matched random-pruning arm nevertheless showed that proposal reallocation can create useful basins, especially on some fresh-D measurements. This final refinement therefore protects multiple native lineages instead of repeatedly pruning to one lineage.

## Exactly authorized project-method arms

Only two new methods may be executed:

- `NP_PE3_SCORE`
- `NP_PE3_RANDOM`

Both use exactly 8,800 total NP UNet evaluations per image, including four root initializations. No checkpoint sweep, score-window sweep, extra method variant, or additional root budget is authorized.

Both methods share the fixed schedule:

1. initialize four frozen NP roots;
2. transitions 0..71: four roots, `k=5` proposals per root;
3. immediately before transition 72, retain exactly three roots and assign one retained root the explorer role;
4. transitions 72..299: explorer uses `k=10`; the two protected roots remain native `k=5` trajectories;
5. there is no pruning at 148 or 300;
6. at projection start, preserve one native continuation from each of the three retained lineages and add exactly one deterministic hard-phase fork from the explorer;
7. transitions 300..998: four terminal trajectories, `k=1` each;
8. clean-free terminal selection remains the frozen post-projection measurement-side LF-MSE selector; GT-best terminal remains offline oracle only.

`NP_PE3_SCORE` uses the frozen trailing-32 LF measurement score at checkpoint 72: drop the worst-scored root and designate the best-scored retained root as explorer. Stable lineage tie-break applies.

`NP_PE3_RANDOM` uses no measurement score for the checkpoint-72 role assignment. A frozen domain hash `B24_METHOD_PE3_RANDOM_ROLE_V1` deterministically orders the four roots: the first becomes explorer, the next two are protected, and the fourth is dropped.

Exact work accounting per image:

`4 + 72*4*5 + 228*(10+5+5) + 699*4 = 8,800` total UNet evaluations.

## Frozen advancement gate versus NP4

Each PE3 arm is evaluated on the same 80 already-exposed development measurements against the already-completed `NP4_INDEPENDENT` result, using canonical raw-orientation 8-bit RGB PSNR. An arm may be considered for a later method freeze only if all four conditions hold:

- median paired PSNR delta versus NP4 is >= 0 dB;
- PE3 Good25 count is at least NP4 Good25 count;
- Good25 rescues are at least Good25 harms;
- >=5 dB rescues are at least >=5 dB harms.

If neither arm passes every condition, B24 method refinement stops. Passing does not authorize confirmation; planner review and a separate freeze/authorization are still required.

## Cross-family compute audit

A development-only compute audit is authorized. It must use only already-exposed DEV80 inputs and prospectively choose any calibration input without method outcomes. It may rerun method trajectories solely for compute instrumentation.

Primary accounting is FLOP/work based, not wall-clock matching. The audit must:

- preserve the exact pinned DAPS, SITCOM, and NP protocols used in DEV80;
- report dynamic FLOPs counted by PyTorch dispatch instrumentation where supported;
- preserve unsupported-operation diagnostics (including FFT counts/shapes and optimizer-step counts) rather than silently treating them as zero-cost;
- report DAPS-1/SITCOM-1 and their independent x4 work ceilings separately;
- report PE3/NP work at its actual 8,800-UNet protocol;
- label wall/GPU-active time as diagnostic only;
- avoid claiming exact cross-family FLOP equivalence when instrumentation has unsupported operators.

## Scope boundary

Authorized:

- the same frozen DEV80 only;
- exactly `NP_PE3_SCORE` and `NP_PE3_RANDOM`;
- development-only compute/FLOP audit;
- reuse all existing DEV80 locked measurements and baseline/NP4 results.

Not authorized:

- any of the 305 confirmation images or confirmation measurements;
- C1-only extra exposure;
- another PE3/EPP/DPS/checkpoint/score sweep;
- changing the prospective NP4 gate after seeing PE3;
- merge/rebase/squash/retarget/force-push/history rewrite;
- modification of PR #37.

Global B24 process/group ceiling remains 52,452 MiB. Workers must never kill or evict unrelated jobs.
