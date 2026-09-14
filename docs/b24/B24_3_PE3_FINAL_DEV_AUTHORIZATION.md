# B24.3 final DEV-only protected-explorer authorization

Planner/user authorization (2026-09-13):

> Authorize B24 DEV-only final protected-explorer refinement plus cross-family FLOP audit. Implement and run exactly PE3_SCORE and PE3_RANDOM on the existing DEV80 measurements at 8,800 NP UNet evaluations each; perform compute accounting using development data only. No confirmation exposure. If neither PE3 arm passes the prospectively frozen NP4 gate, stop B24 method refinement.

## Final status

This stage is complete.

- `NP_PE3_SCORE`: completed on all 80 frozen development measurements.
- `NP_PE3_RANDOM`: completed on all 80 frozen development measurements.
- Cross-family development-only dispatch-supported FLOP audit: complete.
- Confirmation exposure: none.
- Passing PE3 arms under the prospectively frozen NP4 gate: none.

Binding scientific decision:

`STOP_B24_METHOD_REFINEMENT`

The completed summary reported that both PE3 arms satisfied only the nonnegative-median-delta condition and failed the Good25-count, Good25-rescue/harm, and >=5 dB rescue/harm conditions. This verdict must not be changed retrospectively.

## Exactly authorized project-method arms

Only two new methods were executed:

- `NP_PE3_SCORE`
- `NP_PE3_RANDOM`

Both used exactly 8,800 total NP UNet evaluations per image, including four root initializations. No checkpoint sweep, score-window sweep, extra method variant, or additional root budget was authorized.

Both methods shared the fixed schedule:

1. initialize four frozen NP roots;
2. transitions 0..71: four roots, `k=5` proposals per root;
3. immediately before transition 72, retain exactly three roots and assign one retained root the explorer role;
4. transitions 72..299: explorer uses `k=10`; the two protected roots remain native `k=5` trajectories;
5. there is no pruning at 148 or 300;
6. at projection start, preserve one native continuation from each of the three retained lineages and add exactly one deterministic hard-phase fork from the explorer;
7. transitions 300..998: four terminal trajectories, `k=1` each;
8. clean-free terminal selection remains the frozen post-projection measurement-side LF-MSE selector; GT-best terminal remains offline oracle only.

`NP_PE3_SCORE` used the frozen trailing-32 LF measurement score at checkpoint 72: drop the worst-scored root and designate the best-scored retained root as explorer. Stable lineage tie-break applied.

`NP_PE3_RANDOM` used no measurement score for the checkpoint-72 role assignment. Frozen domain hash `B24_METHOD_PE3_RANDOM_ROLE_V1` deterministically ordered the four roots: the first became explorer, the next two protected, and the fourth dropped.

Exact work accounting per image:

`4 + 72*4*5 + 228*(10+5+5) + 699*4 = 8,800` total UNet evaluations.

## Prospectively frozen advancement gate versus NP4

Each PE3 arm was evaluated on the same 80 already-exposed development measurements against the already-completed `NP4_INDEPENDENT` result, using canonical raw-orientation 8-bit RGB PSNR. An arm could advance only if all four conditions held:

- median paired PSNR delta versus NP4 >= 0 dB;
- PE3 Good25 count at least NP4 Good25 count;
- Good25 rescues at least Good25 harms;
- >=5 dB rescues at least >=5 dB harms.

Neither arm passed every condition. Therefore B24 method refinement stopped prospectively as specified.

## Cross-family compute audit result

The development-only audit completed on a prospectively hash-selected DEV80 calibration image. It recorded PyTorch dispatch-supported dynamic FLOPs and retained explicit unsupported-work guards.

Dispatch-supported dynamic FLOPs:

- DAPS-1: `840267171895544`
- DAPS-4 independent equivalent: `3361068687582176`
- PE3_SCORE: `3413820886220800`
- SITCOM-1: `387934191616000`
- SITCOM-4 independent equivalent: `1551736766464000`

Ratios relative to PE3_SCORE were approximately 0.2461, 0.9845, 0.1136, and 0.4545 respectively.

These are not asserted to be exact total FLOPs. Unsupported Fourier/custom kernels remain nonzero and must be inventoried separately before any stronger cross-family compute claim.

## Subsequent authorized work

The only subsequent B24 work currently authorized is zero-GPU development closeout/analysis:

- derive historical Fresh2 from stored DEV80 DAPS trajectories;
- build complementarity/failure-overlap summaries;
- close the compute audit with explicit unsupported Fourier/operator-work accounting;
- package the B24 negative development result.

See:

- `docs/b24/B24_3_ZERO_GPU_CLOSEOUT_AUTHORIZATION.md`
- `configs/b24/b24_3_zero_gpu_closeout.json`

## Scope boundary

Not authorized:

- any of the 305 confirmation images or confirmation measurements;
- any new measurement generation;
- any further NP/EPP/PE3/DPS/checkpoint/score/schedule sweep;
- changing the prospective NP4 gate or the resulting stop verdict;
- C1-only method development;
- merge/rebase/squash/retarget/force-push/history rewrite;
- modification of PR #37.

Confirmation remains locked.
