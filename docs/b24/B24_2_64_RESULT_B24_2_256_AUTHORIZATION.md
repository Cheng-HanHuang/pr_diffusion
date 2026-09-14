# B24.2 64-image result and cumulative 256 authorization

## Completed 64-image checkpoint

Accepted PAC run root:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_2_64_20260826T040303Z`

All four deterministic shards completed the frozen 64-row manifest. Shard 2 required fail-closed resume after an infrastructure-only path-deduplication bug for image `00894` and later after unrelated jobs consumed GPU memory between B24 groups. Completed rows were reused only after frozen row/measurement/solver-seed identity validation; incomplete attempts were preserved.

Primary Good25 class counts over the complete fixed 64-image prefix:

| class | count | point prevalence |
|---|---:|---:|
| A | 57 | 0.890625 |
| B | 2 | 0.031250 |
| C | 3 | 0.046875 |
| D | 2 | 0.031250 |

The primary classifier remains unchanged. Good25 is still raw-orientation RGB PSNR >=25 dB under `CANONICAL_SAVED_RGB_8BIT_RAW_ORIENTATION_V1`.

The 64-image point estimates are too uncertain to freeze a final campaign size for rare classes. A naive plug-in estimate is about 3,200 total screened images for 100 B examples and about 2,133 for 100 C examples. At roughly 16 images/hour four-GPU clean throughput from the fixed-prefix run, 3,200 images would be about 200 wall-hours; this is planning-scale only, not a preregistered final N.

## Planner collection decision

For the immediate development collection, **B and C are the gating rare-class targets**. A is retained naturally and is already abundant. D is still classified and retained whenever encountered, but 100 D examples are no longer a gate for starting method development.

The immediate target is at least 100 B and 100 C development examples. A and D remain scientifically reported. No Good25 threshold, best-of-four rule, image order, measurement seed, solver seed, or class definition may be changed to fill quotas.

The deterministic within-class hash ranking remains the allocation mechanism for eventual development/held-out roles. The current scale stage collects baseline classifications only; it does not execute the B24 NP-native method on held-out data.

## Cumulative 256 authorization

Authorize the deterministic **cumulative 256-image prefix** now. The first 64 rows are immutable and must be reused as the prefix; execute only rows 64--255 as new scientific work.

This 256 checkpoint is intended to:

- refine B/C prevalence before committing to a multi-thousand-image final screen size;
- measure clean long-run throughput and PAC-sharing interruption rate;
- preserve all A/B/C/D baseline outcomes;
- permit immediate continuation of baseline gathering while method design remains separate.

After 256, the executor may prepare larger cumulative prefixes, but no held-out method execution is authorized by this document.

## Shared-GPU execution clarification

The 48,000 MiB normal target and 52,452 MiB hard ceiling are **resource policy bounds, not a CUDA reservation**. The B24.1 smoke observed only about 6.5 GiB of B24 process memory for four concurrent native candidates. Other PAC jobs can therefore start whenever actual free memory permits.

Do not allocate dummy tensors merely to occupy VRAM or block other users. For long baseline runs, wait/retry when the fixed GPU lacks the required free memory, preserve atomic image completions, and resume without rerunning valid completed rows. Additional cross-image concurrency may be benchmarked later only if it represents useful scientific throughput rather than artificial reservation.

## SITCOM-ODE provenance

B24 uses the pinned public SITCOM-ODE implementation/source identity as an external baseline. If the public implementation/configuration differs from the version that produced paper-reported numbers (including a scale change) and the pinned public code performs worse in this use case, document that discrepancy rather than silently tuning the external method to chase the paper table.

B24 therefore claims results for the pinned public SITCOM-ODE implementation/protocol used here; it does **not** claim exact reproduction of every number reported in the SITCOM paper.

## Still not authorized

- changing Good25 or any class definition to increase B/C/D counts;
- using historical B22 `SITCOM-4S` as B24 SITCOM-4;
- modifying the pinned external SITCOM-ODE code merely to improve B24 outcomes;
- NP-native method execution on held-out rows before the method policy is frozen;
- merge/rebase/squash/retarget/force-push/history rewrite.
