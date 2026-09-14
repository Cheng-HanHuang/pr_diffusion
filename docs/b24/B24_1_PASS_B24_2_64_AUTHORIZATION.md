# B24.1 PASS and B24.2 64-image authorization

## Decision

B24.1 passed on PAC and the planner/user directed immediate progression to the first large baseline tranche.

Accepted B24.1 PAC run:

- run root: `/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_1_smoke_20260826T023711Z`
- exposed image: `65082`
- same locked measurement in serial and concurrent execution: **YES**
- overall gate: **PASS**
- next recommendation: `PASS_TO_64_BASELINE`

Observed method gates:

| method | planned concurrency | exact terminal hash equivalence | memory pass | serial wall s | concurrent wall s | speedup | max observed B24 process MiB |
|---|---:|---|---|---:|---:|---:|---:|
| DAPS | 4 | PASS | PASS | 2313.2794 | 478.4246 | 4.8352x | 6632 |
| SITCOM | 4 | PASS | PASS | 1037.4448 | 321.3830 | 3.2281x | 6536 |

B24.1 validates concurrent independent native single-trajectory processes on one GPU. It does not validate or claim solver-internal tensor batching and is not the NP-native method novelty.

## B24.2 authorization

Authorized now:

- exactly the deterministic first **64** rows of the B24 screen order frozen in B24.0;
- one fresh locked B24 measurement per row using the preregistered B24 measurement seed;
- DAPS-4 as four independent Fresh1 terminal-only runs;
- SITCOM-4 as four independent SITCOM-1 runs;
- concurrency 4 within each method on the row's fixed physical GPU, matching the B24.1-passed scheduling form;
- four deterministic shards, 16 images per fixed physical GPU;
- raw-orientation RGB PSNR as the best-of-four selection metric and Good25 class threshold;
- Good26, Good28, SSIM, LPIPS and continuous per-terminal metrics for reporting/sensitivity;
- A/B/C/D prevalence and end-to-end throughput aggregation after all 64 rows complete.

The common B24.2 terminal evaluation representation is frozen before fresh outcomes as `CANONICAL_SAVED_RGB_8BIT_RAW_ORIENTATION_V1`: both DAPS and SITCOM are evaluated from their canonical saved 8-bit RGB terminal PNGs against the same quantized 256x256 ground-truth RGB. This avoids evaluating SITCOM from a higher-precision representation than terminal-only DAPS. Best-of-four selection uses only raw RGB PSNR.

## Still not authorized

- cumulative 256-image expansion or any larger prefix before review of the 64-image prevalence/throughput/integrity summary;
- NP-native method-development execution;
- held-out method execution;
- changing Good25 to fill class quotas;
- historical B22 `SITCOM-4S` as a B24 baseline reference;
- merge/rebase/squash/retarget/force-push/history rewrite.

The scale launcher must fail closed on B24.1 gate mismatch, source identity mismatch, frozen 64-prefix mismatch, LPIPS evaluator unavailability, GPU UUID/free-memory mismatch, per-method memory hard-cap violation, or any missing/invalid terminal artifact.
