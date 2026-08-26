# B24.0 signoff and B24.1 authorization

Planner decision: **B24.0 SIGNED OFF** at closeout head `0ed429cf579ec201c1f9b3dbd6c531f46a4e3ea3`.

The signed B24.0 exposure freeze is `manifests/b24/PRE_B24_EXPOSURE.csv`, 333 rows, SHA-256 `d475c9c29b4f6ab2839ae21f4b19e33a52fa46f2fd7f0a6a7c5fff491e4b3068`. Its compact zero-GPU evidence is `docs/b24/evidence/B24_0_closeout_20260826T013106Z/`. No GPU, model, measurement-generation, reconstruction, or scientific-screening work occurred in B24.0.

## Resource-contract amendment

The accidental 60-GiB/56-GiB text is superseded. Freeze the originally agreed shared-GPU policy:

- B24 aggregate worker/group hard ceiling: **52,452 MiB**, the integer-MiB ceiling below 55 decimal GB;
- normal B24 target: **48,000 MiB**;
- device reserve: **4,096 MiB**;
- minimum free memory immediately before a B24 launch/group: **52,096 MiB**;
- B24.1 concurrency planning budget: **44,000 MiB** so scheduling is chosen conservatively below the normal target;
- explicit fixed physical GPU IDs/UUIDs only;
- sharing with other lab jobs is allowed when memory permits;
- never dynamically select a different GPU and never kill/evict another job.

## B24.1 authorization

**AUTHORIZED NOW:** exposed-image serial-reference versus concurrent-independent-process equivalence plus memory/throughput smoke. This stage may execute Fresh1 and SITCOM-1 on already exposed B23.1 locked inputs only. It must use four independently preregistered B24 solver seeds per method and must not generate a new measurement.

For baseline protocol engineering, “concurrent” means multiple independent native single-trajectory solver processes sharing one explicitly assigned physical GPU. It is not solver-internal tensor batching and is not claimed as B24 method novelty. The scientific novelty remains downstream NP-native branch generation, branch survival/drop, early pruning, and compute reallocation.

Bitwise/content-hash terminal equality between serial and concurrent execution is the preferred B24.1 equivalence criterion and supersedes the numerical envelope when achieved. A memory or equivalence failure stops progression.

## Conditional next action

If B24.1 passes, the executor is directed to move immediately to the frozen **64-image DAPS-4/SITCOM-4 baseline screen** so that A/B/C/D prevalence and measured throughput can be used to estimate the screen size and wall-clock time required to obtain the class quotas. This is a conditional progression instruction, not permission to bypass a failed B24.1 gate.

Method development and any B24 method execution on held-out rows remain unauthorized.
