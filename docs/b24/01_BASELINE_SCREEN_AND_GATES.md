# B24 baseline screen, integrity gates, and execution runbook

## Frozen source identities

The B24 baseline wrapper must verify source state before scientific execution.

| source | HEAD | frozen identity |
|---|---|---|
| DAPS | `e7a77d094167084faed19b599b96673b7bb11447` | tree `e63f9715e4704d9cd7a43a166559496d9d94e781`; index `d5487cdba570dbaac0c1909e549da361a0a0fc3fed81e5c13f59fa12925876b6`; tracked diff `fbb5b42369ecf0d3b9b67f8fc162053bc40ec32aed41dbd92a67e8d81dcfad69` |
| official SITCOM | `275ab67efbd8146bffca20155171ba6be1169c09` | tree `80263442e3606824a06dc003504c28da5c59c2c5`; index `3ef63a8a29d0ba65cc642027a57ec102257fd9b387b0e9a5b4aae7f46d6a949f`; tracked diff `a9f0076d6f852b6898000142c19a09131ffc49ceba0e3d935cd465e85df26e6e` |
| NP/SITCOM | `52f2c37e587576d02e2b27ac971e247f2899fc5e` | tree `cfba2bbce7053d3e8642e5d74a603794bfb56bb2`; index `d0b3e024dee15358dab4ac9350b34f2cd8009eca4df8e4d9fd32a79829cc354e`; tracked diff empty SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

DAPS and official SITCOM are intentionally not assumed clean. Their frozen tracked state is part of provenance. FFHQ checkpoint `/egr/research-pac/huang248/models/ffhq_10m.pt` has SHA-256 `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`.

Control environment is `prdiff_ffhq`; native solver children use their frozen DAPS or SITCOM environment. B24.1 sets `PYTHONDONTWRITEBYTECODE=1` for native children so tracked Python caches are not changed by execution.

## Baseline protocol

For each screened image: one measurement seed is frozen before outcomes; one locked measurement is reused by both methods; four DAPS and four SITCOM solver seeds are independently domain-derived; DAPS-4 is four independent Fresh1 trajectories; SITCOM-4 is four independent SITCOM-1 trajectories; every terminal gets raw RGB PSNR, SSIM, LPIPS, source/config/input hashes, GPU-active time, wall time, and terminal hash; best-of-four is selected only by raw-orientation RGB PSNR; A/B/C/D uses Good25 only.

Historical B22 `SITCOM-4S` is **not** B24 SITCOM-4. It restores one post-model RNG state and consumes a sequential four-trajectory RNG stream. B24 requires four independently preregistered SITCOM-1 seeds.

## B24.1 serial/concurrent equivalence gate — AUTHORIZED

B24.1 uses an already exposed B23.1 image and accepted locked measurement; it generates no new measurement. It runs the same four B24 solver seeds per method twice:

1. serial native single-trajectory reference;
2. controlled concurrent independent native processes on the same explicit GPU.

Concurrent baseline execution is scheduling/throughput engineering only. It is not solver-internal tensor batching and is not claimed as B24 novelty.

Preferred pass criterion is exact terminal-content equality for every seed. Exact equality supersedes the numerical envelope. If a future implementation genuinely cannot be bitwise/content-hash identical, the already frozen fallback envelope remains: terminal max absolute error <= `1e-5`, relative L2 <= `1e-6`, PSNR difference <= `1e-4 dB`, SSIM/LPIPS difference <= `1e-6`, identical Good25/26/28 decisions, candidate count, and seed correspondence. The present B24.1 harness fails closed on terminal-hash disagreement rather than relaxing post hoc.

## Corrected memory/throughput gate

The accidental 60-GiB/56-GiB policy is superseded. Freeze:

- aggregate B24 worker/group hard ceiling: **52,452 MiB**;
- normal target: **48,000 MiB**;
- device reserve: **4,096 MiB**;
- minimum free immediately before each launch/group: **52,096 MiB**;
- B24.1 concurrency-planning budget: **44,000 MiB**.

B24.1 first measures four serial references. For each method it takes the largest measured per-candidate PyTorch peak reserved memory and chooses the largest concurrency in `[1,4]` whose conservative `concurrency × serial_peak_reserved` fits the 44,000-MiB planning budget. It then replays the same four seeds at that width.

Record per-candidate PyTorch peak allocated/reserved MiB, aggregate B24 process memory from `nvidia-smi`, whole-device used/free samples, GPU-active time, group wall time, selected concurrency, and serial/concurrent speedup. Concurrent observed B24 process memory and conservative peak-reserved projection must stay <=48,000 MiB. Any aggregate B24 process observation above 52,452 MiB terminates only the B24 child processes created by that worker and fails the gate. Other lab processes are never killed.

For speed, the authorized B24.1 launcher assigns DAPS to fixed physical GPU 0 and SITCOM to fixed physical GPU 1 and runs the two method smokes simultaneously. Each method remains internally serial-then-concurrent on its own assigned GPU. No dynamic GPU selection is permitted.

## B24.2 conditional 64-image gate

If and only if both B24.1 method summaries pass exact equivalence and corrected memory gates on the same exposed locked measurement, the next executor action is the frozen 64-image DAPS-4/SITCOM-4 baseline screen.

The 64-image result must report class counts A/B/C/D, continuous metrics, Good25/26/28 sensitivity, measured effective DAPS-4 and SITCOM-4 throughput, and class prevalence. Those observations are then used to estimate the screen size and wall time required for 100 development examples per class and, if feasible, an additional 100 held-out examples per class. The Good25 threshold is never changed to fill quotas.

## Atomic completion and resume

Each terminal candidate is reusable only after hashes and metrics validate. Completion records are written atomically. Resume verifies run-manifest, source, measurement, seed, and candidate identities before skipping. Completed candidates are never silently regenerated. Retries are bounded and recorded.

## Four-GPU sharding for scale

The scale manifest is hash-frozen before launch. Four shards are deterministic by row position modulo four after frozen screen ordering:

- GPU 0 `GPU-8c9c6250-7b65-20d8-5c81-d6cb618810c3`
- GPU 1 `GPU-883c037a-34d2-48c4-467f-9a352fd8fdff`
- GPU 2 `GPU-c381c0f4-1dbc-004f-7d3a-1d7f7794dffe`
- GPU 3 `GPU-7d65c050-d7e8-5a6b-ee38-1d72d7a5696a`

Shard sets must be disjoint and their union exactly equal the run manifest. One B24 top-level worker owns each physical GPU; candidate-process concurrency is internal to that worker and bounded by the corrected memory policy.

## Storage policy

Never store full DAPS trajectories in the B24 scale sweep. Retain metrics, hashes, timing, and provenance for every terminal; best DAPS and best SITCOM terminal for every image; all eight terminals for B/C/D; all eight terminals for deterministic 10% A audit subset; measurement identity/seed and enough locked input state to reproduce every row.

## Immediate-stop conditions

Stop on B24 source ancestry mismatch; source/model/input identity mismatch; PRE_B24 exposure mismatch; future-reserve overlap; seed collision; shard overlap/omission; completed-record hash mismatch; unrecorded retry or silent regeneration; prelaunch free memory below 52,096 MiB; normal-target/hard-cap violation; OOM; serial/concurrent disagreement; or any runtime path that lets clean ground truth affect branch dropping, allocation, survival, or routing.
