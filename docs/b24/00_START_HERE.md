# B24 start here

## Current status and authorization boundary

B24 is isolated from B23 and descends from signed-off B23.1 final head `27505e6328157ac9296c95dc5e611cbeef80de98`. B23 cross-family H0 failed: NP-1 and SITCOM-1 remain `BASELINE-ONLY` across family boundaries and no NP/SITCOM cross-family adapter qualified. B24 does not reinterpret that result.

**B24.0 is planner-signed off** at `0ed429cf579ec201c1f9b3dbd6c531f46a4e3ea3`. Its exposure freeze has 333 rows with SHA-256 `d475c9c29b4f6ab2839ae21f4b19e33a52fa46f2fd7f0a6a7c5fff491e4b3068` and compact evidence under `docs/b24/evidence/B24_0_closeout_20260826T013106Z/`.

**B24.1 is authorized now:** exposed-image serial-reference versus concurrent-independent-process equivalence plus memory/throughput smoke. It may execute Fresh1 and SITCOM-1 only on already exposed B23.1 locked inputs and must not generate a new measurement.

B24.2 scientific screening is the conditional next action only after B24.1 passes. Method development and held-out method execution remain unauthorized.

Separate identities:

- branch: `codex/b24-bestof4-failure-sweep`
- required ancestry point: `27505e6328157ac9296c95dc5e611cbeef80de98`
- B24.0 signed head: `0ed429cf579ec201c1f9b3dbd6c531f46a4e3ea3`
- draft PR base: `codex/b23-execution`
- PAC worktree: `/egr/research-pac/huang248/pr_diffusion_b24`
- PAC output root: `/egr/research-pac/huang248/outputs/pr_diffusion/b24`

Do not modify PR #37, `/egr/research-pac/huang248/pr_diffusion_b23`, or the B23 output root. Never rebase, squash, force-push, or rewrite B24 history.

## Scientific claim

The evaluation question is whether an **NP-native batched branch-and-prune method** can recover cases where matched DAPS and SITCOM best-of-four protocols fail.

Best-of-four matching is an apples-to-apples protocol, not the novelty. The intended method novelty is NP-native proposal/branch generation, batched starts, early branch dropping, reallocation of saved compute, branch-survival decisions, where dropping occurs, and eventually clean-free terminal selection.

Ground truth may select the best **terminal** reconstruction during initial oracle candidate studies. Ground truth must never decide early dropping, branch allocation, survival, routing, or runtime stopping.

## Baseline-first execution priority

The first expensive scientific task is the baseline screen, not method development:

1. freeze exposure and deterministic FFHQ screening order;
2. generate one locked measurement per screened image;
3. execute DAPS-4 and SITCOM-4 using preregistered independent solver seeds;
4. compute raw-orientation RGB PSNR, SSIM, LPIPS and continuous per-run metrics;
5. classify each image by the primary Good25 contract into A/B/C/D;
6. expand cumulative screening tranches after throughput/prevalence/integrity review;
7. discuss and freeze the NP-native branch/drop method only after useful baseline class allocation exists;
8. do not execute our method on held-out rows before that policy is frozen.

The first baseline tranche after B24.1 is 64 images. The 256-image tranche is cumulative and reuses those 64 rows. Larger tranches are cumulative prefixes of the same deterministic screen order.

## Primary classes and metrics

`Good25 := raw_orientation_rgb_psnr_db >= 25.0`.

- A: DAPS-4 Good25 and SITCOM-4 Good25.
- B: only SITCOM-4 Good25.
- C: only DAPS-4 Good25.
- D: neither Good25.

Good26, Good28, SSIM, LPIPS, and every continuous per-run metric are recorded for sensitivity analysis. They do not redefine the primary classes to fill quotas.

Target balanced development allocation is at least 100 images per class. Prefer a second held-out 100 per class if prevalence permits. Balanced panels are not FFHQ population estimates; preserve natural-prevalence evaluation and prevalence-weighted reporting.

## Compute accounting

DAPS and SITCOM are not fictitiously equal-cost:

- DAPS-4-equivalent arm: at most four Fresh1-equivalent work units.
- SITCOM-4 efficiency arm: four SITCOM-1 units, reported separately.
- Four-terminal-candidate matching: reported separately.
- Diagnostic union oracle `max(DAPS-4,SITCOM-4)`: eight baseline trajectories.

Every scientific run reports Work-FRE, GPU-active time, wall time, terminal candidate count, and branch count. Full DAPS trajectories are prohibited in the B24 scale sweep.

Historical B22 `SITCOM-4S` is not the B24 SITCOM-4 reference because it consumes one sequential four-trajectory RNG stream. B24 requires four independently preregistered SITCOM-1 seeds. B24.1 therefore compares serial and concurrent execution candidate-for-candidate under those independent seeds.

For baseline protocol engineering, B24.1 “concurrent” means independent native single-trajectory processes sharing one explicit GPU. It is throughput scheduling, not solver-internal tensor batching and not B24 method novelty.

## Exposure and allocation

The signed PRE_B24 exposure freeze contains the 328 pre-B23 images plus B23.1 image-wide exclusions `65082`, `61492`, `62959`, `66821`, and `68142`.

Unexposed FFHQ images are assigned by domain-separated SHA-256:

- `B24_FFHQ_GLOBAL_V1` buckets 0-79: `B24_SCREEN_ELIGIBLE`;
- buckets 80-99: `FUTURE_RESERVE`.

Future reserve is untouched by B24. Baseline screening uses a second deterministic hash order. Once A/B/C/D labels exist, a third domain-separated hash ranks rows within class: first 100 balanced development, next 100 held-out when available, remainder natural-prevalence/evaluation roles. PSNR magnitude or visual appeal cannot alter ranking.

## Corrected resource policy

Hardware is four NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs. B24 uses explicit physical GPU IDs/UUIDs and never dynamically chooses another GPU or kills/evicts another process. Sharing with other lab jobs is allowed whenever memory permits.

The accidental 60-GiB text is superseded. Freeze:

- aggregate B24 worker/group hard ceiling: **52,452 MiB** (integer-MiB ceiling below 55 decimal GB);
- normal target: **48,000 MiB**;
- device reserve: **4,096 MiB**;
- minimum free immediately before each launch/group: **52,096 MiB**;
- B24.1 conservative concurrency-planning budget: **44,000 MiB**;
- record PyTorch peak allocated/reserved memory and B24-process/whole-device `nvidia-smi` samples.

OOM, hard-cap violation, source/input mismatch, or serial/concurrent terminal disagreement is an immediate stop.

## Stage gates

B24.1: **AUTHORIZED** on exposed locked input only. Exact terminal-content equality is preferred and supersedes the numerical envelope when achieved; memory must remain within the corrected policy.

B24.2 64-image baseline: **CONDITIONAL NEXT ACTION AFTER B24.1 PASS**. The first 64 results are used to estimate A/B/C/D prevalence and, together with measured throughput, the total screening time required for class quotas.

Read `docs/b24/01_BASELINE_SCREEN_AND_GATES.md` and `docs/b24/B24_0_SIGNOFF_B24_1_AUTHORIZATION.md` before execution.
