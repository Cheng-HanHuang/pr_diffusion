# B24 start here

## Status and authorization boundary

B24 is a new study isolated from B23. The required ancestry point is the signed-off B23.1 final head `27505e6328157ac9296c95dc5e611cbeef80de98`. B23 cross-family H0 failed: NP-1 and SITCOM-1 are `BASELINE-ONLY` across family boundaries and no NP/SITCOM cross-family adapter qualified. B24 does not reinterpret that result.

B24.0 is zero-GPU only. It may inventory and freeze contracts, create the separate worktree/output layout, construct exposure/allocation/seed registries, implement launch/resume/sharding tooling, and run CPU-only tests/dry rendering. B24.0 does **not** authorize a model load, measurement generation, reconstruction, batch-equivalence smoke, 64-image baseline tranche, method development, or held-out method execution.

Separate identities:

- branch: `codex/b24-bestof4-failure-sweep`
- required ancestry point: `27505e6328157ac9296c95dc5e611cbeef80de98`
- draft PR base: `codex/b23-execution`
- PAC worktree: `/egr/research-pac/huang248/pr_diffusion_b24`
- PAC output root: `/egr/research-pac/huang248/outputs/pr_diffusion/b24`

Do not modify PR #37, `/egr/research-pac/huang248/pr_diffusion_b23`, or the B23 output root.

## Scientific claim

The evaluation question is whether an **NP-native batched branch-and-prune method** can recover cases where matched DAPS and SITCOM best-of-four protocols fail.

Best-of-four matching is an apples-to-apples protocol, not the novelty. The intended method novelty is NP-native proposal/branch generation, batched starts, early branch dropping, reallocation of saved compute, branch-survival decisions, the location of dropping, and eventually clean-free terminal selection.

Ground truth may select the best **terminal** reconstruction during initial oracle candidate studies. Ground truth must never decide early dropping, branch allocation, survival, routing, or runtime stopping.

## Execution priority

The first expensive scientific task is the baseline screen, not method development:

1. freeze exposure and deterministic FFHQ screening order;
2. generate one locked measurement per screened image;
3. execute DAPS-4 and SITCOM-4 using preregistered independent solver seeds;
4. compute raw-orientation RGB PSNR, SSIM, LPIPS and continuous per-run metrics;
5. classify each image by the primary Good25 contract into A/B/C/D;
6. expand cumulative screening tranches only after throughput, prevalence, and integrity review;
7. discuss and freeze the NP-native branch/drop method only after useful baseline class allocation exists;
8. do not execute our method on held-out rows before that policy is frozen.

The first future baseline tranche is 64 images. The 256-image tranche is cumulative and reuses the same 64 rows. Larger tranches are cumulative prefixes of the same deterministic screen order.

## Primary classes and metrics

`Good25 := raw_orientation_rgb_psnr_db >= 25.0`.

- A: DAPS-4 Good25 and SITCOM-4 Good25.
- B: only SITCOM-4 Good25.
- C: only DAPS-4 Good25.
- D: neither Good25.

Good26, Good28, SSIM, LPIPS, and every continuous per-run metric are recorded for sensitivity analysis. They do not redefine the primary classes to fill quotas.

Target balanced development allocation is at least 100 images per class. Prefer a second held-out 100 per class if prevalence permits. Balanced panels are not FFHQ population estimates; preserve a natural-prevalence evaluation and prevalence-weighted reporting.

## Compute accounting

DAPS and SITCOM are not fictitiously equal-cost:

- DAPS-4-equivalent arm: at most four Fresh1-equivalent work units.
- SITCOM-4 efficiency arm: four SITCOM-1 units, reported separately.
- Four-terminal-candidate matching: reported separately.
- Diagnostic union oracle `max(DAPS-4, SITCOM-4)`: eight baseline trajectories.

Every scientific run must report work-FRE, GPU-active time, wall time, terminal candidate count, and branch count. Full DAPS trajectories are prohibited in the B24 scale sweep.

## Exposure and allocation

The populated PAC `PRE_B23_EXPOSURE.csv` is the only accepted source for the 328 pre-B24 historical images. Its frozen SHA-256 from the B24.0 inventory is `a513cb4e3b79b39700ff1d623cb4b2eaf496bc2d6d0fe58bd963709e6a56d288`. The GitHub B23 placeholder is empty and must not be mistaken for the populated PAC registry.

B24 appends image-wide exclusions for `65082`, `61492`, `62959`, `66821`, and `68142`. `PRE_B24_EXPOSURE.csv` is invalid unless it has at least 333 unique image IDs and contains all five.

Unexposed FFHQ images are assigned by a domain-separated SHA-256 rule:

- hash domain `B24_FFHQ_GLOBAL_V1`;
- bucket 0-79: `B24_SCREEN_ELIGIBLE`;
- bucket 80-99: `FUTURE_RESERVE`.

The future reserve is untouched by B24. Baseline screening uses a second deterministic hash order inside `B24_SCREEN_ELIGIBLE`. Once A/B/C/D labels exist, a third domain-separated hash ranks rows within each class: first 100 are balanced development, next 100 are held-out when available, and remaining rows retain natural-prevalence/evaluation roles. No PSNR magnitude or manual visual appeal may influence the ranking.

## Resource policy

Hardware is four NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs. B24 uses explicit physical GPU IDs/UUIDs and at most one B24 worker per physical GPU. It never dynamically chooses a GPU, kills another process, or infers permission merely from free memory.

The user confirmed coexistence with other lab jobs is allowed when VRAM permits. Freeze:

- B24 worker hard ceiling: **52,452 MiB** (55 decimal GB rounded down in MiB);
- B24 normal target: **48,000 MiB**;
- pre-launch fit requirement: requested B24 target plus **4,096 MiB device reserve** must be free;
- both PyTorch peak allocated/reserved memory and whole-device `nvidia-smi` process memory are recorded.

Other lab jobs may start later. B24 does not evict them. OOM, B24-process hard-cap violation, source mismatch, or serial/batched disagreement is an immediate scientific stop.

## Future stage gates

B24.1 (not authorized): exposed-image serial-versus-batched equivalence plus memory/throughput smoke.
B24.2 (not authorized): frozen DAPS-4/SITCOM-4 baseline screen beginning with 64 images, then 256, then larger cumulative tranches only after review.

Read `docs/b24/01_BASELINE_SCREEN_AND_GATES.md` before any future GPU authorization.
