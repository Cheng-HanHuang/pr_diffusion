# A1.0d — CPU-only synthetic restart-plan preflight

**Date:** 2026-10-17. **Status:** ready for user-run **CPU test only**. No native method invocation, no real-image data, no GPU work. Builds on A1.0 V2 source/metadata PASS and A1.0c DAPS `get_start` source-confirmed seed override.

## What this validates

- Four distinct seeds per method per image: DAPS, SITCOM, NP-LF and NP-S2.
- `16` distinct full candidate identities/image; `128` intended trajectories for an **as-yet-unallocated** 8-image development cohort, but **does not** generate or select those images.
- One identical measurement-seed identity per image, separate domain from solver seeds. The actual noisy measurement does not exist and is never accessed.
- NP fixed mixed4 (2LF+2S2) / mixed8 (4LF+4S2) as **offline views** of the exact same LF and S2 candidate identities; `np_mixed8_extra_runs=0`.
- Per-native-run process isolation required; GT access during solver execution forbidden; native commands marked `NOT_READY_SOURCE_PREFLIGHT_PENDING` so no Python from this step can accidentally launch real models.
- Seed32 mapping for legacy NumPy/PyTorch/hydra entrypoints; seed63 retained for domain/provenance audit. Failure on duplicated candidate, modified measurement seed, false extra command, tampered NP mixture or invalid IDs. Only invented `10000`, `10001` examples appear in the two-row summary—they are **synthetic identifiers**, not proposed experiment cohorts.

**Scripts:**
- `scripts/track_a/a1_restart_plan.py`
- `scripts/track_a/test_a1_restart_plan.py`
- `scripts/track_a/a1_run_stage0d_cpu.sh`.

The runner first reruns all A1 protocol and source-preflight synthetic tests, then writes one tiny deterministic manifest *summary* and packages the logs. It uses `CUDA_VISIBLE_DEVICES=""`, imports Python stdlib only and cannot execute native DAPS/SITCOM/NP entrypoints. The exact output manifest intentionally omits real GT, tensors, image paths or measurement files. New exclusive return files inside:

`/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016/`:

- `A1_0D_RESTART_SYNTHETIC_TESTS.log`
- `A1_0D_RESTART_SYNTHETIC_SUMMARY.json`
- `A1_0D_RESTART_SYNTHETIC_RETURN.zip` with printed SHA-256.

No previous A1 V1/V2 or historical source artifact is overwritten. The command must pin source HEAD and refuse dirty tracked files.

## What it does NOT prove or permit

A synthetic plan is not proof of bitwise replay or native RNG independence of the full diffusion trajectory; that requires verified per-process execution on an accepted **development-only** input. This step does not implement a live native launcher, freeze a real measurement, choose eligible real images, reserve hardware, or read the locked B24 confirmation305 registry. The protected confirmation cohort is not guaranteed disjoint merely by excluding B21/B22 final100 and B24 DEV80's combined 180 IDs.

After clean dry-run receipt, the next stage is A1-owned native process wrappers, operator/metric parity checks and **source-independent cohort eligibility governance**, then [DATA RESERVATION] and [RUN RESERVATION] in issue #41 before any explicitly user-executed bounded GPU pilot. Track B owns contested GPUs.

**Study decision:** `GO_CPU_PROTOCOL_SMOKE`; `NO_PROSPECTIVE_COHORT`; `NO_GPU_LAUNCH_AUTHORIZATION`.
