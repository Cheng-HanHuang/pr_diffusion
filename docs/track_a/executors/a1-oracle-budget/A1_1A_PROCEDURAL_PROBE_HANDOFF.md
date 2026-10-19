# A1.1a — procedural input / native Fourier measurement CPU integration probe

Date: 2026-10-18. **Scope:** one wholly newly generated synthetic RGB image, and one frozen noisy observation from the actual native DAPS `PhaseRetrieval.measure`. CPU-only; **NO diffusion model, checkpoint loading, GPU kernels, native solver execution, FFHQ/data cohort allocation or protected B24 confirmation305 access**.

## Why now

A1.0 V2 native source/checkpoint metadata passed and A1.0d synthetic independent-restart manifest passed 14/14 on PAC. Repeated bookkeeping tests are not informative; the next worthwhile gate is real input/observation structure compatible with the pinned Fourier phase retrieval operator. This synthetic engineering probe is independent of all private/protected and already-exposed cohorts. It is **not** an FFHQ distribution experiment, so no paper-quality claims can be extracted from it.

## Frozen, source-pinned procedure

- Source: `scripts/track_a/a1_synthetic_probe_input.py`; tests `scripts/track_a/test_a1_synthetic_probe_input.py`; user runner `scripts/track_a/a1_run_synthetic_probe_input.sh`. Under the existing A1 branch only.
- Input: fully deterministic 256×256 8-bit RGB procedural test pattern comprised of waves and Gaussian blobs; no source images read, and no image/model/data downloads. Single special engineering probe label `A1_SYNTHETIC_PROCEDURAL_PROBE_V1`.
- Transform: same `ToTensor()*2−1` semantics for exactly 256×256 RGB uint8 PNG. Float32 GT shape 1×3×256×256.
- Observation: import **pinned native DAPS** `forward_operator.PhaseRetrieval` from A1.0-verified native source; call `PhaseRetrieval(oversample=2.0,resolution=256,sigma=0.05).measure(x)` once on **CPU** with fixed measurement seed `12061018`, shared single observation across methods in future smoke. Validate float32 finite shape (1,3,384,384) and preserve **signed** raw noisy observation. NP solver will explicitly clamp negatives only in memory, matching its historical method identity.
- Identity: native DAPS HEAD and **full tracked diff** SHA checked before importing native code, A1 checkout SHA pinned. `CUDA_VISIBLE_DEVICES=""`, CUDA availability/initialization check. This project-specific statement does not imply CPU-generated noise has the same exact RNG stream as a GPU-run generator; the frozen saved observation is the actual common tensor.
- Output exclusive A1 root: `.../outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016/a1_1a_synthetic_probe_v1/`. Contains `synthetic_probe.png`, `ground_truth_tensor.pt`, `measurement.pt` and `A1_1A_PROBE_MANIFEST.json` with source SHA, tensor byte/content SHA, measurement seed and schema.
- Return ZIP: `A1_1A_PROBE_METADATA_RETURN_V1.zip` has **only** the test log, run log and small manifest JSON; no image/GT/measurement bytes. Previous A1 0d and stage0 evidence is immutable; runner refuses existing output paths and preserves controlled STOP receipts.
- Python: `/egr/research-pac/huang248/conda-envs/daps/bin/python`, stdlib + installed NumPy/PIL/PyTorch/native operator. No native reconstruction calls, only one operator measurement on CPU; no GPU reservation needed.
- Tests: deterministic RGB bytes, correct model-range tensor, dtype/shape content hash, rejection of malformed input. Test code synthetically checked in local CPU scratchpad for shape/value bounds; the exact pinned PAC unit tests and native measurement path have **not** run yet.

## PASS / STOP

PASS requires tests return 0, native source HEAD/full diff unchanged, CPU operator import and noisy measurement generation return 0, no CUDA initialization, finite tensor of expected shape, and metadata/ZIP hashes returned. If native operator cannot run on CPU, STOP and keep logs; **do not silently substitute a different FFT/magnitude scaling or use GPU**. The next stage would require a separately bounded GPU operator smoke or an approved corrected native CPU wrapper, not an invented measurement.

This probe is an **engineering-only source** with no FFHQ qualification. Before performing even one native DAPS/SITCOM/NP diffusion run, separately implement and review A1-owned native execution adapters, account seed and full model/FFT work, reserve actual GPU/time/memory in issue #41 and have the user run the frozen command. For a prospective FFHQ paper-quality study, still need a legitimate disjoint source-approved cohort, not merely exclusions of exposed B21/B22 and DEV80. Track B owns first priority on contested GPUs.
