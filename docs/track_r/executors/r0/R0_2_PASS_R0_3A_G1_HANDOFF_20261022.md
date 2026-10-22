# Track R R0.2 PASS and R0.3a canonical G1 GPU pilot — 2026-10-22

## PAC evidence: R0.2 successfully completed

- Run: `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_autopilot_20261022T180315Z`
- Verified clean executor head: `45d63071b808193f3cae776d9add8782e0e14adf`
- R0.1: **21/21 CPU tests PASS**.
- R0.2 manifest: **2520/2520 candidates**, across S1 (1400), S2 (960),
  S3 (128), S4 (32). 196 distinct source/image pairs.
- Manifest SHA256: `8ab7dc375a39a94d1f5183dd58d82f6d5c1c20a3d7124c7249df6a685c5ef890`
- G0 CPU: **2520/2520 recorded PSNR replay matches** at the
  frozen 0.01 dB tolerance; 196/196 source GT/measurement checks
  in the 0.0495–0.0505 PASS window; no pairing WARN or FAIL.
- FFT centered-impulse witness, orbit invariance, forbidden-path gates PASS.
- Autopilot final status:
  `PAUSED_GPU_G0_DETERMINISM_AND_R0_3_EXECUTOR_NOT_IMPLEMENTED`.
- **No GPU inference performed so far**.

This is **partial G0 only**. The required 8-candidate, two-physical-GPU
prior-score determinism comparison has not been performed. G1 has not
yet been tested. Neither full scoring, FAST generation nor R0
evaluation is authorized based solely on the completed CPU preflight.

## Important scorer correctness repair

The original R0.1 `PriorScorer` batched **only over eight orientation
members**, so both `batch_size=8` and `batch_size=64` passed
eight images per UNet call. That would make the pre-registered
8-versus-64 determinism gate illusory.

The revised scorer batches across the frozen (timestep, draw, orbit)
tuples, while CPU-noise seeds and paired orbit noise remain identical.
`batch_size=8` and `64` now exercise genuinely different inference
batch shapes, preserving the 256 individual image forwards and
per-(t,j) paired loss definition. Synthetic regression checks assert
the distinct observed batch/call counts.

**No science parameter/score definition was changed.** Full GPU
determinism validation remains pending; CPU tests on this change have
not yet been run on PAC.

## New fail-closed R0.3a G1-only pilot

`scripts/track_r/r0_3a_g1_gpu.py` takes the precise
`S3_a12|DAPS|68539|rep2` canonical image from the validated manifest.

Preflight: clean and pinned Git head; paired G0 CPU receipt and manifest
SHA; pinned `ffhq_10m.pt` SHA; pinned DiffFPR HEAD; exactly one idle
Blackwell physical GPU; cache/output confined to Track R.

Checks the source's canonical RGB8 raw ~6.364 dB and global-180
~31.871 dB before inference. Loads the strict guided-diffusion FFHQ
checkpoint and verifies the frozen schedule, then scores the eight
symmetry members with deterministic CPU noise, 8 timesteps × 4 draws,
batch_size=8 (256 forward images). It applies the frozen prior
argmin/tie rule and uses GT **offline only to check** that the selected
member reaches 25 dB. Saves a receipt whether it passes or fails.

**If G1 fails, the executor stops and escalates to the planner.**
R0.3a by itself does not imply full G0, the complete eight-candidate
R0.3 smoke, or FAST readiness. Stage R0.3b dual-GPU batch determinism,
UnivFD eight-orbit smoke, FAST native replay and compute projection
must all pass before R0.4–R0.6.

R0's fixed 16 aggregate GPU-hour ceiling remains in force.
