# Track R · R0.3b full G0 PASS, R0.3c FAST smoke handoff — 2026-10-22

## Received PAC R0.3b evidence

Run root:
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_3b_smoke_20261022T183528Z`

Executor head: `6767f1850235a3df0c94fea9b53135016dbe271f`.
All stages returned RC=0: frozen selection, batch8 GPU worker, batch64
GPU worker, CPU determinism adjudication. Two distinct physical GPUs:
1 (batch8 + UnivFD) and 2 (batch64). Eight exact same candidates.
Total measured GPU hours: `0.016863963068074857`.

Most discrepant prior per-t score:
`S1_b22|NP|64885|lf_seed100`, orbit `100`, t=700,
batch8=0.00037341981078498065 vs batch64=0.00037342002178775147,
relative diff `5.650547868353224e-07` (well below frozen
`1e-5`). The score, S_mid and per-t metrics passed.
G0 final status: `G0_FULL_PASS_R0_3B_NOT_FULL_R0_3`.

G1 was passed independently on A1.2 DAPS 68539 rep2:
GT-free selected orbit 111, 6.364→31.871 dB on the
previously exposed canonical case.

Measured from R0.3b smoke (not yet full work):
- Scoring the existing 2520: `2.5031679952924604` aggregate GPUh.
- Scoring the 720 future S5: `0.7151908557978459` aggregate GPUh.
- FAST generation remains unmeasured.

These measurements authorize **only** the final native FAST R0.3c
smoke and a compute-budget check, not R0.4–R0.6.

## Committed R0.3c task

Scripts:
- `scripts/track_r/r0_3c_fast_smoke_gpu.py`
- `scripts/track_r/r0_3c_fast_adjudicate_cpu.py`
- `tests/track_r/test_r0_3c_fast_contract.py`

The native sampler is identical to A1.5's external FAST source,
pinned `FAST_DIPS_A15@a9758b3`, executed under
`conda-envs/fast_dips_a15_blackwell` on one explicitly idle
Blackwell GPU. Uses exact signed frozen measurement and model SHA.

Within one model-loaded process the GPU worker runs two independent
roots:

1. **A1.5 replay:** exposed `69073/rep0`, seeded
   `15150000+69073+0`. Crucial: reproduce the historical
   A1.5 RNG order by seeding before native model construction and
   *not reseeding* between model construction and the first sample.
2. **New S2 FAST:** DEV80 image `63070`, rep0, seed from
   frozen `R0FAST|S2_b24_dev80|63070|0` SHA-256 first eight
   bytes little-endian masked to 63 bits. For the long-lived worker,
   this second root resets the RNG just before sample setup.

Each returns a native `reconstruction.pt` plus canonical
rounded `reconstruction.png`, frozen input SHA, wall-only and
CUDA-event sampling seconds, and no GT reads.

The separate CPU adjudicator reads the saved A1.5 reference
`reconstruction.png` and native receipt *after the GPU stage*.
It checks source measurement identity, reference SHA,
historical raw RGB8 replay, new raw RGB8 PSNR, image correlation,
and bit-exact PNG identity. The A1.5 replay passes only if PNG
bit-exact or absolute PSNR difference ≤0.05 dB. Otherwise STOP
and escalate to the planner, with no full scoring.

The budget projection uses the **slower** of the two measured
native FAST single-root wall times for each of all 720 S5 roots,
then adds 2520 + 720 prior scoring projections and measured smoke
overheads. The estimator multiplies the combined estimate by
1.25 for conservative scheduling. Must be ≤16 aggregate GPUh
to pass. This projection is NOT a proof of future runtime and
must be checked against worker ledgers throughout R0.4–R0.6.

Output remains in fresh Track R namespaced directories. The
historical checkpoint, external FAST source, A1.5 and B24 DEV80
exposed results are read-only; the protected B24 cohort remains
inaccessible. Nothing modifies the frozen R0 v1 config/spec.

## Next gate

Cheng runs R0.3c, returns `R0_3C_ADJUDICATION.json`
and logs. On both replay and budget PASS, implement and
test resumable R0.4 scoring/FAST R0.5 workers with a running
**aggregate** GPU-hour watchdog before any long jobs launch.
On mismatch or >16h projection, escalate to Track R planner.
