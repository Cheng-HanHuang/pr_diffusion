# Track R overnight CPU autopilot — scientific scope and stop conditions

## Status for Cheng

The 2026-10-22 PAC run `R0_0B_RC=0`, `ENV_READY` established
the isolated environment and SHA-pinned UnivFD and CLIP weights.
The earlier `numpy` metadata mismatch was verified to have identical
imported NumPy 2.4.6 Python/binary files. The first R0.1 suite then
stopped at collection on an accidental literal backslash-newline in
`prdiffusion/track_r/__init__.py`. The executor fixed it on the
Track R branch; CPU tests have **not yet passed**.

Cheng requested an unattended overnight run. The committed
`r0_autopilot_cpu.py` is **not** permission to skip G0/G1 or execute
unimplemented GPU stages. It executes as far as the current verified
scientific code permits and stores an unambiguous status.

## Automatic chain, one CPU-only process

1. Verify clean pinned checkout, source checkpoint SHA, B22 manifest SHA,
   A1.2 input-freeze SHA, CLIP and UnivFD FC SHA, and exact UnivFD Git HEAD.
2. Run all existing `tests/track_r` using isolated
   `prdiff_track_r`; a failing test stops.
3. Read only S1 B21/B22, S2 B24 DEV80, S3 A1.2, S4 A1.5
   explicitly named/exposed inputs to build `R0_CANDIDATE_MANIFEST.csv`,
   requiring source/family counts: 1400+960+128+32 = **2520**.
4. CPU G0: FFT impulse witness; 16 random + 3 real-image exact-orbit
   invariance checks; source GT/measurement rho (PASS 0.0495–0.0505);
   raw recorded PSNR replay under per-source conventions (≥99% within
   0.01 dB); assert all three per-record paths are permitted.
5. If all CPU G0 checks pass, record
   `PAUSED_GPU_G0_DETERMINISM_AND_R0_3_EXECUTOR_NOT_IMPLEMENTED`.
   **No GPU work is started, even when GPUs are idle.**

Output:
`$RUN/AUTOPILOT_STATUS.json`,
`$RUN/logs/r0_1_cpu.log`,
`$RUN/logs/r0_2_manifest.log`,
`$RUN/logs/r0_2_g0_cpu.log`,
and stage-specific receipts under the fresh Track R output root.

All stages fail closed, preserve their logs and existing historical files,
never regenerate measurements or inspect protected cohort paths, and do not
modify the frozen config/spec. An error or missing source candidate is
reported rather than guessed.

## Why automatic GPU stages are not yet executable

The R0.3 prior/UnivFD/native FAST GPU smoke and dual-GPU determinism
gate, R0.4–6 worker sharding/new FAST generation, 16 aggregate
GPU-hour enforcement and R0.7 analysis/packaging have not been
implemented or reviewed. The planner's R0 handoff forbids launching
GPU work before G0 and G1 and proceeding after G1 failure.

The unattended launcher is deliberately **CPU-first, not a claim
of completing R0 overnight**. Once PAC CPU receipts are returned
and validated, implement the missing GPU stages and independent
hardware budget watchdog, then run an actual end-to-end autopilot
only if all required gates pass. Never use idle resources to infer
scientific authorization.

## Source-layout caveats carried into the manifest builder

- S1 B21 Fresh1/Fresh2 candidate PNGs and measured rows come from
  `fresh2_final_rows.csv`; B22 NP/SITCOM candidate receipts are
  joined by image **ID** rather than divergent `rowNNN` ordering.
- S1 original GT derives from the documented B21
  `ToTensor -> Resize(256) -> CenterCrop(256) -> 2*x-1`;
  saved GT from S2/S3 takes precedence.
- S2 fresh external DAPS4 and SITCOM4 saved candidates come from each
  DEV80 task's `baseline/metrics/METRICS.json:terminal_rows`,
  not `methods/`; NP4 uses named `terminal_{0..3}.pt`.
- S3 A1.2 receipts and S4 A1.5 native roots are read at exact exposed
  locations. If receipt field names differ, the manifest stage stops
  with a specific diagnostic; never invent an outcome.

A successful G0 CPU receipt is **partial G0** because cross-GPU
prior determinism remains pending.
