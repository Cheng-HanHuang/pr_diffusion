# XP GPU co-tenancy policy — operator-approved 2026-10-26

**Intent:** Continue quality experiments despite unrelated GPU jobs, capture the
actual interference, pause only when a new job cannot fit safely in memory,
and automatically resume after the GPU becomes available. *Do not disturb
another user's running job.*

This instruction supersedes the earlier blanket “refuse foreign process above
2 GiB” launch gate **only for the XP executor**. The planner's 80% per-GPU
memory ceiling, replay, configuration and measurement SHA gates, theoretical
outcome embargo, 400 GPU-hour ceiling, and paper-quality timing discipline
remain mandatory. This is a scheduling change, **not** a data/protocol change.

## Per-job admission

Read NVIDIA device total/used/free/utilization and compute-app PIDs immediately
before each launch. Track currently owned worker PIDs/descendants; record
other PIDs as foreign. They **do not block an otherwise fitting job**.
Require `device_used + outstanding_own_reservations + job_vram_estimate
+ safety_margin <= floor(0.8 * device_total)`. Default safety margin 2,048
MiB. If no GPU passes, write `WAIT_VRAM`, rescan every 10 seconds and
automatically resume later. Preserve the same seed and start a new immutable
attempt; do not kill or interrupt any running workers.

When a new foreign process arrives during an active reconstruction, allow
the reconstruction to finish. The result's reconstruction PSNR and physics
checks do not depend on GPU timing cleanliness if all underlying scientific
receipts pass. OOM is a *failed attempt*; the controller can retry from the
same seed after memory is available, without overwriting outputs.

## Per-job observational receipt

Collect GPU occupancy on start, every poll during computation, and after exit.
Attach snapshots and PID visibility to `GPU_EVENTS.jsonl`. An accepted
worker receipt must assert no GT access during generation, pins the code head
and the immutable measurement SHA, and requires `XP_JOB_PASS`.

- `CLEAN`: no observed foreign processes and sufficient process visibility.
- `CONTAMINATED`: foreign GPU process overlapped at any observed instant.
- `UNVERIFIED`: process accounting unavailable or GPU memory above
  the unexplained-memory threshold. **Neither class is eligible for a clean
  runtime/throughput benchmark.**
- All process observations are periodic. A process that enters and exits
  between samples may escape detection; time-aware conclusions must state
  this limitation.

**Quality job:** accept scientifically valid receipts even when
`CONTAMINATED` or `UNVERIFIED`; keep timing metadata as non-publishable
and count actual GPU-hours in the accounting ledger.

**Timing job:** record the first contaminated trial, mark
`TIMING_REMEASURE_PENDING`, and rerun on a demonstrably clean device
when possible, with the same measurement, code and settings. Use a new
attempt directory. Never label a contaminated timing as clean. Limit
remeasure attempts to prevent uncontrolled GPU-hour use. GPU contention
extends the soft expected run duration, but does not confer a new
time/quality performance claim.

## Controller admission and scientific readiness

The scheduler code alone does **not** authorize GPU experiments. Its manifest
requires `status=XP_MANIFEST_FROZEN`, `gates_pass=true`, pinned expected
head, approved workers, worker output receipts and the 400 GPU-hour budget.
No XP-T or replay gate has yet been performed in this branch. Enqueue only
after corresponding native runner, measurement, physics, smoke and replay
prerequisites have been verified. The two lanes remain CRITICAL first,
BACKGROUND second; BACKGROUND units must remain <=20 minutes.

A controller will provide a PID file, stop-after-active sentinel, durable
state, and at most 20-line digest. Graceful stopping waits for currently
running reconstruction jobs. On controller crash, unreconciled running
jobs cause a **stop**, not a duplicate execution. Historical output
directories and other users' processes remain immutable.

## Initial CPU verification

The `xp_gpu_slot_probe.py` command merely reads NVIDIA telemetry and
shows current per-device eligibility. It performs no model inference,
does not reserve GPU memory, and makes no throughput claim. The scheduler
and telemetry synthetic tests cover foreign-PID classification, conservative
VRAM admission, WAIT_VRAM -> eligible transition, contamination classification,
receipt integrity and CRITICAL priority. Real GPU launches require the
separate scientific-validation gate.
