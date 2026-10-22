# R0.4 four-GPU continuation from passed 16-candidate canary
Date: 2026-10-22 · Track R executor

## PAC status and decision

User provided successful canary run:
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_4_score_20261022T193238Z`

- Committed head: `8aee60d98acfeba836fff4ec824137769b0a4698`
- CPU tests: 31/31 PASS
- Sealed 2520 GT-free tasks SHA:
  `38543cc4e4c394ab7c8936303a8d0fee6b46fb81e53f877a5ea6768e151078f7`
- Canary `CANARY_COMPLETE`, 16/16 candidate receipts, no active attempt
- Charged GPU allocation: 0.015001486262513531 GPUh
- Source preparation: 180 signed measurement files and 2520 candidate UIDs
- One GPU used for the canary: physical GPU1
- Each canary candidate has 256 diffusion forward images,
  nine UnivFD scored images and four residual fields

Cheng requested four-way continuation. CPU-only prep and the 16
canary results remain **read-only**. The original single-GPU
`r0_4_autopilot.py --resume` must NOT be run concurrently with
the new parallel controller.

## Four-GPU controller

`scripts/track_r/r0_4_parallel_autopilot.py` uses the *exact same
sealed task bytes*, not a new manifest. The new worker head differs
from the producer head: this is handled by an exact allowlist inside
`scripts/track_r/r0_4_score_gpu.py`, requiring BOTH the original
producer head, immutable task-file SHA and exact canary task path.
No relaxation permits arbitrary task-file substitution.

- Preflight validates the canary completed-shard receipt and all 16
  original candidate outputs, recomputes score/residual self-consistency,
  then checks other exposed candidate tensor/measurement formats on CPU.
- GPU0,1,2,3 must all be physically present and idle (≤1024 MiB
  used) before starting, regardless of previous statuses.
- Frozen remaining index interval [16,2520) has exactly **40**
  non-overlapping chunks: 39×64 plus one×8. Four physical GPUs
  operate concurrently, one chunk per GPU, with no overlap in output
  paths, and dynamic assignment of subsequent chunks after completion.
- Each result is an individual, immutable JSON; the worker verifies
  its inputs by SHA and never reads ground truth or PSNR.
- Parallel controller keeps a single-writer atomic JSON ledger,
  exact owned child PIDs, immutable log per attempt, 900s per-chunk
  timeout, and a `fcntl.flock` duplicate-controller barrier.
- The **6 GPU-hour R0.4 allotment applies to summed GPU allocation
  across *all four simultaneously running child processes*** and
  includes the already spent canary allocation. The controller also
  conservatively charges previous G0/G1/FAST smoke against R0's
  16 GPU-hour ceiling. A four-GPU job does not magically reduce
  total GPU-hours by four; only its elapsed wall-clock changes.
- On failed shard or source mismatch, launch no new shards and
  retain validated completions. When a budget/timeout is reached,
  signal *only* exact owned child PIDs (never kill by process name).
- On process interruption, a still-active ledger cannot be resumed
  automatically: reconcile tracked PIDs and charged allocation before
  selecting `--resume`. Clean STOP states with no active workers
  permit identity-checked resume.
- Final success state: `R0_4_ALL_2520_COMPLETE_4GPU`.
  This includes 16 historical canary + 2504 newly scored,
  all reconciled to the same GT-free input manifest.
- No R0.5/6 generation or R0.7 offline GT analysis is started.

## First-use execution status

The new four-GPU code is committed but **has not been tested on PAC**.
Run CPU tests from the pinned new HEAD first and return if they fail.
A separate new output directory is required; the passed canary root
and its result receipt are never modified.

The GPU0 prior sample had reported ~73589 MiB occupied before the
user subsequently stated all four were idle. Runtime will *recheck*
the actual state, and fail safely if GPU0 is still occupied.
