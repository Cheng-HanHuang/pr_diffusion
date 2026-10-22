# R0.4 recovery after UnivFD duplicate-evaluation discrepancy — 2026-10-22

## Actual PAC failure

Original parallel run (preserve completely, read-only):
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_4_parallel_20261022T200149Z`
at executor head `717100a17a2326f3fbbccb7cbe782ccf4e90126d`.

The original `AUTOPILOT_STATUS.json` verifies:
- status `STOP_WORKER`, 0 active GPU children
- 27 complete 64-candidate parallel shards, RC=0
- 1744 complete incl original 16-candidate canary
- one failed worker RC=8 on physical GPU2, interval [1680,1744)
- 45 candidate receipts saved in its partial `results/` directory
- `charged_parallel_gpu_seconds=5276.953150671441`
- `historical_canary_gpu_seconds=54.005350545048714`
- `total_gpu_seconds_including_canary=5330.95850121649`.

The failed worker log shows:
`STOP_R04_UNIVFD_REPAIRED_MISMATCH`, after 40/64 progress,
then a partial file count of 45. The interrupted item is manifest
index 1725. The failure is from a *duplicate calculation* of UnivFD's
fake probability: the saved eight-member orbit-batch value differs
from evaluating the same repaired orbit image in a single-image batch
by more than an ad hoc 1e-5 absolute consistency limit. This log
alone does **not** quantify the difference or prove its numerical
cause, so the recovery worker is deliberately bounded and diagnostic.

## Bounded FP batch-shape workaround

The scientific object remains `p_fake(R_s(x))` at the eight orbit
members; the native eight-image batch is the canonical evaluation.
`p_fake_repaired` on NEW outputs is assigned to the already
computed `p_fake_orbit[s_star]`. The singleton calculation is
retained as a diagnostic, never used to choose a model/orbit.

- If batch and singleton differ by ≤1e-5, record the agreement.
- Otherwise repeat the **identical eight-image batch** and require
  all 8 scores agree within 1e-5 absolute.
- Reject deviations >1e-3 between the singleton and batch even if
  the same-batch repeat is stable.
- Record the exact differences and FP diagnosis with each new
  candidate. On high drift or batch instability, STOP again rather
  than silently relaxing the scientific scorer.
- Reconcile old/new offline R0.7 results using
  `p_fake_orbit[s_star]` **for every candidate**, including earlier
  immutable receipts. Old successful receipts were already verified
  to have singleton-batch disagreement ≤1e-5, so this rule preserves
  scientific uniformity without overwriting any historical result.

No GT was or will be supplied to the GPU workers. R0 frozen
measurements, orbit definition, prior and classifier weights remain
unchanged. The engineering batch repeat costs at most 8 additional
UnivFD image forwards in candidates where the discrepancy occurs.

## Nonoverlapping recovery

New entrypoint: `scripts/track_r/r0_4_recovery_autopilot.py`.

It validates all **16 canary** receipts, all **27 completed old shards**
and all **45 individually completed candidate receipts** in the
failed shard. This source-identity and numerical-integrity audit
precedes any GPU invocation. It requires the exact original run
location, Git head, task SHA, partial shard GPU2 provenance,
completed shard coverage, and previous charged GPU seconds.

The previous 1789 scored candidates are never regenerated, copied
or modified. Exactly **731** remaining candidates are scored in
a separate output directory:

- partial continuation [1725,1744): 19
- untouched range [1808,2520): 712, divided into 12 shards
- 13 disjoint new shards, 4 physical GPUs, <64 images per shard.

New results carry the new executor head and immutable original task
SHA. The final success state explicitly combines the 1789 audited
historical candidates plus 731 newly completed candidates.
`R0_4_ALL_2520_COMPLETE_RECOVERED_4GPU`.

The recovery controller initializes its aggregate ledger with the
**full old 5276.953150671441 parallel GPU-seconds** (plus 54.00535
canary GPU-seconds) and charges every new GPU child simultaneously.
The 6 GPU-hour R0.4 aggregate cap remains binding, as does the
overall 16 aggregate GPU-hour project cap. The code is committed but
its CPU tests/GPU operation must be verified by the PAC operator.
No automatic R0.5, R0.6 or R0.7 tasks are triggered.

If this recovery fails or its CPU suite fails: stop, retain receipts,
inspect its specific failure before any retry; do not bypass checks.
