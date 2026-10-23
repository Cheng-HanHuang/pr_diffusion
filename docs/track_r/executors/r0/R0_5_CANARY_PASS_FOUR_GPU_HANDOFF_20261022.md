# Track R · R0.5 eight-root native FAST PASS; four-GPU handoff

Date: 2026-10-22 Detroit / 2026-10-23 UTC.

## Observed result (PAC user console)

R0.5 canary:
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_5_canary_20261023T004927Z`

- Git HEAD `c3170587e58e9610c97c6f4ae48964eab2db1b46`.
- Autopilot status `R0_5_FAST_8_ROOT_CANARY_PASS`.
- R0.5 CPU task preflight PASS (4.27s) and native GPU canary PASS (35.01s), both RC0.
- 720 roots from 180 exposed images locked, source-only SHA256:
  `2291aa793b612c973ae76ab3d36b1f20263cd747f3b23cf593e51517a26c3bfe`.
- Two S1 B22 FFHQ images, IDs 60044 and 60140, four independent FAST roots each; all 8 root receipts PASS.
- Native sample-only time (seconds): 3.679, 3.323, 3.348, 3.332,
  3.381, 3.397, 3.398, 3.339; around 3.40 mean.
- Per-8 GPU allocation 35.01077954424545s; internal chunk 28.495s.
- No GT access in GPU workers; roots have immutable PNG/tensor hashes, source measurement SHA and independent frozen seeds.
- No claim about reconstruction quality yet (R0.7 after all FAST scoring).

Earlier R0.4 finished 2520/2520 and used
`2.076199647170207` aggregate GPUh, including recovered work.
Native A1.5 FAST reproducibility achieved bit-exact PNG in R0.3c.

## Remaining 712 native FAST roots (R0.5 parallel)

Scripts:
`scripts/track_r/r0_5_parallel_autopilot.py`,
`scripts/track_r/r0_5_fast_gpu.py` (sealed old-task-head adapter),
`tests/track_r/test_r0_5_parallel_contract.py`.

The new worker permits *only* the exact original R0.5 canary root
manifest path + SHA + producer head under the newer executor HEAD.
Scientific configuration, root ordering, seeds, model and measurements
remain frozen. No source candidates or roots are regenerated.

30 non-overlapping, 4-root-image-aligned chunks: 29×24 and 1×16,
cover indices 8–719 without rerunning the canary 0–7.
One 24-root native FAST chunk per GPU at a time; GPU0–GPU3 must
all be idle (<=1024 MiB used), and each worker is pinned to one
physical Blackwell GPU. The controller dynamically assigns further
chunks, checking source identities and preserving immutable successful
receipts. Restarted partially written chunks must return to their
original GPU, and a preexisting root receipt is hash-checked before reuse.

Budget: overall 16 aggregate GPUhours; R0.5 own 4 aggregate GPUhour
engineering allocation. Controller charges all four running GPU processes
independently (rather than elapsed wall alone), accounts for the
previous 2.0762 R0.4 GPUh and known prior G0/G1/R0.3 smoke allocation,
and adds the 35.01s R0.5 canary. No R0.6 scoring starts automatically.

The new parallel driver and CPU tests are committed but **not yet
executed on PAC**. Run CPU suite first from pinned Git SHA; no
GPU job may start on a CPU failure. If a worker fails or runtime
projection/budget exceeds thresholds, leave immutable receipts and
return logs; do not bypass. A stopped run may resume only after
ledger/PID reconciliation and all GPU children are absent.

Next R0.6: GT-free scoring of all 720 generated FAST roots using
identical frozen prior and UnivFD orbit batch definitions, then R0.7
offline GT P1–P4. All data remain retrospective/development.
The protected confirmation305 source remains unopened through R1.
