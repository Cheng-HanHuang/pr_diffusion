# Track R R0.3c PASS → R0.4 GT-free scoring canary handoff
Date: 2026-10-22

## R0.3c PAC return (verified user-supplied console logs)

Original completed R0.3c run:
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_3c_fast_20261022T185130Z`

- Executor head: `9bc30d9a910d267a45bc791cc43dab8597e4ee61`.
- CPU suite: 26/26 PASS.
- GPU native FAST and CPU replay adjudication: both RC=0.
- Historical A1.5 69073/rep0 replay:
  `reconstruction.png` **bit-identical** SHA
  `c26fd6d7c411f33804adcc096e94e813f79e27b72e1397e79df59f23ac773850`,
  raw RGB8 PSNR 32.38233947753906 dB, diff=0.0 dB,
  correlation≈1.0. Source seed 15219073; sampling 12.984258s.
- New S2 DEV80 63070 rep0:
  native FAST seed 7452755359293807986,
  sampling 3.288141s, no GT access.
- Conservative FAST-generation projection 720 roots:
  2.596851690392941 GPUh based on the **slower** 12.984s
  measured root.
- Scoring projection: 2.5031679952924604 GPUh existing
  2520 + 0.7151908557978459 GPUh future FAST 720.
- Including recorded smoke allocations and 1.25x margin,
  projected aggregate GPUh = **7.308243269557552**, within
  the fixed 16 GPUh ceiling. Budget PASS.
- Final status:
  `R0_3C_PASS_FULL_R0_3_SMOKE_COMPLETE`.

G0 complete (including two-GPU batch8/64 determinism); G1 complete
(canonical DAPS orientation repair 6.364→31.871). Native A1.5
FAST-DIPS reproduced exactly; this is a substantive reproducibility
gate, not held-out evaluation. Dataset remains exposed/development;
confirmation305 remains inaccessible through R1.

## R0.4 implementation

Scripts committed on executor branch:
`r0_4_prepare_cpu.py`, `r0_4_score_gpu.py`,
`r0_4_autopilot.py`, and synthetic CPU tests
`tests/track_r/test_r0_4_scoring_contract.py`.

**The code has not yet been executed on PAC.**

1. CPU preflight verifies the validated 2520-candidate manifest SHA
   `8ab7dc375a39a94d1f5183dd58d82f6d5c1c20a3d7124c7249df6a685c5ef890`;
   CPU G0, GPU G0, G1 and R0.3c receipt statuses and budgets.
2. CPU freezes a fully hashed **GT/PSNR-free** worker input list of
   2520 candidates, with their exact signed-measurement files.
   No source candidates or measurements are overwritten.
3. One idle Blackwell GPU performs native paired-noise prior scoring
   at **batch size 64**, validated by R0.3b, and UnivFD classification
   for all 8 orientation members plus repaired member. Computes both
   signed and clipped measurement residuals before/after repair,
   using canonical rounded RGB8 images and saved signed measurements.
4. Each candidate has a unique immutable JSON receipt containing full
   S(t), S_mid, per-t losses, orbit choice, p_fake orbit values,
   repaired p_fake, signed/clipped residuals and measured run cost.
   No ground truth is imported/loaded by the GPU worker.
5. GPU shards contain at most 64 candidates, are resumable by
   validating already-completed JSON receipts, and stop on source SHA,
   model identity, CLIP/FC identity, type/shape, score or residual
   mismatch. Never substitute, regenerate or relabel candidates.
6. The controller has a **6 aggregate GPU-hour R0.4 allotment**,
   enforced on real GPU child allocation wall time and 900s per shard.
   Remaining stages R0.5 and R0.6 must carry their own limits,
   integrated under the original 16 aggregate GPU-hour ceiling.
   This allocation is an engineering cap, not a scientific change.
7. First run requests ONLY `--max-candidates 16`; inspect receipts,
   distribution, timing, memory and residual identity. Then use
   `--resume --max-candidates 2520` against the SAME run root
   to complete all existing candidates. No full scoring is triggered
   automatically before the canary is adjudicated.

A process interruption leaves prior immutable candidate result files
intact; interrupted shard can be retried with separate immutable
attempt logs. Unexpected parent death during an active child leaves
an explicitly un-reconciled `active_attempt` ledger that must be
reviewed before resume; never erase real GPU allocation costs.

## Remaining work

After R0.4 canary PASS: complete R0.4, implement R0.5 720 native FAST
roots on S1+S2 using the frozen SHA seed, then R0.6 score those roots
with the same physics/detector/prior; finally offline R0.7
GT evaluation, bootstrap and E0–E3 policy analysis. All are
retrospective and exposed-data only. The planner, not the executor,
makes the D-material/D-repair/D-detect scientific decision.

Stop and return any failed R0.4 canary with its logs; do not start
full scoring, native FAST generation, or protected confirmation work
after a gate anomaly.
