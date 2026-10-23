# Track R R0.4 confirmed complete; R0.5 native FAST canary handoff

Date: 2026-10-22 (PAC run time UTC 2026-10-22T23:54:12Z).

## R0.4 PAC results (user-provided verified console)

Recovery run:
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_4_recovery_20261022T235412Z`

Git HEAD `62eaaafe3e22cd6222fc66858cad6dcef532a2cd`.
CPU suite **40 passed**.

- Status `R0_4_ALL_2520_COMPLETE_RECOVERED_4GPU`.
- **2520/2520** historical candidates scored; last recovery 13/13 shards
  completed with RC=0 and no active workers.
- Old canary: 16 candidates. First four-GPU run: 27 complete
  64-candidate shards (1728 candidates), and 45 verified
  partial-shard receipts. Total previous 1789.
- Recovery: 19 resumed from failed source shard + 712 unstarted
  candidates = **731 newly scored**.
- Aggregate R0.4 GPU allocation: `2.076199647170207` GPUh,
  within its 6 GPUh allotment and below the earlier 2.50 GPUh
  smoke-based projection.
- All four physical GPUs used; no candidate duplicated.
- Frozen GT-free historical tasks:
  SHA `38543cc4e4c394ab7c8936303a8d0fee6b46fb81e53f877a5ea6768e151078f7`.
- Forbidden B24 confirmation305 remains untouched.

The controller's status confirms all completed shard receipts;
the R0.5 prep verifies the entire result lineage at per-candidate
receipt level before any new native FAST work.

Scientific outcome remains pending: no R0.7 offline GT labels,
diagnosis, oracle or policy analysis has been interpreted.

## R0.5 implementation (not yet run on PAC)

Scripts:
- `scripts/track_r/r0_5_prepare_cpu.py`
- `scripts/track_r/r0_5_fast_gpu.py`
- `scripts/track_r/r0_5_canary_autopilot.py`
- `tests/track_r/test_r0_5_fast_contract.py`

CPU prep audits the **three** immutable R0.4 result roots, all
2520 scorer receipts, no overlaps, frozen source SHA and total
allocation ledger, before freezing **720** signed-measurement-only
native FAST roots: 100 distinct B22 S1 images ×4, 80 distinct B24
DEV80 S2 images ×4. It never reads protected B24 cohorts.

Seed:
`int.from_bytes(SHA256('R0FAST|{source}|{image_id}|{rep}')[:8],
'little') & (2**63-1)`. Every root has an independently reset
NumPy/torch/Python RNG. Native pixel FAST parameters exactly
match A1.5 (150 steps, ADMM2, S5, rho=200). Model source
`FAST_DIPS_A15@a9758b38906b7c7950fa04491f37f6634948db26`
and `ffhq_10m.pt` weights pinned by SHA.

GPU worker receives **no GT**. Per root it writes immutable
`reconstruction.pt` (float32), canonical RGB8
`reconstruction.png` and `R0_FAST_ROOT_COMPLETE.json`,
including seed, observation hash, output hashes and native
CUDA/event/wall timing. The chunk output can resume by verifying
only complete immutable roots; incomplete partial output stops,
requiring explicit manual audit, never overwrites.

R0.5 canary: 8 roots on one explicitly idle Blackwell GPU after
R0.5 CPU prep PASS. This is `2` S1 images ×4 reps. S2 has
already had one true R0 native FAST root in R0.3c smoke, but
its full four-root image behavior remains subject to later
generation. The canary returns `R0_5_FAST_8_ROOT_CANARY_PASS`
and stops before full 720-root campaign.

**Next after the canary:** inspect both source-group native
provenance, signed measurement/hash, wall clock, PNG,
model-native validity, GPU ledger; then implement/launch the
four-GPU R0.5 controller. That controller must include the exact
R0.4 aggregate 2.0762 GPUh and previous R0.3
smoke GPUh under the binding **16 aggregate GPUh** ceiling,
with separate stage allotment, no GT, immutable retry state.
The frozen R0 spec remains unchanged.

Following R0.5: R0.6 score 720 new FAST roots (same prior/UnivFD
and residual definitions) then R0.7 offline GT and policies.
No claims of prospective performance from exposed data.
