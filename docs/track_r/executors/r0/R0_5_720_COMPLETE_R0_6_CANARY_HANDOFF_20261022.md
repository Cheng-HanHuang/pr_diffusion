# Track R R0.5 PASS, R0.6 scoring canary executor handoff
Date: 2026-10-22 America/Detroit / 2026-10-23 UTC

## R0.5 success evidence received from PAC

R0.5 four-GPU native generation:
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_5_parallel_20261023T012732Z`

- Executor Git HEAD: `2a65698a3855ddcc65071af90f69f287c824fa13`
- CPU suite: 49/49 PASS.
- All 4 NVIDIA RTX PRO 6000 Blackwell GPUs were idle at launch.
- Final status: `R0_5_ALL_720_FAST_ROOTS_COMPLETE_4GPU`.
- 720/720 native FAST candidate roots (100 B22 images ×4, 80 B24 DEV80 images ×4).
- All 30 new native-generation shards PASS (712 roots), plus the
  original 8 canary roots. No failures reported.
- R0.5 aggregate GPU use: **0.7378 GPUh**.
- Known aggregate R0 through R0.5: **2.8454 GPUh**.
- Total budget: at most 16 aggregate GPUh; current headroom
  about **13.15 GPUh**. This is an allocation ledger, not a
  wall-clock metric.
- Original 720-root GT-free manifest SHA:
  `2291aa793b612c973ae76ab3d36b1f20263cd747f3b23cf593e51517a26c3bfe`.
- No GT used by native FAST. Outputs have immutable
  `reconstruction.pt`, `reconstruction.png` and per-root receipts.
- Scientific reconstruction-quality analysis is not yet authorized:
  R0.5 simply completed candidate generation, not scoring or GT analysis.

## R0.6 implementation

Scripts:
- `scripts/track_r/r0_6_prepare_cpu.py`
- `scripts/track_r/r0_6_score_gpu.py`
- `scripts/track_r/r0_6_canary_autopilot.py`
- `tests/track_r/test_r0_6_scoring_contract.py`

CPU preparation performs a **full 720-root integrity audit**:
- Confirms exact R0.5 old canary and 30-shard lineage, strict
  8+712 coverage, native FAST root receipt SHA matches.
- Audits every native float tensor and canonical PNG for exact
  `to_rgb8` equivalence. No GT or historical PSNR is used.
- Freezes an independent task list of 720 scored-candidate inputs,
  including hashed original signed measurements and PNGs.
- Calculates aggregate GPU hours from the immutable R0.5 ledger.
- Fails before GPU launch if coverage, source SHA, quantization,
  budget or forbidden-path checks fail.

The **R0.6 scoring worker reproduces R0.4 scientific definitions**:
- Canonical RGB8 image → signed-measurement residual.
- Frozen 8-timestep ×4-draw ×8-orbit paired-noise diffusion
  prior score (256 UNet image forwards per candidate).
- Pinned CLIP/UnivFD p_fake orbit-batch baseline; the authoritative
  repaired p_fake is the matching eight-orbit batch entry.
- Preserve singleton numerical diagnostic; repeat the same
  eight-orbit batch when singleton differs >1e-5 and fail on
  unstable same-shape scores or discrepancy >1e-3.
- Both signed and clipped residuals for raw and repaired image.
- GT-free immutable candidate/shard receipts; source method
  (S1 B22 versus S2 B24 DEV80) preserved for R0.7 grouping.
- New output paths, never overwrite R0.4 or R0.5 records.

First run: **one-GPU 8-candidate scoring canary**, after the CPU
720-root audit. The canary stops before scoring other roots.
R0.6 score phase engineering budget 4 aggregate GPUhours with
the overall R0 budget of 16 GPUhours.

The R0.6 scripts have been committed but **not run on PAC**.
Do not claim canary or G0/R0.6 completion until receipts return.

## Next scientific tasks

On canary PASS, implement or activate a controlled 4-GPU
continuation for the remaining 712 FAST scoring tasks, retaining
the sealed task SHA and first eight scored-candidate receipts.
After 720 R0.6 scores, R0.7 performs offline GT analysis:
P1 orbit-failure anatomy, P2 prior-repair success/harm,
P3 detector AUROC, P4 E0–E3 policies and paired CIs, with
image-cluster bootstrap and frozen decision thresholds.
All R0 findings remain retrospective and exposed-data only.
No protected B24 confirmation305 access is authorized before R2.
