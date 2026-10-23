# Track R: R0.6 complete; R0.7 frozen CPU analysis handoff

Date: 2026-10-23 UTC, based on PAC operator receipts.

## R0.6 native FAST scoring — complete

PAC four-GPU score run:
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_6_parallel_20261023T023235Z`

- Git HEAD: `99a9e5b6a2544c56490a95f2498dd4c2cb54262e`.
- CPU suite: **60/60 passed**.
- Status: `R0_6_ALL_720_FAST_SCORES_COMPLETE_4GPU`.
- Full FAST scoring: **720/720**, all 12 new shards PASS and
  the eight-R0.6-canary lineage verified; 0 active workers.
- Aggregate R0.6 allocation: **0.585574 GPUh**.
- Known aggregate R0 compute, G0 through R0.6:
  **3.430956 GPUh** out of the hard **16 aggregate GPUh**.
- R0.4 historical score lineage: **2520/2520**.
- Total GT-free scores now available: **3240**.
- R0.6 frozen GT-free task SHA:
  `f268e29b482da10aee31f79de43a6bf19d460f4203f7813c46dbee83b0591596`.
- Prior scientific decisions are **not** inferable from these
  operational receipts. No raw/repaired GT PSNR results have yet
  been computed by R0.7.

## Frozen R0.7 implementation

Scripts:
- `scripts/track_r/r0_7_prepare_cpu.py`: G0-to-R0.6
  provenance audit and CPU-only GT join.
- `prdiffusion/track_r/r0_7_stats.py`: deterministic
  10,000 image-cluster bootstrap, PSNR orbit calculations,
  four raw-GT labels, AUROC and K-run policies.
- `scripts/track_r/r0_7_analyze_cpu.py`: actual frozen
  P1–P4 plus secondary analysis and report generation.
- `scripts/track_r/r0_7_cpu_autopilot.py`: two-stage,
  strictly CPU-only controller.
- `tests/track_r/test_r0_7_frozen_analysis.py`:
  synthetic checks, no source files needed.

**The new R0.7 implementation has not yet run on PAC.**
A full CPU-only test suite is mandatory before execution;
a failed test blocks the launch.

## CPU-only audit

Preflight confirms original 2520-row manifest SHA, every old
candidate UID and source GT manifest row, all immutable R0.4
score receipts (16 canary + 27×64 + 45 valid partial +
13 recovered shards), all 720 native FAST generation receipts,
canonical PNG–float tensor equivalence, all R0.6 score
receipts (8 canary + 12 new shards), and exactly 3240
nonduplicate GT-free scores.

It freezes `prepare/R0_7_OFFLINE_GT_CATALOG.json` with
GT path aliases only for **offline CPU analysis**. This catalog
must never be supplied to a GPU worker or combined with any
other source directory. The scientific code accesses no
B24 confirmation305 registry, path, membership or payload.
All source paths are from the explicit exposed manifest,
not globbed or searched.

## Primary analysis interpretation

- P1: GOOD raw ≥25 dB; ORBIT_ERROR raw <25 and one
  non-identity orbit member ≥25; GLOBAL/PARTIAL s_gt;
  remaining non-orbit failures separated by GOOD-based signed
  residual threshold tau=Q0.95 per frozen source/family.
- P2: repair success/exact orbit match on raw ORBIT_ERROR
  and harm on raw GOOD, compared with UnivFD's argmin-p_fake
  orbit member; S_mid secondary.
- P3: remaining post-repair failure AUROCs for S,
  image-centered S, UnivFD orbit-batch p_fake, signed and
  clipped residuals, by source/family and pooled family.
  Raw non-orbit subtype diagnostic compares surviving
  failures of that subtype with repaired successes; the
  control-set choice is documented explicitly because frozen
  R0 v1 left its operational subset convention unspecified.
- P4: per-image first K candidates in recorded run order;
  E0, E1, E2, E3, O_raw and O_orbit; mean, median,
  Good25, Bad20, Bad15, 5% PSNR quantile and minimum;
  paired image-cluster 10k bootstrap CIs for all seven
  metrics of E1−E0, E2−E1, E2−E3; exact McNemar tests
  on Good25. B22 NP permits K8.
- Secondary: per-t repair and AUROC, Spearman complexity
  bias, native FAST generation CUDA-event/wall timing,
  scoring forward-call counts and GPU timing.

Ground-truth PSNR is always computed from saved source GT
tensors, or the explicit source FFHQ 1024→256 transform
documented in G0, with canonical RGB8. Orbit PSNR
computations have synthetic direct-rotation unit tests.

**Missing GOOD-based tau:** E0/E1 and both GT ceilings
remain evaluable, but E2/E3 and their paired comparisons are
N/A for that source/family. The final status becomes
`R0_7_PARTIAL_TAU_UNAVAILABLE` rather than silently
estimating tau from failures.

**S5 measurement identity:** S5 combines B22 and B24
development measurements. Within-image policies and image
normalization group by (measurement_source,image_id);
image-cluster bootstrap still clusters matching physical
image IDs together across sources.

**Implementation conventions where spec is underspecified:**
- quantile interpolation: NumPy `method='linear'`.
- subtype AUROC control group: repaired successes; matching
  raw subtype among repaired failures supplies positives.
- HF spectral fraction computed on canonical RGB8 [0,1],
  including DC, radius >0.25 cycles/pixel.

These conventions are reported; no post-result tuning or
changes to frozen scientific thresholds are permitted.

## Files written only on PAC

The controller writes a new R0.7 run under
`/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_7_cpu_...`:
- `prepare/R0_7_OFFLINE_GT_CATALOG.json`
- `prepare/R0_7_JOIN_RECEIPT.json`
- `analysis/R0_7_SCORED_GT_ROWS.jsonl`
- `analysis/R0_7_ANALYSIS.json`
- `analysis/R0_7_REPORT.md`
- `analysis/R0_7_COMPLETE_RECEIPT.json`
- `AUTOPILOT_STATUS.json`, stage logs.

All stages are CPU-only and use CUDA_VISIBLE_DEVICES="".
There are **zero additional planned GPU hours**.

## Planner handoff after the CPU report

Report all P1–P4 tables including denominators/CIs, decision
inputs and limitations back to Track R planner through GitHub.
The executor does not declare D-material/D-repair/D-detect
PASS/FAIL without the planner's review. R0 is exposed-data
retrospective analysis only, never an unseen confirmation
claim; protected confirmation305 stays locked through R1.
