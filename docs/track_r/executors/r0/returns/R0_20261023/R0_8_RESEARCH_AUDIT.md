# Track R — R0.8 overnight scientific audit and planner handoff

**Status: independent numerical replay PASS on completed R0.7 outputs.**
This document contains *retrospective* evidence from exposed sources only.
Every D-stage outcome below is a **provisional numeric threshold check**;
the Track R planner retains the scientific decision and R1 design.

## Execution and provenance

- 3,240/3,240 CPU-offline rows checked; 12 source/family groups.
- 37 P4 source/family/K tables independently replayed.
- Final R0 aggregate GPU use: **3.430956 / 16 GPU-hours**.
- No further GPU calls, no new candidates, no new thresholds.
- Original frozen R0 v1 and completed R0.7 files remain read-only.
- No access to B24 protected confirmation305, including membership.

## Independent label counts (single-run)

| Label | Count | Fraction |
|---|---:|---:|
| GOOD | 2040 | 62.96% |
| ORBIT_ERROR | 85 | 2.62% |
| NONORBIT_CONSISTENT | 324 | 10.00% |
| NONORBIT_INCONSISTENT | 791 | 24.41% |

## Frozen decision inputs — provisional

| Gate | Numeric evidence | Provisional reading |
|---|---|---|
| D-material | FAST orbit share 0.036; DAPS 0.131 (threshold 0.20) | False |
| D-repair | Success 0.988 (PASS ≥0.90); harm 0.000 (PASS ≤0.01) | MEETS_PASS_CUTOFFS |
| D-detect | ΔAUROC FAST -0.096; DAPS -0.055 | both superior False; both noninferior False |

- **Provisional branch:** `PLANNER_GREY_ZONE_REVIEW`.
- **R1 is NOT authorized automatically.** Planner must adjudicate
  the statistical evidence, overlap/selection issues, method novelty,
  appropriate branch and locked R1 prospective protocol.

## P1 — anatomy by source and family

| Source | Family | Runs | Good | Orbit | Nonorbit consistent | Nonorbit inconsistent |
|---|---|---:|---:|---:|---:|---:|
| S1_b22 | DAPS | 200 | 165 | 7 | 0 | 28 |
| S1_b22 | NP | 800 | 598 | 20 | 62 | 120 |
| S1_b22 | SITCOM | 400 | 285 | 14 | 59 | 42 |
| S2_b24_dev80 | DAPS | 320 | 170 | 17 | 18 | 115 |
| S2_b24_dev80 | NP | 320 | 163 | 2 | 40 | 115 |
| S2_b24_dev80 | SITCOM | 320 | 129 | 11 | 115 | 65 |
| S3_a12 | DAPS | 32 | 19 | 2 | 0 | 11 |
| S3_a12 | NP_LF | 32 | 21 | 0 | 0 | 11 |
| S3_a12 | NP_S2 | 32 | 22 | 1 | 1 | 8 |
| S3_a12 | SITCOM | 32 | 24 | 0 | 6 | 2 |
| S4_a15 | FAST | 32 | 22 | 1 | 0 | 9 |
| S5_r0_fast | FAST | 720 | 422 | 10 | 23 | 265 |

## P4 — all frozen policy tables

The full P4 comparisons, image-cluster bootstrap intervals and
exact McNemar p-values remain in the accompanying R0.7 analysis
JSON. The companion CSV contains mean, q05, Good25 and failures
for all policies and K values; none of these numbers is re-tuned.

## Exploratory casebook

The casebook lists the largest prior repairs, largest harms,
largest GT-orbit gaps and largest remaining oracle gaps
by frozen candidate UID. These are **EXPLORATORY** case selections.
No GT paths or candidate file paths are in the export.

## Limitations and stop gate

- Existing source cohorts are exposed and tuning-adjacent, with
  correlated runs; source overlap is not independent replication.
- The primary report uses fixed thresholds, RGB8 PSNR and
  image-cluster bootstrap, not unseen-confirmation evidence.
- FAST S4 and S5 are pooled for the prespecified FAST gate,
  but the exposed eight-image S4 cohort overlaps known images.
- The planner must evaluate margins and confidence intervals,
  not treat cutoff crossings alone as a publication claim.
- No access to confirmation305 before R2; no unattended
  GPU experiments or R1 generation without a new planner spec.

## Artifacts and continuation

- `R0_8_PROVISIONAL_DECISIONS.json`: machine-readable gates.
- `R0_8_P4_MATRIX.csv`: all source/family/K policy summaries.
- `R0_8_EXPLORATORY_CASEBOOK.json`: path-free examples.
- `R0_7_REPORT.md`, `R0_7_ANALYSIS.json`: complete frozen report.
- Send `R0_8_PLANNER_RETURN.zip` back for Track R planner review.

**Stop reason: PLANNER_R1_DECISION_REQUIRED.**
