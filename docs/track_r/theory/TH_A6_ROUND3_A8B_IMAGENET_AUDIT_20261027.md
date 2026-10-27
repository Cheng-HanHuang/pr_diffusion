# Track R Round 3: authorized A8b and ImageNet calibration audit (2026-10-27)

**Status: INPUT HASHES AND A8B RESIDUAL QA PASS.** No XP returns, later FFHQ run indices or phase-2 ImageNet outcomes accessed. The following SHA-256 hashes were recomputed from original user-uploaded files in the container, not copied from receipts.

| File | Rows | SHA-256 |
|---|---:|---|
| R0N2_PER_IMAGE_COUNTS.csv | 760 | 197d0c8ea2aa47fa17dd379d518f647d7a47d89b3c614449fb106959b7bb45a6 |
| R0N2_A8B_PER_IMAGE.csv | 760 | e099138f6abaf46b0826377be04be2b0a47de0e5c20fa0847bc62dfd074cd901 |
| R0N2_A8B_TAU_BY_SOURCE_FAMILY.csv | 12 | a86d625ec574614ad42131ae648b84beefe7ce8cb340516ce6a2ae65f20db9f5 |
| XP_D1_P1_K4_A8_STYLE_PER_IMAGE.csv | 100 | c21778803a17821ecf51857af6378273c2b583d4d19938d0c42e401f20178bc1 |

A8b's first 19 columns match the old A8 values and row order exactly in all 760 rows. A8b raw signed-rho ordering matches e0_rank_order for all 540 eligible DAPS/SITCOM/FAST image-source rows, and C/I classifications exhibit ZERO contradictions to per-source tau. Where printed 6-sigfig residuals are equal, use the original stored rank permutation as authoritative.

## Within-image G/C gap audit

The 205 paired comparisons share some images and runs, so they are NOT statistically independent:

| Family | Paired G/C comparisons | C wins | median rho_C - rho_G | min gap |
|---|---:|---:|---:|---:|
| DAPS | 21 | 0 | 0.00016520 | +0.00002760 |
| SITCOM | 158 | 5 | 0.00034185 | -0.00001180 |
| FAST | 26 | 0 | 0.00011035 | +0.00000410 |

Five inversions all come from SITCOM; even its negative raw residual margins are extremely small in physical units. The predeclared pooled 5/205 rare-C tail remains a stress scenario for DAPS/FAST, not a fitted family-specific heavy tail. A8b supplies additional *permitted* source score information, but alone cannot identify a K64 extreme tail.

## ImageNet v2 status (Good20)

The authorized XP-D1 phase-1 theory input gives 100 fixed images, four native FAST-DIPS runs each: G100, global O67, partial O3, C188, I42. The receipt states that CLIP G1 decisions are still pending, no phase-2 outcomes were evaluated, and no forbidden FFHQ higher-K results are included. Two calibrated original ImageNet ranks violate strict I-last: image IDs ending 49005 (rank CCIO) and 49099 (rank CIGI). Preserve those known prefix rankings and treat strict S1 as approximate. The v2 target is a completed 16-run panel with K=(8,12,16), and E1_CLIP must be explicitly ABSTAINED if no ImageNet G1 decision data are released before freeze; FFHQ's 58/58 repaired global orbit examples do not transfer automatically.

## A6 control variate and QA status

Planner's six-line correction was committed to fit_th1_prefix_v1.py on theory branch @83220764b1f0aa862b60f46e3a0f32e1a369a1a7. The exact oracle and selector predicted values are now shifted by the same per-panel Monte Carlo oracle discrepancy, so small fluctuations cannot make the selector forecast exceed its own oracle. A separate local Python/NumPy/SciPy reproduction of the core conditional-prefix model using authentic A8, exact subset weights, PL class rates and the same four variants passed smoke controls in DAPS primary n8, SITCOM primary n16 and FAST primary n64, for both B22 and DEV80.

**Distinction:** The local smoke was independently coded, not a complete run of the current GitHub production file. The planner previously ran the immediately preceding version end-to-end (3m17s), observing the 0.00044 false bound violations; its supplied six-line patch is now applied. The full production validator check_a6_panel_contract.py still must be run on the updated A6 output and report PASS 976/252/60 before the final SHA-256/commit freeze. No TH1_PREDICTIONS_v1 file is represented as frozen by this report.

**Next priority:** produce/register the TH1 v1 forecast on or before Oct 28 18:00 ET; proceed independently with ImageNet v2 Good20 calibration within the 48-hour phase-1 deadline. Do NOT read XP return directories or out-of-prefix runs until the planner registers the forecast artifacts.
