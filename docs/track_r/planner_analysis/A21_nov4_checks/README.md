# Nov 4 checks (planner, 2026-11-03)

Planner reproduction of the XP executor's Nov 4 publication (XP `953e2f0`), plus one exploratory follow-up.

## Inputs (SHA-256)

| Input | Source | SHA-256 |
|---|---|---|
| Face-test CSV, 11,520 rows | XP `953e2f0`, `returns/XP_A25_NOV4_20261102/XP_A25_FAST_FACE_CLIP_11520.csv` | `1be37fa9…` |
| Executor exact curves | same folder, `XP_A25_A9_2D_FAST_E1_EXACT_CURVES.json` | `6e470cc7…` |
| Executor v1 comparison | same folder, `XP_A25_A9_2D_FAST_E1_V1_COVERAGE.json` | `1b55c007…` |
| 64-run strings with residuals | XP `codex/prdiff-track-r-xp-int-20261028`, `XP_A9_EXCHANGEABILITY_20261028T100727Z/A9_FAST_64_PER_IMAGE_STRINGS.csv` | `d616dd50…` |
| v1 forecasts | theory `8ef5f94`, `TH1_PREDICTIONS_v1.json` | `c19798f9…` |
| 15 panel PNGs | XP `953e2f0`, `png_256/` | each matches the receipt (`../../../paper/track_r_cvpr27/fig/unsolved/SOURCE_PANELS_SHA256SUMS`) |

## 1. A9.2(d) reproduction: `code/a92d_repro.py` → `out/A92D_PLANNER_REPRODUCTION.json`

**Consistency checks.**
- Every face row's label and cohort equal the frozen 64-run strings.
- `f_chosen_good` follows from label and choice in every row.

**Exact E0 and E1 curves.** E1 is the frozen N2 face test applied to the residual pick, averaged exactly over all K-subsets. Maximum difference from the executor: **6.7e-16**.

**v1 E1_CLIP comparison.** 120 comparisons, **all identical** to the executor's.

| Variant | Primary coverage | Secondary coverage |
|---|---|---|
| race | **3/21** | 4/9 |
| S2 | **21/21** | 9/9 |
| race_tau1 | 10/21 | 7/9 |
| tail | 12/21 | 4/9 |

The race forecast is high at every primary point, by 0.8–8.0 points.

## 2. Selection-only check for E1 (exploratory, post hoc): `code/a92d_conditional_e1.py` → `out/A92D_CONDITIONAL_E1.json`

**Design.** This extends `../A9_2b_FAST_conditional_selection/` from E0 to E1.
- Each image's realized classes and each run's realized face-test decision are held fixed.
- The frozen rank laws predict only which class the residual pick falls in.
- Within a class, runs are exchangeable, so predicted E1 = (g1/g)·P(pick G) + (o1/o)·P(pick O).

**Results (pooled).**
- The race is **+0.6 to +0.8** points above the observed E1 (B22 −0.3 to +0.5; DEV80 +1.0 to +1.8).
- race_tau1 is within 0.3.
- S2 is **2.5–4.7** points short.
- The registered race exceeds the conditional race by 2.9–3.5 points.

**Reading.** Test 1's E1 miss is generation; selection is predicted well. This is the same as for E0.

## 3. Figure inputs

- **Fig. 3 (main text):** the first image of each frozen list:
  - FFHQ: B22 `65003`, 15.33 dB; its best run is the lowest-residual run;
  - ImageNet: `ILSVRC2012_val_00049022`, 18.35 dB.
- **Supplement S6:** all 15 images.
- **Tiles:** cropped losslessly by `fig/unsolved/crop_tiles.py`.
- **Orientation check:** where the receipt gives oracle orientation 111 on the lowest-residual run (D4 #4), the oracle tile equals that run's tile rotated by 180°, exactly.
