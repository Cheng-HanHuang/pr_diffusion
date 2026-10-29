# XP-D1 phase-2 prospective v2 / v2-E1 readout (2026-10-29)

**Status: PASSED implementation and SHA/schema audit on PAC, as reported by executor.** This is an explicitly *aggregate-only* immediate scientific return from the pasted local readout; the full frozen comparison JSON, per-image CSVs, and receipts remain on PAC (paths and hashes below). No claim of independent extraction of full local bytes from GitHub. No B16/F8 or E′ prospective outcome was opened.

## Locked evaluation

- 100 ImageNet images, phase 1: 400 historical candidates, phase 2: 1,200 newly generated and receipt-verified candidates (1,600 total).
- CPU scoring and ViT-L/14 CLIP G1/G2: 100/100 images. Frozen v2 and v2-E1 registry/SHA checks passed *before* scoring. Scoring code `052b7886272aa90806e22b731f617990ae7dd8ee` fixes a report-only variable shadowing bug. No change to registered theory forecasts or generated candidates.
- Frozen forecast comparisons: 36 v2 and 24 v2 E1; 60 total. Local `D1P2_SHA_AND_SCHEMA_PASS` verified all three file hashes from completion receipt.
- On PAC: `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/xp_a10_d1p2_fixed_20261029T044818Z/`.
- Comparison JSON SHA-256: `e2c4d4b4993bf8f42a5e87a9978090823ea0aad4c28698ca40a626e9e68ec1e5`. Source of all figures in this return is the executor's validated console extract; full JSON comparison must be kept intact for subsequent paper table.

## Observed Good@20 panel performance (fraction of images with at least one GOOD outcome under each policy)

| K | E0 | E1_CLIP (G1) | E1_CLIP_G2 | O_raw | O_orbit |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.43531 | 0.58769 | 0.58014 | 0.61309 | 0.69977 |
| 12 | 0.47270 | 0.62147 | 0.60847 | 0.66781 | 0.73204 |
| 16 | 0.50000 | 0.65000 | 0.63000 | 0.70000 | 0.75000 |

Rates for K<16 are exact-subset expected values, **not** direct independent cohort observations or per-root confidence intervals. E1_CLIP/G2 evaluates frozen orientation decisions; O_raw and O_orbit are GT-based oracle ceilings.

## Prediction coverage by frozen model variant

| Forecast source | Variant | Covered / total | Maximum |standardized residual| |
|---|---|---:|---:|
| v2 | S2 | 3/9 | 4.873 |
| v2 | race | 6/9 | 2.953 |
| v2 | race_tau1 | 6/9 | 2.965 |
| v2 | tail | 6/9 | 3.151 |
| v2 E1 | S2 | **0/6** | **5.752** |
| v2 E1 | race | **6/6** | 1.140 |
| v2 E1 | race_tau1 | **6/6** | 1.366 |
| v2 E1 | tail | **6/6** | 1.276 |
| **Overall v2** | | **21/36** | |
| **Overall v2 E1** | | **18/24** | |

These are jointly dependent targets across K, policy, and model variant; coverage fractions are descriptive model checks, not independent binomial experiments. The S2 disagreement is substantive; do **not** refit v2/v2 E1 after observing outcomes. Race/tail E1 results are substantially more consistent than S2 for these six predictions, whereas base v2 race/tail remain imperfect.

## Phase-1 versus phase-2 class composition

| Class | P1 rate (n=400) | P2 rate (n=1200) | Paired image-signflip p |
|---|---:|---:|---:|
| G | 0.2500 | 0.2317 | 0.46259 |
| O | 0.1675 | 0.1767 | 0.67192 |
| P | 0.0075 | 0.0092 | 0.75223 |
| C | 0.4700 | 0.4683 | 0.96839 |
| I | 0.1050 | 0.1142 | 0.59880 |

No clear composition shift detected by the reported paired tests. This **does not prove exchangeability of full trajectories or score distributions**; it is a coarse class-frequency check. There are **30 images with zero GOOD candidates among 16**. The on-PAC JSON contains nine composition-conditional E0 comparisons and per-image data not included here.

## Interpretation and next gates

1. Keep the v2/v2 E1 predictions frozen. Present S2 failures, not only race successes. Inspect the nine composition-conditioned E0 comparisons to separate class-count fit from within-class rank/selection errors; audit the 30 never-GOOD images.
2. First planner-requested scientific return now exists in `xp/returns/` at aggregate granularity. For paper use, independently re-read the full PAC comparison JSON and persist full verified detailed digest without printing sealed F8/B16 outcomes.
3. Separate GPU priority: historical **DAPS FFHQ 400×5**, eight-root replay: original frozen B21/DEV80 seeds and code; 0.1 dB raw PSNR and identical-label gate; F8 production only after PASS. As of this report, wave1 Hydra preflight timed out on index 1, **without any GPU worker launched**. Fix timeout and rerun wave1; do not bypass native configuration checks. B32 stays blocked.
4. Maintain A9.4 embargo on B16 and F8 future-candidate outcomes. Timeline: A10 expires Oct 30 noon EDT, shared B16+replay+F8 budget 70 GPU-h; B16 consumed 38.021 GPU-h.
