# Track R theory response to A6 planner guidance — October 27, 2026

**Status:** A6 source changes committed; TH2 v1.1 preregistered on theory branch; TH1 v1 is UNFROZEN pending A8b deadline and original-file end-to-end QA. No XP return or post-calibration seed was read.

## A6 contract delivered (code, awaiting full-file validation)

The A6 forecaster is at scripts/track_r/theory/fit_th1_prefix_v1.py. It preserves the observed A8 labels and original within-image rank order and simulates unknown future runs, then computes exact K-subset U statistics on completed panels.

| Family | Panel | Full nmax | K grid |
|---|---|---:|---|
| DAPS | primary | 8 | 4,6,8 |
| SITCOM | primary | 16 | 8,12,16 |
| SITCOM | secondary | 32 | 8,12,16,24,32 |
| FAST | primary | 64 | 8,12,16,24,32,48,64 |
| FAST | secondary | 16 | 8,12,16 |

Four variants: (i) race as primary E0/E1 with family-specific weights; (ii) S2 exchangeable-rank reference; (iii) pooled 5/205 extreme-C tail sensitivity, explicitly STRESS ONLY for DAPS and FAST because all five calibration C-over-G wins belong to SITCOM; (iv) race_tau1 smaller Gaussian prior standard deviation, 1 instead of 2.5, to expose complete-separation shrinkage.

The JSON marks interval_type as predictive_realized_panel. The registered comparison rule counts primary-panel E0 and E1_CLIP grid points within [lo95,hi95], with mean abs(observed-point)/max((hi95-lo95)/3.92,.005) as tie-break. Secondary panels are separate and no sensitivity variant can replace the predeclared primary retrospectively.

Anatomy fields report generation failure = 1-O_orbit (no correct-orbit candidate among the K subset), selection loss = O_orbit-E1_CLIP, and first strict grid crossover; null means no crossing within that grid. The A6 output contract validator scripts/track_r/theory/check_a6_panel_contract.py expects 976 policy rows, 252 anatomy rows, 60 crossover records, and tests the per-policy selection inequalities and decomposition. The validator MUST be run after the modified forecaster on the actual authorized raw A8; these new edits are not yet end-to-end executed here. The earlier A5 script was run successfully by the planner, but this does not certify the A6 edits.

## TH2 v1.1 mechanism file (separate freeze)

- File: docs/track_r/theory/TH2_MECH_PREDICTIONS_v1_1.json
- Commit SHA: 5abad48dfeb759550c0954f286f73c0ac2eeeb58
- File SHA-256: 11262301fc00bb0c9483639719564c7f2e04bd8fa0fc146cef2c8477324b758d
- UTC commit time: 2026-10-27T15:56:47Z
- No preregistered A5.5 outcomes seen. **Planner: register this new mech v1.1 row before sharing A5.5.**

TH2_A6_DC_DELTA_LAW_20261027.md rigorously gives, in a *flat local Gaussian posterior sampler*, the independent chi-square law Delta = (1/kappa) chi_square_r - chi_square_r, together with E D/r=1+1/kappa, E C/r=1, E Delta/r=-(1-1/kappa), all three variances, an exact F sign probability and a bound. This is not asserted of the finite-step native solvers. For a prior-dominated FAST denoiser output that correlates negligibly with noise, C~0 and Delta~D is a qualitative conditional test. An exact rotation of a candidate has identical D and C, so DAPS/FAST G-vs-O score preference is attributable to differing *generated* off-orbit perturbations rather than discrimination between exact copies.

## Data and input blocker

- Original A8 was verified by planner from actual raw bytes with source SHA-256 197d0c8ea2aa47fa17dd379d518f647d7a47d89b3c614449fb106959b7bb45a6.
- Current permitted GitHub R0N return tree does **not yet** show A8b; new permitted ImageNet theory_inputs directory also not present.
- Local mount has an anonymized minimal fixture, /mnt/data/track_r_A8_minimal_authorized.csv, containing 540 image-family label/rank rows but **NOT** the original raw A8 bytes. It cannot pass the original-file hash guard, and it lacks original index order and other decision fields.
- **Request for Cheng:** upload original permitted A8 CSV and, when released by Oct 28 12:00 ET, A8b K<=4 CSV into this conversation if an independent raw-file verification/freeze in this runtime is desired. Do not upload embargoed XP returns or seeds beyond calibration.

TH1 v1 is NOT frozen. The A8b gate (Oct 28 noon ET), A6 end-to-end run, final code commit, output file hash, UTC time, and planner registration are all needed before releasing XP K>4 outcomes.

## Paper positioning

The differentiated contribution remains an experimentally tested, cross-native-solver, physics-specific decomposition and prospective held-out K predictions. Classical weighted-score racing, Stein identities and Gaussian/Fourier symmetry are credited. No further generic toy proofs proposed.


## Update after Cheng uploaded original A8 (2026-10-27)

**The original-A8 access blocker is RESOLVED.** The file /mnt/data/R0N2_PER_IMAGE_COUNTS.csv was mounted and the byte-level SHA-256 independently recomputed: 197d0c8ea2aa47fa17dd379d518f647d7a47d89b3c614449fb106959b7bb45a6, exactly the prereg source checksum. It has 79,404 bytes and 760 data rows. Retaining the 540 allowed rows reproduces 1,960 run counts and family/cohort sizes. The text above about the minimal anonymous fixture as the only local file was accurate at initial A6 handoff but has now been **superseded**.

A separate independent local Python/NumPy/SciPy reconstruction of the fitted alpha, regularized Bradley--Terry rank weights, observed-rank-conditional exponential scoring, completed-panel simulation, and exact K-subset selection estimators **passed a raw-A8 primary-panel smoke test for all four variants**. No real future-run results or XP returns were read. The current upstream GitHub forecaster file itself has not been executed locally; the planner's production A6 run and check_a6_panel_contract.py must still be completed.

The A6 theory has also improved: the posterior conditional mean of every K-subset oracle on the completed panel is an **exact beta-hypergeometric finite sum**, independent of the rank-selection model. See TH1_A6_EXACT_PREFIX_ORACLE_20261027.md and newly committed tests. The GitHub fitter now inserts these exact O_raw and O_orbit point forecasts and recenters each model-conditional *realized-panel predictive interval* around them, keeping actual future-panel sampling uncertainty intact. The A6 validator now requires oracle point-forecast equality across variants.

The original per-run CLIP calibration has additionally been inspected: all **58/58 allowed global ORBIT runs** have clip_fix_ok=1. The primary E1_CLIP idealized global-orientation repair is therefore consistent with observed A8 for this family set (there is one partial-ORBIT, not globally correctable). This does NOT establish future perfect repair from the small calibration sample.

**Do not freeze v1 before the A8b timing gate:** its authorized GitHub publication directory remained absent at last check, and it may be used only if it arrives before Oct 28 noon ET. This appendix does not change the separate TH2 mech v1.1 freeze SHA or access rules.
