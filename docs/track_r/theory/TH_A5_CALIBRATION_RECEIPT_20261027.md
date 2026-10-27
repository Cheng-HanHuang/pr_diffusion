# Track R TH-A5 authorized calibration receipt (2026-10-27)

**Status: CALIBRATION VERIFIED; TH1 v1 forecasts NOT frozen.** Theory branch: theory/prdiff-track-r-th-20261026. No XP return or out-of-calibration run index opened. Source data were read via GitHub connector, not PAC.

## Source and scope

Authorized source: GitHub branch codex/prdiff-track-r-xp-20261026, path docs/track_r/executors/r0n/returns/R0N2_20261026T191819Z/R0N2_PER_IMAGE_COUNTS.csv.
GitHub file: 79,404 characters and 760 data rows. Companion SHA256SUMS.txt records source SHA-256 197d0c8ea2aa47fa17dd379d518f647d7a47d89b3c614449fb106959b7bb45a6, matching planner guidance. The agent read the sidecar and verified structural counts, but could not independently hash original bytes in the connected-tool environment; do not claim an independently computed SHA256.

Retain only 540 image--solver rows: S1_b22 DAPS/SITCOM; S2_b24_dev80 DAPS/SITCOM; S5_r0_fast FAST. B22 100 images and DEV80 80 per family; DAPS/B22 has two runs, others four, yielding 1,960 labeled runs. NP, NP_LF, NP_S2, S3 and S4 excluded.

## Exact A8 calibration counts

| Family | Runs | G | O global | O partial | C | I |
|---|---:|---:|---:|---:|---:|---:|
| DAPS | 520 | 335 | 23 | 1 | 18 | 143 |
| SITCOM-ODE | 720 | 414 | 25 | 0 | 174 | 107 |
| FAST-DIPS | 720 | 422 | 10 | 0 | 23 | 265 |
| Total | 1960 | 1171 | 58 | 1 | 215 | 515 |

From same-image residual-rank pair comparisons (dependent observations):
- G beats C: 200/205, DAPS 21/21, SITCOM 153/158, FAST 26/26.
- G beats O: 50/69, DAPS 21/24, SITCOM 14/30, FAST 15/15.
- G beats I: 570/571, one FAST exception.
- Exactly 100 images/family pairs have G and some other passing class; residual E0 selects G in 92. Under S2, the conditional expected G selections sum to 51.9166667.
Thus S2 is rejected at calibration depth. Pairwise rank examples are dependent within images; they must be block-resampled for uncertainty and 5/205 is not automatically a per-C-run independent extreme-tail probability.

## Preliminary regularized composite Bradley--Terry race fit

G has score-race weight one. O and C logweights have independent Gaussian N(0, 2.5^2) penalties; P is tied to O (only one P observed), I last under approximate S1. Within-image label pairs are pooled by solver, not treated as independent for final inferential intervals.

| Family | wO/wG | wC/wG |
|---|---:|---:|
| DAPS | 0.170403 | 0.020109 |
| SITCOM | 0.953582 | 0.045380 |
| FAST | 0.036579 | 0.023600 |

These are in-calibration composite estimates; not physical-residual scale fits or held-out accuracy claims.

## Theorem and forecasting implications

The exponential class-race integral, limiting weighted ceiling and source-specific interpretation are proved in TH1_RANK_RACE_MATH_20261027.md. The S2 formula remains only an explicit exchangeable-ranks reference.

**Counterexample to a proposed race gap monotonicity:** with P(G)=1/2, wG=1, P(bad)=1/2, wbad=.03, the oracle-minus-race gap equals 0.0386195271 at K=6 and 0.0385396532 at K=7. Hence no nondecreasing-gap constraint should be imposed on the primary PL race or tail forecasts.

Heavy-tail sensitivity is calibrated to the 5/205 C-over-G rank proportion; class-label frequencies and the image-conditioned Dirichlet posterior are shared across reference, PL and tail models. Sensitivity does not identify empirical residual-tail shape from four ranks.

The paper-audit decomposition is exactly:
  1-E1(K) = (1-q)^K + [O_orbit(K)-E1(K)].
The first term is no orbit-good candidate, *not* no residual passer.

Scripts: scripts/track_r/theory/audit_a8_rank_race.py and scripts/track_r/theory/fit_th1_rank_race_v1.py. The forecaster has explicit no-freeze gating pending A8b timing and end-to-end validation; no predictions_v1 have been committed.

## Date gate and related TH2 work

Target v1 freeze: October 29, 2026 18:00 America/Detroit; hard deadline October 31. Check authorized A8b per-run residual data by October 28, 18:00; otherwise use A8 ranks-only and disclose it. Preserve the XP and later-run embargo until planner registers the prediction hash.

The new related exact local-linear sampled-residual analysis is in TH2_OVERWEIGHTED_SAMPLER_NOISE_FLOOR_20261027.md. It verifies the planner's formula for over-weighted posterior samples with flat active directions and explains paired ranking by common-noise cancellation, without claiming the native solvers satisfy Gaussian posterior exactness.
