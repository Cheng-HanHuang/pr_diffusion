# H2 pre-score: F8 DAPS equal-compute stopping advantage (2026-10-29)

**PROSPECTIVE PRE-CP1 DECLARATION.** Authored before this theory executor was provided or accessed the DAPS F8 CP1 stopping outcome, and **without** opening xp/returns or any DAPS run beyond its original A8 historical prefix. The separate SITCOM B16 and ImageNet D1-P2 protocol outcomes are already released; all analyses concerning them below are explicitly **post hoc calibration checks**, not fresh predictions. This entry is not a refit of any frozen v1/v3 forecast or of GTFREE_PROTOCOL_v1.

**H2 locked directional prediction:** On 180 FFHQ F8 DAPS images, for stop-at-M=2, Nmax=8 versus frozen linearly interpolated fixed-K E1 at *the same pooled mean number of runs*, **a positive gain of roughly +2 to +4 percentage points**, with **central rough estimate +3 pp**. Thus the pre-score qualitative prediction is **H2 positive** (>=2 pp) and a larger advantage than the already observed ImageNet D1-P2 **+0.3 pp** and SITCOM B16 **-0.4 pp**. This is a **low-to-moderate confidence** rough forecast, not a 95% predictive interval: the needed physical-residual conditional ranking and per-image latent rate distribution are not identified by the prefix.

## Authorized calibration and zero future labels

- Original A8 SHA-256 197d0c8ea2aa47fa17dd379d518f647d7a47d89b3c614449fb106959b7bb45a6; A8b per-run signed residual SHA-256 e099138f6abaf46b0826377be04be2b0a47de0e5c20fa0847bc62dfd074cd901.
- Retain exactly source S1_b22 100 DAPS images x 2 runs and S2_b24_dev80 80 DAPS images x 4 runs. Do **not** use source S3_a12 eight additional B22 cases, any released FAST64 cross-solver difficulty, B16/F8 later-run labels, or 4-run extrapolations as calibration.
- **DAPS B22 historical 200 runs:** G165 / global O7 / partial O0 / C0 / I28. At frozen signed-residual threshold, pass counts G156/O4/C0.
- **DAPS DEV80 historical 320 runs:** G170 / global O16 / partial O1 / C18 / I115. Threshold pass counts G161/O16/P1/C18.
- Frozen A8 residual race rates (wG=1): wO=wP=0.1704026, wC=0.0201086; I-last surrogate wI=0. Here threshold information is used **only from the historical A8b**, not from future F8 runs.

## Computation and two alternative reduced approximations

Use the frozen S1 Sec.5 iid G/O/P/C/I marked-run truncation, \(s=\sum_j\theta_j v_j\), \(M=2\), \(N=8\), and
\[
\mathbb E T(s)=\frac{2-8(1-s)^7+6(1-s)^8}{s},
\quad
S_{stop}=(1-r^8-8sr^7)H_2^+ + 8sr^7H_1^+ +r^8 H_8^- .
\]
The fixed comparator is the **same fixed-K cohort curve** linearly interpolated at \(\bar K=(100 E T_{B22}+80 E T_{DEV80})/180\). Estimate each cohort's per-class probabilities by its pooled historical counts and \(v_j\) by the historical A8b threshold passes. Orient global O successfully as an *optimistic E1 surrogate*, partial P unsuccessfully, GOOD successfully; the deployed G1 policy is not actually fitted in this reduced model. For conditional passing/nonpassing residual ranks, compare (i) class-exchangeable passing ranks (S2 reduced control) and (ii) passing ranks under the frozen family class-exponential weights. This transfer of unconditional PL weights to threshold-conditioned ranks is an explicit modeling **approximation**, not a consequence of the class-race theorem. A tiny finite-rate exception permits an occasional rejected GOOD candidate to win among the nonpassing scores; no future data are involved.

| Historical-prefix reduced model | Mean runs | Stop E1 | Equal-run fixed E1 | Gain |
|---|---:|---:|---:|---:|
| DAPS, exchangeable passing ranks | 2.835 | 0.9568 | 0.9303 | **+2.65 pp** |
| DAPS, frozen family race weights | 2.835 | 0.9927 | 0.9534 | **+3.93 pp** |

The underlying all-G-pass simplification (vG=vO=1) would give +3.40 pp (exchangeable) and +4.67 pp (race), but **is not the primary prediction**: the actual tau rejects 9/165 B22 and 9/170 DEV80 historical GOOD reconstructions. The two displayed calculations are homogeneous *within each cohort*; the unknown per-image heterogeneity and rank/tau dependence are the major error sources. Do not report 0.9927 as a calibrated probability of success on held F8.

## Already-released post-hoc controls (not prospective)

The **same** approximation applied to frozen historical SITCOM and ImageNet FAST prefixes (same respective solver class-race fits; Nmax=16, threshold pass counts from authorized A8b/P1 K4) yields:

| Solver (held outcome already known) | Exchangeable gain | Race-weighted gain | Actual protocol gain |
|---|---:|---:|---:|
| SITCOM B16 | +0.73 pp | +3.53 pp | **-0.4 pp** |
| ImageNet D1-P2 FAST | +0.15 pp | +1.76 pp | **+0.3 pp** |

These **post-hoc misses of the crude race-weight transfer** warn against treating DAPS +3.93 as a point-calibrated answer. They motivate the central **+3 pp** estimate and substantial uncertainty. DAPS has **far fewer CONSISTENT failures** than SITCOM (DAPS DEV80 C18/I115; SITCOM DEV80 C115/I65), so its pass threshold is more selective of correctness, supporting a larger gain. But a strong full residual rank race can itself decrease the incremental stopping benefit, and image-level mixture effects can reverse the sign.

**Scoring guard:** Assess H2 against the planner's *frozen* equal-mean-run interpolation, using actual F8 full panels only after this note is committed and timestamped. Report sign and numerical miss as a genuine pre-score test, whether H2 passes or fails. This note must not be modified after CP1; any error is a separate erratum.
