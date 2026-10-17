# A1.0 — B22 exposed offline-PSNR oracle evidence and uncertainty

**Study owner:** continuing Track A executor (former A0 executor; role consolidated 2026-10-17). **Status:** `DESCRIPTIVE_ARCHIVE_VERIFIED / NO_NEW_DATA`. This is NOT a prospective study or GPU experiment.

## Data and reproducibility

The original user-returned `A0_B22_METADATA_REPLAY_V3_RETURN.zip` was read directly in a local CPU Python session (no PAC access). Independently verified whole-ZIP SHA-256: `e0641c4c78e05489d8df53dd42587995b2af03460403b4ac3b22928c4b20d242`. Loaded `A0_CANDIDATE_ROWS.csv` (1400 rows, 100 image IDs × 14 archived candidate records) and `A0_POLICY_ROWS.csv` (3600 rows), verified each image's 14 candidates share the same **raw measurement tensor content hash**, checked exact method/candidate cardinalities and replayed the five full oracle policies against archived policy-level PSNR values within 1e-12. The retrospective A0 execution that produced these data passed all 12 synthetic tests and all three historical selected-policy Good25 replays.

**Definition:** for each image, `Q* = max_j raw_RGB_PSNR(candidate_j,GT)` over the *already-archived* candidates in a given named group. Good25 = `Q* >= 25 dB`; Bad20 = `Q* < 20 dB`; q05 = **linear interpolation** of sorted 100-image oracle PSNR values at probability 0.05. These are offline GT diagnostics; cannot be chosen without ground truth. All inputs are float-tensor B21/B22 metric representation `B21_B22_FLOAT_RAW_RGB_V1`, not B24 8-bit saved PNG metrics. No metric recomputation or GT reading.

## Exact B22 exposed-panel summary

| Offline oracle portfolio | K | Good25 | Bad20 | Mean dB | Median dB | q05 dB | Minimum dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| DAPS2 historical Fresh2 candidates | 2 | 92/100 | 8/100 | 29.427 | 30.948 | **15.883** | 6.482 |
| SITCOM4 historical sequential RNG stream | 4 | 95/100 | 3/100 | 26.667 | 27.224 | **24.944** | 6.787 |
| NP LF-only4 | 4 | 94/100 | 4/100 | 29.022 | 29.992 | **23.895** | 10.263 |
| NP S2-only4 | 4 | 94/100 | 4/100 | 29.061 | 30.009 | **23.816** | 11.918 |
| NP LF4+S2-4 mixed8 | 8 | 96/100 | 2/100 | 29.407 | 30.047 | **25.474** | 12.610 |

**Historical runtime/representation caveats:** DAPS4 unavailable; B22 SITCOM4 is four trajectories in one sequential RNG stream, **not** independently reseeded four runs; NP mixed8 contains 4 LF plus 4 S2 candidates, **not** eight i.i.d. retries of a single NP policy. B22 DAPS2, SITCOM4, NP S2-only4, NP mixed8 total *candidate reconstruction time sums* respectively ~312.64, 196.07, 206.46, 411.78 seconds/image; these are not calibrated full end-to-end cost or equal-work claims.

## Predeclared descriptive paired comparisons

**Uncertainty computation:** Paired image bootstrap with 10,000 resamples, NumPy `default_rng(5401)`, common random resample index matrix, percentile 95% intervals for mean and q05 contrasts and Good25 success-rate differences. Good25 rescues/harms and two-sided exact McNemar are computed on original 100 paired images. Bootstrap percentile intervals for q05 with only a few catastrophes are particularly unstable and are **not** a power demonstration for future work. Values are rounded for presentation.

| Comparison, first minus second | Mean PSNR Δ (95% CI), dB | Good25 Δ (95% bootstrap CI), pp | Discordant rescue/harm; McNemar p | q05 Δ (95% bootstrap CI), dB |
|---|---:|---:|---:|---:|
| NP mixed8 − DAPS2 | −0.020 [−0.951, +1.058] | +4 [−2,+10] | 7/3; 0.344 | +9.592 [−3.352,+17.701] |
| NP mixed8 − SITCOM sequential4 | +2.740 [+2.261,+3.308] | +1 [−3,+5] | 3/2; 1.000 | +0.531 [−2.281,+12.245] |
| **NP S2-only4 − SITCOM sequential4** | **+2.394 [+1.714,+3.094]** | **−1 [−6,+4]** | **3/4; 1.000** | **−1.127 [−9.875,+11.260]** |
| NP LF4 − NP S2-only4 | −0.038 [−0.726,+0.635] | 0 [−4,+4] | 2/2; 1.000 | +0.078 [−11.400,+9.572] |
| NP mixed8 − NP S2-only4 | +0.347 [+0.009,+0.859] | +2 [0,+5] | 2/0; 0.500 | +1.658 [0,+10.377] |

At **equal K4**, NP S2 has an obvious retrospective *mean* advantage over the B22 sequential SITCOM stream but **not** an established Good25 or q05 advantage. NP mixed8 at approximately twice SITCOM4's recorded candidate seconds cannot be called equal-compute superior merely because its Good25 score is one point higher. NP mixed8 oracle q05 is above 25 dB, whereas DAPS2's is below it, but the paired q05 bootstrap difference crosses zero and DAPS4 was not measured.

The nested mixed8 vs S2K4 oracle has no Good25 harms by construction because S2K4's candidate set is a subset of mixed8; an apparent bootstrap improvement in such a paired comparison is not a fresh independent-validation conclusion. Any paper conclusion must use a **new separately reserved and frozen cohort**, independent native restarts, and total charged multi-run cost.

## Published benchmark convention checked against source text

- DAPS, §4.1 ([arXiv v3](https://arxiv.org/html/2407.01521v3)): phase retrieval reports the best of four **independent** samples, oversampling 2.0.
- SITCOM, Appendix G ([arXiv v2](https://arxiv.org/html/2410.04479v2)): phase retrieval reports best results from four **independent** runs for baselines; its Table 1's displayed runtime is **one run**, not the total charge for a best-of-four evaluation.

These conventions make an **offline oracle best-of-K** result legitimate paper-style quality reporting, provided full multi-run cost is clearly charged for computational comparisons. The oracle is **not an executable final selector**. Historical native GT-free selection remains a separate secondary estimand.

**Scientifically binding next step:** read-only source/provenance and metadata preflight for A1 prospectively; *no* GPU, new cohort selection, new measurement or B24 confirmation305 access at this stage.
