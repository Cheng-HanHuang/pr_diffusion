# Track A A1.2 — 128 independently seeded native trajectories: independently verified scientific analysis

**Run:** A1.2 B22 EXPOSED DEV8, eight fixed *previously exposed* FFHQ images, 4 native starts × 4 method families/image = 128 independent full processes. **Analysis date:** 2026-10-20. **Status:** `PASS_EXECUTION_128_OF_128 / ORACLE_FINDINGS_EXPLORATORY`. **No pristine/held-out claims.**

## 1. Independent archive and candidate audit

Received PAC stdout terminal (start `2026-10-20T02:34:01Z`/Oct19 22:34 EDT) and uploaded `A1_2_K4_GPU_METADATA_RETURN_V1.zip`. SHA256 of **actual uploaded ZIP bytes** exactly `e82ad0c0314b4cff0fed82db656500307cd82f6927f42c7bc667f94878d7e9fc`, matches terminal; ZIP CRC `testzip()==None`; **368 ZIP members**: 128 per-candidate COMPLETE.json, 128 CUDA timers, 96 native SITCOM/NP result.json, 8 image-level COMPLETE, `RUN_IDENTITY.json`, `A1_2_COMPLETE.json`, `A1_2_OFFLINE_ORACLE_SUMMARY.json`, main log and four tail-only native logs. Console's `GPU_RECORD_COUNT 365` is number of metadata path list entries before dedup and before four tail entries (not a candidate count); the ZIP member count of 368 is expected.

`A1_2_COMPLETE.json` status `PASS_A1_2_EXPOSED_B22_128_COMPLETE`; all 128 candidates reported PASS; `DAPS,SITCOM,NP_LF,NP_S2` each 32. Independently checked the 128 candidate receipts against all 128 original plan entries for `candidate_id,method,image_id,rep,native_seed32,measurement_tensor_sha256,ground_truth_tensor_sha256`: **zero discrepancies**. There are 128 distinct recorded terminal SHA256 digests, all 128 valid CUDA timers with `status=0`, no `A1_2_STOP.json` in archive, and no terminal error. Fixed per-image measurement content hashes agree with earlier frozen A1.2 input manifests.

For **all 96** available SITCOM/NP native result.json records, independently verified `measurement_raw_content_sha256`, GT tensor content SHA, source HEAD, model SHA, frozen method-specific observation-preprocessing descriptor, native seed and finite-output flag against associated A1 candidate receipt: **zero discrepancies**. DAPS per-trajectory log files and terminal images are not in this metadata-only ZIP; their **parent-reported** success, source pin, recorded terminal hash and CUDA timer are validated, but image/tensor bytes cannot be independently rehashed absent PAC access. Archived native float-result PSNR differs from uniform saved-PNG8 PSNR by <0.018 dB across inspected native SITCOM/NP result records, as expected from 8-bit quantization; use homogeneous PNG8 for cross-method comparisons.

Source identity A1 `f4cd28dcd3ec4860cc9f3990fb24964bb97e97bd`, original frozen DAPS `e7a77d094167084faed19b599b96673b7bb11447`, SITCOM `275ab67efbd8146bffca20155171ba6be1169c09`, DiffFPR `a45ffe58f18fed8a63d3446600424e2b08733524`, model SHA256 `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`. Physical GPU3 `GPU-7d65c050-d7e8-5a6b-ee38-1d72d7a5696a` initially 0 MiB used / 97,250 MiB free; peak observed full-device usage **1,667 MiB**, peak Torch reserved **966 MiB**, both below reservation limits. Full wall 5,432.002s=90.53min; sum subprocess walls 5,417.332s and CUDA event elapsed 4,833.754s. CUDA events measure bracketed GPU-stream elapsed (not direct SM utilization); procedural runtime estimate ~1.53 GPUh was accurate for this dev panel (observed ~1.51h).

All 8 image source IDs in frozen hash order: `69073,68539,67948,60140,67293,67174,61101,66631`. No claim of new data: they are *already-exposed B22 validation images*. The protected B24 confirmation305 cohort remains entirely off limits; no inferred independence from that cohort.

## 2. Native per-method compute

| Method | Candidates | Sum process wall s | Mean per candidate s | Sum event elapsed s | Peak Torch reserved MiB | Peak device used MiB |
|---|---:|---:|---:|---:|---:|---:|
| DAPS | 32 | 2080.85 | 65.03 | 1905.41 | 966 | 1667 |
| SITCOM | 32 | 979.85 | 30.62 | 856.52 | 942 | 1643 |
| NP-LF | 32 | 1179.24 | 36.85 | 1043.57 | 946 | 1647 |
| NP-S2 | 32 | 1177.40 | 36.79 | 1028.25 | 946 | 1647 |

K4 process wall **per image** DAPS 260.11s, SITCOM 122.48s, NP-LF 147.40s, NP-S2 147.17s. DAPS K4 ~1.76× NP-LF K4 wall at this fixed native configuration, so highest oracle PSNR is NOT highest quality per unit time without additional analysis.

## 3. Frozen uniform saved-PNG8 raw-PSNR offline oracle summaries

These are ground-truth **oracle maxima across independent full trajectories**, *not executable restart selectors*. Primary scoring raw orientation of saved 8-bit RGB PNG against quantized GT, rotation-180-aware separately.

| Family | K | Mean raw oracle PSNR dB | Good25/8 | Good30/8 | Sum charged wall s |
|---|---:|---:|---:|---:|---:|
| DAPS | 1 | 21.805 | 4 | 4 | 555.39 |
| DAPS | 2 | 28.634 | 7 | 7 | 1068.65 |
| DAPS | 4 | **31.518** | **8** | **8** | 2080.85 |
| SITCOM | 1 | 22.072 | 5 | 0 | 245.93 |
| SITCOM | 2 | 27.425 | **8** | 0 | 487.83 |
| SITCOM | 4 | 27.439 | **8** | 0 | 979.85 |
| NP-LF | 1 | 21.145 | 4 | 2 | 295.82 |
| NP-LF | 2 | 23.963 | 5 | 2 | 612.38 |
| NP-LF | 4 | **30.388** | **8** | 4 | 1179.24 |
| NP-S2 | 1 | 21.273 | 4 | 3 | 279.37 |
| NP-S2 | 2 | 26.409 | 6 | 3 | 583.59 |
| NP-S2 | 4 | 30.222 | **8** | 4 | 1177.40 |
| NP mixed | 4 (LF0,S20,LF1,S21) | 26.467 | 6 | 3 | 1195.97 |
| NP mixed | 8 (LF0,S20,...,LF3,S23) | 30.392 | **8** | 4 | 2356.63 |

Good25 is raw PSNR ≥25dB, Good30 ≥30dB. Arithmetic independently recomputed from all candidate metrics: **all 8×12 homogeneous method/K oracles and 8×2 NP mixed oracles** and each selected-prefix wall charge matched the reported outcomes with zero discrepancies. Mixed portfolios merely reuse NP candidates; the *marginal* GPU cost of computing NP mixed8 after both NP sets already exist is zero, but **the compute charged to NP mixed8 as a standalone algorithm is all eight candidates**, i.e. 2356.63s across eight images, roughly 2× NP-LF K4.

## 4. Detailed per-image K4 raw-PSNR GT oracle (dB)

| Exposed ID | DAPS K4 | SITCOM K4 | NP-LF K4 | NP-S2 K4 | NP mixed8 |
|---|---:|---:|---:|---:|---:|
| 69073 | 32.18 | 27.82 | 31.17 | 31.19 | 31.19 |
| 68539 | 32.23 | 28.04 | 31.31 | 31.24 | 31.31 |
| 67948 | 30.82 | 27.08 | 29.91 | 29.82 | 29.91 |
| 60140 | 30.78 | 27.06 | 29.63 | 29.51 | 29.63 |
| 67293 | 30.54 | 26.93 | 29.48 | 29.27 | 29.48 |
| 67174 | 30.40 | 27.11 | 29.11 | 28.23 | 29.11 |
| 61101 | 33.31 | 28.07 | 31.86 | 31.86 | 31.86 |
| 66631 | 31.89 | 27.41 | 30.64 | 30.64 | 30.64 |

DAPS K4 strictly outperforms NP-LF K4 on all eight images; mean paired margin **+1.130 dB**, range **+0.904 to +1.454 dB**. DAPS exceeds SITCOM K4 on all eight by a mean **+4.079 dB**. However DAPS K4 is ~1.76× as expensive as NP-LF K4, so compare equal wall budgets and executable selectors separately. These eight images are exposed, small, and not randomized for inference. No population p-value should be used as a publication claim.

SITCOM K2 → K4 improves mean raw oracle PSNR **only 0.014 dB** for double compute; it already achieves Good25 8/8 at K2, but has lower PSNR ceiling (~26.9–28.1). Threshold Good25 at K4 is saturated for *all four families*, making stricter thresholds (Good30), actual executable selection, and compute-quality curves more discriminating.

NP-LF K4 vs NP-S2 K4 mean **30.388 vs 30.222 dB**. NP mixed8 vs NP-LF K4 yields only **+0.0040 dB mean** despite doubling charged cost. LF K4 is already best or tied on 5/8 images and mixed8 only improves 3/8 by ≤0.0213dB. This suggests no useful cross-variant complementarity on the current saturated oracle score; stronger S2 advantage is NOT evidenced by this sample.

## 5. Why multi-start gains are so dramatic: symmetry vs genuine poor basins

Across 32 independent draws/family, raw Good25 candidates: DAPS **19/32**, SITCOM **24/32**, NP-LF **21/32**, NP-S2 **22/32**. When also allowing an offline 180-degree rotation GT comparison, Good25 candidates become DAPS **21/32**, SITCOM **24/32**, NP-LF **21/32**, NP-S2 **23/32**. Thus both orientation-ambiguous and other failures occur. A low raw score is **not synonymous with a poor physical measurement fit or wrong basin**.

Extreme example image **68539**: DAPS rep0 raw **6.36 dB**, 180°-rotated **31.83 dB**; DAPS rep2 raw ~6.36 but rotated ~31.87. A raw oracle at K2 sees poor outcomes (K2 best raw **9.39 dB**), while an ambiguity-aware K2 oracle reaches high-quality ~31.83dB. Thus raw DAPS K2 mean **28.634dB** with Good25 7/8, but *offline ambiguity-aware K2* mean **31.44dB** and Good25 **8/8**. DAPS ambiguity-aware K4 mean **31.52dB**, an additional ~0.08dB over K2, not the raw +2.88dB appearance. Any GT-oracle rotation is **not a deployable orientation selector**. Fourier magnitude for real signals is invariant under conjugate inversion/180° reversal (modulo discretization and centering), so raw measurement fitting cannot resolve orientation alone; learned prior or explicit nonoracle orientation policy is necessary. Rotation-180 is only one of the possible phase-retrieval ambiguities and need not repair all low-PSNR trajectories.

There are genuine/other wrong-basin cases as well: for image **67293** DAPS rep1–rep3 have raw ~11.4–11.6dB and rotated only ~14–14.5dB, contrasting DAPS rep0 ~30.54dB. On image **67174**, NP-LF K1–K3 remain raw ~12.7–13.1dB, only rep3 reaches **29.11dB**; 180° rotation does not explain that low-score sequence. On image **68539**, NP LF rep2 reaches 31.31, NP S2 rep2 reaches31.24; both require ≥3 independent runs for raw success.

Consequence: avoid interpreting K1→K4 raw PSNR improvements as all genuine optimization gains; separate symmetry/orientation flips, other basins, and GT-free candidate selection. The vast K improvements are GT oracle maxima, not evidence the native method can pick the right solution.

## 6. Method/source quality audit and limitations

- Source frozen native solvers and same per-image saved observation hashes; **method-specific preprocessing differs by design**. SITCOM uses raw signed magnitude; NP clamps negatives to zero in memory. Across the eight real images, NP native logs report clipping roughly 161,780–174,249 / 442,368 entries per image, vs ~46.8% negatives on the prior procedural probe. This must stay visible in protocol discussion rather than calling numerical solver inputs bit-identical.
- NP and SITCOM child `result.json` files report `finite_output=true`, consistent raw measurements/source/checkpoint/seed hashes for all 96 child records. Legacy native methods have GT tensors available for evaluation / shape-based initialization, so this pipeline identity check alone does not certify a new end-to-end GT firewall.
- Native output float PSNR vs uniform PNG8 offline PSNR differs at most ~0.018dB on available SITCOM and NP child records. DAPS unquantized float output is still not captured uniformly across arms; don't silently replace original prospective float metric with PNG8 after outcomes.
- **Crucial central research question remains unanswered:** Can NP's **GT-free native selection** outperform DAPS/SITCOM with their GT-free native choices at comparable GPU/time costs? This archive provides candidate trajectories and GT oracle maxima, not a validated measurement/prior-based best-of-K **deployable selector** across restarts. Native within-trajectory NP selector scores are not the same as selecting one of 4 independent runs. Different methods' objectives contain different regularizers, so comparing raw measurement error as a universal ranking score might be systematically unfair.
- Cohort n=8 was drawn from the **already exposed** B22 final100; cannot be called prospective heldout, confirmation305-disjoint, or statistically representative. B24 protected confirmation305 IDs/metadata must never be opened even for membership checks.

## 7. Proposed next stage (NO new GPU needed initially)

1. **A1.3 GT-free selection audit on archived A1.2 reconstructions:** Work only within A1 own `a1_2_exposed_b22_k4_v1` and already-allowed B22 inputs. Extract per-candidate physically correct measurement losses and method-specific native scoring/regularization where available (do not assume a single measurement-error rule represents native DAPS/SITCOM/NP). Freeze one candidate-selection rule **without GT**, then evaluate the chosen candidate offline against PNG8 PSNR, also reporting regret to the GT oracle and 180° ambiguity-aware metrics. A missing selector must be honestly stated, not replaced with the oracle.
2. **Same-cost frontier:** compare SITCOM K2 (~61s/image), DAPS K2 (~134s/image), NP-LF K4 (~147s/image), NP-S2 K4 (~147s/image), along with K1 and K4 at their actual costs. Include orientation-aware GT diagnostic *separately* from raw primary (never pretend this orientation correction is executable).
3. **Basin classification:** from saved images/tensors classify low-PSNR candidates into near-exact 180° conjugate-reflection vs other measurement-consistent basins vs high residual/nonconvergence, rather than attributing every low raw PSNR to numerical optimization failure. Compare raw/rot180 measurement residuals and visually inspect a small sanctioned set if required.
4. **Publication pathway:** resolve provenance-independent truly pristine dataset/custodian without touching protected B24 confirmation305 and prospectively freeze common float-vs-PNG8 scoring before any new GT. Next GPU experiments should answer a specific selection or basin hypothesis, not simply accumulate more GT-best-of-K oracle wins on eight already-exposed faces.

**Decision:** `A1_2_GPU_PASS`; Track A candidate-generation engineering complete. No need for another K4 GPU run now. Prioritize GT-free policy/regret and symmetry decomposition to determine whether NP has an actual method advantage; DAPS K4 oracle currently leads in reconstruction PSNR at higher compute.
