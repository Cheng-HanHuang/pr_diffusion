# A1.6 postrun scientific audit — four-GPU NP × FAST engineering study

**Date:** 2026-10-21. **Status:** Independent metadata audit of the user-operated completed A16 run. **No new GPU experiments executed by auditor.** Historical execution branch and artifacts remain unchanged. Analysis branch: `codex/prdiff-track-a-a16-results-audit-20261021` (separate from A16 execution HEAD).

## 0. Provenance, completion, and verification

- The user uploaded `A16_COMPLETE_METADATA_RETURN.zip`; SHA-256 independently computed: `12768e9eedd49b73ba6608c81a8e7521414e666e650b81f6760c0fd307d897df`. ZIP CRC `testzip()` PASS; **1063/1063 manifest-listed metadata entries have matching SHA-256 and byte lengths**, 1064 total ZIP entries including the manifest. No image, checkpoint, or GT tensor supplied in this metadata archive.
- `A16_CPU_PLAN_PASS` receipt SHA `b7a87c185a88cb1fee1d86c18aae456c58dbd97461be3c01fa1f6a6e720e5db6`; four-lane `A16_STAGE0_PASS`, `A16_STAGE1_PASS`, `A16_STAGE2_PASS`; `A16_AUTOPILOT_COMPLETE`.
- **260/260** independent candidate completion receipts are `A16_NP_CANDIDATE_PASS`, all source HEAD **`66fad4a6fafcf7f224b2e8cb6e388137b18d7569`**, all `gt_used_only_after_sampling=true`; 260/260 full subprocess wall receipts and 260/260 worker logs present, no reported traceback or worker error. Native NP-LF reproduction smoke `np_native_replay_maxabs=0.0`, original 2200 U-Net calls.
- All four assigned physical GPU UUIDs represented. Count and aggregate per-lane subprocess time: GPU0 72 roots/~2248s; GPU1 72/~2228s; GPU2 56/~995s; GPU3 60/~483s. The sum of worker walls is NOT total elapsed time because GPUs run concurrently; does not include controller interstage overhead.
- Dataset is **exact eight already-exposed FFHQ B22 engineering images**, repeated rep0–3 for promoted arms. Stage0 selected four images based on known FAST successes/failures, and Stage1/2 promotion optimized a GT-informed development-only utility; **no prospective validation**. The protected B24 confirmation305 cohort was not part of the stage receipts.

## 1. Principal final-stage comparison — same eight exposed images

All K4 quality values below are mean *per-image* raw oriented rounded-RGB8 **GT best of four**, **nondeployable oracle**. Times are **total four subprocess walls per image**, including repeated model loads/imports. **Important smoke-accounting exception:** the first `lf_native/69073/rep0` subprocess included a dedicated 27.644962-second FULL native NP replay for source equivalence. The archive's `lf_native` 127.29265 s/K4-image is therefore the actual observed research resource bill; subtracting 27.644962/8=3.45562 seconds per K4 image gives ~**123.84 s** for the comparable *method-only four-root subprocess cost*. No other arms include this replay. The code's stage promotion uses sampling-wall time (not this inflated subprocess figure), so its selections are not influenced by this bookkeeping difference. U-Net counts in this table represent the method's four scored roots, not the extra fidelity-only replay. Failure rates in the next column are across all **32 generated individual roots**, not image-level K4 outcomes.

| A16 arm | Mean raw K4 oracle PSNR | Bad20 per root | K4 recorded subprocess wall/image | K4 U-Net calls/image | Good25 among eight at K4 |
|---|---:|---:|---:|---:|---:|
| `lf_fast_post` | **30.4151** | 9/32 | 123.30s | 8800 | 8/8 |
| `lf_native` | 30.2307 | 9/32 | 127.29s **(123.84s adjusted)** | 8800 | 8/8 |
| `lf_fast_prescore` | 30.1910 | 12/32 | 123.54s | 8800 | 8/8 |
| `lf_beam2` | 30.2209 | 9/32 | **71.51s** | **4392** | 8/8 |
| `s2_native` | 25.7190 | 17/32 | 124.24s | 8800 | 6/8 |
| `lf_150_fast` | 27.0045 | 16/32 | **28.31s** | 696 | 7/8 |

**Separate prior A1.5 native FAST-DIPS reference:** K4 GT oracle 31.5251dB; Good25 8/8; K4 sampling-only 14.851s/image, and four-root **program wall after imports** 32.4396s/image. Do not divide these 32.44s by the A16 *whole subprocess* wall and call an exact hardware-matched speed ratio; FAST's actual external four subprocess walls on previous K1 subset were roughly 4×9s, still far below 71–127s for A16 methods. FAST K2 already achieved 30.5253dB and Good25 7/8 with **two** roots costing 7.46s/image *sampling-only*; this is a separate budget and cannot be directly compared with four-root subprocess wall.

## 2. Paired K4 comparison by image

| image | NP-LF native | NP-LF + FAST post | NP corrected prescore | NP two-branch | NP-S2 native | NP 150-fast |
|---|---:|---:|---:|---:|---:|---:|
| 69073 | 31.20 | 31.19 | 31.20 | 31.15 | 31.20 | 30.83 |
| 68539 | 31.33 | 31.36 | 31.33 | 31.38 | 31.39 | **9.85** |
| 67948 | 29.85 | 29.83 | 29.79 | 29.59 | **14.30** | 26.18 |
| 60140 | 29.56 | 29.65 | 29.56 | 29.61 | 29.56 | 29.19 |
| 67293 | 29.49 | 29.52 | 29.49 | 29.24 | 29.49 | 28.93 |
| 67174 | 28.04 | **29.24** | 27.77 | 28.32 | 26.93 | 27.53 |
| 61101 | 31.86 | 31.87 | 31.86 | 31.80 | 31.86 | **32.95** |
| 66631 | 30.52 | 30.66 | 30.52 | 30.67 | **11.02** | 30.57 |

- `lf_fast_post` vs exact paired `lf_native` K4: **+0.1844dB** mean, 6/8 images improve. **Image 67174 contributes +1.206dB**; on the other 7 images the mean gain is only **+0.038dB**. Root-level mean improvement is **+0.032dB**, and **Bad20 remains 9/32** in both. This is an *image-specific recovery*, not evidence of a broad, robust improvement or reduced catastrophic probability. One correction arm has 16 effective updates/root = 80 reverse-mode VJP operations and 336 correction forward magnitudes, but original U-Net count stays 2200.
- `lf_fast_prescore` is **−1.694dB per-root mean** versus `lf_native`, with **12/32 vs 9/32 Bad20**. K4 change **−0.040dB**. Correcting before ranking can perturb basin decisions and is not an automatic benefit. Per-root comparison uses matching nominal image/rep seeds, although diverged proposal paths can consume RNG differently.
- `lf_beam2` K4 is **−0.0099dB** relative to native LF, yet costs **1098 vs 2200 U-Net calls per root** (~50.1% of calls) and 71.51s vs **123.84s adjusted** native LF K4 subprocess wall (~42.3% saving; vs observed 127.29s including one verifier it is ~43.8%). Root-level mean is **−0.697dB**, Bad20 still 9/32. K4 oracle picks among independent roots and hides unstable individual trajectories. **Critically, branch + reduced proposals + FAST correction all change together**; without a cost-matched sparse-no-branch/no-FAST control at K4, it is unsound to attribute savings or quality retention to branching itself.
- `lf_150_fast` K4 27.004dB is **−3.226dB** vs NP-LF native, and **−4.521dB** vs FAST native, despite 696 U-Net calls per image at K4 (FAST native 600). Four independent roots fail on image **68539** (best 9.85dB); this is not a reliable 150-step substitute. Importantly this A16 method is not a faithful 150-step FAST sampler; it is a shortened VP/NP with sparse FAST-inspired corrections.
- `s2_native` K4 25.719dB vs native LF 30.231dB for THESE new paired seeds. Two images (67948, 66631) lack even one Good25 root; earlier A1.2 NP-S2 K4 was 30.222dB on DIFFERENT seed set. Do not claim S2 is universally worse, merely that it exhibits basin sensitivity.

## 3. The most intriguing practical diagnostic: signed final-residual selection

The algorithm's intermediate NP measurement target was generally `y^+ = max(y,0)` (frozen B22 NP convention). However, the **original signed** noisy amplitude `y` remains available for scoring the generated *final* candidate reconstructions without GT.

For each method, select among **all four already-generated roots**, paying their entire generation cost, using the minimum full Fourier-magnitude residual either against original signed `y` or clipped `y^+`. This is a **post-hoc GT-free policy counterfactual**, not a selector trained or validated on fresh data.

| A16 arm | GT oracle K4 raw PSNR | signed-residual K4 | clipped-residual K4 | signed Good25 | signed mean oracle regret |
|---|---:|---:|---:|---:|---:|
| `lf_fast_post` | 30.415 | **30.387** | **23.619** | 8/8 | 0.0277dB |
| `lf_native` | 30.231 | 30.179 | 30.154 | 8/8 | 0.0513dB |
| `lf_fast_prescore` | 30.191 | 30.177 | 30.117 | 8/8 | 0.0140dB |
| `lf_beam2` | 30.221 | 30.153 | **22.952** | 8/8 | 0.0682dB |
| `s2_native` | 25.719 | 25.390 | 21.534 | 6/8 | 0.3291dB |
| `lf_150_fast` | 27.004 | 26.646 | 21.447 | 7/8 | 0.3585dB |

**Concrete failure mode:** In `lf_fast_post`, selecting by clipped residual returns catastrophic raw images at 66631 (10.62dB), 67174 (13.16dB), and 67293 (11.66dB), even though the four-root sets include good candidates at ~30.66, 29.24, 29.52dB respectively. Signed residual chooses good candidates on all eight and its mean oracle regret is only .0277dB. Similar clipped-ranking catastrophes occur for `lf_beam2` on images 69073/67174/67293.

These are meaningful **GT-free selection engineering observations**, but not a theoretical solution to phase-retrieval ambiguities. Exact Fourier-magnitude conjugate-reflection is observationally indistinguishable in idealized models; a likelihood residual alone cannot guarantee the desired orientation. Some A16 failures have high rot180 quality (10/260 raw-Bad20 cases across all arms were rot180-Good25), especially branch and ranking arms, but **raw PSNR remains primary**. Nor does within-method signed residual success guarantee safe selection BETWEEN DAPS/SITCOM/NP/FAST (original A1.3 reports cross-method failures).

A15 native FAST baseline signed-residual selection achieved 31.4834 vs GT K4 oracle 31.5251 (5/8 exact winners) on the same exposed cohort; source data are NOT an independent confirmation of A16 result.

## 4. Screening/promotion limitations and mechanisms not identified

- Stage0 screened only four preselected images (two earlier FAST successes and two FAST failures) with **one root** per arm. E.g. `lf_250_moments` looked strong at 25.93dB, then declined to 19.52dB with all eight×2 roots at Stage1; `lf_150_fast` displaced it only after Stage1. This is an adaptive GT-informed experiment, and a "stage winner" is exposed-data selection.
- GPU0 Stage0 post-correction K1 screen appeared to gain ~3.83dB over native across four images because it rescued 67174; Stage2 averaged over 32 roots shrank this to a +0.032dB root gain. Do not extrapolate Stage0 mean.
- GPU2 Stage0/1 `lf_sparse` vs `lf_beam2` demonstrate an interesting ranking change, but `lf_sparse` was not retained to K4; cannot causally isolate carrying two branches.
- Some `lf_beam2` catastrophic raw roots have high rotated-PSNR (e.g. image69073 rep2/3 raw ~7.08dB, rotated >30dB); this is a real orientation-basin concern. K4 oracle is not deployable, whereas signed-residual selection appears to recover good raw candidates in this exposed study.
- The FAST-inspired correction is a source-derived *effective K2 iteration*, using NP VP `gamma=(1-alpha_bar)/alpha_bar`. It is **not** experimentally demonstrated bitwise-identical to original FAST source on arbitrary inputs. No causal evidence from A16 alone attributes improvement to ADMM projection rather than a generic local gradient or stochastic trajectory perturbation; gradient control `lf_grad_post` only received Stage0 (4 roots). Longer controlled tests needed for claims.

## 5. Scientific recommendation and next experiment

**Result accepted as completed exploratory engineering work. Do NOT claim new NP scientific superiority or a new algorithm paper now.** Native FAST-DIPS still has a substantially better observed quality-compute frontier than our NP hybrids on these eight exposed images; method ingredients (particle exploration, adaptive K, FAST correction) have significant prior art.

Recommend a narrow but decisive **factorial follow-up**, not another unbounded omnibus sweep:

1. **Selection-focused prospective gate (highest priority):** Freeze the signed-original-measurement vs clipped-residual selection policy *before* any fresh cohort, using the same four generated roots and fully charged time. Compare across individual method families, not just within NP; primary GT-free selected raw PSNR, Bad20/Good25, and oracle regret, keeping raw orientation. Do not access reserved B24 confirmation305 without Track A planner/metaplanner approval and a new frozen run receipt.
2. **Cost-matched basin exploration ablation:** On fresh or clearly separated development images, run a 2×2 with `branch_hold∈{1,6}` (no persistence vs six-step persistence) and `FAST correction∈{off,on}`, PLUS original sparse-K no-branch; match U-Net evals and count FFT/VJP/wall. If K4 performance for branched runs remains near NP-LF while NFE halves, that is a worthwhile engineering improvement; compare against native FAST K1/K2/K4 to see if any actual frontier advance exists.
3. **Corrector mechanism check:** preserve exactly identical native seed and measurement and compare original FAST K2 correction, analytically simplified effective second update, generic matched FFT/VJP plain-gradient control, and final x0hat vs x0y. At 1000-step NP, correcting after selection at only 16 steps made no broad quality difference; therefore avoid longer hyperparameter sweeps unless causal evidence emerges.
4. **A decision gate:** If best corrected/sparse NP remains slower and lower-quality than FAST on independently fixed inputs, **park NP as a standalone quality-efficiency paper** or reframe toward principled GT-free basin selection; continue Track B basin-transition development without pretending we found a new method.

## 6. Mandatory reporting conventions

- Every reported K4 raw PSNR is the average of per-image **maximum PSNR among four roots**, and is explicitly **offline GT oracle**. Runtime includes ALL four candidates, regardless of selector.
- Good25 among K4 images is *not* the same as Bad20 among all 32 roots. Both must be shown.
- Signed-vs-clipped selection is applied **only after** candidate generation in this replay (except separately labeled `lf_250_signed` experimental arm). It is not proof the sampler's correction should use signed input.
- Timing scopes: FAST 32.44s = four program-after-import walls, A16 71–127s = four recorded full subprocess walls, and A16 native LF's 127.29s uniquely includes the extra replay guard; **method-only native LF is 123.84s** after subtracting the precisely recorded 27.645s verifier. Report hardware/software and CUDA sample walls before precise speedup factors.
- Keep B22/A1.2/A1.5 checkpoints, source commits, and completed receipts immutable. This report is committed on a **separate analysis branch**.
