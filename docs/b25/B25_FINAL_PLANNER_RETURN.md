# B25 final planner return

## Executor status

B25 is complete. The scientific planner accepted the CPU evidence **with required reporting changes**, and those qualifications are incorporated here. This return is archival/reporting only and does not authorize another experiment.

The single returned direction remains:

**a precisely defined NP conditional-selection correction, to be investigated prospectively at fixed compute**

This is a conditional-inference hypothesis and implementation target. **B25 does not establish a better FFHQ reconstruction method or a PSNR improvement.** B24 remains closed under `STOP_B24_METHOD_REFINEMENT`; confirmation remains locked.

## PR / branch identity

- repository: `Cheng-HanHuang/pr_diffusion`
- draft PR: `#39 — [B25] noise-selection bias and phase-retrieval failure mechanisms`
- branch: `codex/b25-noise-selection-mechanisms`
- base: `codex/b24-bestof4-failure-sweep`
- immutable B24 start: `ed162c2f97430804fddb5d9a0bfec7abde201ca0`
- signed B23.1 ancestor: `27505e6328157ac9296c95dc5e611cbeef80de98`
- pre-run B25 scientific commit: `32453db6445acec4fc19a4a928142a412d67f1ae`
- PR remains draft, open, and unmerged

## Completed execution and corrected resource accounting

PAC run:

`/egr/research-pac/huang248/outputs/pr_diffusion/b25/B25_cpu_20260914T224132Z`

The worker completed PASS. Scientific work was CPU-only with `CUDA_VISIBLE_DEVICES=""`.

The process receipts, rather than the smaller inner-script timers, give:

- total stage-process wall time: `166.2627886198461 s`;
- maximum recorded process RSS: `0.6489639282226562 GiB`;
- GPU work: false;
- pretrained-model inference: false;
- new FFHQ measurement generation: false;
- new FFHQ reconstruction generation: false;
- confirmation305 payload access: false.

For provenance only, the earlier published `159.6690 s` and `0.5504 GiB` are inner measurements and should not be used as the full process-accounting figures.

## Experiment 1 — accepted mechanism result, with boundary

The frozen iid Gaussian proposal study passed the analytic order-statistic checks. All four `K>1` hard-selection fits supported the prospectively frozen `h^0.5` band, while all four likelihood-weighted fits supported the `h^1` band:

- linear `K=4`: hard `0.4916`, weighted `1.0325`;
- linear `K=8`: hard `0.4932`, weighted `1.0600`;
- quadratic `K=4`: hard `0.4752`, weighted `0.9833`;
- quadratic `K=8`: hard `0.4690`, weighted `0.9725`.

Interpretation: hard best-of-K selection changes the leading small-step behavior in the controlled iid model. Matching the `h` scaling of the weighted arm **does not prove that weighted selection has the correct posterior dynamics**, and this is not a theorem about native NP because historical NP includes incumbent reuse and non-iid proposal semantics.

## Experiment 2 — posterior fidelity is not reconstruction accuracy

All exact-reference validation checks passed. The relevant `K=8` results must be reported with both posterior total-variation error and empirical reconstruction MSE to the one frozen truth:

| Prior family | hard TV | exact-intermediate weighted TV | denoised-point weighted TV | hard truth-MSE | exact-intermediate weighted truth-MSE | denoised-point weighted truth-MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| distinguishable equal | **0.00000** | 0.06213 | 0.00004 | **0.000000** | 0.004128 | 0.000002 |
| exact ambiguity, unequal weights | 0.26242* | **0.00475** | 0.24250 | 0.049744* | 0.024272 | **0.000728** |
| near ambiguity, equal weights | 0.22087 | **0.03401** | 0.24467 | **0.000000** | 0.024769 | 0.045241 |
| near ambiguity, unequal weights | 0.24983 | **0.00567** | 0.22762 | **0.000000** | 0.024829 | 0.002158 |

Lower is better in both metric blocks.

The near-ambiguity rows support the conditional-inference hypothesis: exact-intermediate weighting is much closer to the target posterior probabilities. They **do not demonstrate better reconstruction of the single frozen ground truth**; hard selection has lower truth-MSE in both near-ambiguity examples. The distinguishable family further prevents any claim that weighting universally wins.

`*` The exact-ambiguity hard-selection row is **not evidence for an intrinsic hard-selection defect**. The theoretically identical likelihoods differ at floating-point precision, and a hard `argmax`/`argmin` can turn that numerical difference into a systematic preference. This case remains descriptive until explicit tie handling is examined. The near-ambiguity evidence is not affected by that qualification.

The accepted conclusion is therefore narrower: if an NP correction is pursued, it should target an estimator of the intermediate observation likelihood `p(y|z_t)` rather than merely soften the historical denoised-point score. B25 does not show that such an estimator will improve FFHQ PSNR.

## Experiment 3 — tested symmetry family only

The offline eight-mask channelwise reversal oracle is valid under the tested channelwise Fourier-magnitude operator and reveals several isolated orientation failures.

Across all DEV80:

- DAPS: `1` Good25 symmetry recovery;
- SITCOM: `3` Good25 symmetry recoveries;
- NP four-candidate oracle: `57/80` Good25 before/after symmetry, with `0` new Good25 recoveries.

The `57/80` NP figure is the **four-candidate oracle**, not the historical clean-free selected method. Historical clean-free NP selection is `55/80` Good25.

On the frozen 10-image shared DAPS4/SITCOM4 failure subset, the tested transformations give:

- DAPS: `0` Good25 recoveries;
- SITCOM: `0` Good25 recoveries;
- NP4 oracle: `0` additional Good25 recoveries.

Thus the **tested eight transformations** do not explain the shared failures. This does not rule out every possible phase-retrieval ambiguity or symmetry.

## Experiment 4 — measurement preprocessing is a material difference, not a quality result

DAPS and SITCOM both predict nonnegative Fourier magnitude but evaluate their main Gaussian squared residual against the supplied signed noisy measurement. DAPS's local B20 low-frequency amplitude projection separately uses `abs(measurement)` only in that auxiliary projection path.

Historical B24 DEV80 NP verifies the locked raw measurement and then uses:

`measurement_np = measurement_raw.clamp_min(0.0)`.

All `80/80` DEV observations contain negative values. The raw-to-clamped relative L2 change has median `0.08223` and range `0.05596` to `0.14793`.

This establishes a material observation/preprocessing change. **It does not establish its effect on reconstruction quality.** A future experiment must isolate this factor rather than attributing a combined change to selection.

## Precisely defined next investigation

The next scientific question is:

> **Can we estimate the intermediate conditional likelihood well enough to improve reconstruction at a fixed computational budget?**

Before any new FFHQ experiment, the planner must specify:

1. the implementable estimator of `p(y_raw | z_t)` for the pretrained-model setting;
2. the proposal distribution used to generate candidates;
3. the finite-proposal approximation and normalization/resampling rule;
4. lineage handling for the selected/resampled state and noise;
5. any proposal-density correction if incumbent reuse or nonexchangeable proposals remain;
6. the complete compute accounting, including denoiser, selector/likelihood-estimator, projection, FFT/custom-operator, and branching costs.

The observation and selection changes must then be isolated against frozen NP with at least these arms:

- frozen historical NP;
- raw-measurement correction only;
- selection-rule/intermediate-likelihood correction only;
- combined raw-measurement + selection correction.

Fresh2 remains an important efficiency comparator. A combined improvement without this ablation would not identify which mechanism mattered.

This specification is a planner target, **not authorization to implement or run it**. No new GPU stage is authorized by the B25 review.

## Evidence capsule

Archive:

`/egr/research-pac/huang248/outputs/pr_diffusion/b25/B25_cpu_20260914T224132Z.tar.gz`

Verified SHA-256:

`5b1e785ba46e6be952ffdbc768c5d9bab365e495cf9efd5b154b075ce0081354`

The existing capsule is preserved unchanged. It has 26 safe members; `SHA256SUMS.txt` contains 25 run-file digests and all 25 were independently verified. No rerun or repackaging is required.

## Final executor return

B25 supplies a defensible next conditional-inference hypothesis, not a validated reconstruction improvement. Return control to the planner with PR #39 left draft/open, B24 closed, confirmation locked, and no new GPU execution authorized.
