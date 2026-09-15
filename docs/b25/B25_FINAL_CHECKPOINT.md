# B25 final checkpoint

## Status

`B25 COMPLETE — ACCEPTED WITH REPORTING QUALIFICATIONS — RETURN TO PLANNER`

The authorized CPU-only B25 mechanism study completed successfully. The scientific planner accepted the evidence with reporting corrections. No additional B25 scientific execution is authorized by this checkpoint.

B25 supports a focused NP conditional-selection **investigation**. It does **not** establish a better FFHQ reconstruction method or a PSNR improvement.

## Identity

- repository: `Cheng-HanHuang/pr_diffusion`
- branch: `codex/b25-noise-selection-mechanisms`
- draft PR: `#39`
- immutable B24 start: `ed162c2f97430804fddb5d9a0bfec7abde201ca0`
- signed B23.1 ancestor: `27505e6328157ac9296c95dc5e611cbeef80de98`
- pushed pre-run scientific commit: `32453db6445acec4fc19a4a928142a412d67f1ae`
- PAC run: `/egr/research-pac/huang248/outputs/pr_diffusion/b25/B25_cpu_20260914T224132Z`

B24 remains closed under `STOP_B24_METHOD_REFINEMENT`; confirmation remains locked.

## Safety and corrected resources

The completed worker reported PASS. Full stage-process receipts give:

- total stage-process wall time: `166.2627886198461 s`;
- maximum recorded process RSS: `0.6489639282226562 GiB`;
- GPU work: `false`;
- pretrained-model inference: `false`;
- new FFHQ measurements: `false`;
- new FFHQ reconstructions: `false`;
- confirmation305 payload access: `false`.

The previously reported `159.6690109781921 s` and `0.5503883361816406 GiB` are inner measurements, not the full process-receipt accounting. The DEV80/confirmation305 allowlist gate passed before any development payload was opened.

## Experiment 1 — conditional-selection distribution

All analytic engineering checks passed. The prospectively frozen small-step fits supported the predicted mechanism in every `K>1` cell:

| Energy | K | hard-min exponent | weighted exponent |
| --- | ---: | ---: | ---: |
| linear | 4 | 0.4916 | 1.0325 |
| linear | 8 | 0.4932 | 1.0600 |
| quadratic | 4 | 0.4752 | 0.9833 |
| quadratic | 8 | 0.4690 | 0.9725 |

Within the iid toy model, hard minimum selection exhibits the predicted `O(sqrt(h))` directional displacement while likelihood-weighted selection exhibits `O(h)` displacement. This shows a change in small-step behavior; it does **not** prove the weighted rule has correct posterior dynamics, and it is not a theorem about native NP because native NP includes incumbent reuse.

## Experiment 2 — posterior fidelity versus one-truth reconstruction

All exact-reference validation checks passed. At `K=8`, all four frozen prior families give:

| Prior family | hard TV | exact-intermediate weighted TV | denoised-point weighted TV | hard truth-MSE | exact-intermediate weighted truth-MSE | denoised-point weighted truth-MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| distinguishable equal | **0.00000** | 0.06213 | 0.00004 | **0.000000** | 0.004128 | 0.000002 |
| exact ambiguity, unequal weights | 0.26242* | **0.00475** | 0.24250 | 0.049744* | 0.024272 | **0.000728** |
| near ambiguity, equal weights | 0.22087 | **0.03401** | 0.24467 | **0.000000** | 0.024769 | 0.045241 |
| near ambiguity, unequal weights | 0.24983 | **0.00567** | 0.22762 | **0.000000** | 0.024829 | 0.002158 |

Lower is better. The near-ambiguity rows support better posterior fidelity from exact-intermediate weighting, but hard selection has lower error against the single frozen truth in both near-ambiguity examples. The distinguishable family also prevents any claim that weighting universally wins.

`*` The exact-ambiguity hard-selection result is excluded from evidence for an intrinsic hard-selection defect until tie handling is examined: theoretically identical likelihoods differ at floating-point precision, and hard `argmax`/`argmin` can convert that difference into a systematic preference.

The accepted implication is therefore conditional-inference only: a future NP investigation should estimate `p(y|z_t)` rather than merely soften the historical denoised-point score. B25 does not demonstrate a reconstruction-quality benefit.

## Experiment 3 — tested symmetry diagnostic

The exact eight-mask channelwise reversal invariance check passed to maximum relative L2 `2.641774875941548e-16` against tolerance `1e-10`.

Across all DEV80 images, the GT-assisted offline symmetry oracle produced Good25 recoveries for one DAPS image and three SITCOM images. NP's `57/80` figure in this diagnostic is the **four-candidate oracle**; historical clean-free NP selection is `55/80` Good25.

On the prospectively frozen 10-image shared DAPS4/SITCOM4 failure subset:

- DAPS: `0/10` recoveries;
- SITCOM: `0/10` recoveries;
- NP4 oracle: `0/10` additional recoveries, despite a maximum symmetry gain of `3.412863 dB`.

The tested transformations therefore do not explain the shared failures. This conclusion is restricted to the tested eight transformations and does not rule out every possible phase-retrieval ambiguity.

## Experiment 4 — observation/preprocessing audit

All 80 locked DEV measurements contain negative stored amplitudes. NP's `clamp_min(0)` changes those observations by median relative L2 `0.08222852` (range `0.05595935` to `0.14793063`).

Post-run source inspection established:

- DAPS phase retrieval predicts a nonnegative FFT magnitude and its main Gaussian loss is the squared residual against the supplied signed observation;
- SITCOM uses the same main signed-observation semantics and its wrapper records no measurement preprocessing;
- DAPS's local B20 low-frequency amplitude projection applies `abs(measurement)` only inside that auxiliary projection;
- B24 NP verifies the same locked raw observation and then uses `measurement_raw.clamp_min(0.0)` in its NP path.

The discrepancy is real and material. B25 does **not** establish its effect on reconstruction quality; that requires an isolated ablation.

## Final evidence capsule

Archive:

`/egr/research-pac/huang248/outputs/pr_diffusion/b25/B25_cpu_20260914T224132Z.tar.gz`

SHA-256:

`5b1e785ba46e6be952ffdbc768c5d9bab365e495cf9efd5b154b075ce0081354`

The accepted review independently verified all 26 safe members and all 25 internal checksums. The existing capsule is preserved unchanged; no rerun or repackaging is needed.

## Final recommendation and next question

Exactly one B25 direction is returned:

**a precisely defined NP conditional-selection correction, investigated prospectively at fixed compute**

The unresolved question is:

> Can we estimate the intermediate conditional likelihood well enough to improve reconstruction at a fixed computational budget?

Before any new FFHQ experiment, specify the estimator, proposal distribution, finite-proposal approximation, lineage handling, and full compute cost. The eventual experiment must separately test raw-measurement correction, selection-rule correction, and their combination against frozen NP; Fresh2 remains an efficiency comparator.

No new GPU stage is authorized. Control returns to the scientific planner.
