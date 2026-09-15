# B25 final checkpoint

## Status

`B25 COMPLETE — RETURN TO PLANNER`

The authorized CPU-only B25 mechanism study completed successfully. No additional B25 scientific execution is authorized by this checkpoint.

## Identity

- repository: `Cheng-HanHuang/pr_diffusion`
- branch: `codex/b25-noise-selection-mechanisms`
- draft PR: `#39`
- immutable B24 start: `ed162c2f97430804fddb5d9a0bfec7abde201ca0`
- signed B23.1 ancestor: `27505e6328157ac9296c95dc5e611cbeef80de98`
- pushed pre-run scientific commit: `32453db6445acec4fc19a4a928142a412d67f1ae`
- PAC run: `/egr/research-pac/huang248/outputs/pr_diffusion/b25/B25_cpu_20260914T224132Z`

B24 remains closed under `STOP_B24_METHOD_REFINEMENT`.

## Safety and resources

The completed worker reported PASS with:

- aggregate scientific wall time: `159.6690109781921 s`;
- maximum observed RSS: `0.5503883361816406 GiB`;
- GPU work: `false`;
- pretrained-model inference: `false`;
- new FFHQ measurements: `false`;
- new FFHQ reconstructions: `false`;
- confirmation305 payload access: `false`.

The DEV80/confirmation305 allowlist gate passed before any development payload was opened.

## Experiment 1 — conditional-selection distribution

All analytic engineering checks passed. The prospectively frozen small-step fits supported the predicted mechanism in every `K>1` cell:

| Energy | K | hard-min exponent | weighted exponent |
| --- | ---: | ---: | ---: |
| linear | 4 | 0.4916 | 1.0325 |
| linear | 8 | 0.4932 | 1.0600 |
| quadratic | 4 | 0.4752 | 0.9833 |
| quadratic | 8 | 0.4690 | 0.9725 |

Within the iid toy model, hard minimum selection therefore exhibits the predicted `O(sqrt(h))` directional displacement while likelihood-weighted selection exhibits `O(h)` displacement. This is not promoted to a theorem about native NP because historical NP includes incumbent reuse and non-iid proposal semantics.

## Experiment 2 — exact finite-support posterior

All exact-reference validation checks passed. The central mechanism result is that exact-intermediate likelihood weighting can approach the correct posterior as the candidate count increases, while both hard selection and likelihood evaluated only at a denoised point can remain badly mode-distorting.

Representative `K=8` total-variation errors:

| Family | exact-intermediate weighted | hard exact-intermediate | denoised-point weighted |
| --- | ---: | ---: | ---: |
| ambiguity unequal | 0.00475 | 0.26242 | 0.24250 |
| near ambiguity equal | 0.03401 | 0.22087 | 0.24467 |
| near ambiguity unequal | 0.00567 | 0.24983 | 0.22762 |

This rejects a simplistic correction that merely replaces hard argmin with a softmax of the historical denoised-point score.

## Experiment 3 — symmetry diagnostic

The exact eight-mask channelwise reversal invariance check passed to maximum relative L2 `2.641774875941548e-16` against tolerance `1e-10`.

Across all DEV80 images, the GT-assisted offline symmetry oracle produced Good25 recoveries for one DAPS image and three SITCOM images. However, on the prospectively frozen 10-image shared DAPS4/SITCOM4 failure subset:

- DAPS: `0/10` recoveries, maximum gain `0 dB`;
- SITCOM: `0/10` recoveries, maximum gain `0 dB`;
- NP4: `0/10` recoveries, maximum gain `3.412863 dB`.

Symmetry is therefore a real isolated failure mode but not the primary explanation of the common B24 catastrophic subset and is not the returned B25 recommendation.

## Experiment 4 — observation/preprocessing audit

All 80 locked DEV measurements contain negative stored amplitudes. NP's `clamp_min(0)` changes those observations by median relative L2 `0.08222852` (range `0.05595935` to `0.14793063`).

Post-run source inspection established:

- DAPS phase retrieval predicts a nonnegative FFT magnitude and its main Gaussian loss is the squared residual against the supplied signed observation;
- SITCOM uses the same main signed-observation semantics and its wrapper records no measurement preprocessing;
- DAPS's local B20 low-frequency amplitude projection applies `abs(measurement)` only inside that auxiliary projection;
- B24 NP verifies the same locked raw observation and then uses `measurement_raw.clamp_min(0.0)` in its NP path.

The discrepancy is real and nontrivial. B25 does not claim that clipping alone caused historical reconstruction failures.

## Final evidence capsule

Archive:

`/egr/research-pac/huang248/outputs/pr_diffusion/b25/B25_cpu_20260914T224132Z.tar.gz`

SHA-256:

`5b1e785ba46e6be952ffdbc768c5d9bab365e495cf9efd5b154b075ce0081354`

Packaging checks:

- archive safe paths: PASS;
- archive members: `26`;
- internal checksummed run files: `25`;
- internal checksum verifications: `25`;
- post-run operator audit SHA-256: `1d4ebc132d7e4b92e7a9f7be14bec20ef6715a438b18a34096836fb58c6d7d0c`.

## Final recommendation

Exactly one B25 recommendation is returned:

**a precisely defined NP conditional-selection correction**

The detailed correction contract is recorded in `docs/b25/B25_POSTRUN_NATIVE_NP_AUDIT.md`. No new execution is authorized; control returns to the scientific planner.
