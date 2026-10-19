# A1.1b V2 native four-method procedural GPU smoke — VERIFIED PASS

**Date:** 2026-10-19. **Executor:** continuing Track A. **Scientific disposition:** `GO_PROSPECTIVE_DEV_ENGINEERING_DESIGN` (not a face-domain benchmark). **Execution disposition:** `FOUR_NATIVE_PIPELINES_PASS`. No longer GPU campaign is implicitly authorized.

## 1. Independent PAC receipt audit

User uploaded `A1_1B_GPU_NATIVE_RETURN_V2.zip`; independently opened via `zipfile`. Exact archive SHA-256 **`875fae9cc3a546c82d72c15cf31964dc8c5728c52528b4e2f5005b5f7b63df3a`**, `testzip()==None`, **18 valid members** including `RUN_IDENTITY.json`, `A1_1B_COMPLETE.json`, DAPS/SITCOM/NP-LF/NP-S2 completion JSON, all four CUDA timers, native method logs and B22 NP/SITCOM results. Internal `A1_1B_COMPLETE.json` status `PASS_A1_1B_PROCEDURAL_FOUR_NATIVE_PIPELINE`, exactly four unique methods. No `A1_1B_STOP.json`. Source A1 HEAD `586e4d617759dca4900832e8b6f39b2c4a066028`; physical GPU3 UUID `GPU-7d65c050-d7e8-5a6b-ee38-1d72d7a5696a`, start device free 97,250 MiB / used 0 MiB, same frozen DAPS/SITCOM/DiffFPR source identities and FFHQ model SHA `81d53574f0614074d7793a16c8fa9e306237faa` (full hash `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`).

**Same saved raw signed observation across methods**: content SHA `992a33efa9bfb8e81d97ed61cf28369263ae1f11aeae19273c53973c2470acdb`, float32 (1,3,384,384), from wholly procedural RGB synthetic input, sigma=0.05 and oversample=2.0. Native SITCOM result.json confirms raw/used SHA same; NP LF/S2 confirm raw SHA same, but `clamp_min_zero_in_memory` used SHA is **`214a7b7752fb9cce59f704d98f3c057f756f1c905166c37e972f3562613cadcd`**, clipping **206,939/442,368 (46.78%)** negative entries (expected historical NP-specific convention on sparse Fourier magnitudes; cannot silently change method). DAPS console includes `[B20 fixed measurement] loaded ... measurement.pt` and `[B21 start seed] seed=463500061`.

## 2. Results: 1 full native trajectory per method

All quality figures below are **same saved 8-bit RGB PNG** against quantized procedural GT; raw orientation primary, 180° rotation sensitivity auxiliary. **None** support FFHQ superiority or reliability estimates.

| Method | Native seed (uint32) | Process wall (s) | CUDA-event elapsed (s) | Peak PyTorch reserved (MiB) | Max device used observed (MiB) | Raw 8bit PSNR (dB) | Rotation-180-aware PSNR (dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DAPS | 463500061 | 61.988 | 58.379 | 966 | 1667 | 17.315 | 28.325 |
| SITCOM | 2178904464 | 39.251 | 33.156 | 942 | 1643 | 17.011 | 26.802 |
| NP LF | 4046706865 | 37.234 | 32.593 | 946 | 1647 | 17.165 | 27.898 |
| NP S2 | 1462844243 | 33.084 | 29.592 | 946 | 1647 | 17.291 | 26.562 |

Total 4-method wall **173.028 s** (2 min 53 s), sum subprocess walls **171.557 s**, sum CUDA-event elapsed **153.719 s**. CUDA events bracket CPU+GPU work and should **not** be described as precise GPU-SM active utilization. All four method terminals are non-empty, have distinct recorded SHA-256, and all required saved PNGs are present according to the scored completed receipt (actual image bytes were not sent, so their content digest is verified as archived metadata, not independently rehashed from image payload). All four same-measurement content hashes verified in available native result/parent outputs.

Native runtime counters:
- DAPS `torch_randn_like` 40,401, `torch_randn` 1.
- SITCOM `torch.optim.SGD.step` 20,000, `torch_randn_like` 20,201.
- NP LF/S2 `torch_randn_like` 1,900 each, `torch_randn` 1 each.
- `torch_fft2_calls=0` and `torch_ifft2_calls=0` for all four; **this is an instrumentation coverage limitation**, not evidence of zero Fourier operator calls. Native operators may dispatch via other FFT APIs, and the current wrapper instruments only `fft2` and `ifft2`. Do not state exact full FFT work from these zero counts.

Native warnings:
- DAPS log shows two torchvision deprecated pretrained/weights warnings; no fatal model/solver error, 400/400 anneal stages completed and output saved.
- NP LF/S2 `undefined_diagnostics=["selector_stats.selector_post_lf_mse_margin_mean"]` because one candidate per selector/post-step can leave margin undefined; explicitly represented as null, not an error or hidden dropped candidate.
- DAPS native results print progress/evaluation PSNR in raw orientation and produced valid PNG. Source-level GT leakage caveat unchanged: native DAPS and legacy SITCOM B22 wrapper may have GT available for logging or shape; this is a **procedural integration test**, not a new prospective blind method-certification.

## 3. Methodological takeaways and next stage gates

All four solver families can consume the **same native DAPS-generated saved measurement** with correct source/seed checks, complete their original frozen reconstruction configurations in separate processes, and produce outputs suitable for a uniform PNG8 quality evaluator. Every run was much smaller than the conservative resource reservation (under 1 GiB torch reserved / ~1.6 GiB maximum device usage). 180°-aware PSNR substantially exceeds raw orientation for all four; this is a Fourier phase-retrieval ambiguity diagnostic and **must not silently replace the preregistered raw-orientation primary endpoint**.

Measured 16-candidate full K4/image wall projection from this single procedural test, assuming linearity and no rescheduling: `4*(61.988+39.251+37.234+33.084)=686.2 seconds/image`, `8*686.2=5489.8 seconds` (about **1.53 allocated GPU-hours** sequential one-GPU wall) before data generation and orchestration; 128 trajectories. This is a **planning extrapolation** based only on one procedural input and not an empirical FFHQ timing distribution; allow meaningful slack, e.g. 3h total and bounded per-method failure policy. No hidden candidates for NP mixed4/mixed8 offline portfolios.

**Data governance:** A1 study preregistration prohibits interpreting B21/B22 final100 and B24 DEV80 as untouched held-out test; protected B24 confirmation305 membership/data/metadata must not be opened, even for exclusion. Real-image development needs independently approved disjoint source/split mechanism; no such new cohort is yet certified. If choosing to use exposed images, label them transparently as exploratory/engineering development and formally amend the preregistered pilot protocol first. The procedural A1.1b is not sufficient to claim source-disjoint FFHQ pilot readiness.

**Evaluation fidelity decision pending BEFORE new GT:** Original `A1_STUDY_PREREGISTRATION.md` specified common **unquantized float PSNR** across all arms, while current native DAPS outputs only 8-bit PNG without float tensor in our wrapper. For A1.1b we used **common saved PNG8** for fairness. A longer K4 work plan must explicitly preregister PNG8 as primary (with quantization sensitivity on NP/SITCOM) or implement a verified non-invasive DAPS float exporter; never silently swap metrics after seeing real GT. This is the main scientifically material change to reconcile. Raw orientation and paired GT oracle remain locked.

**Decision:** A1.1b `PASS`; work on 8-image 128-candidate *development engineering* design and provenance first, with one new explicit `[DATA RESERVATION]` and `[RUN RESERVATION]` before user invokes a longer real-image run; Track B GPU priority remains. Preserve V1 expired-window STOP and V2 successful output directories. No auto-launch.
