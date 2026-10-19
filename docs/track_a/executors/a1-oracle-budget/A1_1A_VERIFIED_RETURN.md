# A1.1a returned archive — independently verified PASS

**Date:** 2026-10-19. **Owner:** continuing Track A executor. **Status:** `A1_1A_VERIFIED_PASS`.

The user supplied the correct `A1_1A_PROBE_METADATA_RETURN_V1.zip` (earlier mistakenly attached ZIP was from A1.0d). Independently opened the actual A1.1a archive in the sandbox and checked SHA-256:

- Full ZIP: `a2f61731638602f38563502c0c86e3d0896fa9a95f8fbf6fbb7937c507bc290f`, exact match to PAC terminal.
- Three members only: `A1_1A_SYNTHETIC_INPUT_TESTS_V1.log` SHA `5cbf6eb647369e3dd69a4183062c57a2a42291e09c9e89a39b6f1ab770480202`; `A1_1A_NATIVE_MEASUREMENT_V1.log` SHA `883526e992327f7ae94b72d42d798cd84f3fb6945e82d7966c607968d0706c8a`; `A1_1A_PROBE_MANIFEST.json` SHA `8f3ba584c9daf4f3ec62f27cda171fc9aaa7a43cd331e372d6d5a062cae38fd8`.
- Synthetic unit tests 3/3 PASS. CPU-native measurement log `status=A1_1A_SYNTHETIC_NATIVE_MEASUREMENT_PASS`. Native DAPS HEAD `e7a77d094167084faed19b599b96673b7bb11447`; tracked source diff SHA `fbb5b42369ecf0d3b9b67f8fc162053bc40ec32aed41dbd92a67e8d81dcfad69`; A1 execution HEAD `0b94d455f5d82389a400313776bea411c3a488f8`.
- The only input is `A1_SYNTHETIC_PROCEDURAL_PROBE_V1`, procedurally generated RGB (256x256), with file SHA `ab7fdedf0fe8672244ba3d3723bc31c48cb7bbb2ceb4d8fed03e227867d97187` and model-range GT float tensor content SHA `6eda5926b1cc56e23cdfed5fd70b743bdfe61067c955f778900683a42a6a1123` (file SHA `5b16ce5d63f5d19f07c62058c59a0e2c40e85912d69ce18826ca915c4bfe518b`).
- Native `DAPS PhaseRetrieval.measure` produced one shared **signed noisy** float32 observation shape [1,3,384,384] for sigma=0.05, oversample=2.0, seed 12061018; tensor content SHA **`992a33efa9bfb8e81d97ed61cf28369263ae1f11aeae19273c53973c2470acdb`**, file SHA `50f253dc72ae22aaf3eff4b688e8c2e6e3a3e3c97b32d359f167bccaca383ba4`. Observed minimum `-0.23671559989452362`, max `92.4192123413086`; NP-specific `clamp_min_zero_in_memory` does not alter frozen raw observation.
- Manifest flags: `synthetic_only=true`, `quality_inference_valid=false`, `ffhq_accessed=false`, `gpu_used=false`, `checkpoint_loaded=false`, `solver_executed=false`, `locked_confirmation_access=false`.

This confirms A1.1a procedural *input/measurement integration*, **NOT** native reconstruction, method superiority, prospective FFHQ quality or complete GPU-pipeline fidelity. The 180 exposed images and protected B24 confirmation305 remain entirely untouched. Subsequent A1.1b native pipeline smoke on this same saved measurement needs its **own** physically gated GPU run; Track B retains priority.

A1.1b uses already pinned B22/B23/B24 **read-only source as process-launch helpers** under an A1-owned controller. DAPS scratch dataset/config symlink under the separate `external/daps` directory is transient, namespaced `a1-1b-procedural-...` and removed conservatively after execution; no historical B19–B26 run artifacts are modified. The common **saved RGB 8-bit** PSNR convention is frozen before comparing synthetic results, but the values must **not** be interpreted as face-distribution performance.
