# A1.2 exposed B22 DEV8 — CPU input byte-freeze handoff

**Date:** 2026-10-19. **Track:** A continuing executor. **Stage status:** `CODE_PREPARED_NOT_YET_RUN`. **No GPU allocation.**

## Completed predecessor

A1.2 metadata-only eight-image inventory independently verified SHA `720f30adfa04a7629b766e597afbdc4ae1d853cca7e98ac20757020ebfd2b90b`, 2/2 tests PASS, frozen B22 source-manifest SHA `94184e36b27aac0defb93b31ae7b47ea332909bcb0c0a80445be2949f422a219`. Inventory selected `69073 68539 67948 60140 67293 67174 61101 66631` in fixed hash ranking, selected-list digest `faa2e292e915690dc8dd7d477591bb47a12a5f5977fd1c9648849e25c2d09f32`. **All eight are already exposed B22 images**, NOT new pristine held-out cohort and cannot support a publication reliability claim.

## Immediate CPU-only step

A1-owned code:
- `scripts/track_a/a1_2_freeze_exposed_inputs.py`
- `scripts/track_a/test_a1_2_freeze_exposed_inputs.py`
- `scripts/track_a/a1_2_run_input_freeze_cpu.sh`

The script opens **only eight explicitly identified B22 exposed image/measurement files** pinned by the SHA in the inventory. It checks source PNG and archived .pt byte SHA256, loads the original signed noisy `torch.float32 [1,3,384,384]` measurements, then independently recomputes content SHA using frozen B22 helper `b22_smoke_common.tensor_content_sha256`. It applies the original `torchvision.transforms.ToTensor -> Resize(256) -> CenterCrop(256) -> 2*x-1` **on CPU** to each 1024² source image and verifies the original frozen B22 GT tensor-content SHA. Does not regenerate measurement, load any network checkpoint, use CUDA, perform any solver or access B24 confirmation305 data, and does not inspect nonenumerated FFHQ image paths.

Success writes one B22-compatible GT `ground_truth_tensor.pt` per image into new A1-owned outroot `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016/a1_2_exposed_b22_inputs_v1/`; one item.json per image and `A1_2_INPUT_FREEZE_COMPLETE.json` with provenance/hashes/flags. Return ZIP `A1_2_INPUT_FREEZE_METADATA_RETURN_V1.zip` includes **only metadata JSON/logs**, no face PNG, GT tensor or measurement tensor bytes. Nonoverwrite: if a byte/source mismatch or failed run occurs, preserve the output directory/log and return a STOP. Do NOT delete/repair original B21/B22 files.

## Development-only data governance

An explicit `[DATA RESERVATION]` in issue #41 tags the IDs as `A1_2_B22_EXPOSED_ENGINEERING_ONLY`, not independent DEV or untouched final. No B24 confirmation305 registry/membership/hash-query/path/content accessed. Historical measurement noise is reused exactly; no new measurement seed namespace. Independent native solver seeds for future exploratory GPU stage will use existing A1 domain-separated `seed32_numpy(family,image_id,r)` for r=0..3, distinct family names `DAPS,SITCOM,NP_LF,NP_S2`. Never make group member selection based on GT, PSNR, failure taxonomy.

## Scope of proposed longer study

The successor **GPU** stage is still not launched and requires new implementation and a `[RUN RESERVATION]`. Candidate matrix 8 × 4 families × 4 independent fresh native processes = **128 trajectories**; NP mixed8 reuses the 8 NP candidates per image without hidden inference. Offline GT-oracle *best-of-K* and raw Good25 can be computed on **saved RGB8 PNG representation equally for every method**, but these are **engineering diagnostics on exposed data only**. Original A1 paper preregistration requests homogeneous **float** PSNR; the engineering extension must be reported separately and not used to modify preregistered confirmatory criteria after GT exposure. A1.1b procedural runtime projects ~1.53 GPUh serial; a ~3 GPUh cap is a proposal, not approval; Track B gets priority.

**Next user action:** checkout pinned new A1 source SHA, run `A1_EXPECTED_HEAD=... bash scripts/track_a/a1_2_run_input_freeze_cpu.sh` in fresh PAC shell and return output ZIP/log. Do not try a larger GPU launch until this PASS is independently verified.
