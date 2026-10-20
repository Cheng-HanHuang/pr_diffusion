# Track A A1.3 executor — implementation and read-only inventory handoff

**Date:** 2026-10-20, America/Detroit.  
**Owner branch ONLY:** `codex/prdiff-track-a-a13-candidate-audit-20261020`.  
**Status:** `IMPLEMENTATION_COMMITTED / PAC_NOT_INVENTORIED / CPU_TESTS_NOT_EXECUTED / NO_GPU / NO_NEW_DATA`.  
**Next action:** researcher runs **only the read-only** first PAC block in [A13_STAGE0_SOURCE_AND_CPU_PROTOCOL.md](A13_STAGE0_SOURCE_AND_CPU_PROTOCOL.md) and returns its bounded stdout. No stage0 output directory, old A1 artifact or source checkout has been modified by this executor.

## Frozen science and implementation

Evidence anchors: A1.2 execution commit `f4cd28dcd3ec4860cc9f3990fb24964bb97e97bd`, A1.2 complete 128 unique terminal SHA claims, A1 source report/index, Track A decision `ac88a3eabc4a58652de4b9913873eb01185d6315`, issue #41. Cohort frozen to exposed B22 IDs `69073,68539,67948,60140,67293,67174,61101,66631`, 4 methods × 4 reps per image. 8×16=128 only. All previously generated reconstruction outputs must survive unchanged. The already-exposed data are not a new held-out confirmation.

Dedicated changes:
- `configs/track_a/a13_frozen_analysis_v1.json` freezes allowed IDs/roots/methods, A1 execution identity, native HEADs and patch hashes, exact signed/clipped operator definition and stable earliest-tie argmin, PNG8 raw PSNR, calibration seed and thresholds.
- `scripts/track_a/a13_physics.py`: unnormalized forward-model pixels `x_model -> 0.5*x+0.5`, explicit 64px symmetric zero pad, fastMRI centered `fftn(norm="ortho")` channelwise magnitude, signed and clipped data-fit scores, per-channel SSE, PNG8 PSNR and rotation witnesses, noise-only synthetic reference and isolated GT-free selector.
- `scripts/track_a/test_a13_physics.py`: pure synthetic CPU tests for impulse FFT normalization, rotation, signed negative observation handling, exact PNG8 metric, selector direction/ties/NaN/leakage and calibration reproducibility. **Not yet executed on PAC.**
- `scripts/track_a/a13_inventory.py`: source-pin and native patch audit, numerical reconciliation with DAPS/SITCOM CPU native operators *without loading a checkpoint*, exact 8 input-file checks, all 128 terminal and comparable-PNG SHA checks, candidate/seed uniqueness and A1 receipt link, with write-only-new stage0 results after PASS.
- `scripts/track_a/a13_analysis.py`: **GATED, NOT EXECUTED**. Reload/rehash exact stage0 receipt, all 128 original bytes and native identities, CPU-only residual extraction on PNG8 for common within-family policy, additional separately charged float residual on 96 SITCOM/NP terminals, GT evaluation only after selector choices, per-candidate and K-policy tables, exact A1.2 GT oracle replay, rotation/fitting diagnosis, explicit unsupported cases and new metadata-only outputs. No native step160 SITCOM or NP within-trajectory selector is fabricated as a cross-root rule.
- `scripts/track_a/a13_run_inventory_v1.sh` and `a13_run_cpu_forensics_v1.sh`: source-HEAD argument, fixed CPU interpreter and `CUDA_VISIBLE_DEVICES=""`, no terminal-closing flags, no overwrite, stage0 gate before stage1, compact metadata ZIP only after PASS. These are **not launched**.

## Why the residual design matters

DAPS and SITCOM original operators both use centered FFT magnitude after converting RGB model range to physical range, and orthonormal FFT, as identified by their pinned native `forward_operator` and `fastmri_utils` sources. They measure signed noisy magnitudes, while historical NP internally clamps negative observations. Consequently A1.3 computes separate `r_signed` and `r_clipped`; no method-native likelihood/prior is claimed to equal either one. Common terminal scores use the **saved PNG8 representation** to be comparable across all four methods; results on available 96 unquantized SITCOM/NP endpoints are *additional* and cannot be compared to unavailable DAPS native float outputs without qualification.

The descriptive failure tags distinguish raw Good25, verified 180° rotation-equivalent candidates, low-raw candidates that fit at or below a *pre-calibrated independent Gaussian-noise RMS reference*, larger-than-3×-reference measurement misfit, and unresolved middle cases. They are **not** topological basin identification, GT-free orientation recovery or evidence of posterior multimodality.

## STOP/caveats before analysis

This is source- and PAC-unverified software until the user runs and returns the first inspection and independent stage0 CPU tests. Native patched operator code can differ from public HEAD; numeric reconciliation with **both** native CPU operators is deliberately required. If import/source-diff/FFT/operator/complete-byte/GT-PSNR replay fails, STOP, report exact failure and preserve artifacts. No model/denoiser import, new diffusion trajectory, new measurement, B24 confirmation305 access even for membership, Track B basin transition, or GPU work is authorized.

Expected stage0 audit status `A13_SOURCE_AND_INPUT_AUDIT_PASS` (not observed). Expected stage1 audit status `A13_CPU_FORENSICS_COMPLETE` (not observed). Do not cite either as results until user PAC execution and independent review.

## Next larger research decision

After stage0 and later staged scientific audit, submit a final `A13_EXECUTOR_RETURN.md` or `A13_STOP_REPORT.md` with actual observations and metadata archive SHA. The Track A lead will determine any A1.4 next stage; this handoff does not greenlight OOD GPU pilots or prospective generalization claims. Issue #41 records this implementation and future return.
