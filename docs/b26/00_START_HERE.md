# B26 — NP observation correction and intermediate-likelihood feasibility

Status: **AUTHORIZED / IMPLEMENTATION AND PRE-RUN FREEZE IN PROGRESS.**

B26 starts exactly from signed B25 closeout `c906d36e0e396a6abbf761e3d65c91433428170f` on `codex/b25-noise-selection-mechanisms`. B25 scientific pre-run is `32453db6445acec4fc19a4a928142a412d67f1ae`; its sealed capsule SHA-256 is `5b1e785ba46e6be952ffdbc768c5d9bab365e495cf9efd5b154b075ce0081354` and must remain unchanged.

B26 branch: `codex/b26-np-conditional-correction`. Draft PR: #40, based on `codex/b25-noise-selection-mechanisms`.

## Authorized scientific stages

### B26.0 — implementation and integrity
Freeze `configs/b26/b26_spec.json`, the PAC-derived `configs/b26/b26_dev80_manifest.json`, RNG namespaces, code, tests, analysis, resource limits, and exact source identities in one pushed pre-run commit before scientific execution.

### B26.1 — native NP observation correction
Compare historical H against signed-score R on the existing B24 DEV80 observations. H uses `y_plus=max(y_raw,0)` for both scoring and amplitude projection. R uses `y_raw` in every historical measurement discrepancy used for scoring/selector statistics, while projection alone uses `y_plus`. All proposal generation, denoiser calls, K schedule, incumbent-noise reuse, hard winner rule, projection schedule, four frozen roots, and initialization remain unchanged.

The sequence is fixed: 8 root-level smoke trajectories (H/R on root0 of one hash-ranked image per A/B/C/D), then all four roots on frozen DEV16, then all DEV80 if and only if replay/integrity/resource/budget gates pass. Continuation cannot depend on PSNR favorability.

### B26.2 — CPU finite-estimator feasibility
On the four frozen B25 finite-support families, generate 32 new synthetic observations/family, use K=5 exact unconditional outer proposals, and compare hard exact, categorical exact, denoised-point, and `L_hat_M` for M=1,4,16 over 512 reverse trajectories/method/observation. This stage is CPU-only with CUDA hidden.

## Hard boundaries

- B24 remains closed under `STOP_B24_METHOD_REFINEMENT`.
- confirmation305 payloads remain locked; only registry IDs may be read for exclusion proof.
- no new FFHQ measurement may be generated.
- no DAPS/SITCOM/Fresh rerun.
- no weighted FFHQ reconstruction or large screen.
- no NP/SITCOM cross-family adapter.
- no mutation of historical worktrees, outputs, PRs #37–39, or `main`.
- no merge, rebase, squash, retarget, force push, or history rewrite.

## PAC locations

- worktree: `/egr/research-pac/huang248/pr_diffusion_b26`
- output root: `/egr/research-pac/huang248/outputs/pr_diffusion/b26`
- accepted role directory: `/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_2_7424_extension_20260907T231303Z/case_freeze/method_stage`
- accepted DEV80: `/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_dev80_overnight_20260913T082146Z`
- corrected B24 closeout: `/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_corrected_20260914T060600Z`
- model: `/egr/research-pac/huang248/models/ffhq_10m.pt`, SHA-256 `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`
- environment: `/egr/research-pac/huang248/conda-envs/prdiff_ffhq`

## GPU/resource contract

Physical GPU IDs 0–3 are the recorded authorized set, but every launch rechecks live UUID/free-memory/process state. Admission requires at least 10,240 MiB free; B26 process-group usage must remain <=52,452 MiB per physical GPU; one controlled worker per physical GPU; no dynamic reassignment. B26.1 has a 24 aggregate GPU-hour reservation-time ceiling including setup and failures. B26.2 has <=4 CPU threads, <=16 GiB process RSS, and <=4 hours wall time.

## Required reading

1. root `AGENTS.md`
2. this file
3. `docs/b26/B26_EXECUTOR_CONTRACT.md`
4. `configs/b26/b26_spec.json`
5. `docs/b26/B26_NATIVE_ESTIMATOR_SPEC.md`
6. `docs/b26/B26_LITERATURE_AND_ESTIMATOR_BOUNDARY.md`
7. signed B25 final return/audit/model/novelty documents
8. frozen B24 DEV80/native-NP sources referenced by the PAC-derived manifest

## Pre-run gate

Do not start B26.1 or B26.2 until the PAC inventory/freeze script has produced the exact DEV80 manifest, zero-GPU tests pass, the manifest is committed to B26, and that exact commit is pushed and recorded as the B26 pre-run scientific SHA. Thereafter launch/resume scripts fail closed if local or remote B26 identity changes.
