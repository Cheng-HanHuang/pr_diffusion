# A1.0d verified PAC retrospective synthetic restart plan — EXECUTOR RETURN

**Date:** 2026-10-18. **Owner:** continuing Track A executor, inherited A1 study. **Disposition:** `A1_0D_CPU_SYNTHETIC_PASS`; GO next engineering phase, **not** GPU campaign permission.

## Signed machine receipt

User executed `scripts/track_a/a1_run_stage0d_cpu.sh` on pinned A1 Git HEAD `ca17ef8113588ff1b909f0b648d440740c1ab13b`, environment `/egr/research-pac/huang248/conda-envs/prdiff_ffhq/bin/python`, and uploaded original `A1_0D_RESTART_SYNTHETIC_RETURN.zip`. Independent ZIP SHA-256: **`88060ce25af74360bb6284d94b472848409d85e70fcbcd77a0d77a458ba170b5`**, exactly matches terminal SHA.

Independently opened ZIP with Python's `zipfile` and checked all members and JSON:
- `A1_0D_RESTART_SYNTHETIC_TESTS.log`, 2247 bytes, SHA-256 `40652549275c063e07107a61fa038938b3df458f0d77e530a598f9c9256b4c31`.
- `A1_0D_RESTART_SYNTHETIC_SUMMARY.json`, 569 bytes, SHA-256 `3c38069cce1a5d9898112436fb0ee949da487441ef3b27061bd2d5c4faaeb5c4`.

The test log explicitly shows **14/14 tests PASS**, 0.027 s; `A1_0D_TEST_RC=0`, `A1_0D_SUMMARY_RC=0`, `A1_0D_PACK_RC=0`, `A1_0D_FINAL_STATUS=0`, `A1_0D_RUNNER_RC=0`.

The committed deterministic dry-run summary (metadata-only) declares:
- `schema=prdiff.a1.synthetic-restart-manifest.v1`; `kind=INVENTED_FAKE_IDS_NO_DATA`; `status=DRYRUN_NOT_LAUNCHABLE`.
- **2 invented image IDs** (not an actual FFHQ dataset allocation), **32 unique candidate IDs**, K=4 per family DAPS, SITCOM, NP_LF, NP_S2.
- `manifest_content_sha256=49f2b8f994e851d726aaa97e2f6fe30eac5a6101e16ea0efba84c45374d0c31e` — an internal digest of the full synthetic plan computed by the frozen source; since the full plan was not included in the return archive, that internal digest was **read and checked for presence/format but not independently recomputed from archived raw plan**. The ZIP checksum and included JSON are independently verified.
- `mixed8_additional_gpu_work=0`; correctly reuses NP LF+S2 portfolio IDs rather than charging hidden additional candidates.
- `checks`: no cohort allocated, no GPU work, no measurement created, no model loaded, no B24 confirmation access, and no executable native commands.

This is a full PASS of the **synthetic manifest contract only**. No native reconstruction, independent stochastic stream execution, float PSNR comparability, actual compute budget, or real image cohort has been demonstrated.

## Publication and method implications

A0 retrospective B22 score statistics remain exposed/nonconfirmatory; A1.0 source/metadata and A1.0b/0c DAPS-SITCOM GT/seed semantics passed. A1.0d closes the candidate/seed bookkeeping substage. Repeating another synthetic restart-plan stage without a specific detected defect would have little scientific value.

**Next work:** A1.1 preparation of real native execution/measurement adapter and common metric evaluator, without changing B19–B26 historical sources. Native DAPS returns saved 8-bit PNG while SITCOM/NP write float tensors in historical paths; a prospective oracle PSNR comparison must use one **common representation** (ideally unquantized float tensors from each method, if a noninvasive DAPS adapter supports this; if impossible, preregister a comparable 8-bit-derived evaluation for **all** methods). Never silently compare DAPS PNG PSNR to NP/SITCOM float PSNR.

A development engineering image can be an entirely synthetic procedurally generated RGB image from a new independent source, **only for pipeline integration**, with explicit distribution-shift warning; it cannot count as an FFHQ quality/failure benchmark. A scientifically meaningful prospective FFHQ benchmark still requires a separately provenance-disjoint allowed image pool without reading B24 confirmation305 membership. No GPU is reserved; before any active run, report source SHA, data source/seed, physical GPU UUID/time, memory and compute budget and stop gates in issue #41 and provide user-specific launch commands.

**Stage disposition:** `A1_0D_PASS_GO_A1_NATIVE_ADAPTER_ENGINEERING`, `A1_DATA_COHORT_UNALLOCATED`, `A1_GPU_NOT_RESERVED`.
