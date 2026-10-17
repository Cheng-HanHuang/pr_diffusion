# Track A A1.0 — PAC V1 STOP and bounded V2 checkpoint-link correction

**Date:** 2026-10-17. **Owner:** continuing Track A executor. **Status:** `V1_STOP_CHECKPOINT_LINK / V2_CORRECTION_COMMITTED_NOT_EXECUTED`.

## Returned V1 PAC evidence

User executed the pinned branch at `da9ae473f972b7ce771fa460b073c33b0824be72`. Returned `A1_0_PREFLIGHT_RETURN.zip` with independently checked SHA-256 `2ba9a4fc9c211b1d20efae484f28248c49610b581cc683ec5fddebfcdf9d4f4f`; ZIP members: `A1_0_SYNTHETIC_TESTS.log`, `A1_0_PREFLIGHT.log`, `a1_0_preflight_20261017/A1_0_PREFLIGHT.json`. Cross-checked the JSON and original stdout.

- **Seven of seven** synthetic tests passed, `A1_SYNTHETIC_TEST_RC=0`.
- Git HEAD and tracked-diff source gates: **DAPS, DiffFPR and SITCOM `SOURCE_SIGNATURE_MATCH`**. DAPS diff SHA `fbb5b42369ecf0d3b9b67f8fc162053bc40ec32aed41dbd92a67e8d81dcfad69`; DiffFPR clean HEAD `a45ffe58f18fed8a63d3446600424e2b08733524`; SITCOM B22 one-line patched mkdir identity reproduced at HEAD `275ab67efbd8146bffca20155171ba6be1169c09`. The SITCOM **B24 full tracked diff remains uncertified**, not resolved by this preflight.
- NP/DAPS model `/egr/research-pac/huang248/models/ffhq_10m.pt` verified **374,417,833 bytes**, SHA-256 `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`.
- Metadata digests **PASS** for B21 exposed Fresh2 rows, B22 full manifest, B24 **DEV80 only** manifest and original A0 B22 replay ZIP. Frozen exposed cohorts are **100 B21/B22 unique images and 80 B24 DEV80 unique images with zero overlap**, 180 total unique exposed IDs, **not** a newly eligible test population.
- Available Python environment paths observed; the public FFHQ image-root directory exists, but image names/bytes were not listed/read.
- One-shot `nvidia-smi`: **four NVIDIA RTX PRO 6000 Blackwell Server Edition** devices indexed 0–3, memory.total 97887 MiB each; 0 MiB used, 0% utilization at that instant. Neither idle status nor apparent capacity constitutes a GPU reservation.
- **Only STOP cause:** `SITCOM_native: missing/nonregular/symlink required file: /egr/research-pac/huang248/external/SITCOM_ODE/checkpoint/ffhq256.pt`. No checkpoint SHA was produced in V1. **Do not assert it is definitely a symlink**; strict `file_sha()` rejects both missing and symlink names. A0 previously verified the named SITCOM checkpoint content SHA, making a normal target-changing scenario less likely, but current V1 alone cannot prove it.
- Honest machine exit: `A1_PREFLIGHT_RC=3`, `A1_FINAL_STATUS=4`, wrapper `A1_STAGE0_RUNNER_RC=1`. **Overall `STOP_PREFLIGHT`, not PASS.**
- No GPU work, new reconstruction, image data, new measurements, cohort selection, or confirmation305 access.

## Minimal V2 repair with source provenance maintained

The error originates at A1's new *checker*, which called `file_sha()` with `reject-symlink` enabled on a **checkpoint** path. Many valid model installations install checkpoints as links. The historical read-only metadata check **must continue rejecting symlinks**, so V2 adds `checkpoint_sha()` **only for the two explicitly named checkpoint files**:

1. Require checkpoint name to exist as a file or symbolic link.
2. Resolve links with `Path.resolve(strict=True)` and fail closed for dangling links or cycles.
3. Require final content path to be a **regular file within `/egr/research-pac/huang248`**. An escaped link is STOP.
4. SHA-256-hash **resolved file contents** under same size bound (2 GB), compare to the frozen native model SHA `81d53574…`.
5. Record the original path, `is_symlink`, literal `link_text` if linked, the resolved target and file size/digest in the receipt.
6. Keep `file_sha()` unchanged for B21/B22/B24 DEV80 manifests and the A0 capsule.

V2 synthetic regressions cover a valid in-workspace symlink, ordinary checkpoint file, dangling symlink and target outside the allowed workspace; **V2 tests have not yet run on PAC**.

No prior file is overwritten. V2 output is `a1_0_preflight_20261017_v2/A1_0_PREFLIGHT.json`; synthetic/preflight logs are `A1_0_SYNTHETIC_TESTS_V2.log` / `A1_0_PREFLIGHT_V2.log`; archive `A1_0_PREFLIGHT_RETURN_V2.zip`. The existing user checkout can be strictly fast-forwarded from V1 head to the new pinned commit, then the new V2 runner invoked.

## Next gate

If V2's SITCOM resolved checkpoint **matches SHA**, other source and archive gates still PASS and tests PASS, source/metadata preflight is complete **for the bounded dimensions it tested**. This is **not** a ready-to-run GPU experiment. Remaining gates: independently seeded SITCOM/DAPS native startup, method/measurement preprocessing and float-PSNR equivalence, new eligible cohort allocation without touching B24 confirmation305, then Track A/Track B ledger and actual physical device/time/resource reservations. No development images selected in V1, and V2 must not silently select any.
