# A1.3 stage0 initial PAC attempt — STOP on SITCOM tracked diff identity

Date: 2026-10-20 America/Detroit. Branch: `codex/prdiff-track-a-a13-candidate-audit-20261020`.

## Actual observed PAC receipt (user-provided stdout; no direct PAC access)

- New isolated clone succeeded: A1.3 commit `6639992b0abfcee5a032e708b74067bf6f066faf`, correct dedicated branch and worktree; existing A1 source HEAD previously verified as `f4cd28dcd3ec4860cc9f3990fb24964bb97e97bd`.
- Seven synthetic CPU regression tests **PASS** (1.617 seconds): min/max/ties, nonfinite/GT-leakage, PNG8 PSNR and rotation, signed-negative handling, inversion symmetry, impulse/scale, frozen noise reference.
- Stage0 inventory **STOPPED**, Python return code `7`, reason `STOP native dirty patch mismatch SITCOM`. The seven tests' PASS **does not imply native SITCOM/DAPS operator equivalence or 128 hashes passed**, because source gate executed first.
- At this stop, no A1.3 inventory receipt, candidate forensic analysis, GPU work, new measurements, or model inference has been reported. Existing 128 A1.2 outputs remain untouched by our code.
- Historical B24 documentation pinned SITCOM local tracked diff SHA256 `a9f0076d6f852b6898000142c19a09131ffc49ceba0e3d935cd465e85df26e6e`, but A1.2 original CPU preflight source controller pinned SITCOM **HEAD**, not this full SITCOM dirty-diff hash. Therefore this mismatch could be an overstrict A1.3 inherited historical B24 patch expectation, a later local change, or genuine source drift; **cannot distinguish without read-only evidence**. Do not repin based on a guessed hash.
- Original A1.3 shell wrapper returned status 0 after echoing `A13_INVENTORY_RC=7`, masking failure. This is an engineering reporting bug. Fixed in later dedicated branch commits `44bf0ab254bc04619b556603e6527eb2dab2bc78` (stage0) and `5083b56e15ee94034d049897024ecf6e77721354` (stage1). Both now return nonzero without closing the interactive PAC terminal; neither has been rerun on PAC. **Do not pull and rerun stage0 until the SITCOM source discrepancy is diagnosed.**

## Next read-only PAC evidence (no source mutation, no new output)

SITCOM path exactly `/egr/research-pac/huang248/external/SITCOM_ODE`. Read exact `git rev-parse HEAD`, `git diff HEAD -- | sha256sum`, `git diff HEAD --name-status --`, `git diff HEAD --numstat --`, tracked `git status --short --untracked-files=no`, and targeted diff of native `forward_operator/__init__.py` and `forward_operator/fastmri_utils.py` if present (bounded to 100 lines). Also print native operator file hashes and tracked names. No reset, checkout, merge, install, or `git add`. Report the stdout and preserve the 128 saved reconstructions.

If the mismatch is benign and fully source-explained, commit a precise, justified new A1.3 source identity/check protocol; otherwise produce an explicit source-uncertain STOP report. Never silently disable patch checks or claim A1.2 candidate provenance was reverified by this STOP.

No A1.3 stage1 or GT-based candidate inspection is authorized from this receipt. Protected B24 confirmation305 remains off-limits, including metadata/membership.

## Targeted read-only SITCOM diagnostic received

The user subsequently ran the bounded SITCOM diagnostic on PAC:
- Native SITCOM HEAD **PASS**, exact `275ab67efbd8146bffca20155171ba6be1169c09`.
- Current **full tracked** dirty diff SHA256 `0a5587945acbbb86da46ca8e49996bb79ff4cf142c821f268515777ad379ab2a` (mismatch with historical B24 full diff `a9f0076d6f852b6898000142c19a09131ffc49ceba0e3d935cd465e85df26e6e`).
- Exactly **11** modified tracked files: 10 `__pycache__/*.cpython-38.pyc` bytecode files, each differing by 26 bytes; **one** changed source file `posterior_sample.py` with 1 deletion and 1 insertion. No other tracked Python source changes reported.
- Native measurement source `forward_operator/__init__.py` **unchanged** relative to HEAD; Git blob `adee8c03c7041e2c63346ea0e7aed59ad25f47dd`. Native `forward_operator/fastmri_utils.py` **unchanged** relative to HEAD; Git blob `f669ea2dd5c48fe9798ce1ce800b9789e22e0657`. No forward-operator diff output.

**Interpretation:** The earlier full tracked patch gate is overly sensitive to tracked bytecode. The native operator source appears unchanged; however, because the actual `posterior_sample.py` one-line diff was not included, its possible provenance/logic implications remain **unresolved**. There is no basis to claim the whole SITCOM application source is historically identical; nor evidence here that measurement physics changed. Do **not** reset bytecode, assume current full diff is canonical, or repin a full digest without reviewing the remaining one-line source patch. Next step: `git -C "$SIT" diff --no-ext-diff --unified=4 HEAD -- posterior_sample.py` and `git -C "$SIT" diff --no-ext-diff HEAD -- posterior_sample.py | sha256sum`, read-only. Then decide a relevance-scoped gate: pinned HEAD, pristine forward operator source git blobs, explicit observed posterior patch identity, ignore/report bytecode as non-source metadata, and native CPU numerical equivalence, while avoiding false provenance claims about historical sampler execution.
