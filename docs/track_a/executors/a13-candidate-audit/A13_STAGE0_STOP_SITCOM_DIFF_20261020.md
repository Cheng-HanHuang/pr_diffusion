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
