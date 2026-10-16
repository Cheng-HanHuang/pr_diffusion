# A0 source recheck correction — 2026-10-15

**Current gate:** `STOP_SOURCE_IDENTITY_NOT_RECONCILED`, still binding for the comprehensive B22/B24 native/oracle replay. The new user-executed source inventory JSON has SHA-256 `2c811247ae8cd077aac57ab23ce19f219ef3169a2b69cb04943ffbdccffa16ed` and remains on PAC in the A0-owned output directory.

## Facts established from actual PAC output
- Native official SITCOM Git HEAD `275ab67efbd8146bffca20155171ba6be1169c09`: **MATCH**.
- Native SITCOM checkpoint `checkpoint/ffhq256.pt`: 374417833 bytes, SHA-256 `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`: **RECORDED CURRENT IDENTITY**, same SHA as the expected FFHQ checkpoint.
- Modified tracked source files: only `posterior_sample.py`. Ten additional modified tracked `.pyc` files under `__pycache__`.
- Frozen B24 **full** tracked diff SHA `a9f0076d6f852b6898000142c19a09131ffc49ceba0e3d935cd465e85df26e6e`.
- Current **full** tracked diff SHA `0a5587945acbbb86da46ca8e49996bb79ff4cf142c821f268515777ad379ab2a`.
- Current **source-only** diff SHA `d0bb0292e341abe3f6b99984bb2a98acb5919c41efe0f4bd8fd0ffb60b464fc5`.

## Invalid prior test corrected

The A0 source-only check wrongly compared the current source-only hash to the **historical full-diff** hash. These are hashes of differently defined byte streams; unequal hashes **do not prove source-code drift**. Equally, seeing only modified bytecode besides `posterior_sample.py` is **not proof** that the Python source matches the historical B24 file. Source patch content must be compared against a genuinely archived historical patch or source-file content SHA.

The historical B22 `full/manifest.json` **does** preserve the exact `posterior_sample.py` patch (one `Path(dir).mkdir(parents=True, exist_ok=True)` change). This is an independent B22 reference, **not** automatically the frozen B24 patch. The B24 frozen source ledger currently records a full tracked-diff hash rather than a complete source-only patch; therefore equivalence to B22, if shown, is not sufficient by itself to certify B24's launch-time full tracked source identity.

The fixed tool `scripts/track_a/a0_source_reconcile.py` now hashes both current full and source-only diffs, checks exact B22 archived source patch text separately, and refuses to mark B24 source PASS from a cross-definition hash comparison. The `scripts/track_a/a0_historical_compare.py` analysis gate was likewise corrected to require a genuine full-diff identity match. This cannot PASS on the current evidence.

## Correct next read-only step

Show the bounded current `git diff -- posterior_sample.py` patch, with SHA and status; compare against the separately archived B22 patch. Do not edit/reset/restore tracked source or bytecode. If the original B24 version cannot be reconstructed from permitted archived patch/source-file hashes, report `SOURCE_UNCERTAIN` and stop A0.1 rather than claiming verified B24 SITCOM native results. If a trusted frozen full historical diff is later available, reconcile it separately.

All previous A0 inventory and documented historical headline results remain unchanged. No candidate or terminal data, model inference, GT-payload, FFT, measurement generation, GPU or confirmation305 data was accessed by this source recheck.
