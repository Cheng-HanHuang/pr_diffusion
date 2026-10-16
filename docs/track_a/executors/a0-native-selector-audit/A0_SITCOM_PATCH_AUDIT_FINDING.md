# A0 SITCOM historical-patch comparison — archived B22 vs PAC current

This is a metadata-only, **content-level comparison** of the previously received PAC `A0_SITCOM_PATCH_AUDIT.json` against `B22_FULL.identities.sitcom_posterior_sample_diff` in the 2026-10-15 A0 read-only inventory manifest. Its B22 source is the validated B22 `full/manifest.json` (SHA `94184e36b27aac0defb93b31ae7b47ea332909bcb0c0a80445be2949f422a219`).

## Observed checks

| Evidence | Result |
|---|---|
| PAC patch check reported | `matches_frozen_B22_patch=false` |
| Current patch bytes SHA | `d0bb0292e341abe3f6b99984bb2a98acb5919c41efe0f4bd8fd0ffb60b464fc5` |
| Archived B22 patch bytes SHA | `b1d679f61d178e51624e930f2c574cb4daa4b326578892aaebc7f5cc428662f0` |
| Exact added/removed lines | **IDENTICAL** |
| Archival/current diff after `rstrip("\\n ")` | **EQUAL** |
| Current SITCOM HEAD | frozen `275ab67efbd8146bffca20155171ba6be1169c09` |
| Currently verified native checkpoint | SHA `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa` |

Both patches change exactly one tracked Python line in `posterior_sample.py`:

```diff
-        Path(dir).mkdir()
+        Path(dir).mkdir(parents=True, exist_ok=True)
```

The current `git diff` includes **two trailing unchanged blank context lines**, represented as `\n \n \n`; the archived B22 patch text ends at `return Path(dir)` with no final newline. A byte-exact digest comparison fails because the archive shortened the context. The diff's changed lines and nontrailing text match. This is a **B22 source-change reconciliation**, not a byte-for-byte equality claim.

## Scope of provenance finding

The original B22 manifest independently preserves the Python source patch. Combined with matching HEAD and checkpoint, the **change-level B22 source identity** is supported for a **read-only B22-only metadata replay**.

This does **not** prove frozen **B24** full tracked source identity. B24 recorded `a9f0076d...` from a different *full tracked* diff byte stream (tracked `.pyc` files may contribute), which current `0a558794...` does not match. Without the historical B24 file content/patch evidence, source-specific B24 conclusions remain SOURCE_UNCERTAIN. Do not reset or modify any native repo to alter the hash.

The PAC command exited 0 and created a new audit JSON without reading reconstructions, images, models for inference, operators, FFTs, measurements or confirmation305. Candidate-level analysis has not yet run.
