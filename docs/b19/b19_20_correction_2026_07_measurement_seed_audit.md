# Correction note: B19.20 measurement-seed independence

Date: 2026-07-08  
Related original report: `docs/b19/b19_20_ffhq100_detector_failure_decomposition.md`  
Audit report: `docs/b21/b21_0_measurement_integrity_audit.md`

## Correction

B19.20 should be treated as an **FFHQ100 image-level diagnostic panel**, not as `100 x 10 = 1000` independent image-measurement cases.

B21.0 audited the saved locked measurement payloads and B19.20 result CSVs:

```text
Measurement payloads loaded: 1010
Measurement load errors: 0
Duplicate SHA rows across measurement seeds: 0
Per-case CSVs found: 12
Eligible PSNR seed-variation groups: 5303
Groups with PSNR sd < 0.01 dB: 4583
Fraction with PSNR sd < 0.01 dB: 0.8642277955874034
```

The measurement payload files themselves are distinct, so this is **not** a duplicate-payload-generation failure. However, the B19.20 CSV metrics are degenerate across measurement seeds for most groups, which fails the B21 protocol's seed-variation sanity check.

## Reporting rule

Do not cite B19.20 as `n=1000` independent measurement-seed validation. Until a fresh audited rerun is completed, cite B19.20 only as an `n=100` image-level panel.

The main qualitative B19.20 lesson remains useful as a diagnostic: the detector layer exposes candidate-generation/oracle-init failures and final-exact-selector failures on a broader FFHQ100 image set. But rates that depend on measurement-seed independence should be marked provisional or corrected.

## Follow-up

Before any B22 FFHQ100 rerun:

1. Verify that the runner logs the exact measurement payload path and SHA loaded for each run.
2. Verify that output directories are measurement-seed-specific.
3. Run a one-image/two-measurement smoke test and confirm final PSNR varies across genuinely distinct measurement payloads.
4. Only then launch full FFHQ100 multi-measurement validation.
