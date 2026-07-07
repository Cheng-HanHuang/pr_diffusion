# B21.0 measurement integrity audit

Status: complete.  
Generated from PAC rerun output pasted on 2026-07-08.  
Script: `scripts/b21/audit_measurement_integrity.py`.

## G0 verdict

**G0 FAIL. Treat B19.20 as `n = 100` images, not `n = 1000` independent image-measurement cases.**

The locked measurement payloads for seeds `5001--5010` are distinct by SHA, but the B19.20 result CSVs show degenerate PSNR variation across measurement seeds for most eligible groups. Therefore the issue is not duplicate measurement payload files; it is downstream of payload generation, most likely in the runner/loader/output/analyzer path that produced the B19.20 CSVs.

Planner consequence: B19.20 can still be used as an image-level FFHQ100 diagnostic panel, but measurement-seed independence claims from B19.20 should not be cited. FFHQ100 multi-measurement revalidation must move to B22 or a later audited rerun.

## Inputs found

```text
Requested measurement seeds: 5001,5002,5003,5004,5005,5006,5007,5008,5009,5010
Expected image count:       100
Measurement payloads loaded: 1010
Measurement load errors:     0
Distinct images with payloads: 110
Expected image-seed pairs:   1000
Per-case CSVs found:         12
```

The extra ten measurement payload rows are not by themselves an integrity failure; the audit inventory searched all available measurement payloads under the B19 measurement root and found 110 image IDs with seeds in the requested range. The expected B19.20 image-seed count remains 100 x 10.

## Measurement distinctness

```text
Measurement verdict: PASS
Images with duplicate SHA across measurement seeds: 0
Duplicate SHA rows: 0
```

Artifact paths on PAC:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_0_measurement_integrity_audit/measurement_inventory.csv
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_0_measurement_integrity_audit/measurement_duplicate_sha.csv
```

## PSNR seed-variation check

```text
PSNR verdict: FAIL
Eligible groups with >1 measurement seed: 5303
Groups with sd < 0.01 dB:       4583
Fraction sd < 0.01 dB:          0.8642277955874034
```

This violates the protocol requirement that fewer than 10% of fixed `(image, run)` groups should have PSNR standard deviation below `0.01 dB` across multiple measurement seeds. The observed fraction is about `86.4%`.

Example rows from the audit include many groups with all listed seeds sharing identical PSNR to full CSV precision, e.g.:

```text
F4_full_exact, image 00046, run 3, seeds 5001--5010, psnr_mean=min=max=20.97223472595215, sd=0.0
F4_full_exact, image 00171, run 0, seeds 5001--5010, psnr_mean=min=max=14.304004669189451, sd=0.0
F4_full_exact, image 00272, run 0, seeds 5001--5010, psnr_mean=min=max=29.452560424804688, sd=0.0
```

Artifact paths on PAC:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_0_measurement_integrity_audit/case_rows_extracted.csv
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_0_measurement_integrity_audit/case_psnr_seed_variation.csv
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_0_measurement_integrity_audit/runner_measurement_path_snippets.txt
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_0_measurement_integrity_audit/summary.json
```

## Interpretation

Facts:

1. The locked measurement payload files are distinct across measurement seeds.
2. The B19.20 result CSVs largely do not vary across measurement seeds.
3. Therefore the measurement-generation step is not the direct failure mode identified by this audit.
4. The suspect location is the path between saved payloads and final B19.20 CSV rows: runner measurement-path templating, payload loading, output reuse, or analyzer grouping.

Conservative reporting rule:

```text
Do not cite B19.20 as 1000 independent image-measurement cases.
Treat it as an FFHQ100 image panel until an audited rerun proves measurement-seed variation.
```

## Follow-up required before B22 FFHQ100 rerun

1. Identify the B19.20 launcher used for runseed `4400`.
2. Check whether `measurement_path` varied with `meas_seed` at execution time.
3. Confirm whether the runner output directory included `meas_seed` in every path, not just in labels.
4. Run a tiny one-image/two-measurement smoke test that logs:
   - measurement payload SHA loaded by the runner;
   - measurement tensor mean/std/first values;
   - output directory path;
   - final PSNR.
5. Only then launch a fresh FFHQ100 measurement-seed panel.
