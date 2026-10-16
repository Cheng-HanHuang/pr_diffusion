# A0 source and artifact manifest — review of PAC inventory, 2026-10-15

**Disposition:** **PARTIAL INVENTORY / SOURCE-ID STOP**, not an A0.1 analysis PASS. The raw PAC-generated JSON, text log, and checksums are retained under the exclusive A0 PAC output directory. This repository contains the compact reviewed `A0_SOURCE_AND_ARTIFACT_MANIFEST.json`.

## Source and integrity

The user-executed command returned `A0_SCRIPT_RC=0`; all five ZIP contents listed in PAC `SHA256SUMS.txt` independently match after mapping their absolute PAC filenames to ZIP basenames. Received ZIP SHA-256: `77342e210f70580f4148860fc2de6bb89441aa92935296ab9cf337d7bccb07bf`. PAC JSON SHA-256: `bc749c7bd92c956585934d2bd1dcd5e03583982ef7a242b5b818301a62584203`. Source inventory did not claim an executor PASS (`AWAITING_SOURCE_AND_SCHEMA_REVIEW`).

| Source | PAC HEAD | Full tracked diff | Decision |
|---|---|---|---|
| B19 checkout | `0c3c2ec9...` | `b57536b4...` | Preserve dirty historical worktree |
| B22 checkout hint | `ba78c06e...` | empty | Run manifest instead records `191dbd70...`; don't substitute checkout HEAD |
| B24 checkout | `19d1054d...` | empty | Later reporting-correction checkout, not the original development launch |
| DAPS external | `e7a77d09...` | `fbb5b423...` | **MATCHES** frozen B24 recorded tracked diff |
| SITCOM external | `275ab67e...` | `0a558794...` | **NOT MATCHED** to frozen `a9f0076d...` |
| DiffFPR | `a45ffe58...` | empty | Matches B22 DiffFPR identity |

SITCOM status includes tracked `posterior_sample.py` **and ten modified tracked `.pyc` files**. This is an identifiable possible explanation, not a proven reconciliation; compare source-only diff and inspect the distinct native SITCOM checkpoint SHA. **Do not reset, clean, restore, or copy anything in native repositories.** The model SHA `81d53574...` is the separately documented FFHQ DiffFPR/DAPS model, and must not be copied into the SITCOM checkpoint identity.

## Cohorts, fields and representation

| Cohort | Verified inventory evidence | Available clean-free native | Offline GT oracle | Missing/qualification |
|---|---|---|---|---|
| B21 Fresh2 | 100 CSV rows, two candidate PSNR/loss/seed/wall-time columns | DAPS K2, loss margin 0.7 | K1/K2 possible | Source/measurement content hashes not in B21 CSV; no K4 |
| B22 full100 | 100 validated policy records per method, **sampled** first/last candidate records (SITCOM four each, NP eight each) | SITCOM K4 step160 norm; NP LF K1..4, mixed K8 | SITCOM K1..4, NP homogeneous K1..4 and mixed K8 | Full candidate-file enumeration and per-item identity not done in inventory; B22 NP signed terminal residual not established |
| B24 DEV80 | 80 PASS receipts, 20/stratum, 16 Pilot16 NP reused; 320 DAPS, 320 SITCOM, 320 NP roots in receipts | DAPS Fresh2 K2 exact persisted loss; NP LF K1..4 native score | DAPS/SITCOM K1..4, NP LF K1..4 | DAPS native K4 **does not exist**; SITCOM native step160 K4 **not recorded**; source drift unresolved |
| B24 corrected | Fresh2 80-row CSV and stored signed loss0/loss1, 29 rep1 acceptances | DAPS K2 | DAPS K2/K4 | Reading metadata only; never recompute signed loss |

**Metric fairness:** B22 tensor raw RGB PSNR is a different pipeline from B24. Within B24, DAPS/SITCOM external baselines are evaluated on saved **8-bit quantized RGB PNGs**; native NP4 stores float-tensor PSNR. The analysis specification keeps those representation labels distinct and prohibits claiming a precisely matched continuous paired PSNR comparison without actual compatibility verification. Do not pool B22 final100 and selected, stratified B24 DEV80 for significance.

**Cost:** B22 per-candidate reconstruction seconds, B21 DAPS arm wall times, and B24 DAPS/SITCOM per-candidate GPU-active seconds are persisted. B24 NP4 has **8,800 UNet calls** and recorded whole-arm wall time, not separable per-root GPU-active time. Cross-family dispatch-supported FLOPs are partial and not exact whole-algorithm FLOPs.

## Explicit A0.1 gate

Pure metadata selector functions, tests and a fail-closed extractor are staged on this branch, **but they have not been run on PAC**. The extractor requires a read-only SITCOM source-only diff reconciliation plus separately verified checkpoint SHA; without them it returns STOP. It further fails on incomplete candidates, differing measurement IDs, metric representation changes, missing native selectors and headline-replay drift.

The intended frozen replay numbers, **not recomputed in this audit**, are B22 DAPS K2 **92/100**, SITCOM K4 **93/100**, NP mixed K8 **95/100**; B24 DAPS Fresh2 K2 **57/80** and NP4 **55/80**. No new PSNR, GT metric, FFT, measurements, reconstructions, or GPU work was performed.

B24 confirmation305 remains locked and untouched. The cross-track ledger is GitHub issue #41.
