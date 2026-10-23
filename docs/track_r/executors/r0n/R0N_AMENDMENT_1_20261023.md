# R0-N amendment 1 (planner, 2026-10-23, after the planner verified the R0.8 ZIP)

**Scope.** This amendment adds to the R0-N handoff. It changes no frozen R0 or R0-N v1 definition. It adds no GPU work and no data access beyond what is listed here.

## A1 — N0 checksum list

The planner verified the uploaded ZIP: SHA-256 `13c008f3bca59b7d8d717ec3215fa1f783ca453b31b9001886c84633b513efab`, `testzip` clean. When committing the N0 files, they must match these hashes exactly:

```text
5b4b14fd9d104a7f1d5dd6f228adba9046f7f466ebc7e06f33a7282f575586b6  R0_7_ANALYSIS.json
50ff1550d3bb6fa64bac75d286c5cfab240e54ff391dba4ff9b0c3820f3e79c8  R0_7_COMPLETE_RECEIPT.json
10a9ac826f15b970997043ce65ebba58c4827655b08fa9bee0b76517bdbfd738  R0_7_REPORT.md
ff892a48d6000339b53d2747db5fee41d38090b4289d32a1e66a006c2b344c35  R0_8_EXPLORATORY_CASEBOOK.json
d73a9693c825bd58b881bad6c871964fad387151013d91cd9b1ed00f16c411d9  R0_8_P4_MATRIX.csv
aaed4b654aa41ca5e20fdf4ddb73e8655b1d15a272d205ae267298525d916f32  R0_8_PROVISIONAL_DECISIONS.json
e8cf5cb141c97272c44dfc78ccf1bd5a5d1abe4f1529841c6e77b3ee5e48b9a1  R0_8_READINESS.json
26dc399ab4a5926c5e2be55c04ec2064ff6222bb331e655902beabbe75e42ec3  R0_8_RESEARCH_AUDIT.md
```

## A2 — new task N1h: training-membership check (CPU, bounded)

**Purpose.** Rule out memorization as the reason the prior repairs orientation so well.

**Steps:**
1. For every image id in the B22_random and DEV80_enriched cohorts, look up the FFHQ metadata `category` field ("training" or "validation") in `/egr/research-pac/huang248/data/ffhq/ffhq-dataset/ffhq-dataset-v2.json`.
   - Look up by id only. Do not enumerate or scan the image tree.
2. Report the counts per cohort.
3. Report repair success and harm (prior S and S_mid) separately for validation-category and training-category images.
4. Record any documentation of which images `ffhq_10m.pt` was trained on, found in the DPS, DiffPIR or DiffFPR repositories or papers already on PAC (`external/dps`, `external/DiffFPR`). Use targeted `grep` only. If none is found, write "UNKNOWN".

**Output:** a short table in `R0N_REPORT.md` and fields in `R0N_LETTER_EVIDENCE.json`.

## A3 — reporting note for N1b/N1g

The planner has already verified E1 − O_raw at maximal K from the R0.8 P4 matrix: B22 DAPS +3, B22 SITCOM −1, B22 NP −2, DEV80 DAPS +1, DEV80 SITCOM +3, DEV80 NP −1, FAST 0 (Good25 images).

N1b/N1g must reproduce these exactly. They must also add:
- the FAST split by cohort;
- the image identities behind each post-selection fix and generation failure.
