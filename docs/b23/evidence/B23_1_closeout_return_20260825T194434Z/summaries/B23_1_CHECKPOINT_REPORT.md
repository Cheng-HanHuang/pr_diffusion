# B23.1A/B checkpoint report

Status: **PASS_RECOMMEND_B23_1A_B_SIGNOFF_ONLY**

GPU work performed during evidence closeout: **NO**

The planner accepted the scientific execution at `3ffb237818e1bfa4921b3f4f8bc9a3bd24b7e406` and the repaired packaging at
`fad055d40d5bd0eaf4c9471359177c321958d2d7`. This zero-GPU closeout revalidated the immutable `B23_1_return_20260825T184922Z` archive,
all 300 internal checksums, and all 29 scientific prerequisite rows without
launching a parent process, generating a measurement, or reconstructing an image.

The accepted bounded execution contains 32 trajectories: 12 native repeats, four wrapper runs, and
16 heterogeneous smoke runs. One completed Fresh1 native trajectory was recovered rather than
rerun. All four replay reports are BITWISE/PASS, all four compute ledgers are calibrated, and all
four smoke images completed under all four parents.

The cross-family H0 **failed**: zero NP/SITCOM cross-family adapters qualified. NP-1 and SITCOM-1
remain baseline-only across family boundaries. The evidence supports only the narrowed statement
`CONTINUE DAPS-NATIVE ONLY UNDER NARROWED CLAIM`; it does not authorize a schedule.

B23.2, B24 execution, large panels, and adaptive schedules remain **NOT AUTHORIZED**. Full raw
scientific artifacts remain on PAC and are identified by absolute path, byte size, producer commit,
and SHA-256 in `ARTIFACT_MANIFEST.tsv`.
