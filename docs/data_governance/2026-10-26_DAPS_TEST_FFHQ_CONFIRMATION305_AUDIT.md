# DAPS `test-ffhq` versus B24 `confirmation305`: filename-level split audit

**Audit date:** 2026-10-26 (UTC timestamp: `2026-10-26T06:46:59.195529+00:00`)  
**Status:** **NOT DISJOINT at filename-ID level**. Count-only integrity audit; not an experiment or change to any frozen split.  
**Scope:** Official DAPS test-FFHQ archive versus frozen B24 image-role metadata. No image-pixel equivalence check.

## Result

| Comparison | Left-hand cohort | B24 `confirmation305` | Matching FFHQ filename IDs | Disjoint? |
| --- | ---: | ---: | ---: | --- |
| DAPS configured `test-ffhq` evaluation | 100 | 305 | **1** | **No** |
| Complete downloaded DAPS test-FFHQ archive | 1,000 | 305 | **1** | **No** |

The one matching filename ID lies inside DAPS's configured 100-image evaluation subset, not merely its 1,000-image archive. Do not publish or infer the individual confirmation member from this count-only audit.

## Provenance and exact selection rule

- **Official DAPS source:** [`zhangbingliang2019/DAPS` `configs/data/test-ffhq.yaml`](https://github.com/zhangbingliang2019/DAPS/blob/main/configs/data/test-ffhq.yaml) configures `root: dataset/test-ffhq`, `start_id: 0`, `end_id: 100`. Its [`data.py`](https://github.com/zhangbingliang2019/DAPS/blob/main/data.py) collects and sorts image paths before slicing by those positions. Therefore indices `0:100` select the first 100 files **after sorting**, not numeric FFHQ IDs 0–99.
- **Official archive:** [DAPS download instructions](https://github.com/zhangbingliang2019/DAPS/blob/main/download.sh), Google Drive file ID `1IzbnLWPpuIw6Z2E4IKrRByh6ihDE5QLO`, local read-only ZIP `/egr/research-pac/huang248/outputs/pr_diffusion/split_integrity/daps_official_test_ffhq.zip`. ZIP size 401,303,051 bytes; SHA-256 `3fc8073bda2ecc021485cbef917a5025de5cae38bae0c0e8872d8eee49c8a919`. Its 1,000 image member filenames follow `test-ffhq/49000.png` through `test-ffhq/49999.png`; first 100 sorted filenames are `49000.png`–`49099.png`. The source ZIP was not extracted for the audit.
- **B24 frozen source:** `/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_2_7424_extension_20260907T231303Z/case_freeze/method_stage/B24_METHOD_IMAGE_ROLES.csv`; SHA-256 `7c05dd67cc39263db74e58b7255297d014e7d674da0c0bebdee317bcbff809bf`. Role counts verified as 80 development and 305 confirmation, 385 unique image IDs. Its location was obtained from `/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_METHOD_STAGE_LATEST_FREEZE.txt`.
- **PAC audit receipt:** `/egr/research-pac/huang248/outputs/pr_diffusion/split_integrity/DAPS_TEST_FFHQ_CONFIRMATION305_AUDIT.json`. The JSON records both intersection counts, source hashes, source sizes and scope.

## Validity and limits

The comparison used five-digit image filename stems and the B24 role-registry `image_id` fields. **Matching original-looking filename IDs are not, by themselves, an independent pixel-identity proof**; the audit JSON marks `pixel_identity_independently_verified: false`. Image resampling or renaming provenance was not independently validated. No confirmation pixels, measurements, solver outputs, reconstruction scores or outcome metrics were read, and no GPU work was done. The *role registry* was read to compute a set intersection; it is inaccurate to say confirmation membership metadata was not accessed.

This count-level check is a narrow user-directed split-integrity exception to the prior Track R rule barring confirmation membership checks; **it does not authorize** any additional confirmation cohort analysis, performance evaluation, model tuning or outcomes access.

## Interpretation and handling

- The historical B24 305-image confirmation cohort should remain **frozen unchanged**. Do not swap, silently reallocate or relabel the overlapping image.
- Do not describe all 305 cases as independent of DAPS's published `test-ffhq` evaluation images.
- If a future comparison requires DAPS-`test-ffhq` identity exclusion, specify a prespecified **304-image DAPS-disjoint analysis subset**, while retaining the original 305-case cohort and disclosing the single exclusion. Any analysis or release of confirmation cases requires separate authorization; this note itself does not authorize that.
- This is an overlap with an **external published test set**, not evidence that B24 confirmation reconstruction measurements or outcomes were examined. It neither establishes a training leak nor proves that the same underlying image pixels were used without further provenance verification.
- Optional further audit: establish provenance of the DAPS ZIP images versus the canonical FFHQ IDs; do not perform protected outcome inspection.

**No frozen manifests, code, experiments or B24/Track R verdicts were modified by this documentation note.**
