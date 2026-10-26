# Track R XP executor return — resource conflict and CPU gate repair

**Branch:** `codex/prdiff-track-r-xp-20261026`
**Planner merged:** `92e4c4d05356fa5bda27273776a2eeaa039c29b4`
**Status:** CPU gates not yet validated on PAC; no new XP GPU jobs launched.

## Operator update, 2026-10-26

The XP-0/R0-N.2 attempt passed the five original exact-estimator unit tests.
The next attempt failed two **new synthetic symlink tests** before reaching B2/A8
or XP-0. The inventory helper still rejected all symlinks unconditionally
(`STOP_SYMLINK`). Fix: permit read-only aliases only when both lexical and
resolved paths avoid prohibited tokens and the real target stays beneath
`/egr/research-pac/huang248`. Re-run all 10 CPU tests before accepting either
CPU receipt. Historical outputs are immutable.

## GPU scheduling clarification from Cheng

On the preceding inventory, GPU memory used (MiB) was 43,370 / 43,312 /
23,708 / 4,156 across the four Blackwell devices. Cheng reports that other
jobs are currently using the GPUs and directs **time-aware XP jobs not to
compete with those jobs**. Reschedule internal XP milestones around available
capacity. Quality-oriented BACKGROUND experiments remain in scope, but must
not preempt existing work or bypass the >2-GiB foreign-process gate without
an explicit operator override. CPU-only A8 and XP-0 are unaffected.
The fixed external CVPR paper/supplement deadlines are not represented as
changed, and all experiment runtime / 400 GPU-hour accounting remains binding.

## Next gate

Commit and push R0-N.2 per-image label counts (2,040 G; 85 orbit;
324 consistent; 791 inconsistent) after CPU replay. Complete the XP-0
measurement-SHA and FFHQ-ID inventory, then implement/validate the full
no-new-code runners and two-lane controller before any XP-T launch.

## XP-0 inventory, first successful PAC pass (2026-10-26)

The PAC executor reports `XP0_RC=0`: `XP_0_NEEDS_IMAGENET`,
180 FFHQ identities and 180 unique measurement SHA files verified, with
oversample 4.0 producing pad 128 and canvas 512. The expected
`test-imagenet` dataset directory is absent (ImageNet manifest count 0);
this blocks ImageNet launches but not the FFHQ input verification.
GPU use remains unchanged (no XP GPU launch).

The initial R0-N.2 B2/A8 full pass reached the aggregation of the
3,240 historical R0.7 records and then raised a Python tuple-unpacking
error. Cause: a previous migration to the 4-key
`(source,cohort,family,category)` stratum retained three-key writes for
other B2 counters. Corrected in the next executor commit; tests for
same-key aggregation and output integrity are added. **No A8 counts
have yet been published**, and the theory executor should await a
successful commit of `R0N2_PER_IMAGE_COUNTS.csv`.

## M1 update — A8 accepted and published

PAC reported 14/14 CPU tests passing, and the R0-N.2 B2/A8 run succeeded:
`R0N2_B2_A8_CPU_PASS`, 760 source/family/image rows and 3,240
historical candidates. Immutable label totals replay as
2,040 GOOD / 85 ORBIT / 324 CONSISTENT / 791 INCONSISTENT.
Committed and pushed on the XP branch at
`83abf2614b30f8e02d63ca15886e6e23c3604187`, with
`R0N2_PER_IMAGE_COUNTS.csv` SHA-256
`197d0c8ea2aa47fa17dd379d518f647d7a47d89b3c614449fb106959b7bb45a6`
and B2 membership report SHA-256
`cec48fc01d99eebcb8983f84efd99400fdac8d8cb76aa98a8cbb928086a278da`.

The theory executor can now read the pushed A8 data to fit its
pre-registered FFHQ predictions; this publication is historical
and does not violate the XP run-index embargo.

The CPU-only XP-0 inventory completed:
`XP_0_NEEDS_IMAGENET`, with 180 R0 FFHQ measurements checked for SHA
equality and the exact 512-canvas pad=128 formula verified.
No ImageNet inputs were present at the two inventoried paths.
Next: import only the official DAPS 100-image ImageNet test archive
(File ID `1pqVO-LYrRRL4bVxUidvy-Eb22edpuFCs`), validate every ZIP member
before extraction, then repeat XP-0 against the new asset.
The R0-N.2 A4.3 historical protocol ladder is also queued CPU-only,
without any new inference.

## M1 update: R0-N.2 A4.3 and ImageNet archive correction (2026-10-26)

PAC returned 23/23 CPU unit tests passing and
`R0N2_A43_CPU_PASS`: 14 strata and 224 aggregated policy-ladder rows.
Ladder CSV SHA-256:
`2797220bb489bba620392435480996b663901373081331937907b138e96d0115`.
The four-file return is committed to the XP executor branch at
`b4ca7fe65ca78ca664e92dc11582cb7ee4c0994a`, under
`docs/track_r/executors/r0n/returns/R0N2_A43_20261026T193644Z/`.

Historical cohort-wise anatomy from this frozen analysis, *not* new XP outcomes:
- S1 B22 DAPS (100 images, K2): 95 selected repaired Good25, 5 generation failures;
  SITCOM (100, K4): 94 selected Good25, 4 generation failures, 2 selection misses;
  NP (100, K8): 95 selected Good25, 4 generation failures, 1 selection miss.
- S2 DEV80 DAPS (80, K4): 68 selected Good25, 11 generation failures,
  1 orientation miss; SITCOM: 58 selected Good25, 22 generation failures;
  NP: 56 selected Good25, 23 generation failures, 1 selection miss.
- R0 FAST pooled (180, K4): 155 selected Good25, 24 generation failures,
  1 selection miss. These use the GT-free CLIP orientation selector on
  the E0-residual chosen reconstruction and a GT-only label for analysis.

**ImageNet source mismatch identified.** The official Google Drive archive
for the DAPS `test-imagenet` images
(`1pqVO-LYrRRL4bVxUidvy-Eb22edpuFCs`) was successfully downloaded
once to
`outputs/pr_diffusion/track_r/xp_imagenet_download_20261026T193644Z/test-imagenet.zip`.
Its ZIP member audit found **1,000 ImageNet images**, although DAPS README
describes a supplied subset of 100. The importer correctly refused its
previous exact-100 assumption without extracting any images. The XP
preregistration *already* specifies indices 0 through 99 of the files
sorted in DAPS loader order; the subsequent guarded importer implements
this first 100 of the 1,000, storing original archive SHA, total image
count, canonical relative paths, selected file SHAs, and a manifest.
The importer does **not** assert that this first-100 sample equals the
published DAPS paper's exact evaluation sample; note the distinction when
comparing reported publication PSNRs.

No extra archive download, dataset-bundle FFHQ import, new GPU work, or
time-aware experiment is authorized during this recovery.
