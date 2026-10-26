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
