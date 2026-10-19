# A1.1b V2 — post-expiry restart instructions, **NOT a continuation**

**Date:** 2026-10-19 evening, America/Detroit. **Status:** `V2_CODE_READY_NO_GPU_WORK_YET`.

## Why V2 exists

The user's V1 run on pinned commit `f745898c4073466af8ca41d58b1979017d93fea3` returned before creating the GPU run directory. The sole failure was the deliberate hard date gate (`run_start_in_approved_window` for 12:30–16:00 EDT Oct 19), at an evening execution. No native subprocess started; V1 ZIP SHA256 `ea631b9b581adc07b86511197574f247262ccf6dc62931c84f4574fa3b681672` verified. See `A1_1B_V1_EXPIRED_WINDOW_STOP_REPORT.md`. Preserve V1 log and ZIP forever; do NOT unlink them.

## Fix without relaxing compute protection

- `scripts/track_a/a1_1b_native_one_each.py`: uses unique V2 output dir `a1_1b_one_each_v2`, includes updated UTC hard-start window `2026-10-19T23:30:00Z` through `2026-10-20T03:30:00Z` (= Oct 19, 19:30–23:30 EDT), refuses any execution outside this date range. A one-hour execution started in-window may finish up to Oct 20 00:30 EDT at latest. The per-method 1500-second limit, aggregate 3600-second limit, physical GPU3 UUID pin, <=1024 MiB initial use, >=52096 MiB free, <=52452 MiB live device use, subprocess self-kill on limit and frozen-source checks remain unchanged.
- `scripts/track_a/a1_1b_run_gpu_smoke_v2.sh`: new non-overwriting logs `A1_1B_GPU_NATIVE_ONEEACH_V2.log` and archive `A1_1B_GPU_NATIVE_RETURN_V2.zip`. Does **not** rerun or overwrite old `A1_1B_GPU_NATIVE_RETURN_V1.zip`. Caller must pass explicit `A1_EXPECTED_HEAD`, physical GPU3, and `A1_RUN_RESERVATION_ACK`; only user execution can launch GPU.
- `scripts/track_a/test_a1_1b_native_one_each.py`: updated explicit UTC early/on-time/late boundary unit tests; the V1 CPU preflight artifact stays valid for unmodified data/model/method semantics, but this V2 source/date change must be validated by the user on PAC before GPU launch.

## Scientific scope

Exactly **one full native candidate per method**: DAPS, SITCOM, NP LF, NP S2, sequential, on the **same** procedural engineering-only frozen observation SHA256 `992a33efa9bfb8e81d97ed61cf28369263ae1f11aeae19273c53973c2470acdb`. Native source/model hashes unchanged. A1-owned output only. Common offline saved RGB8 PSNR on procedural synthetic data is diagnostic and cannot support face-domain reliability claim. B24 locked confirmation305 remains inaccessible, no real FFHQ data cohort allocated.

## Resource governance and launch STOP

The GitHub issue #41 V1 reservation is **expired and not renewed automatically**. A new conditional `[RUN RESERVATION]` is required for V2. Track B retains GPU priority and a single idle reading does not guarantee exclusive access. Before running GPU shell, check physical GPU3 UUID and running compute apps; do not launch if any Track B or other job uses the GPU. No changing GPU number without a corresponding new reservation/source update.

A failed native subprocess should **STOP**, preserve `A1_1B_STOP.json` and full result logs, and never auto-run the next method or retry V2 over its own directory. A followup investigation should create V3 only after inspecting V2 receipt. If the date window expires again, renew deliberately instead of bypassing the guard.

**Next:** user fast-forwards from V1 `f745898c...` to pinned V2 head, runs updated CPU unit tests and `--mode plan`; if both pass, confirms current GPU3 availability and issue reservation, then runs the explicit `a1_1b_run_gpu_smoke_v2.sh` one time. Return ZIP and log; long GPU campaign remains separate and not authorized.
