# A1.1b — CPU native-integration plan verified; first runtime-parent correction

**Date:** 2026-10-19. **Executor:** continuing Track A. **Status:** `CPU_PLAN_VERIFIED_PASS / RUNTIME_CODE_CORRECTED / GPU_NOT_EXECUTED`.

## PAC receipt independently verified

User ran `scripts/track_a/a1_1b_run_cpu_preflight.sh` on Git HEAD `bfe5202f7f1b839f45c35602e8f21410788e1f82` in the dedicated A1 checkout, returned `A1_1B_CPU_PLAN_RETURN_V1.zip`. Independently opened the ZIP and verified SHA256 **`f923c0666ca4c744f31af68e23becfb76e4062354316925238ace5dac07f640d`**, exactly matching the PAC terminal receipt. Two contents:

- `A1_1B_CPU_TESTS_V1.log` SHA256 `0df2dc329b46e49c1c78e5cd0f4ecf1197474b80a7d1d28cf8e68fc53b57a7bb`. Exactly **13/13** unit tests PASS; no errors; `A1_1B_CPU_TEST_RC=0`.
- `A1_1B_CPU_SOURCE_PLAN_V1.json` SHA256 `51219d23275b0bd861bdea9530488932284b8cfb612bb9b5385c143e5f0bcceb`. `A1_1B_CPU_PLAN_RC=0`. Plan reports `A1_1B_CPU_PREFLIGHT_PASS`; four one-candidate families in order `DAPS, SITCOM, NP_LF, NP_S2`; native seeds `463500061,2178904464,4046706865,1462844243` respectively; all distinguishable uint32. Probe noisy measurement content SHA `992a33efa9bfb8e81d97ed61cf28369263ae1f11aeae19273c53973c2470acdb`; native DAPS/SITCOM/DiffFPR source HEAD and model SHA match frozen source. `gpu_run_authorized=false`, one procedural non-FFHQ probe, no protected confirmation access, common 8-bit saved RGB offline evaluation. `A1_1B_CPU_FINAL_STATUS=0`, wrapper RC=0.

## Runtime-source correction AFTER the user CPU PASS

Inspected imported frozen B24 helper `scripts/b24/run_b24_1_method_smoke.py`. Its `child_spec(...parent,...)` creates `parent/rep0` using `mkdir(parents=True,exist_ok=False)`? **Actual frozen line is `run_dir.mkdir(parents=True, exist_ok=False)`** (see line 195). That source **already creates intermediate parents**, so parent creation is not essential for this historical helper, but remains a harmless fail-closed A1-owned explicit directory construction. New helper `make_native_parent` tests exclusive A1-owned `daps/` and `sitcom/` paths, and both subprocess calls use those parent directories. No historic code changed. This correction was made **after** the original 13/13 tests and therefore requires a fresh, nonoverwriting inline CPU test before GPU launch.

The controller's physical GPU3 launch now also checks a dated **conditional** window (Oct 19, 2026, local 12:30–16:00 start), with one-hour maximum run, separate CPU test regression, and exact physical GPU UUID/memory gates. These enhancements were not covered by the user's previous CPU receipt. The pending new execution commit is pinned separately at handoff.

## Physical resource/data governance

The prior A1.1a procedural probe is verified; this is **not** FFHQ, thus no B24 protected confirmation panel touched and no FFHQ data reservation required for this engineering test. The reserved real FFHQ sample and confirmation305 roster must remain entirely inaccessible. Before GPU launch, post a `[RUN RESERVATION]` in shared issue #41 containing physical GPU3 UUID, dated conditional launch window, hard device memory cap 52,452 MiB, max per-candidate 1500 seconds and total 3600 seconds, native work estimate, env/model SHA, own output path, STOP rules, and Track B priority. The reservation is *conditional* upon the GPU being idle and uncontested and the user executing the disclosed bounded job. Previous `A1_1B_CPU_PLAN_RETURN_V1.zip` is immutable; do not rerun its named runner over that file.

**Decision:** Verify corrected A1-owned tests/plan inline; then bounded 1+1+1+1 sequential native GPU smoke only if the issue reservation and live GPU fit gates permit. Do not launch longer experiments automatically. Failure/STOP must preserve the one-shot A1.1b logs and outputs.
