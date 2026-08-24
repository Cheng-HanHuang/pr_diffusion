# B23.0 checkpoint report

Status: **PASS_RECOMMEND_PLANNER_REVIEW**
GPU work performed: **NO**

The clean execution worktree was checked at `77f964c7cd6034246f6e2b60e599e0582c001c95` on
`codex/b23-execution`. It descends from the pinned operational handoff `d1119e37fa688ac07f48ffc87ce19b13dbfb1c27`.

The bounded PAC freeze records the historical checkout and its preserved dirty patch, DAPS and its
preserved B21 patch, official SITCOM, the NP/SITCOM fork, DiffFPR, all three named environments,
the FFHQ model/checkpoint, dataset-root existence, and hardware inventory. No data or output tree was
recursively scanned and Python probes ran with `CUDA_VISIBLE_DEVICES` empty.

The merged exposure manifest contains 328 images and
328 image/measurement rows: 100
truly resolved rows, 990 unresolved measurement-tag
mentions, and 228 image-wide unknown rows. Every
unresolved identity excludes all measurements for that image. The future B23 split registry remains empty.

The bounded untracked-path inventory classifies 497 paths across historical
DAPS, official SITCOM, and the NP/SITCOM fork. All 2 importable
files are hashed; any unresolved importable source is a hard stop.

Numeric atomic-operation weights are intentionally absent: B23.1 microbenchmarks must measure and
freeze them before any hybrid execution. B23.1 and B23.2 remain unauthorized.

Replay schema `b23.replay-report.v3` requires a pre-wrapper tolerance-freeze SHA-256 identity, the
exact frozen native-run set, one declared numerical floor per numeric comparison, mechanical
envelope-plus-floor acceptance, and a complete or explicitly justified-unavailable deterministic
audit. Compute schema `b23.compute-ledger.v2` rejects executed calibrated atomic/coupled work with
nonpositive GPU-active or wall time or a placeholder timer method. The B23.0 tests use synthetic
values only: no numerical tolerance, atomic weight, or calibration result is scientifically frozen.

The fail-closed wrapper records the repository tests, contract validator, dry renderer, and PAC
collector in `ZERO_GPU_STEP_RESULTS.tsv`. The evidence publisher independently requires all four
rows to be `PASS` with return code zero.

Failures: NONE
