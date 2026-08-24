# B23.0 checkpoint report

Status: **REVISE_BEFORE_SIGN_OFF — AUTHORIZED ZERO-GPU CONTRACT CLOSEOUT PENDING**
GPU work performed: **NO**

The clean execution worktree was checked at `4cf026d834650b90a1bfc6b8951a69500d190b21` on
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

The planner accepted the prior PAC evidence and required one further contract-only closeout. Replay
schema `b23.replay-report.v3` now requires a pre-wrapper tolerance-freeze identity, exact frozen
native-run set, per-metric numerical floors, mechanical envelope-plus-floor comparisons, and a
complete or explicitly justified-unavailable deterministic-flag audit. Compute schema
`b23.compute-ledger.v2` now rejects any executed calibrated atomic/coupled work unless GPU-active
time and wall time are strictly positive and the timer method is non-placeholder. B23.0 assigns no
scientific tolerance and performs no calibration; the new rejection tests are synthetic and
zero-GPU.

The fail-closed wrapper records the repository tests, contract validator, dry renderer, and PAC
collector in `ZERO_GPU_STEP_RESULTS.tsv`. The evidence publisher independently requires all four
rows to be `PASS` with return code zero.

Pending action: push the clean corrective pre-run commit, rerun the complete fail-closed B23.0
zero-GPU PAC suite with CUDA hidden, and publish a later evidence commit/capsule. No B23.1 action is
authorized.
