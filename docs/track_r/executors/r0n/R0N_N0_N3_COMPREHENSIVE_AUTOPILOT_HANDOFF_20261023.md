# Track R R0-N — comprehensive PAC executor handoff (N0–N3)

Date: 2026-10-23. Source and decision provenance:
- Accepted R0 parent: `c8eb60877fadeb1b9c7be63b0ff389f7681e7107`.
- Frozen scientific planner decision: `6686322b919e891b5dd37d14ce5dddfa96a97dcd`.
- Two-parent, conflict-free merge commit:
  `1936e79acd4040f67e754276415bdac503b1121c`.
- Branch: `codex/prdiff-track-r-r0n-20261023`.

## Scientific scope and non-negotiable gates

R0 remains accepted; **Track R method paper STOP, no R1**.
R0-N is a small letter evidence package. No new reconstruction or
measurement, no edits to frozen R0.x outputs, and no access, listing,
hashing, or membership query concerning protected B24 confirmation305.

Authorized work includes all N0 and N1 analysis, optional N2 CLIP
orientation scoring limited to **≤0.25 aggregate GPU-hours** and
strictly bounded N3 ImageNet asset inventory. Any N1 scientific
replay discrepancy stops the pipeline before N2. A missing optional
N2/N3 environment or 0.25-hour forecast violation produces a
clear PARTIAL return, not a falsely passing K2 claim.

## Single unattended workflow

Launcher: `scripts/track_r/r0n_all_launch.sh <exact HEAD>`.
Controller: `scripts/track_r/r0n_all_autopilot.py`.

Before background launch the launcher:
1. Checks Git worktree clean, fetches and checks out exact requested HEAD.
2. Parses six Python source modules with Python AST.
3. Runs the entire Track R CPU test suite with a fixed environment.
4. Verifies original R0.8 ZIP SHA and all seven member SHA-256 digests.
5. Prints GPU inventory and selects GPU 1 or another idle GPU.

The controller:
1. **N0:** verifies and stages seven R0.8 aggregate tables and
   SHA256SUMS; commits and pushes on the new executor branch.
   The private R0.7 per-run GT JSONL and offline GT-path catalog
   are NEVER committed.
2. **N1:** reruns CPU tests at the actual N0 SHA, performs K3
   replay and 3,240 SHA-verified score receipt checks, emits the
   N1a–N1g tables, 60-configuration paired-loss ablation, bootstrap
   CIs, projected costs and three figure designs (PNG+PDF).
3. **N2 (optional):** on CPU, loads only frozen candidate paths
   to SHA-verify and convert already-saved RGB8 reconstructions
   to canonical PNG copies; checks lossless PNG roundtrips;
   emits a strict GT-free model-scoring manifest containing only
   those PNG paths and hashes. The GPU worker loads local SHA-pinned
   CLIP ViT-L/14 via the UnivFD-bundled OpenAI CLIP module,
   uses the exact six frozen prompts (no tuning), scores identity
   versus global-180 on 3,240 candidates, and commits to GPU time
   below 900 seconds including model loading. It performs a 64-root
   throughput canary and stops if conservative cost projection
   exceeds 900 seconds. A separate CPU-only evaluator joins
   the CLIP choices to already-frozen R0.7 PSNR labels, compares
   CLIP, primary prior-global, half-grid prior-global and UnivFD-global
   success/harm with 10,000 image-cluster CI draws and records
   partial-orbit error counts and per-candidate GPU time.
4. **N3:** checks only the two specified ImageNet model paths,
   data/imagenet* at depth 1, and relevant tracked FAST-DIPS
   config files. Reports presence and sizes; no downloads or
   image-tree enumeration.
5. **Return:** commits compact N1 numeric report/CSVs/figures,
   optional N2 evidence and N3 inventory, and
   `R0N_EXECUTOR_RETURN.md` to
   `docs/track_r/executors/r0n/returns/R0N_<UTCSTAMP>/`,
   plus local SHA256SUMS and a path-free PAC ZIP suitable for
   planner review. Pushes the return commit. **No R1** is triggered.

Return statuses:
- `R0N_ALL_STAGES_COMPLETE_PLANNER_READY` if N0–N3 all finish.
- `R0N_PARTIAL_OPTIONALS_PLANNER_READY` if required N0/N1
  succeed but N2 or N3 cannot safely complete.
- `STOP_R0N_N0_*` or `STOP_R0N_K3_OR_N1` on required failure.

Stages and PIDs are recorded in
`$BASE/r0n_all_<UTCSTAMP>/AUTOPILOT_STATUS.json`, with a per-stage
log and stdout controller log. Final committed report includes both
the planner and R0 parent SHAs, K3 counts, K1 B22 fixes, N1d
DEV80-selected / B22-tested cost configuration, N2 K2 inputs and
N3 inventory.

## Preflight confidence and limitations

The original uploaded `R0_8_PLANNER_RETURN.zip` has been inspected
locally: overall SHA
`13c008f3bca59b7d8d717ec3215fa1f783ca453b31b9001886c84633b513efab`,
and **all seven** aggregate member digests match the N0 importer's
constants. The seven predeclared P4 Good25 witnesses also match the
frozen R0.7 source analysis JSON exactly.

New N0–N3 programs and synthetic tests are committed, but they **have
not executed on PAC**. The launcher is therefore deliberately
fail-closed on syntax, tests, ZIP SHA, changed/dirty Git HEAD,
frozen dataset replay, unexpected CLIP model SHA or source paths,
GPU budget and external availability. None of these successes
should be claimed before the user returns PAC receipts.

R0-N letter go/no-go and K1/K2 scientific decisions remain the
planner's responsibility. A successful numerical run is not a
decision to restart method development.
