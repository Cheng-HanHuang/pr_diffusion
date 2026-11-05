# A24 / XP-K — CPU-first continuation under co-tenancy (2026-11-05)

**Planner:** round 20 @ `aafaf7f35e9d9d0beafa459259ca10b7fda4e8f5`, notes `docs/track_r/executors/xp/20_PLANNER_NOTES_20261104_PM.md`. **Governing:** A24, A23.5b XP-K gates. **Main-paper cut closed**.

## GPU policy — do not confuse a preference with a hard gate

Round-20 exact instruction: **“co-tenant-safe work may run; clean-GPU work waits. For the XP-K identity replays, a quiet GPU is preferable, but the tolerance must not be weakened.”**

- Pure CPU checks can run immediately without waiting for idle GPUs.
- The XP-K historical two-B22-root identity replays **are GPU jobs**, but are **not strictly clean-GPU-only**. Use an isolated/pinned native checkout and original-parameter source, live VRAM/throughput admission, no signals to foreign jobs, exact output comparison. On a failure **stop and ask**; do not relax replay tolerances or silently move to fresh-only runs.
- XP-K prospective **calibration generation** may be co-tenant-safe if and only if native parameter diff, historical replay, required F8 predecessor (for DAPS), source/seed/budget/telemetry gates all pass. No production is authorized by CPU snapshot checks.
- Do not run time-sensitive clean-GPU benchmark jobs under co-tenancy; those must wait.

## Existing trustworthy prerequisite

`docs/track_r/executors/xp/returns/XP_F8_LEGACY_DAPS_8ROOT_REPLAY_EXACT_PASS_20261029.md` documents the previous original historical DAPS lineage: four B22 plus four DEV80 roots, **8/8 PNG pixel-exact** on corrected native original settings, 400 annealing × 5 ODE, original native tau 0.01. This is evidence toward the *prior F8 replay* requirement. It does **not** prove the XP-K newly adapted DAPS calibration and does **not** replace XP-K's own two-B22-root identity replay.

## Gate and resource inventory

| Family | Target single parameter | CPU config diff | Native binding proof | Original-parameter XP-K B22 identity replay | XP-K generation |
| --- | --- | --- | --- | --- | --- |
| SITCOM-ODE | `lgvd_tau: 0.01 → 0.05`; Euler step unchanged | A39 prospective one-leaf screen | **Pending** native sampler binding source diff | **Pending**, two roots | **Not authorized** |
| FAST-DIPS | `epsilon: 0.05 → 33.26` | A39 prospective one-leaf screen | **Pending** native sampler binding source diff | **Pending**, two roots | **Not authorized** |
| Historical FFHQ DAPS | Langevin `tau: 0.01 → sqrt(2)*0.05` | **Pending** exact original Hydra key/source proof | **Pending** native sampler binding source diff | **Pending**, two roots; F8 prior receipt documented | **Not authorized** |

A39 files: `scripts/track_r/xp_a39_a24_xpk_cpu_preflight.py` and `scripts/track_r/xp_a39_a24_xpk_cpu_launch.sh`; CPU-only unit tests in `tests/test_track_r_xp_a39_a24_xpk_cpu.py`. The CPU run produces immutable historical/proposed JSON config snapshots and a non-authorizing provenance/coverage receipt under a new `outputs/.../xp_a39_xpk_cpu_*` output root. It reads native Git HEADs and existing approved source files only. **It intentionally marks the native binding/source-diff gate as pending even if the two JSON single-leaf screens pass.**

The next science-engineering step is to trace the actual SITCOM LGVD `tau`, FAST-DIPS projection `epsilon`, and DAPS historical task Hydra `tau` through their native sampler, freeze a source diff/trace for each, and only then prepare the original-parameter two-B22-root replay adapters (prefer quiet GPU but co-tenancy permitted with unchanged strict tolerances). Observe the registered family ordering, the separate 30 GPU-hour XP-K cap, and the global 400 GPU-hour project cap.

## CP4 inventory, deadline Nov 15 12:00 ET

- G7 **planner-accepted**, complete; use consolidated G7 return `4d1bd609f70428c767a95a51b37f8c3a7a5d0c71`. Do not repeat G7.
- XP-K against registered mechanism v1/v1.1 forecasts remains open.
- Final **project-wide** GPU ledger must be reconciled from all charged campaigns, not merely the accepted G7 checkpoint (last G7 project ledger `247.80191737622505 /400`).
- Record explicit not-run list, including optional A21.10 and any XP-K arms that fail/are withheld.
- No protocol v2, new selector, matched-compute claim or new main-text addition is authorized.

**This is a CPU-source handoff, not an XP-K scientific acceptance or GPU launch order.**
