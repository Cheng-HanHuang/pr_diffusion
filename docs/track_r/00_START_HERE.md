# Track R — start here

**Updated:** 2026-10-21 by the planner (Claude).

**Track R** = reliability per compute for diffusion-prior Fourier phase retrieval. The core idea is "repair, don't repeat":
- fix exact-symmetry failures in place with the diffusion prior's likelihood ratio;
- flag the remaining failures with the prior;
- judge the pipeline on tail reliability per unit of compute, against DAPS, SITCOM, NP, FAST-DIPS and FA-FPR.

**Relation to earlier tracks:**
- Track R supersedes the parked Track A (NP method paper).
- Track B is paused.

**Roles:**
- Planner: Claude.
- Executor: a GPT/Codex session chosen by Cheng (no PAC access).
- PAC operator: Cheng.

## Current stage

**R0 — failure anatomy, prior-ratio orbit repair, prior-consistent detection.** Offline on existing outputs, plus cheap FAST-DIPS runs.
**Status:** authorized for executor implementation. GPU sub-stages run only by Cheng, inside the gates.

## Documents (read in this order)

1. Planner roadmap — decisions, short- and long-term plan, publication image: [`docs/planning/track_r/2026-10-21_TRACK_R_PLANNER_ROADMAP.md`](../planning/track_r/2026-10-21_TRACK_R_PLANNER_ROADMAP.md)
2. Literature basis — what is and is not new: [`docs/planning/track_r/2026-10-21_LITERATURE_BASIS.md`](../planning/track_r/2026-10-21_LITERATURE_BASIS.md)
3. R0 spec — binding, FROZEN v1: [`docs/planning/track_r/R0_EXPERIMENT_SPEC.md`](../planning/track_r/R0_EXPERIMENT_SPEC.md)
4. Executor handoff: [`docs/track_r/executors/r0/00_EXECUTOR_HANDOFF.md`](executors/r0/00_EXECUTOR_HANDOFF.md)
5. Return template: [`docs/track_r/executors/r0/R0_RETURN_TEMPLATE.md`](executors/r0/R0_RETURN_TEMPLATE.md)
6. Configs:
   - `configs/track_r/r0_frozen_v1.json` — frozen science.
   - `configs/track_r/pac_paths_track_r_v1.json` — paths from records, unverified.
   - `configs/track_r/pac_paths_track_r_verified.json` — written by the executor after R0.0.

## Stage ledger

| Stage | Content | Status | Who |
|---|---|---|---|
| R0.0 / R0.0b | PAC inventory; create missing env and assets (`prdiff_track_r`, UnivFD, CLIP weights) | NEXT | executor writes, Cheng runs |
| R0.1 | Modules, scripts, CPU tests | NEXT | executor |
| R0.2 | Candidate manifest and G0 validity gates | pending | executor writes, Cheng runs (CPU) |
| R0.3 | GPU smoke and G1 canonical case; throughput and budget projection | pending | Cheng runs (1 GPU) |
| R0.4 | Prior and UnivFD scoring of existing candidates (S1–S4) | pending (pre-authorized if G0/G1/smoke pass) | Cheng runs (4 GPUs) |
| R0.5 | FAST-DIPS K=4 generation on B22 and DEV80 measurements | pending (pre-authorized as above) | Cheng runs |
| R0.6 | Scoring of the FAST outputs (S5) | pending (pre-authorized as above) | Cheng runs |
| R0.7 | Analysis and executor return | pending | executor |
| R0-D | R0 decision, with "Planner's view for Cheng" | pending | planner |
| R1 | FAST-based repair and detection pipeline, FA-FPR baseline | not authorized | — |
| R2 | Prospective frozen evaluation (new FFHQ validation images; ImageNet if warranted) | not authorized | — |
| T1 | Theory note on per-step hard selection (√h drift) | parked, desk only | planner |

## Branches

| Role | Branch |
|---|---|
| Planner | `claude/prdiff-track-r-planner-20261021` (this branch; contains copies of the final Track A/B, B26 and A1/A13 records) |
| Executor | `codex/prdiff-track-r-r0-<YYYYMMDD>`, created from the planner HEAD |
| Historical, read-only | Track A parked: `docs/planning/track_a/TRACK_A_PARKED_README_2026-10-21.md`. Track B TB0 STOP: `docs/track_b/executors/tb0-homometric-bridge/TB0_STOP_REPORT.md`. |

## PAC namespaces (verify in R0.0)

| Item | Path |
|---|---|
| Checkout | `/egr/research-pac/huang248/pr_diffusion_track_r` |
| Outputs | `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/` |
| Logs | `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/logs/` |
| Caches | `/egr/research-pac/huang248/outputs/.cache/track_r/` — never `/home` |
| New env | `/egr/research-pac/huang248/conda-envs/prdiff_track_r`, a clone of `prdiff_ffhq` (only if absent) |
| FAST env | `/egr/research-pac/huang248/conda-envs/fast_dips_a15_blackwell` |

## Governance (lighter than the 2026-10-15 metaplanner charter, same safety)

- No metaplanner charter or issue #41 reservations are required for Track R; Cheng may still log there.
- **Kept:**
  - no agent launches GPU jobs;
  - runs only from pushed, clean commits;
  - historical outputs are read-only and nothing is overwritten;
  - bounded inventory, no recursive scans of large trees;
  - safe shell style (no `set -e` / `exit` / broad `pkill`; `nohup` with PID/status/stop);
  - full compute accounting;
  - raw-oriented RGB8 PSNR is primary;
  - no GT in deployable steps;
  - **B24 confirmation305 locked**.
