# Track R — start here

**Updated:** 2026-10-24 by the planner (Claude).
- After concurrent work PC-DAPS (arXiv 2610.31592): see the positioning note (document 2 below).
- After the N2 CLIP result: K2 triggered, so the paper is now an audit (decision §10, LETTER_PLAN v1.3).

**Track R** is reliability per compute for diffusion-prior Fourier phase retrieval.

**Status:**
- R0 and R0-N are complete.
- The method-paper line is **stopped**: no R1, no R2.
- K2 removed diffusion-prior orientation repair as a contribution.
- The remaining deliverable is one short **audit paper**, recommended for EUSIPCO 2027.
- After that, the project winds down.

**Relation to earlier tracks:** Track R superseded the parked Track A (NP method paper). Track B is paused.

**Roles:**
- Planner: Claude.
- Executor: a GPT/Codex session chosen by Cheng (no PAC access).
- PAC operator: Cheng.

## Current stage

**R0-N.2 — last CPU block for the audit paper** (amendment 2, revision B):
- N1h (membership lookup);
- the protocol ladder with mean-of-K and E1_CLIP;
- the PC-DAPS-label crosswalk;
- stored decisions only, no new inference.

**Status:** authorized (decision §10, 2026-10-24). Return by 2026-11-07.

## Documents (read in this order)

1. **R0 decision** — results, gate rulings, the plan from here, publication image: [`docs/planning/track_r/2026-10-23_R0_PLANNER_DECISION.md`](../planning/track_r/2026-10-23_R0_PLANNER_DECISION.md)
2. **PC-DAPS positioning (2026-10-24)** — full read, crosswalk, the gap; read its K2 update box first: [`docs/planning/track_r/2026-10-24_PC_DAPS_POSITIONING.md`](../planning/track_r/2026-10-24_PC_DAPS_POSITIONING.md)
3. **Paper plan (v1.3, audit):** [`docs/planning/track_r/LETTER_PLAN.md`](../planning/track_r/LETTER_PLAN.md)
4. **R0-N executor handoff:** [`docs/track_r/executors/r0n/00_EXECUTOR_HANDOFF.md`](executors/r0n/00_EXECUTOR_HANDOFF.md), with config `configs/track_r/r0n_letter_analysis_v1.json` (FROZEN v1) and amendments 1–2
5. **R0 record:**
   - executor return on `codex/prdiff-track-r-r0-20261021`: `docs/track_r/executors/r0/R0_FROZEN_COMPLETE_SCIENTIFIC_RETURN_TO_PLANNER_20261023.md`;
   - R0 spec [`docs/planning/track_r/R0_EXPERIMENT_SPEC.md`](../planning/track_r/R0_EXPERIMENT_SPEC.md);
   - R0 handoff [`docs/track_r/executors/r0/00_EXECUTOR_HANDOFF.md`](executors/r0/00_EXECUTOR_HANDOFF.md).
6. **Background:**
   - roadmap (historical as of 2026-10-21) [`docs/planning/track_r/2026-10-21_TRACK_R_PLANNER_ROADMAP.md`](../planning/track_r/2026-10-21_TRACK_R_PLANNER_ROADMAP.md);
   - literature basis [`docs/planning/track_r/2026-10-21_LITERATURE_BASIS.md`](../planning/track_r/2026-10-21_LITERATURE_BASIS.md).

## Stage ledger

| Stage | Content | Status | Who |
|---|---|---|---|
| R0.0–R0.8 | Inventory, environment, G0/G1, scoring of 2,520 historical candidates, 720 new FAST roots and their scoring, frozen analysis, independent audit (3.43 GPU-hours) | COMPLETE, accepted | executor and Cheng |
| R0-D | R0 decision | COMPLETE (2026-10-23) | planner |
| R0-N | Letter evidence package: N0 tables, N1 analyses, N2 CLIP baseline, N3 ImageNet inventory | COMPLETE, accepted 2026-10-24 (N1 `5af47db`; N2 recovery `33a34bd`). K1 not triggered, **K2 TRIGGERED**, K3 PASS | executor writes, Cheng runs |
| R0-N.2 | One CPU block (amendment 2, revision B): N1h; protocol ladder with mean-of-K and E1_CLIP (required); PC-DAPS-label crosswalk (recommended); A3 identities (optional) | NEXT (return by 2026-11-07) | executor writes, Cheng runs |
| N4 | Clean-image orientation probe | WITHDRAWN (K2) | — |
| Paper | GT-free audit of diffusion PR (LETTER_PLAN v1.3). Recommended venue EUSIPCO 2027 (deadline 2027-02-06, to verify); draft submission-ready by 2026-12-11 | drafting | planner drafts, Cheng and coauthors edit |
| R1 / R2 | Pipeline and prospective evaluation | CANCELLED | — |
| T1 | Theory note on per-step hard selection (√h drift) | parked | planner (desk only) |

## Branches

| Role | Branch |
|---|---|
| Planner | `claude/prdiff-track-r-planner-20261021` |
| Executor, R0 (finished) | `codex/prdiff-track-r-r0-20261021` @ `c8eb608` |
| Executor, R0-N | `codex/prdiff-track-r-r0n-<YYYYMMDD>`, created from `c8eb608` and merged with the planner branch |
| Historical, read-only | Track A parked: `docs/planning/track_a/TRACK_A_PARKED_README_2026-10-21.md`. Track B TB0 STOP: `docs/track_b/executors/tb0-homometric-bridge/TB0_STOP_REPORT.md`. |

## PAC namespaces (verified in R0.0; see `configs/track_r/pac_paths_track_r_verified.json` on the executor branch)

| Item | Path |
|---|---|
| Checkout | `/egr/research-pac/huang248/pr_diffusion_track_r` |
| Outputs | `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/` |
| Logs | `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/logs/` |
| Caches | `/egr/research-pac/huang248/outputs/.cache/track_r/` — never `/home` |
| Env | `/egr/research-pac/huang248/conda-envs/prdiff_track_r` |
| FAST env | `/egr/research-pac/huang248/conda-envs/fast_dips_a15_blackwell` |

## Governance (unchanged)

- No agent launches GPU jobs.
- Runs only from pushed, clean commits.
- Historical outputs are read-only; nothing is overwritten.
- Bounded inventory only; no recursive scans of large trees.
- Safe shell style: no `set -e`, `exit` or broad `pkill`; long jobs use `nohup` with PID, status and stop.
- Full compute accounting.
- Raw-oriented RGB8 PSNR is primary.
- No GT in deployable steps.
- **B24 confirmation305 is locked.**
