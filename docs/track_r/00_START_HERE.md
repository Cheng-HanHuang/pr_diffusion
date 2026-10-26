# Track R — start here

**Updated:** 2026-10-25 by the planner (Claude).
- After PC-DAPS (arXiv 2610.31592): see the positioning note (document 3).
- After the N2 CLIP result: K2 triggered, so the paper is an audit (decision §10).
- **2026-10-25:** a top-venue attempt (CVPR 2027) with theory plus extended experiments. EUSIPCO 2027 and SPL are the fallbacks (document 1).

**Track R** is reliability per compute for diffusion-prior Fourier phase retrieval.

**Status:**
- R0 and R0-N are complete. The method-paper line is **stopped**: no R1, no R2.
- K2 removed diffusion-prior orientation repair as a contribution.
- **The deliverable is an audit paper with a selection theory.**
  - Attempt: CVPR 2027 (enrollment Nov 10, paper Nov 16, planner go/no-go Nov 8).
  - Fallbacks: SPL, or EUSIPCO 2027 (Feb 6).
- After that, the project winds down, except a possible separate TH2 theory study.

**Relation to earlier tracks:** Track R superseded the parked Track A (NP method paper). Track B is paused.

**Roles:**
- Planner: Claude.
- PAC-side executor: a GPT/Codex session (no PAC access itself).
- **Theory executor:** a separate chat or agent (no PAC).
- PAC operator: Cheng.

## Current stages

| Stage | What | Status |
|---|---|---|
| **R0-N.2** | Amendment 2 **revision D**, CPU: N1h → **A8 per-image counts** → ladder (vs O_orbit) → A7 export with SSIM/LPIPS → crosswalk | **AUTHORIZED, urgent; return by 2026-10-29** |
| **XP** | `XP_PREREG_v1.md`: XP-0/T inventory and timing; XP-A FAST K → 16; XP-C orientation tests on ImageNet and FFHQ ground truth; XP-D ImageNet audit; XP-B optional SITCOM K → 16; XP-E optional bridge | **FROZEN v1 — awaiting Cheng's "GO XP"** (CPU prep may start) |
| **TH1** | Selection theory for the paper; predictions frozen by **Oct 31**; note due Nov 6 | **AUTHORIZED** (theory executor) |
| **TH2** | Inner-functional theory (feasibility memo due Nov 13) | AUTHORIZED (memo only) |

## Documents (read in this order)

1. **Top-venue plan (2026-10-25):** [`docs/planning/track_r/2026-10-25_TOP_VENUE_PLAN.md`](../planning/track_r/2026-10-25_TOP_VENUE_PLAN.md)
2. **Experiments:** [`docs/planning/track_r/XP_PREREG_v1.md`](../planning/track_r/XP_PREREG_v1.md), with handoff [`executors/xp/00_XP_EXECUTOR_HANDOFF.md`](executors/xp/00_XP_EXECUTOR_HANDOFF.md)
3. **Theory:** [`docs/planning/track_r/THEORY_PROGRAM_v1.md`](../planning/track_r/THEORY_PROGRAM_v1.md), with handoff [`executors/theory/00_THEORY_EXECUTOR_HANDOFF.md`](executors/theory/00_THEORY_EXECUTOR_HANDOFF.md)
4. **R0 decision and rulings** (§§1–10; §10 is K2): [`docs/planning/track_r/2026-10-23_R0_PLANNER_DECISION.md`](../planning/track_r/2026-10-23_R0_PLANNER_DECISION.md)
5. **PC-DAPS positioning** (read its K2 update box first): [`docs/planning/track_r/2026-10-24_PC_DAPS_POSITIONING.md`](../planning/track_r/2026-10-24_PC_DAPS_POSITIONING.md)
6. **Fallback paper plan** (EUSIPCO/SPL condensed): [`docs/planning/track_r/LETTER_PLAN.md`](../planning/track_r/LETTER_PLAN.md). Extension options: [`2026-10-24_EXTENSION_OPTIONS.md`](../planning/track_r/2026-10-24_EXTENSION_OPTIONS.md)
7. **R0-N executor handoff and amendments 1–2:** [`executors/r0n/00_EXECUTOR_HANDOFF.md`](executors/r0n/00_EXECUTOR_HANDOFF.md)
8. **R0 record:**
   - executor return on `codex/prdiff-track-r-r0-20261021`;
   - R0 spec [`docs/planning/track_r/R0_EXPERIMENT_SPEC.md`](../planning/track_r/R0_EXPERIMENT_SPEC.md);
   - R0 handoff [`executors/r0/00_EXECUTOR_HANDOFF.md`](executors/r0/00_EXECUTOR_HANDOFF.md).
9. **Background:**
   - roadmap (historical) [`2026-10-21_TRACK_R_PLANNER_ROADMAP.md`](../planning/track_r/2026-10-21_TRACK_R_PLANNER_ROADMAP.md);
   - literature basis [`2026-10-21_LITERATURE_BASIS.md`](../planning/track_r/2026-10-21_LITERATURE_BASIS.md).

## Stage ledger

| Stage | Content | Status | Who |
|---|---|---|---|
| R0.0–R0.8 | Inventory → frozen analysis and audit (3.43 GPU-hours) | COMPLETE, accepted | executor and Cheng |
| R0-N | N0 tables, N1 analyses, N2 CLIP baseline, N3 inventory | COMPLETE, accepted (N1 `5af47db`; N2 `33a34bd`). K1 not triggered, **K2 TRIGGERED**, K3 PASS | executor and Cheng |
| R0-N.2 | Amendment 2 revision D (CPU) | NEXT, urgent (Oct 29) | executor and Cheng |
| XP | `XP_PREREG_v1` (GPU, ≤ 45 GPU-hours) | FROZEN, awaiting GO | executor and Cheng |
| TH1 / TH2 | Theory program v1 | AUTHORIZED | theory executor |
| Paper | CVPR 2027 attempt (go/no-go Nov 8) → SPL / EUSIPCO fallback | drafting from Oct 27 | planner drafts, Cheng and coauthors edit |
| N4; X5/X6 | Clean-image probe (withdrawn); PC-DAPS audit and operator robustness (deferred to a journal version) | — | — |
| R1 / R2 | Pipeline and prospective evaluation | CANCELLED | — |
| T1 | √h hard-selection theory | parked | — |

## Branches

| Role | Branch |
|---|---|
| Planner | `claude/prdiff-track-r-planner-20261021` |
| Executor, R0 (finished) | `codex/prdiff-track-r-r0-20261021` @ `c8eb608` |
| Executor, R0-N / R0-N.2 | `codex/prdiff-track-r-r0n-20261023` (N2 return @ `33a34bd`) |
| Executor, XP | `codex/prdiff-track-r-xp-<YYYYMMDD>`, from `33a34bd`, merged with the planner branch |
| Theory executor | `theory/prdiff-track-r-th-<YYYYMMDD>`, from the planner branch; writes only `docs/track_r/theory/` and `scripts/track_r/theory/` |
| Historical, read-only | Track A parked; Track B TB0 STOP |

## PAC namespaces (verified in R0.0)

| Item | Path |
|---|---|
| Checkout | `/egr/research-pac/huang248/pr_diffusion_track_r` |
| Outputs | `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/` |
| Logs | `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/logs/` |
| Caches | `/egr/research-pac/huang248/outputs/.cache/track_r/` — never `/home` |
| Env | `/egr/research-pac/huang248/conda-envs/prdiff_track_r` |
| FAST env | `/egr/research-pac/huang248/conda-envs/fast_dips_a15_blackwell` |

## Governance (unchanged)

- No agent launches GPU jobs; Cheng runs every PAC block.
- Runs only from pushed, clean commits.
- Historical outputs are read-only; nothing is overwritten.
- Bounded inventory only; no recursive scans of large trees.
- Safe shell style: no `set -e`, `exit` or broad `pkill`; long jobs use `nohup` with PID, status and stop.
- Full compute accounting.
- Raw-oriented RGB8 PSNR is primary.
- No GT in deployable steps.
- **B24 confirmation305 is locked.**
