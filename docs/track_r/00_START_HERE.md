# Track R — start here

**Updated:** 2026-10-27 by the planner (Claude).
- After PC-DAPS (arXiv 2610.31592): see the positioning note (document 5).
- After the N2 CLIP result: K2 triggered, so the paper is an audit (decision §10).
- **2026-10-25:** a top-venue attempt (CVPR 2027) with theory plus extended experiments. EUSIPCO 2027 and SPL are the fallbacks (document 1).
- **2026-10-26:** Cheng's decisions, recorded in the update box of document 1:
  - **GO XP, aggressively.** `XP_PREREG_v1_1.md` supersedes v1: tiers A/B/C, 400 GPU-h cap;
  - CVPR;
  - the theory executor starts now (handoff revision B; program v1.1; literature digest);
  - the A8 label strings may be committed (amendment 2 revision D.1).
- **2026-10-26, later:**
  - **XP-E is cancelled.** DAPS `test-ffhq` overlaps the locked holdout by one id.
  - **Hard rule:** never load any bundled FFHQ set (`test-ffhq`, `demo-ffhq`, DAPS FFHQ archives). Never merge `main` or the data-governance audit branch into Track R branches.
  - TGD has been read in full, and the research pass on phase-retrieval theory is digested. This produced prereg amendment A1, theory program v1.2 and handoffs B.1.

- **2026-10-27, 17:15 (planner):**
  - **Executor:** A8b and the D1-P1 strings are published (v2 due Oct 29, 12:01 ET). D1-P2 and D3 are generated. E′ stopped at 45/720 because labmates' jobs took the GPUs; resume as audited.
  - **Theory:** mech v1.1 registered. The planner ran the A6 forecaster: the validator fails without a control-variate fix and passes with it.
  - **Prereg A7:** co-tenancy for generation stages, the B16 lineage gate, F8 seeds.
  - **Notes, round 3:** [`executors/theory/03_PLANNER_GUIDANCE_20261027_EVE.md`](executors/theory/03_PLANNER_GUIDANCE_20261027_EVE.md) and [`executors/xp/03_PLANNER_NOTES_20261027_EVE.md`](executors/xp/03_PLANNER_NOTES_20261027_EVE.md).
- **2026-10-27, 12:30 (planner):**
  - **Overnight generation certified:** XP-A16, XP-A64, XP-D1 phase 1, XP-D2 (about 39 GPU-h in total).
  - **TH2 mechanism forecasts v1 registered.**
  - **Prereg A6:**
    - A4.4 = 400 × 5, to be confirmed by replay;
    - XP-E′ eligible;
    - evaluation panels;
    - predictive-interval rule;
    - the `docs/track_r/executors/xp/theory_inputs/` directory.
  - **Notes, round 2:** [`executors/theory/02_PLANNER_GUIDANCE_20261027_PM.md`](executors/theory/02_PLANNER_GUIDANCE_20261027_PM.md) and [`executors/xp/02_PLANNER_NOTES_20261027_PM.md`](executors/xp/02_PLANNER_NOTES_20261027_PM.md).
- **2026-10-27, 00:00 (planner):**
  - **NP dropped** from the paper (Cheng, 2026-10-26 23:01). Paper counts are now on DAPS, SITCOM and FAST-DIPS only (1,960 runs; 540 image–solver pairs).
  - **Prereg amendment A5:**
    - XP-I (PC-DAPS) becomes CRITICAL-if-ready, and XP-I2 adds ImageNet;
    - new XP-K: a parameter-only calibration of each solver's data term to σ;
    - new analysis A5.5 (the noise floor) and the A8b residual strings;
    - A7 is now required;
    - the XP-F8 replay gate;
    - v1 freezes both the S2 reference and a rank-calibrated forecast.
  - **New facts from A8 and N1c:**
    - S2 is rejected: within an image, GOOD beats CONSISTENT in 200/205 pairs;
    - DAPS and SITCOM fit the noise: every GOOD run has ρ < σ.
  - **Notes:** [`executors/theory/01_PLANNER_GUIDANCE_20261027.md`](executors/theory/01_PLANNER_GUIDANCE_20261027.md) and [`executors/xp/01_PLANNER_NOTES_20261027.md`](executors/xp/01_PLANNER_NOTES_20261027.md).
  - **Freeze:** v1 target Oct 29, 18:00 ET; hard deadline Oct 31.

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
| **R0-N.2** | Amendment 2 **revision D.1**, CPU: N1h → **A8 per-image strings (by Oct 28)** → ladder (vs O_orbit) → A7 export with SSIM/LPIPS → crosswalk | **AUTHORIZED, urgent; return by 2026-10-29** |
| **XP** | `XP_PREREG_v1_1.md` (+ amendments A1–A7; A2 = DAPS-4K for the published-protocol replication; A5 = XP-I promoted, XP-K, A5.5, A8b, F8 gate; A6 = A4.4 ruling, XP-E′, panels, interval rule, theory inputs; A7 = co-tenancy, B16 lineage gate, F8 seeds, v2 deadline):<br>• **Tier A (Nov 2):** XP-0/T; FAST FFHQ n=16; XP-C; XP-C2 equivariance defect; FAST ImageNet n=16<br>• **Tier B (Nov 6):** FAST n=64; SITCOM n=16 on 180 ids; DAPS ImageNet n=4 on 100; SITCOM ImageNet; DAPS FFHQ n=8<br>• **Tier C (Nov 20):** SITCOM n=32, DPS, PC-DAPS, noise robustness, 512-canvas control, FAST ImageNet n=64, conditional XP-E′<br>• XP-E cancelled | **GO given (2026-10-26); FROZEN v1.1 + A1–A7; two lanes (CRITICAL / BACKGROUND); cap 400 GPU-h.** A16, A64, D1-P1, D2 generated (about 39 GPU-h). Planner notes 2026-10-27 (two rounds) |
| **TH1** | Selection theory for the paper; predictions v1 (S2 reference + rank-calibrated race) — target **Oct 29 18:00 ET**, hard **Oct 31** (embargo on K > 4 outcomes until registered); note due Nov 6 | **IN PROGRESS** (interim `12e0ff3` accepted; guidance 2026-10-27) |
| **TH2** | Noise-floor theory (over-weighted data terms; within-image residual races; orientation leak); optional mechanism forecasts for XP-K/XP-G by Oct 31 | AUTHORIZED, refocused 2026-10-27 |

## Documents (read in this order)

1. **Top-venue plan** (2026-10-25; read the 2026-10-26 update box first): [`docs/planning/track_r/2026-10-25_TOP_VENUE_PLAN.md`](../planning/track_r/2026-10-25_TOP_VENUE_PLAN.md)
2. **Experiments:** [`docs/planning/track_r/XP_PREREG_v1_1.md`](../planning/track_r/XP_PREREG_v1_1.md) (v1 is superseded; amendments A1–A5: lanes, NP dropped, SITCOM settings, DiffFPR; A5 = PC-DAPS promoted, XP-K, noise floor), with handoff revision C (autopilot) [`executors/xp/00_XP_EXECUTOR_HANDOFF.md`](executors/xp/00_XP_EXECUTOR_HANDOFF.md) and the planner notes [`executors/xp/01_PLANNER_NOTES_20261027.md`](executors/xp/01_PLANNER_NOTES_20261027.md)
3. **Theory:**
   - program v1.2 [`docs/planning/track_r/THEORY_PROGRAM_v1.md`](../planning/track_r/THEORY_PROGRAM_v1.md);
   - literature digest [`2026-10-26_THEORY_LIT_DIGEST.md`](../planning/track_r/2026-10-26_THEORY_LIT_DIGEST.md);
   - kickoff prompts: [`executors/xp/KICKOFF_PROMPT.md`](executors/xp/KICKOFF_PROMPT.md), [`executors/theory/KICKOFF_PROMPT.md`](executors/theory/KICKOFF_PROMPT.md);
   - handoff revision B.1 [`executors/theory/00_THEORY_EXECUTOR_HANDOFF.md`](executors/theory/00_THEORY_EXECUTOR_HANDOFF.md) and the planner guidance [`executors/theory/01_PLANNER_GUIDANCE_20261027.md`](executors/theory/01_PLANNER_GUIDANCE_20261027.md);
   - predictions registry [`theory/PREDICTIONS_REGISTRY.md`](theory/PREDICTIONS_REGISTRY.md).
4. **R0 decision and rulings** (§§1–10; §10 is K2): [`docs/planning/track_r/2026-10-23_R0_PLANNER_DECISION.md`](../planning/track_r/2026-10-23_R0_PLANNER_DECISION.md)
5. **PC-DAPS positioning** (read its K2 update box first): [`docs/planning/track_r/2026-10-24_PC_DAPS_POSITIONING.md`](../planning/track_r/2026-10-24_PC_DAPS_POSITIONING.md)
6. **Fallback paper plan** (EUSIPCO/SPL condensed): [`docs/planning/track_r/LETTER_PLAN.md`](../planning/track_r/LETTER_PLAN.md). Extension options: [`2026-10-24_EXTENSION_OPTIONS.md`](../planning/track_r/2026-10-24_EXTENSION_OPTIONS.md)
7. **R0-N executor handoff and amendments 1–2:** [`executors/r0n/00_EXECUTOR_HANDOFF.md`](executors/r0n/00_EXECUTOR_HANDOFF.md)
8. **R0 record:**
   - executor return on `codex/prdiff-track-r-r0-20261021`;
   - R0 spec [`docs/planning/track_r/R0_EXPERIMENT_SPEC.md`](../planning/track_r/R0_EXPERIMENT_SPEC.md);
   - R0 handoff [`executors/r0/00_EXECUTOR_HANDOFF.md`](executors/r0/00_EXECUTOR_HANDOFF.md).
9. **Paper draft:** [`docs/paper/track_r_cvpr27/`](../paper/track_r_cvpr27/README.md). Read `CLAIMS_LEDGER.md` before editing numbers.
10. **Background:**
   - roadmap (historical) [`2026-10-21_TRACK_R_PLANNER_ROADMAP.md`](../planning/track_r/2026-10-21_TRACK_R_PLANNER_ROADMAP.md);
   - literature basis [`2026-10-21_LITERATURE_BASIS.md`](../planning/track_r/2026-10-21_LITERATURE_BASIS.md).

## Stage ledger

| Stage | Content | Status | Who |
|---|---|---|---|
| R0.0–R0.8 | Inventory → frozen analysis and audit (3.43 GPU-hours) | COMPLETE, accepted | executor and Cheng |
| R0-N | N0 tables, N1 analyses, N2 CLIP baseline, N3 inventory | COMPLETE, accepted (N1 `5af47db`; N2 `33a34bd`). K1 not triggered, **K2 TRIGGERED**, K3 PASS | executor and Cheng |
| R0-N.2 | Amendment 2 revision D.1 (CPU) | NEXT, urgent (A8 Oct 28; return Oct 29) | executor and Cheng |
| XP | `XP_PREREG_v1_1` (GPU, ≤ 400 GPU-hours, tiers A/B/C) | **GO given 2026-10-26** | executor and Cheng |
| TH1 / TH2 | Theory program v1.2 | AUTHORIZED, starting | theory executor |
| Paper | CVPR 2027 attempt (go/no-go Nov 8) → SPL / EUSIPCO fallback. Draft in `docs/paper/track_r_cvpr27/` (claims ledger, figures plan) | revised Oct 27: NP-free counts, E1_CLIP table, S2 rejection, noise floor, theory corrections (builds; ≈7.5 pp. of text with placeholders) | planner drafts, Cheng and coauthors edit |
| N4 | Clean-image probe (withdrawn) | — | — |
| X5/X6 | PC-DAPS audit and noise robustness: now XP-I and XP-G (Tier C, supplement) | in XP v1.1 | executor and Cheng |
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
- **B24 confirmation305 is locked.** Because DAPS `test-ffhq` contains one of its images, **no bundled FFHQ set is ever loaded** in Track R.
