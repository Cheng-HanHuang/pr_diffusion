# Track R · XP executor handoff: evaluation-only extensions (GPU), revision C (autopilot)

> **2026-10-27: read [`01_PLANNER_NOTES_20261027.md`](01_PLANNER_NOTES_20261027.md) and prereg amendment A5.** They give the Tier B correction, the CPU block (A4.4, A8b, A7, A5.5), the one-launch queue, the XP-I and XP-K engineering, and VRAM sizing.
> **2026-10-27, 12:30: then read [`02_PLANNER_NOTES_20261027_PM.md`](02_PLANNER_NOTES_20261027_PM.md) and prereg A6.** It covers the decisions on issue #42, the A4.4 sources, the GPU queue, `theory_inputs/`, and XP-D2 scoring.

**From:** planner (Claude), 2026-10-26 14:30.
- **Revision C replaces B.2.** It adds:
  - **two lanes** (CRITICAL / BACKGROUND);
  - an **autopilot protocol** that minimizes back-and-forth with Cheng;
  - prereg amendment A4: NP dropped from the paper, SITCOM settings resolved, XP-J (DiffFPR), and the prior-separation analysis.
- **Kickoff prompt** for a new or continuing executor chat: `docs/track_r/executors/xp/KICKOFF_PROMPT.md`.

**To:** the PAC-side executor (GPT/Codex). You have **no PAC access**. Cheng runs every PAC block you write.

**GO is given** (Cheng, 2026-10-26: *"GO aggressively … Try to get as much as possible out."*). Proceed through all lanes without asking again. Stop only on a §7 condition.

**Binding:**
- `docs/planning/track_r/XP_PREREG_v1_1.md`, FROZEN v1.1 with amendments A1–A4. Read the amendment boxes first.
- `docs/track_r/executors/r0n/R0N_AMENDMENT_2_20261024.md`, revision D.1 (R0-N.2, CPU).
- All R0 and R0-N data, PAC-safety and accounting rules (`AGENTS.md`, Track R section).

**Hard data rules:**
- Never load the DAPS, FAST-DIPS or PC-DAPS bundled FFHQ sets (`test-ffhq`, `demo-ffhq`, DAPS FFHQ archives): one of those images belongs to the locked holdout. FFHQ inputs come only from the R0 manifests of the 180 ids.
- Never access confirmation305.
- Never merge `main` or `docs/daps-test-ffhq-confirmation305-audit-*` into a Track R branch.

## 0. Planner's view

**Results.** The FFHQ audit is done. Residual selection plus an orientation test recovers 93–97% of the orbit oracle's gain for the three published solvers; what remains is mostly generation failure.

**What XP adds.** The paper argues that measurement-based selection fails in exactly three ways:
1. orientation (provably invisible to the residual);
2. consistent-but-wrong runs;
3. no good run at all.

XP measures each of these:
- K-scaling against frozen predictions (FAST to 64, SITCOM to 32, DAPS to 8);
- ImageNet, including the published DAPS-4K protocol;
- the denoiser's equivariance defect;
- whether a prior score can separate type-2 failures (A4.5).

**Publication image.** CVPR 2027: paper Nov 16, go/no-go Nov 8. The binding constraint is **time**, not GPUs.

## 1. Branch and environment

- **Branch.** `codex/prdiff-track-r-xp-<YYYYMMDD>`, from `codex/prdiff-track-r-r0n-20261023` @ `33a34bd` (or your R0-N.2 head), then merge `origin/claude/prdiff-track-r-planner-20261021`. **Re-merge the planner branch before each new launch** and report the planner commit you ran against.
- **Environments.** Use the historical ones read-only (DAPS env, `sitcom_ode_bw`, `fast_dips_a15_blackwell`, `prdiff_track_r`). For new needs (DPS, PC-DAPS, DiffFPR), clone a new environment; never modify a historical one.
- **Paths.**
  - outputs: `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/xp_<stage>_<UTC>/`;
  - caches: `/egr/research-pac/huang248/outputs/.cache/track_r/`;
  - new external code: `/egr/research-pac/huang248/external/<name>_xp/` (Cheng clones it).

## 2. The two lanes (prereg A4.1)

| Lane | Stages | Data cut |
|---|---|---|
| **CRITICAL** | XP-0, XP-T, replay gates → XP-A16, XP-C, XP-C2, XP-D1 (phase 1, then 2) | **Nov 2** |
| **CRITICAL** | XP-B16, XP-D2 (DAPS-4K), XP-D3, XP-A64, XP-F8 | **Nov 6** |
| **BACKGROUND** (fills idle slots) | **No new code** (start as soon as XP-T is done): XP-G, XP-G2 (FAST part), XP-D4, then XP-B32 once XP-B16's replay gate passes | Supplement if by Nov 20 |
| **BACKGROUND** | **New code** (build while the CRITICAL lane runs): XP-H (DPS), XP-I (PC-DAPS), XP-J (DiffFPR), XP-G2 (SITCOM part); XP-E′ only if A4.4 says so | Supplement if by Nov 20; otherwise the journal version |

**Scheduling rules:**
- A free GPU slot takes the highest-priority **ready CRITICAL** job; only if none is ready does it take **BACKGROUND** work.
- BACKGROUND jobs are units of at most about 20 minutes.
- Never kill a running job to free a slot.
- **Target:** all 4 GPUs busy from the end of XP-T to Nov 20, with W\* workers per GPU from XP-T.

## 3. Autopilot protocol (minimal back-and-forth)

1. **Front-load the code.** Before the first GPU launch, push in one go:
   - the runners for every no-new-code stage;
   - the manifests;
   - the two-lane controller;
   - the scoring pass;
   - the analysis and publish scripts;
   - CPU unit tests on synthetic tensors.

   Build the new-code BACKGROUND items afterwards, while the GPUs run.
2. **One block per launch.** Each message to Cheng contains:
   - (a) one PAC block (`git fetch`, checkout of the pushed SHA, clean-tree check, then the command);
   - (b) its expected duration and what success looks like;
   - (c) the exact command whose output Cheng pastes back.

   No questions unless you are blocked; batch any questions you have.
3. **Pre-authorized auto-continue.** Inside a launch, a stage's smoke test passes into its full run **without a stop** if all of these hold:
   - runtime within ±50% of the XP-T projection;
   - the frozen settings replay;
   - the replay gate passes (historical families);
   - the measurement SHA matches.

   A failing family stops alone; everything else continues.
4. **Controller.**
   - Resumable: skip jobs whose outputs and receipts verify.
   - Every job carries a time limit of 3× its projection; on breach it is marked failed and retried once with the **same** seed.
   - New manifests (more stages) enter through an inbox directory, with no restart.
   - New **code** means a graceful restart from the new pushed commit: running jobs finish first, then the controller resumes from receipts.
   - `nohup` with a PID file, `status`, and a graceful `stop`.
5. **Status digest.** `status` prints **at most 20 lines**:
   - per lane: done / running / failed / queued;
   - GPU-h used against the 400 cap, and the projection;
   - ETA per data cut;
   - any STOP flags;
   - a final line, either `ACTION NEEDED: …` or `OK`.

   Cheng pastes it when it says ACTION, at milestones, or about twice a day.
6. **Publishing.** `publish_return.sh <stage>` (run by Cheng) copies only committable aggregates and per-image strings (prereg §7) into `docs/track_r/executors/xp/returns/XP_<UTC>/`, writes SHA256SUMS, **enforces the embargo** (§5 below), and commits and pushes from the PAC checkout on your branch. You then read the pushed return from GitHub and write the report.
7. **Analysis runs automatically** on CPU when a stage completes: U-statistics, bootstrap, anatomy, the A4.5 prior-separation analysis. Exploratory items come last.

## 4. Order of work

1. **Now:**
   - R0-N.2 **B2 + A8** (CPU), committed **by Oct 28**. It gates the theory freeze on Oct 31.
   - In parallel, write the XP-0 block, the runners and the controller (§3.1).
2. **XP-0 block** (CPU):
   - inventory;
   - y-SHA equality check;
   - pad formula (oversample 4.0 → 512);
   - `test -d` checks for `test-imagenet`, DPS, PC-DAPS and DiffFPR. For anything absent, write the exact command for Cheng; if the DAPS download bundles FFHQ, Cheng keeps only `test-imagenet`.
   - **Same block:** the R0-N.2 A8 job, if not already run.
3. **XP-T plus first launch** (one block). XP-T, then the replay gates; then the controller starts the **CRITICAL** Tier A queue and the **BACKGROUND** no-new-code queue automatically, with the auto-continue rules.
4. **R0-N.2 remainder** (CPU, while the GPUs run): A4.3 → A7 → A4.1–A4.2 → A3.
5. **CRITICAL Tier B:** enqueue after XP-T projections (prereg §3.6). XP-D2 uses **DAPS-4K** (A2), and XP-D3 uses the FFHQ SITCOM settings with the ImageNet model (A4.3).
6. **BACKGROUND new-code items** as their smoke tests pass.

## 5. Embargo and freezes (prereg §5)

- **FFHQ:** no outcome data for runs with index ≥ 4 in XP-A/B/F (index ≥ 2 for B22 DAPS) until v1 is listed in `docs/track_r/theory/PREDICTIONS_REGISTRY.md` (due Oct 31).
- **ImageNet FAST:** no outcome data for runs with index ≥ 4 until v2 is listed, or until 48 h after the phase-1 counts if no v2 is made.
- **Commit the XP-D1 phase-1 counts promptly.** The theory executor needs them.
- **Runs are never embargoed; only outcomes are.** Run everything as soon as possible.
- `publish_return.sh` refuses embargoed files.

## 6. PAC command style (unchanged)

- One consolidated block per launch.
- `nohup` with PID, status and graceful stop.
- Bounded logs; output saved to file, with only a short printed summary.
- No `set -e`, `set -u`, `set -o pipefail`, `exit`, `logout` or broad `pkill`.
- Explicit return-code checks.
- Run from a pushed, clean commit, and print the SHA at start.
- Print `nvidia-smi` first, and refuse a GPU with foreign processes above 2 GiB unless Cheng passes an override.

## 7. Stop conditions (affected family or stage only)

- smoke runtime > ±50% off the projection;
- a frozen-setting mismatch;
- a replay-gate failure;
- a measurement-SHA mismatch;
- a projected cumulative total > 400 GPU-h;
- a path with a forbidden token ("confirmation", "confirm305");
- any code path that would load a bundled FFHQ set.

**Drop order if late or over budget:** prereg §3.5 (CRITICAL Tier A is never dropped).

## 8. Milestone returns (`XP_EXECUTOR_RETURN.md`)

| Milestone | Content | Target |
|---|---|---|
| M1 | XP-0 + XP-T projections; A8 committed; both lanes running | Oct 28 |
| M2 | CRITICAL Tier A receipts; XP-C/C2; XP-D1 phase-1 counts | Oct 30 |
| M3 | After the v1 (v2) freeze: FAST FFHQ (and ImageNet) K-curves vs predictions | Nov 2 |
| M4 | CRITICAL Tier B, with the SITCOM/DAPS prediction checks and the A4.5 analysis | Nov 5–6 |
| M5 | BACKGROUND results | Nov 20 |

**Each return contains:**
1. Status for Cheng (≤ 8 lines).
2. Identity: branch, SHAs, planner commit, environments, GPU.
3. GPU-h per stage and lane, cumulative against 400.
4. Tables with n and CIs, figures, anomalies, failed-job counts.
5. Theory check: registry entries, SHA/time verification, consistent/rejected per prediction (including shape).
6. Artifacts: committed aggregates and strings, SHA256SUMS; the planner ZIP (SHA only).
7. Executor opinion (optional, separate).
