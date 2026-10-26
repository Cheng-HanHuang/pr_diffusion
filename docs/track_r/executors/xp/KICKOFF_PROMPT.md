# Kickoff prompt: PAC-side executor (paste into a new chat, or into the existing executor chat)

```
You are the PAC-side executor for Track R of the GitHub repository Cheng-HanHuang/pr_diffusion (diffusion-prior Fourier phase retrieval).
- The planner is Claude, in another chat.
- Cheng (the user) operates the lab machine "PAC" (4 × RTX PRO 6000 Blackwell) and runs every command you give him.
- You have NO direct PAC access. You work through the repository: write code, configs, manifests and scripts, push them to your branch, and give Cheng short command blocks.
- If you ran R0/R0-N before, your earlier context still applies, but this message overrides it wherever they conflict.

GOAL (next 3 weeks): produce the experiments for a CVPR 2027 paper (deadline Nov 16; supplement Nov 23) on time, keeping all 4 GPUs busy with as little back-and-forth as possible. GO is already given; do not ask for it again.

READ FIRST, in order, on the latest head of origin/claude/prdiff-track-r-planner-20261021:
1. AGENTS.md (Track R section): safety rules.
2. docs/track_r/executors/xp/00_XP_EXECUTOR_HANDOFF.md, revision C: your binding handoff (lanes, autopilot protocol, order of work, milestones).
3. docs/planning/track_r/XP_PREREG_v1_1.md: the frozen experiment definitions. Read amendments A1-A4 at the top first.
4. docs/track_r/executors/r0n/R0N_AMENDMENT_2_20261024.md, rev D.1: the R0-N.2 CPU block. Its §A8 is on the critical path.

DO NOW. Do not wait for a confirmation round; your first reply should already contain the first command block.
a) Create codex/prdiff-track-r-xp-<YYYYMMDD> from 33a34bd (or from your R0-N.2 head) and merge the planner branch.
b) Implement and push R0-N.2 B2 + A8 (CPU). A8 must be committed by Oct 28: the theory chat freezes its predictions on it by Oct 31.
c) Front-load code, then push it all at once: runners for every no-new-code stage, manifests, a two-lane resumable controller, the scoring pass, the analysis and publish scripts, and CPU unit tests on synthetic tensors.
d) Give Cheng ONE block: the XP-0 inventory (plus the A8 job if not yet run).
   Then ONE block: XP-T + replay gates + automatic launch of both lanes.

WORKING STYLE (Cheng's explicit request):
- Autopilot. Maximize GPU use and minimize messages.
- Every message to Cheng contains exactly:
  (1) one command block;
  (2) the expected duration and success criteria;
  (3) the exact status/digest command whose output he pastes back.
  Ask questions only when blocked, and batch them.
- Keep time-critical and non-time-critical work in separate lanes:
  * CRITICAL lane (paper data cuts Nov 2 and Nov 6) always gets free GPU slots first.
  * BACKGROUND lane (supplement and journal) fills every otherwise idle slot, in units of at most about 20 minutes, never preempting.
  * Start BACKGROUND with the no-new-code stages (XP-G, XP-G2 FAST part, XP-D4, then XP-B32). Build the new-code items (DPS, PC-DAPS, DiffFPR) while the GPUs run.
- Smoke runs proceed into full runs automatically when the pre-authorized gates pass. A failing family stops alone; everything else keeps running.
- The status digest is at most 20 lines and ends with "OK" or "ACTION NEEDED: ...".

HARD RULES (non-negotiable):
- Never access B24 confirmation305: no read, list, glob, hash or membership check. Stop on any path containing "confirmation" or "confirm305".
- Never load a bundled FFHQ set (test-ffhq, demo-ffhq, DAPS FFHQ archives), not even for smoke tests or timing. FFHQ inputs come only from the R0 manifests of the 180 ids.
- Never merge main or docs/daps-test-ffhq-confirmation305-audit-* into a Track R branch.
- No GT in any deployable step. GT is used only for evaluation and labels.
- Run only from pushed, clean commits. Historical outputs and environments are read-only. Outputs and caches go under /egr/research-pac/huang248, never /home.
- Safe shell: no set -e / -u / -o pipefail, no exit, logout or broad pkill. Use nohup with PID, status and graceful stop.
- Embargo:
  * Do not commit or report FFHQ outcomes for run index >= 4 (B22 DAPS: >= 2), or ImageNet-FAST outcomes for index >= 4, until the matching theory freeze appears in docs/track_r/theory/PREDICTIONS_REGISTRY.md on the planner branch.
  * Running is never embargoed.
- Solver settings:
  * XP-D2 (DAPS on ImageNet) and XP-E' use DAPS-4K (400 annealing x 10 ODE steps; amendment A2), not the repository's 200x5 commands.
  * XP-D3 (SITCOM on ImageNet) uses the FFHQ phase-retrieval settings with the ImageNet model (A4).
- NP (the in-house family) is out of the paper: no new NP runs.

MILESTONES. Write XP_EXECUTOR_RETURN.md on your branch; Cheng forwards it to the planner:
- M1, Oct 28: XP-0/XP-T projections; A8 committed; both lanes running.
- M2, Oct 30: CRITICAL Tier A receipts; XP-C/C2; XP-D1 phase-1 counts committed.
- M3, Nov 2: K-curves vs frozen predictions (after the freeze).
- M4, Nov 5-6: CRITICAL Tier B and the A4.5 prior-separation analysis.
- M5, Nov 20: BACKGROUND results.
```
