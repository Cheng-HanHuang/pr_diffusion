# Track R · XP executor handoff: evaluation-only extensions (GPU), revision B.1

**From:** planner (Claude), 2026-10-26.
- Revision B.1 follows prereg **amendment A1**:
  - XP-E is cancelled;
  - a bundled-FFHQ data ban;
  - new Tier C items XP-G2 (512 canvas) and the conditional XP-E′;
  - exploratory CPU analyses.
- Revision A (2026-10-25) was never started.

**To:** the PAC-side executor (the one that ran R0 and R0-N). Cheng runs every PAC block.

**GO is given.** Cheng, 2026-10-26 00:57: *"GO aggressively … even if some runs failed it's no hurts. Try to get as much as possible out."*
- You may proceed through Tiers A → B → C without asking again.
- **You still stop** on any §6 condition. Any external download or clone is done by Cheng.

**Hard data rule (A1).** Never load the DAPS, FAST-DIPS or PC-DAPS bundled FFHQ sets (`test-ffhq`, `demo-ffhq`, any DAPS FFHQ archive). This includes smoke tests, timing and debugging; the repositories' default commands point to them, so override them. One of those images belongs to the locked holdout. FFHQ inputs come only from the R0 manifests of the 180 ids.

**Binding:**
- `docs/planning/track_r/XP_PREREG_v1_1.md` (FROZEN v1.1; supersedes v1; self-contained);
- `docs/track_r/executors/r0n/R0N_AMENDMENT_2_20261024.md`, revision D.1 (R0-N.2, CPU);
- `docs/planning/track_r/2026-10-25_TOP_VENUE_PLAN.md`;
- all R0 and R0-N data, PAC-safety and accounting rules.

## 0. Planner's view

**Results.** The FFHQ audit is done. GT-free selection recovers 89–97% of the orbit oracle's gain, and what remains is mostly generation failure. K2 means orientation repair is generic (CLIP), not a method.

**Plan.** XP adds what a top venue needs:
- a second dataset (ImageNet), including the published DAPS protocol on all 100 images;
- K-scaling to 64 (FAST) and 16–32 (SITCOM), plus DAPS to 8. These **test frozen theory predictions** in both regimes the theory distinguishes;
- same-GPU timing;
- the network's equivariance defect (XP-C2), which tests the theory's mechanism for orbit errors.

**Publication image.**
- CVPR 2027 attempt: paper Nov 16, go/no-go Nov 8. SPL and EUSIPCO 2027 are the fallbacks.
- **Data cuts:** Tier A by Nov 2, Tier B by Nov 6 (main paper), Tier C by Nov 20 (supplement, due Nov 23).

## 1. Branch and environment

- **Branch.** Create `codex/prdiff-track-r-xp-<YYYYMMDD>` from `codex/prdiff-track-r-r0n-20261023` @ `33a34bd` (or from your R0-N.2 branch head, if R0-N.2 is already committed). Then merge `origin/claude/prdiff-track-r-planner-20261021`. Record both SHAs.
  - **Merge nothing else.** In particular, never merge `main` or the data-governance audit branch `docs/daps-test-ffhq-confirmation305-audit-*`: its file path contains a forbidden token.
- **Environments.** Reuse the historical ones read-only: DAPS env, `sitcom_ode_bw`, `fast_dips_a15_blackwell`, `prdiff_track_r`. If a package is missing (e.g. for DPS or PC-DAPS), create a **new** environment by cloning; never modify a historical one.
- **Paths.**
  - Outputs: `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/xp_<stage>_<UTC>/`.
  - Caches: `/egr/research-pac/huang248/outputs/.cache/track_r/`.
  - New external code, if Cheng clones it: `/egr/research-pac/huang248/external/<name>_xp/`.

## 2. Order of work (critical path first)

1. **R0-N.2, B2 + A8 (CPU).** Commit A8 **by Oct 28 if at all possible**. The theory executor's frozen predictions (due Oct 31) are fitted on it.
2. **XP-0 (CPU), in parallel.** Inventory per prereg §2, plus:
   - the per-id y-SHA equality check (prereg §1);
   - the pad formula for oversample 4.0, which must give a 512 canvas (XP-G2);
   - `test -d` for `test-imagenet`, DPS and PC-DAPS. For anything absent, **write the exact command for Cheng** (the DAPS README gdown link; `git clone` + `git checkout <pinned SHA>`), then continue with what is present.
   - If the DAPS download bundles FFHQ and ImageNet together, ask Cheng to place **only** `test-imagenet` in the new data path.
3. **XP-T (GPU, ≈1 h).** Batch-1 timing for every (family, dataset) available, plus the W ∈ {1, 2, 4} throughput test. Output: a projection table per stage, which replaces the planner's estimates.
4. **Tier A queue.**
   - XP-A16, after the FAST replay gate (prereg §4.4);
   - XP-C and XP-C2;
   - XP-D1: ImageNet measurements, then phase 1 (runs 0–3), then phase 2 (runs 4–15).
5. **R0-N.2 remainder (CPU) while the GPUs run:** A4.3 → A7 → A4.1–A4.2 → A3.
6. **Tier B queue.**
   - Replay gates for SITCOM and DAPS first (prereg §4);
   - then XP-A64, XP-B16, XP-D2, XP-D3 and XP-F8, in the prereg §3 priority, work-conserving.
7. **Tier C queue:** XP-B32 (no new code, so it can start first), then XP-I, XP-H, XP-G2, XP-G and XP-D4. XP-E′ runs only if A4.4 shows that the historical FFHQ DAPS configuration differs from DAPS-1K. XP-E is cancelled.
8. **Analysis (CPU).** Prereg §6, respecting the §5 embargo and freeze checks. The exploratory analyses (GT-free reliability diagnostics; mixed-twin anatomy) come last and are optional.

## 3. Controller (recommended design)

The aim is one PAC launch per tier, not one per stage.

- **One job manifest per tier.** One row per (stage, family, dataset, uid, run index, seed, settings hash).
- **Workers.** W\* per GPU per family (from XP-T), each bound by `CUDA_VISIBLE_DEVICES`.
- **Work-conserving.** A worker takes the highest-priority ready job.
- **Resumable.** Skip any job whose output and receipt already exist and verify.
- **Receipts.** Write per-job receipts: status, seconds, GPU, commit.
- **Shell and logs.** `nohup` with a PID file, a status command (jobs done/failed/remaining, GPU-h used, projection) and a graceful stop that finishes running jobs. Logs are bounded.
- **Scoring is a separate GPU pass:** batched CLIP, the 2-forward prior, S_mid, and LPIPS. Solver workers only write candidates and residuals.
- **Failures.** A failed job is retried once with the same seed, then marked failed. Failures are counted and reported, **never silently resampled with a new seed**.

## 4. Embargo and freeze checks (prereg §5)

- **Before v1 is registered** (`docs/track_r/theory/PREDICTIONS_REGISTRY.md` on the planner branch), commit and report **no outcomes** for:
  - FFHQ XP-A, XP-B and XP-F runs with index ≥ 4;
  - B22 DAPS runs with index ≥ 2.

  Receipts with run counts and failures only are fine.
- **Before v2 is registered, or 48 h after the phase-1 counts if none is made,** the same applies to ImageNet FAST runs with index ≥ 4.
- **Do commit XP-D1 phase-1 per-image counts promptly.** The theory executor needs them for v2.
- **For each prediction file used,** the analysis verifies:
  - its registry entry;
  - the SHA-256 of `git show <commit>:<path>`;
  - that its commit time precedes your first commit containing the matching outcomes.

  Report all three.

## 5. PAC command style (unchanged)

- One consolidated block per launch.
- `nohup` with PID file, status command and graceful stop.
- Logs are bounded, and outputs saved to file with only a short printed summary.
- No `set -e`, `set -u`, `set -o pipefail`, `exit`, `logout` or broad `pkill`.
- Explicit return-code checks.
- Run from a **pushed, clean commit**. Print the SHA at start.
- Before any GPU block, print `nvidia-smi`. Refuse a GPU with foreign processes using more than 2 GiB unless Cheng passes an explicit override.

## 6. Stop and escalate

**Stop the affected family or stage on any of:**
- a smoke-test runtime more than 50% off the XP-T projection;
- a frozen-setting mismatch;
- a replay-gate failure;
- a measurement-SHA mismatch;
- a projected cumulative total above 400 GPU-h;
- any path matching the forbidden tokens ("confirmation", "confirm305");
- any code path that would load a bundled FFHQ set (`test-ffhq`, `demo-ffhq`, a DAPS FFHQ archive).

Other stages continue.

**Drop order if late or over budget:** prereg §3.5. Tier A is never dropped.

## 7. Returns (`XP_EXECUTOR_RETURN.md`, one per batch)

**Schedule:**

| Return | Content | Target |
|---|---|---|
| 1 | XP-0 + XP-T (projections) | Oct 28 |
| 2 | Tier A receipts; XP-C, XP-C2; XP-D1 phase-1 counts | Oct 30 |
| 3 | After the v1 (and v2) freezes: FFHQ FAST K-curves vs predictions; ImageNet FAST | Nov 2 |
| 4 | Tier B, with the SITCOM and DAPS prediction checks | Nov 5–6 |
| 5 | Tier C | Nov 20 |

**Each return contains:**
1. **Status for Cheng:** at most 8 lines.
2. **Identity:** branch, SHAs, environments, GPU model.
3. **GPU-hours** per stage and cumulative, against the 400 cap.
4. **Per stage:** tables (numbers only, with n and CIs), figures, anomalies, failed-job counts.
5. **Theory check:** the registry entries used, the SHA and commit-time verification, and per-prediction consistent/rejected, including the shape predictions.
6. **Artifacts:**
   - committed: aggregates, per-image strings (prereg §7), SHA256SUMS;
   - planner ZIP (SHA only in git): per-candidate tables, contact sheets.
7. **Executor opinion:** optional, and kept clearly separate.
