# Track R · XP executor handoff: evaluation-only extensions (GPU)

**From:** planner (Claude), 2026-10-25
**To:** the PAC-side executor (the one that ran R0 and R0-N). Cheng runs every PAC block.

**Binding:**
- `docs/planning/track_r/XP_PREREG_v1.md` (FROZEN v1);
- `docs/planning/track_r/2026-10-25_TOP_VENUE_PLAN.md`;
- all R0 and R0-N data, PAC-safety and accounting rules.

**Gate.** **Do nothing on GPU until Cheng writes "GO XP".** CPU preparation (code, tests, XP-0 inventory) may start at once.

## 0. Planner's view

**Results.** The FFHQ audit is done: GT-free selection recovers 89–97% of the orbit oracle's gain, and what remains is generation failure. K2 means orientation repair is generic (CLIP), not a method.

**Plan.** XP adds three things a top venue needs:
- a second dataset (ImageNet);
- K-scaling that **tests frozen theory predictions**;
- same-GPU timing.

The R0-N.2 CPU block (amendment 2 revision D) runs **first or in parallel**. Its A8 per-image counts feed the theory predictions.

**Publication image.** A CVPR 2027 attempt (paper due Nov 16; go/no-go Nov 8), with SPL and EUSIPCO 2027 as fallbacks. XP-A and XP-C must be back by **Nov 2**; XP-D (FAST part) by **Nov 5**.

## 1. Branch and environment

- **Branch.** Create `codex/prdiff-track-r-xp-<YYYYMMDD>` from `codex/prdiff-track-r-r0n-20261023` @ `33a34bd`, then merge `origin/claude/prdiff-track-r-planner-20261021`. Record both SHAs.
- **Environments.** Reuse the historical ones read-only: DAPS env, `sitcom_ode_bw`, `fast_dips_a15_blackwell`, `prdiff_track_r`. If a package is missing, create a **new** environment by cloning; never modify a historical one.
- **Paths.** Outputs go under `/egr/research-pac/huang248/outputs/pr_diffusion/track_r/xp_<stage>_<UTC>/`. Caches go under `/egr/research-pac/huang248/outputs/.cache/track_r/`.

## 2. Tasks, in order

1. **XP-0 (CPU, bounded).**
   - Record:
     - GPU count, model and free memory;
     - whether `test-imagenet` is present (`test -d` only);
     - the `imagenet256.pt` SHA-256;
     - an environment smoke import for each solver.
   - If `test-imagenet` is absent, **write the exact download command for Cheng** (DAPS README gdown link) into a new data path, then stop for him. Do not download from agent code.
2. **XP-T (GPU, ≈0.3 h).** Timing calibration exactly as in pre-registration §2: one GPU model, CUDA events plus process wall. This also yields measured CLIP and 2-forward-prior times per candidate.
3. **XP-A (GPU, ≈2 h).** FAST on 180 FFHQ ids, run indices 4–15. Reuse the S5 measurement files. Continue the S5 seed rule.
   - **Scoring on GPU:** CLIP (frozen face set and G1/G2) and the 2-forward prior.
   - **Scoring on CPU:** everything else.
4. **XP-C (GPU, minutes).** Orientation tests on clean GT images (100 ImageNet, 180 FFHQ).
5. **XP-D (GPU).**
   - ImageNet measurements first (CPU or GPU; seed rule in §1).
   - Then FAST n = 8 on all 100; SITCOM n = 4 on all 100; DAPS n = 4 on 25.
   - Smoke each family before its full launch. Parallelize across free GPUs, one process per GPU.
6. **XP-B (optional, GPU ≈14 h).** SITCOM n = 16 on the 60 pre-registered FFHQ ids. Launch only if XP-A, XP-C and the FAST part of XP-D are done and the budget allows.
7. **XP-E (optional).** Only after Cheng's written confirmation about `test-ffhq` and confirmation305 (pre-registration §2).
8. **Analysis (CPU).** Pre-registration §3, with U-statistic estimators and image-cluster bootstrap.
   - Before computing any K > 4 curve, check that `docs/track_r/theory/TH1_PREDICTIONS_v1.json` exists on the planner branch **with a commit time earlier than your return commit**. If it does not exist, compute K ≤ 4 only and report "predictions not frozen".

## 3. PAC command style (unchanged)

- One consolidated block per stage.
- `nohup` with PID file, status command and graceful stop.
- Logs are bounded, and outputs saved to file with only a short printed summary.
- No `set -e`, `set -u`, `set -o pipefail`, `exit`, `logout` or broad `pkill`.
- Explicit return-code checks.
- Run from a **pushed, clean commit**. Print the SHA at start.

## 4. Stop and escalate

**Stop on any of:**
- a smoke-test runtime more than 50% off the XP-T projection;
- a frozen-setting mismatch;
- a measurement or replay mismatch;
- a GPU projection that pushes the cumulative total above 45 h;
- any path matching the forbidden tokens ("confirmation", "confirm305").

**Drop order if late:** XP-B → DAPS on ImageNet → SITCOM on ImageNet.

## 5. Return (`XP_EXECUTOR_RETURN.md`, one per stage batch)

1. **Status for Cheng:** at most 8 lines.
2. **Identity:** branch, SHAs, environments, GPU model.
3. **GPU-hours** per stage and cumulative.
4. **Per stage:** tables (numbers only, with n and CIs), figures, anomalies.
5. **Theory check:** the predictions-file SHA and commit time, and per-prediction consistent/rejected (K > 4 only).
6. **Artifacts:**
   - committed: aggregates, per-image label counts, SHA256SUMS;
   - planner ZIP (SHA only in git): per-candidate tables, contact sheets.
7. **Executor opinion:** optional, and kept clearly separate.
