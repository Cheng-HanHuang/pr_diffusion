# PRDiffusion B26 executor handoff — 2026-10-15

## Mission / operating relationship
You are the **B26 execution and recovery executor** in a separate ChatGPT chat. The user runs all commands on MSU PAC; **you have no direct PAC access**. The existing chat is the scientific **planner**, responsible for deciding scientific continuation after reading your compact verified return.

**FIRST ACTION: Read this handoff, repository `AGENTS.md`, `docs/b26/00_START_HERE.md`, `docs/b26/B26_EXECUTOR_CONTRACT.md`, `docs/b26/B26_NATIVE_ESTIMATOR_SPEC.md`, `docs/b26/B26_CHECKPOINT_REPORT.md`, `configs/b26/b26_spec.json`, and the exact launch/status/finalization source. Then perform a read-only PAC inventory. DO NOT start another run until you know whether the old one exists, is live, failed, or finished.**

## Known GitHub facts (verified October 15)
- Repo: https://github.com/Cheng-HanHuang/pr_diffusion
- Frozen execution branch: `codex/b26-np-conditional-correction`; draft PR #40: https://github.com/Cheng-HanHuang/pr_diffusion/pull/40
- Remote frozen B26 head at handoff: `5beb7f6485c5a7e44be19bcf4dc1cc5cd47603d2` (2026-09-15; exact DEV80 manifest freeze).
- B26 signed B25 base: `c906d36e0e396a6abbf761e3d65c91433428170f`; B25 pre-run `32453db6445acec4fc19a4a928142a412d67f1ae`.
- This *handoff* is published on a separate docs-only branch `codex/planner-b26-resume-handoff-20261015`, based on the frozen B26 head. **Do not switch the PAC execution checkout to this handoff branch**: B26 launchers require the pristine execution branch and exact remote pre-run identity.
- The GitHub B26 checkpoint still said "implementation / PAC manifest freeze pending" although the later branch head now contains `configs/b26/b26_dev80_manifest.json`. Trust file/commit/run receipts over the outdated checkpoint sentence.
- No completed B26 science evidence was found in the GitHub branch by the planner. **This does not mean PAC was idle**: check PAC for unpushed outputs.

## User-provided PAC leads (verify existence, branch and hashes on PAC; do not infer)
Root: `/egr/research-pac/huang248`.
Checkout likely `/egr/research-pac/huang248/pr_diffusion_b26`; main checkout `/egr/research-pac/huang248/pr_diffusion_repo`; other historical checkouts `pr_diffusion_b22`, `pr_diffusion_b23`, `pr_diffusion_b24`, `pr_diffusion_b25`.
Outputs under `/egr/research-pac/huang248/outputs/pr_diffusion`, B26 subdir `.../b26`. Other historical output subdirs `b22_baselines`, `b23`, `b24`, `b25`.
Conda envs under `/egr/research-pac/huang248/conda-envs`; B26 native NP Python `.../prdiff_ffhq/bin/python`; also `.../daps` and `.../sitcom_ode_bw`. **Never use conda installed in /home or assume default Python has packages.**
Frozen B24 DEV80: `/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_dev80_overnight_20260913T082146Z`.
B25 sealed capsule: `/egr/research-pac/huang248/outputs/pr_diffusion/b25/B25_cpu_20260914T224132Z.tar.gz`, expected SHA-256 `5b1e785ba46e6be952ffdbc768c5d9bab365e495cf9efd5b154b075ce0081354`.

## Frozen B26 scientific work (authorization inherited; do not modify)
**B26.1** native NP scoring correction on 80 development images, four frozen roots, two arms: H uses `y_plus = max(y_raw,0)` for scoring and projection; R uses **signed `y_raw`** for historical score/selector, but `y_plus` for nonnegative amplitude projection. Everything else unchanged: pretrained guided model, four roots, RNG, K=5 pre-projection, K=1 post-projection, step=1000, projection start 300, LF score radius 0.6, projection radius 0.2, candidate selection, etc. `2200` UNet evaluations per root; total final 80*4*2 = **640 root-arm trajectories**, with smoke and DEV16 included in the final count (not added twice). Stages: smoke root0 across four frozen strata = 8 trajectories; DEV16 cumulative 16*4*2 = 128; DEV80 cumulative 640. H must replay B24 terminal hashes/selector/noise lineages. Continuation gates are **identity, technical and budget gates only, NOT PSNR-favorability**. GPU budget ceiling 24 aggregate reserved GPU hours; four physical GPU IDs 0–3 only when runtime gates confirm availability; minimum free 10,240 MiB; hard process/group maximum 52,452 MiB per GPU. No dynamic reassignment.
**B26.2** CPU finite-support intermediate-likelihood estimator: 4 families *32 observations *6 methods *512 reverse trajectories = **393,216 complete synthetic reverse trajectories**. Conditional likelihood estimates M=1,4,16, exact alternatives, separate RNG namespaces, randomized exact ties. CPU only (`CUDA_VISIBLE_DEVICES=""`), <=4 threads, <=16 GiB RSS, <=4h wall.
**No weighted FFHQ campaign**: future continuation estimator was costed around 50,200 UNet evaluations/root versus 2,200 historical, and was NOT authorized for this experiment.

## Hard boundaries
- B24 method refinement remains closed under `STOP_B24_METHOD_REFINEMENT`.
- The frozen 305 confirmation items must never be opened, measured, sampled or analyzed.
- No new FFHQ measurements; no DAPS/SITCOM/Fresh rerun; no NP/SITCOM adapter; no weighted FFHQ reconstruction.
- Do not retune B26 using interim PSNR, or alter the frozen 80-image split or selection thresholds.
- Preserve B25 sealed capsule and PRs #37–39, `main`, historical branches, worktrees, and results.
- No force push, rebase, squash, merge, reset, or destructive cleanup. Never modify a dirty checkout automatically. If a code fix is required, STOP and return an explicit patch proposal for planner approval; a changed pre-run SHA invalidates the current launch identity.
- Never use `set -e`, `set -u`, `set -o pipefail`, `exit`, `logout`, broad `pkill`, or commands that close the user's terminal. Persist full logs, print bounded summary; prefer existing committed scripts.

## First PAC command: READ-ONLY inventory (the user's next action)
This only writes a new inventory text log under the known output parent, and reads existing paths and statuses. It must not start, stop or modify a scientific run.

```bash
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_b26"
OUT="$ROOT/outputs/pr_diffusion/b26"
AUDITPARENT="$ROOT/outputs/pr_diffusion"
if [ ! -d "$AUDITPARENT" ]; then
  echo "STOP: output parent not found: $AUDITPARENT"
else
  AUDIT="$AUDITPARENT/B26_INVENTORY_$(date -u +%Y%m%dT%H%M%SZ).txt"
  {
    echo "=== TIME / SYSTEM ==="
    date -u
    hostname
    echo "=== PAC WORKTREES ==="
    for d in "$ROOT/pr_diffusion_repo" "$ROOT/pr_diffusion_b22" "$ROOT/pr_diffusion_b23" "$ROOT/pr_diffusion_b24" "$ROOT/pr_diffusion_b25" "$ROOT/pr_diffusion_b26"; do
      if [ -d "$d/.git" ] || [ -f "$d/.git" ]; then
        printf 'WORKTREE %s\n' "$d"
        git -C "$d" rev-parse --abbrev-ref HEAD 2>&1
        git -C "$d" rev-parse HEAD 2>&1
        git -C "$d" status --short 2>&1 | head -n 20
      else
        printf 'MISSING_OR_NON_GIT %s\n' "$d"
      fi
    done
    echo "=== PYTHON ==="
    for n in prdiff_ffhq daps sitcom_ode_bw; do
      p="$ROOT/conda-envs/$n/bin/python"
      if [ -x "$p" ]; then "$p" --version; echo "FOUND $p"; else echo "MISSING $p"; fi
    done
    echo "=== B26 ROOT ==="
    if [ -d "$OUT" ]; then
      find "$OUT" -mindepth 1 -maxdepth 1 -type d -name 'B26_*' -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort -r | head -n 12
      echo "=== RECENT RUN STATUSES ==="
      find "$OUT" -mindepth 1 -maxdepth 1 -type d -name 'B26_*' -printf '%T@ %p\n' | sort -nr | head -n 4 | cut -d' ' -f2- | while IFS= read -r RUN; do
        echo "--- RUN $RUN ---"
        if [ -f "$REPO/scripts/b26/status_b26.sh" ]; then
          bash "$REPO/scripts/b26/status_b26.sh" "$RUN" 2>&1 | head -n 100
        else
          echo "STATUS_SCRIPT_MISSING"
          find "$RUN" -mindepth 1 -maxdepth 2 -type f -name '*.json' -print | head -n 25
        fi
      done
    else
      echo "B26_OUTPUT_DIR_NOT_FOUND"
    fi
    echo "=== GPU SNAPSHOT (NO ADMISSION ASSUMPTION) ==="
    nvidia-smi --query-gpu=index,uuid,memory.total,memory.free --format=csv,noheader,nounits 2>&1
  } > "$AUDIT" 2>&1
  echo "AUDIT_FILE=$AUDIT"
  tail -n 140 "$AUDIT"
fi
```

**Important:** the inventory only checks that candidate path names exist, not source hashes or whether any GPU is available for a future run. Do not claim RUN complete based only on directory existence.

## After the user returns the audit
1. Classify state: no run / live / partial / stage-complete / B26 machine-complete / contaminated. Examine `PRE_RUN_IDENTITY.json`, GPU gate files and results, `CPU_RESOURCE.json`, `CPU_SUMMARY.json`, `B26_COMPLETE.json`, and fresh logs. Match local/remote branch HEAD to the recorded pre-run identity and frozen manifest SHA. Check data/process integrity, no confirmation access, and full resource receipts. Use targeted paths, no broad recursive scan.
2. If B26 already completed, run only verification/analysis/package steps justified by current files; **do not rerun**. If partial, preserve completed PASS jobs and failure receipts and use committed resume/gate scripts; do not overwrite.
3. If no B26 run exists, inspect existing `scripts/b26/launch_b26_initial.sh`, `launch_b26_gpu_stage.sh`, `finalize_b26_gpu_stage.sh`, `status_b26.sh`, `stop_b26.sh`, `gate_b26_gpu.py`, `run_cpu_stage.py`, `finalize_b26_run.py`. Verify B25 capsule SHA, B26 local/remote exact identity, manifest/spec hashes, zero-GPU tests, free GPUs, and preserved source identity. Only then guide the user through the **frozen** launch, beginning smoke plus CPU, and prospectively gated DEV16 and DEV80. Do not bypass gates, invent runner options, start simultaneously conflicting workers, or estimate runtime from memory.
4. After successful stages verify all frozen metrics, paired image-level bootstrap, Good25/bad20, signed-vs-clamped selector cross-evaluation, H replay fidelity, M=1/4/16 synthetic estimator error and truth-risk, total GPU-hours/UNet calls/CPU work; distinguish positive/negative/mixed evidence. Run exact finalization and package audit, push only compact results to an approved clean branch without moving the pre-run execution SHA.
5. Return an explicit `PLANNER_RETURN` message with exact branch/head, PAC run path, stage progress, all gate outcomes, resource totals, counts, missing/failure rows, output artifacts/checksums, confirmation flag, raw-vs-clamped results, finite-estimator results, and one recommendation: `stop`, `isolate another identified defect`, or `request a precisely specified weighted-NP pilot`. Executor **does not** choose the next scientific method or promote a publication claim.

## Baseline and research background (for context, not B26 execution authorization)
B22 fixed 100-image panel: Fresh2 two full DAPS runs, clean-free exact loss margin 0.7: mean PSNR 29.299, q05 15.860, Good25 92/100, 312.6 GPU-s/image; SITCOM-4S four SITCOM-ODE runs, correction_norm tau=0.8 selector: 26.442, q05 23.415, Good25 93/100, 196.1 GPU-s/image; NP-8-RS: **eight NP runs from 2 scoring configurations * 4 seeds** (LF and S2 pre-proj lambda 0.01; seeds 100--103), minimum post-projection LF winner MSE selector: mean 29.269, q05 25.146, Good25 95/100, 411.8 GPU-s/image. B22 oracle4/oracle8 are **separate diagnostic rows**. B22 NP-8-RS, despite its name, is not a ground-truth "best of 8". See `configs/b22/b22_2_overnight.json` and `docs/b22/b22_3_scientific_closeout.md` on `codex/b22-fixed-baseline-comparison`.
Original DAPS paper CVPR 2025 explicitly uses the *best of four independent phase-retrieval samples*: https://arxiv.org/html/2407.01521v3; original SITCOM ICML 2025 likewise uses best of four, including for baseline PR runtime: https://arxiv.org/html/2410.04479v2. Do not equate published oracle-style best-of-four with our executable, clean-free selectors.
The user/planner may separately explore whether NP's reliability improvements support a small-venue paper. **Do not divert B26 execution to paper reruns or fresh method search** without planner authorization.

## First response in new executor chat
State that no PAC commands have been executed by you. Summarize exact B26 fork/spec and safety gates in 1–2 paragraphs. Give the read-only consolidated PAC inventory block above and ask the user to paste the bounded summary or upload the audit file. Based on results, provide stage-specific guidance without asking them to repeat known PAC paths.
