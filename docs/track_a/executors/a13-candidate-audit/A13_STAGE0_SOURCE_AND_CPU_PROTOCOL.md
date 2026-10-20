# A1.3.0 source inventory and CPU-only test protocol (2026-10-20)

**State:** Implementation committed for review. **PAC UNINSPECTED; tests NOT RUN; no saved candidate opened by this executor; no GPU.** The eight images are already exposed B22 development samples, not prospective validation.

## Gates and ownership

- Source branch exclusively: `codex/prdiff-track-a-a13-candidate-audit-20261020`; pinned execution HEAD must be supplied explicitly in every runner (not inferred from old A1 source).
- New worktree reserved: `/egr/research-pac/huang248/pr_diffusion_track_a_a13_candidate_audit`. New sole output namespace: `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a13_candidate_audit_20261020/` with stage IDs `a13_0_inventory_v1`, `a13_1_cpu_forensics_v1`. These are reservations, **not assertions that they exist**.
- A1.2 sources are read-only: `.../a1_oracle_budget_20261016/a1_2_exposed_b22_k4_v1` and `.../a1_2_exposed_b22_inputs_v1`. All 128 original terminals, receipts, measurements and GT tensors remain in place, unmodified.
- The A1.2 GPU controller at execution commit `f4cd28dcd3ec4860cc9f3990fb24964bb97e97bd` is checked by Git blob identity in the A1 source checkout (which may now have later *documentation* commits).
- Native HEADs: DAPS `e7a77d094167084faed19b599b96673b7bb11447`, SITCOM `275ab67efbd8146bffca20155171ba6be1169c09`, DiffFPR `a45ffe58f18fed8a63d3446600424e2b08733524`. DAPS/SITCOM historical tracked patch hashes are pinned in config. No native source is edited.
- No source/model imports beyond native CPU `forward_operator.PhaseRetrieval` **after source identity succeeds**; no checkpoints loaded. Always `CUDA_VISIBLE_DEVICES=""`, `PYTHONDONTWRITEBYTECODE=1`, the explicit `prdiff_ffhq` interpreter. No `torch.cuda` compute.
- B24 confirmation305 is off limits entirely, including path/ID/membership discovery. No B21–B26 source or artifact writes, no GT-derived selectors, no Track B algorithm changes.

## First PAC command: bounded, READ-ONLY inspection ONLY

Run in current interactive shell; this command DOES NOT clone, create outputs, read image pixels or tensors, run native models, or launch CPU scoring. Inspect results and return the full short stdout before taking further steps.

```bash
ROOT=/egr/research-pac/huang248
BR=codex/prdiff-track-a-a13-candidate-audit-20261020
WORK="$ROOT/pr_diffusion_track_a_a13_candidate_audit"
A1SRC="$ROOT/pr_diffusion_track_a_a1_oracle_budget"
A1="$ROOT/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
RUN="$A1/a1_2_exposed_b22_k4_v1"
INPUT="$A1/a1_2_exposed_b22_inputs_v1"
A13="$ROOT/outputs/pr_diffusion/track_a_np_paper/a13_candidate_audit_20261020"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
printf 'EXPECTED_BRANCH=%s\n' "$BR"
if [ -e "$WORK" ] || [ -L "$WORK" ]; then
  echo 'WORKTREE_OCCUPIED_STOP: inspect owner before any clone or worktree operation'
  git -C "$WORK" branch --show-current 2>/dev/null
  git -C "$WORK" rev-parse HEAD 2>/dev/null
  git -C "$WORK" status --short --untracked-files=no 2>/dev/null | head -8
else
  echo 'WORKTREE_ABSENT: reservation appears free (not created)'
fi
if [ -e "$A13" ] || [ -L "$A13" ]; then
  echo 'OUTPUT_NAMESPACE_OCCUPIED_STOP: preserve, do not create/overwrite'
else
  echo 'OUTPUT_NAMESPACE_ABSENT: reservation appears free (not created)'
fi
printf 'PYTHON_EXISTS='; [ -x "$PY" ] && echo yes || echo no
printf 'A1_SOURCE_EXISTS='; [ -d "$A1SRC" ] && echo yes || echo no
printf 'RUN_EXISTS='; [ -d "$RUN" ] && echo yes || echo no
printf 'INPUT_EXISTS='; [ -d "$INPUT" ] && echo yes || echo no
git -C "$A1SRC" rev-parse HEAD 2>/dev/null | sed 's/^/A1_DOC_HEAD=/'
"$PY" - "$RUN" "$INPUT" <<'PYTHON'
import json,sys
from pathlib import Path
run,inp=map(Path,sys.argv[1:])
for path in (run/"RUN_IDENTITY.json",run/"A1_2_COMPLETE.json",
             run/"A1_2_OFFLINE_ORACLE_SUMMARY.json",
             inp/"A1_2_INPUT_FREEZE_COMPLETE.json"):
    if not path.is_file() or path.is_symlink():
        print("MISSING_STOP",path);continue
    try:
        z=json.loads(path.read_text())
        print("RECEIPT",path.name,"KEYS",",".join(sorted(z)[:25]),
              "N_IMAGE_IDS",len(z.get("image_ids",[])),"COUNT",z.get("candidate_count","n/a"),
              "STATUS",z.get("status","n/a"))
    except Exception as e:print("UNREADABLE_STOP",path.name,type(e).__name__)
for ident,method,rep in [("69073","DAPS",0),("68539","DAPS",0),
                         ("67174","NP_LF",3),("66631","SITCOM",3)]:
    receipt=run/ident/f"{method}_rep{rep}_COMPLETE.json"
    if receipt.is_file() and not receipt.is_symlink():
        try:
            row=json.loads(receipt.read_text())
            term=Path(row["terminal"])
            print("TERMINAL_PATH_ONLY",f"{ident}:{method}:{rep}",
                  "EXISTS",term.is_file(),"RECEIPT",receipt.name)
        except Exception as e:print("RECEIPT_STOP",receipt.name,type(e).__name__)
    else:print("RECEIPT_MISSING_STOP",receipt)
PYTHON
```

**STOP and report** if unknown worktree/output occupied, missing receipts, unexpected branch, or missing interpreter. The initial command deliberately does not enumerate arbitrary B21 or B24 files.

## Subsequent authorized CPU path (do NOT execute before reviewing first receipt)

After checking the read-only inspection, create only the reserved independent A1.3 checkout from the dedicated GitHub branch, leaving the A1 executor checkout untouched. Confirm actual branch and commit; do not merge, rebase or reset historical branches. First run focused synthetic tests:
`$PY -m unittest scripts.track_a.test_a13_physics -v`.

Then execute `bash scripts/track_a/a13_run_inventory_v1.sh <PINNED_A13_COMMIT>`, which independently:

1. Verifies A1.3 branch, exact HEAD, no tracked changes, A1 frozen controller blob, native repo HEADs and historical dirty-source patch hashes, then compares independent FFT output to **both** CPU native `PhaseRetrieval` operators. Requires orthonormal, centered FFT, symmetric 64-pixel zero pad, [-1,1] -> [0,1] without clamp, channelwise shape 1×3×384×384, impulse, zero, and inversion witness.
2. Checks frozen eight-ID list/manifest and original signed measurement and saved GT **file digests**, then all 128 receipt/seed/measurement identities, unique candidate IDs, unique terminal SHA and exact terminal bytes, and comparable original PNG SHA (DAPS saved PNG included). No tensor pixels are decoded by stage0.
3. Only after all gates PASS, creates exactly one new `a13_0_inventory_v1` result directory and a compact JSON audit/receipt. No output on failed precreate gate, no partial filtering, and no fallback paths.

**Hard gate:** Do not run `a13_run_cpu_forensics_v1.sh` until the source audit and first-stage receipt are returned and independently checked. Stage1 revalidates source and all 128 terminal bytes before decoding the saved candidates/observations/GT. It computes independent signed-noise calibration *before inspecting real candidates* (32 generated CPU Gaussian draws; fixed seed 720611 and sigma 0.05), then evaluates two **predeclared** non-GT policies, separately: min normalized signed final full FFT residual and min clipped final full FFT residual (stable earliest index), compared to raw RGB8 PSNR GT oracle of exactly the same within-family K1/K2/K4 or frozen interleaved NP K4/K8 portfolios. The candidate selector residual is computed **from uniform saved PNG8 outputs**; DAPS does not have a uniformly archived comparable float endpoint. The 96 SITCOM/NP float terminals additionally get individually verified and scored, distinctly labeled/charged as native-float diagnostics. Neither score is a replacement for archived method-specific native costs or ranking rules.

A1.3.1 reports 128 per-candidate signed/clipped scores (including count/fraction of negative raw observations and per-channel SSE), raw/rot180 PSNR replay, 112 per-image policy prefix oracle replays, 224 score-policy rows, reconstruction and external FFT costs, residual/PSNR ranks, failure taxonomy against a locked synthetic Gaussian-noise RMSE baseline
(consistent if at/below max of 32 synthetic draws, poor fit if above 3× maximum,
otherwise unresolved), unsupported cells, and a metadata-only ZIP. Both selector indices are chosen from features **before any GT tensor is loaded**. The fit taxonomy intentionally marks non-rotation low-quality cases **unresolved** until an independent known-noise threshold is calibrated; do not overinterpret low raw PSNR or low residual. All eight samples are already exposed and any association is descriptive only.

### Mandatory return/review/STOP

Return the initial PAC stdout first. For stage0, return the concise receipt/sha256 and full test output; on STOP, return error without modifying paths. After verified stage0 and explicit user continuation, return the stage1 metadata ZIP with printed SHA256, unit-test/stdout logs, source HEAD, and any non-PASS receipts. Final forensic results/limitations and next-stage GO/STOP must be committed in `A13_EXECUTOR_RETURN.md` or `A13_STOP_REPORT.md` and registered in issue #41, where Track A planner decides whether A1.4 exists. No new GPU/data stage follows automatically.
