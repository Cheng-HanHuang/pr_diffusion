#!/usr/bin/env bash
# Recovery of the EXACT Python UnboundLocalError, no science change.
# Must be run from NEW detached A27 worktree (not frozen A26).
# Original A20 source/code freeze remains byte-for-byte intact.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: bash $0 REPAIR_HEAD BRANCH" >&2;exit 8; }
REPAIR_HEAD="$1";BRANCH="$2"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
REC_REL="docs/track_r/executors/xp/XP_C3_A20_RECOVERY_R1.json"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$WT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
cd "$WT"
test "$(git rev-parse HEAD)" = "$REPAIR_HEAD"
test -z "$(git status --porcelain)"
test -z "$(git branch --show-current)"
git fetch --no-tags origin "$BRANCH"
test "$(git rev-parse FETCH_HEAD)" = "$REPAIR_HEAD"
test ! -e "$REC_REL"

# Runs previously missing behavioral test before touching any candidate.
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a25_xpc3_a20.py' -v
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a26_xpc3_recovery_r1.py' -v

# Registrable evidence from ORIGINAL failed run and frozen A20 code.
"$PY" - <<'PY'
from pathlib import Path
import hashlib,json,subprocess
root=Path.cwd()
base=Path("/egr/research-pac/huang248/outputs/pr_diffusion/track_r")
orig=base/"xp_a25_xpc3_prod_20261103T002834Z"
f=orig/"XP_C3_A20_FINAL_STATUS.json"
log=orig/"logs/XP_N_1.log"
source=root/"docs/track_r/executors/xp/theory_inputs/XP_C3_A20_SOURCE_MANIFEST.json"
freeze=root/"docs/track_r/executors/xp/XP_C3_A20_CODE_FREEZE.json"
worker=root/"scripts/track_r/xp_a25_xpc3_mid_gpu.py"
control=root/"scripts/track_r/xp_a25_xpc3_autopilot.py"
reg=root/"docs/track_r/executors/xp/XP_C3_A20_RECOVERY_R1.json"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert f.is_file() and log.is_file()
old=json.loads(f.read_text())
spec=json.loads(freeze.read_text())
attempts=old["attempts"]
source_hash="298b88cdd8f4880fe35fd727c6d7c874dabcd9d09da55cd40a469788d5d19b27"
model_hash="a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1"
original="25152a3d6ffcc4611f45d12bf24d891331976d91"
original_scorer_bytes=subprocess.check_output(
    ["git","show",original+":scripts/track_r/xp_a25_xpc3_mid_gpu.py"])
assert spec["scorer_script_sha256"]==hashlib.sha256(original_scorer_bytes).hexdigest()
assert sha(source)==spec["source_manifest_sha256"]==source_hash
assert spec["checkpoint_sha256"]==model_hash
assert spec["timesteps"]==[300,400,500,600]
assert spec["draws"]==[0,1,2,3]
assert spec["no_2forward_secondary"] is True
assert old["status"]=="STOP_A20_XPC3_UNVERIFIED_OR_BUDGET"
assert old["completed_panels"]==[] and len(attempts)==1
assert (attempts[0]["gpu"],attempts[0]["panel"],
        attempts[0]["limit"],attempts[0]["rc"])==(0,"XP_N",1,8)
assert "UnboundLocalError" in log.read_text() and \
       "cannot access local variable 'source'" in log.read_text()
assert all(not list((orig/p).glob("*.json"))
           for p in ("XP_N","G5","XP_D4"))
prior=float(old["charged_gpu_h"])
assert 0<prior<0.01
assert abs(prior-attempts[0]["wall_seconds"]/3600)<1e-7
i2=base/"xp_a24_xpi2_prod_20261102T085625Z"/"XP_A24_XPI2_FINAL_STATUS.json"
i2doc=json.loads(i2.read_text())
assert i2doc["status"]=="A24_XPI2_COMPLETE"
assert i2doc["n_completed_bags"]==100
assert abs(float(i2doc["project_total_gpu_h"])+prior-old["project_gpu_h"])<1e-6
src=worker.read_text()
ctl=control.read_text()
assert 'saved_reconstruction=torch.load(' in src
assert 'source=torch.load(' not in src
assert 'ALLOWED_PHYSICAL_GPUS=(1,2,3)' in src
assert 'ALLOWED_PHYSICAL_GPUS=(1,2,3)' in ctl
assert 'cap_gpu_h=CAP_H,charged_gpu_h=previous_charge' in ctl
assert 'c.index in ALLOWED_PHYSICAL_GPUS' in ctl

registration=dict(
    status="A20_XPC3_ZERO_OUTPUT_R1_CODE_RECOVERY",
    planner_amendment="A20; original frozen prior unchanged",
    recovery_scope="Python local variable source shadowing bug; GPUs 1,2,3 only; carry original 4-second failed cost",
    original_execution_head=original,
    original_freeze_path=str(freeze.relative_to(root)),
    original_freeze_sha256=sha(freeze),
    original_scorer_sha256=spec["scorer_script_sha256"],
    original_source_commit=original,
    source_manifest_sha256=source_hash,
    source_manifest_relpath=str(source.relative_to(root)),
    source_manifest_rows=13600,
    corrected_scorer_sha256=sha(worker),
    corrected_controller_sha256=sha(control),
    failed_final_sha256=sha(f),
    failed_worker_log_sha256=sha(log),
    failed_charged_gpu_h=prior,
    failed_project_gpu_h=float(old["project_gpu_h"]),
    failed_worker_rc=8,
    failed_worker_gpu=0,
    failed_candidate_outputs=0,
    allowed_physical_gpus=[1,2,3],
    disallowed_physical_gpus=[0],
    no_new_method=True,
    scientific_definition_unchanged=True,
    exact_only_S_mid_global2=True,
    no_two_forward_secondary=True,
    model_checkpoint_sha256=model_hash,
    timesteps=[300,400,500,600],
    draws=[0,1,2,3],
    choice="111 iff L000-L111 > 1e-7",
    original_stage_cap_gpu_h=8.0,
    recovery_stage_remaining_gpu_h=8.-prior,
    project_cap_gpu_h=400.0,
    scoring_not_performed_before_this_registration=True)
assert not reg.exists()
reg.write_text(json.dumps(registration,indent=2,sort_keys=True,allow_nan=False)+"\n")
print("A20_R1_CORRECTED_SCORER_SHA",sha(worker))
print("A20_R1_CORRECTED_CONTROLLER_SHA",sha(control))
print("A20_R1_ORIGINAL_FAILED_ATTEMPT_CHARGED_GPU_H",prior)
print("A20_R1_UNCHANGED_13600_SOURCE_SHA",source_hash)
print("A20_R1_REGISTRATION_STAGED",str(reg))
PY

git add -- "$REC_REL"
test "$(git diff --cached --name-only)" = "$REC_REL"
git commit -m "freeze(track-r): register A20 XP-C3 zero-output R1 code-only repair, GPU1-3 and charged failure"
FROZEN_R1_HEAD="$(git rev-parse HEAD)"
git push origin "HEAD:refs/heads/$BRANCH"
git fetch --no-tags origin "$BRANCH"
test "$(git rev-parse FETCH_HEAD)" = "$FROZEN_R1_HEAD"
test -z "$(git status --porcelain)"
RUN="$BASE/xp_a25_xpc3_prod_r1_$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$BASE/XP_C3_A20_R1_SUPERVISOR_$(basename "$RUN").log"

# Read-only full ledger/controller preflight BEFORE detach.
"$PY" -u scripts/track_r/xp_a25_xpc3_autopilot.py \
    --expected-head "$FROZEN_R1_HEAD" \
    --run "$RUN" --branch "$BRANCH" --preflight-only

# All science/source/code recovery SHAs are pushed. Launch once.
nohup env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$WT" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    "$PY" -u scripts/track_r/xp_a25_xpc3_autopilot.py \
    --expected-head "$FROZEN_R1_HEAD" \
    --run "$RUN" --branch "$BRANCH" \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "A20_XPC3_R1_REGISTERED_HEAD=$FROZEN_R1_HEAD"
echo "A20_XPC3_R1_SUPERVISOR_PID=$PID"
echo "A20_XPC3_R1_RUN=$RUN"
echo "A20_XPC3_R1_LOG=$LOG"
echo "A20_XPC3_R1_ONLY_PHYSICAL_GPUS=1,2,3"
echo "A20_XPC3_R1_FROZEN_SOURCE_UNCHANGED=298b88cdd8f4880fe35fd727c6d7c874dabcd9d09da55cd40a469788d5d19b27"
echo "A20_XPC3_R1_OLD_A26_UNTOUCHED"
