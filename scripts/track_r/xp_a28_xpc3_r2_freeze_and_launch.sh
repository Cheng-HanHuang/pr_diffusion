#!/usr/bin/env bash
# Recovery of native NATIVE import error following R1 UnboundLocalError, no science change.
# Must be run from NEW detached A28 worktree (not frozen A26 or R1 A27).
# Original A20 source/code freeze remains byte-for-byte intact.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: bash $0 REPAIR_HEAD BRANCH" >&2;exit 8; }
REPAIR_HEAD="$1";BRANCH="$2"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
REC_REL="docs/track_r/executors/xp/XP_C3_A20_RECOVERY_R2.json"
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
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a28_xpc3_recovery_r2.py' -v
"$ROOT/conda-envs/daps/bin/python" - <<'PY'
from scripts.track_r.xp_t_daps4k_prepare_cpu import IMAGENET_MODEL_SHA
from scripts.track_r.xp_t_daps4k_native_cpu import NATIVE,verify_clean_sparse
from scripts.track_r.xp_t_daps4k_worker_gpu import model_config,load_model_native,validate_weight_trace
from prdiffusion.track_r.prior_score import frozen_alpha_schedule,prior_seed
assert NATIVE.name=="DAPS_XP_NATIVE_4K" and len(IMAGENET_MODEL_SHA)==64
assert callable(verify_clean_sparse) and callable(model_config) and callable(load_model_native)
assert callable(validate_weight_trace) and len(frozen_alpha_schedule())==1000
assert isinstance(prior_seed("x",300,0),int)
print("A20_R2_NATIVE_DAPS_PYTHON_IMPORT_PASS")
PY

# Verify original and R1 failed receipts before signing recovery R2.
"$PY" - <<'PY'
from pathlib import Path
import hashlib,json,subprocess
root=Path.cwd()
base=Path("/egr/research-pac/huang248/outputs/pr_diffusion/track_r")
o=base/"xp_a25_xpc3_prod_20261103T002834Z"
r=base/"xp_a25_xpc3_prod_r1_20261103T163302Z"
src=root/"docs/track_r/executors/xp/theory_inputs/XP_C3_A20_SOURCE_MANIFEST.json"
freeze=root/"docs/track_r/executors/xp/XP_C3_A20_CODE_FREEZE.json"
r1=root/"docs/track_r/executors/xp/XP_C3_A20_RECOVERY_R1.json"
r2=root/"docs/track_r/executors/xp/XP_C3_A20_RECOVERY_R2.json"
w=root/"scripts/track_r/xp_a25_xpc3_mid_gpu.py"
c=root/"scripts/track_r/xp_a25_xpc3_autopilot.py"
of=o/"XP_C3_A20_FINAL_STATUS.json"
rf=r/"XP_C3_A20_FINAL_STATUS.json"
ol=o/"logs"/"XP_N_1.log"
rl=r/"logs"/"XP_N_1.log"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
blob=lambda ref,path:subprocess.check_output(["git","show",ref+":"+path])
O="25152a3d6ffcc4611f45d12bf24d891331976d91"
R="eac8b4e044f04e513d136081d5c398da480bd550"
assert all(x.is_file() for x in (src,freeze,r1,of,rf,ol,rl))
spec=json.loads(freeze.read_text())
rec=json.loads(r1.read_text())
old=json.loads(of.read_text())
prior=json.loads(rf.read_text())
source_sha="298b88cdd8f4880fe35fd727c6d7c874dabcd9d09da55cd40a469788d5d19b27"
assert sha(src)==spec["source_manifest_sha256"]==source_sha
assert spec["no_2forward_secondary"] is True
assert spec["timesteps"]==[300,400,500,600]
assert spec["draws"]==[0,1,2,3]
osp=hashlib.sha256(blob(O,"scripts/track_r/xp_a25_xpc3_mid_gpu.py")).hexdigest()
rsp=hashlib.sha256(blob(R,"scripts/track_r/xp_a25_xpc3_mid_gpu.py")).hexdigest()
rcp=hashlib.sha256(blob(R,"scripts/track_r/xp_a25_xpc3_autopilot.py")).hexdigest()
assert osp==spec["scorer_script_sha256"]
assert rsp==rec["corrected_scorer_sha256"] and rcp==rec["corrected_controller_sha256"]
assert rec["original_freeze_sha256"]==sha(freeze)
assert rec["source_manifest_sha256"]==source_sha
assert old["status"]==prior["status"]=="STOP_A20_XPC3_UNVERIFIED_OR_BUDGET"
assert old["completed_panels"]==prior["completed_panels"]==[]
assert len(old["attempts"])==len(prior["attempts"])==1
assert old["attempts"][0]["gpu"]==0 and old["attempts"][0]["rc"]==8
assert prior["attempts"][0]["gpu"] in (1,2,3) and prior["attempts"][0]["rc"]==8
assert prior["attempts"][0]["panel"]=="XP_N" and prior["attempts"][0]["limit"]==1
assert "UnboundLocalError" in ol.read_text()
assert "ImportError" in rl.read_text(), "STOP_R2_PRIOR_FAILURE_DIFFERENT"
assert "cannot import name 'NATIVE'" in rl.read_text(), "STOP_R2_IMPORT_NAME_DIFFERENT"
assert all(not list((path/p).glob("*.json")) for path in (o,r) for p in ("XP_N","G5","XP_D4"))
assert prior["source_manifest_sha256"]==source_sha
assert prior["scorer_freeze_sha256"]==sha(freeze)
assert prior["recovery_registry_sha256"]==sha(r1)
initial=float(old["charged_gpu_h"])
total=float(prior["charged_gpu_h"])
assert 0<initial<total<.02
assert abs(initial-old["attempts"][0]["wall_seconds"]/3600)<1e-7
assert abs(total-initial-prior["attempts"][0]["wall_seconds"]/3600)<1e-7
i2=base/"xp_a24_xpi2_prod_20261102T085625Z"/"XP_A24_XPI2_FINAL_STATUS.json"
i2d=json.loads(i2.read_text())
assert i2d["status"]=="A24_XPI2_COMPLETE" and i2d["n_completed_bags"]==100
assert abs(float(i2d["project_total_gpu_h"])+total-prior["project_gpu_h"])<1e-6
assert float(i2d["project_total_gpu_h"])+8<=400.
assert 'ALLOWED_PHYSICAL_GPUS=(1,2,3)' in w.read_text()
assert 'ALLOWED_PHYSICAL_GPUS=(1,2,3)' in c.read_text()
assert 'charged_gpu_h=previous_charge' in c.read_text()
assert 'from scripts.track_r.xp_t_daps4k_native_cpu import NATIVE,verify_clean_sparse' in w.read_text()
assert not r2.exists()
data=dict(
 status="A20_XPC3_ZERO_OUTPUT_R2_IMPORT_RECOVERY",
 original_execution_head=O,r1_execution_head=R,
 source_manifest_sha256=source_sha,
 original_freeze_sha256=sha(freeze),
 original_scorer_sha256=osp,
 r1_registration_sha256=sha(r1),
 r1_scorer_sha256=rsp,r1_controller_sha256=rcp,
 corrected_scorer_sha256=sha(w),corrected_controller_sha256=sha(c),
 failed_r0_final_sha256=sha(of),failed_r1_final_sha256=sha(rf),
 failed_r0_worker_log_sha256=sha(ol),failed_r1_worker_log_sha256=sha(rl),
 failed_total_gpu_h=total,failed_project_gpu_h=prior["project_gpu_h"],
 failed_candidate_outputs=0,
 allowed_physical_gpus=[1,2,3],disallowed_physical_gpus=[0],
 scientific_definition_unchanged=True,no_2forward_secondary=True,
 exact_only_S_mid_global2=True,original_stage_cap_gpu_h=8.,
 remaining_gpu_h=8.-total,project_cap_gpu_h=400.,
 timesteps=[300,400,500,600],draws=[0,1,2,3],
 checkpoint_sha256=spec["checkpoint_sha256"],
 scoring_not_performed_before_this_registration=True)
r2.write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n")
print("A20_R2_REGISTRATION_STAGED",r2)
print("A20_R2_PREVIOUS_FAILURES_TOTAL_GPU_H",total)
print("A20_R2_ORIGINAL_SOURCE_SHA",source_sha)
print("A20_R2_WORKER_SHA",sha(w),"CONTROLLER_SHA",sha(c))
PY

git add -- "$REC_REL"
test "$(git diff --cached --name-only)" = "$REC_REL"
git commit -m "freeze(track-r): register A20 XP-C3 zero-output R2 native import repair, GPU1-3 and both failed charges"
FROZEN_R2_HEAD="$(git rev-parse HEAD)"
git push origin "HEAD:refs/heads/$BRANCH"
git fetch --no-tags origin "$BRANCH"
test "$(git rev-parse FETCH_HEAD)" = "$FROZEN_R2_HEAD"
test -z "$(git status --porcelain)"
RUN="$BASE/xp_a25_xpc3_prod_r2_$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$BASE/XP_C3_A20_R2_SUPERVISOR_$(basename "$RUN").log"

# Read-only full ledger/controller preflight BEFORE detach.
"$PY" -u scripts/track_r/xp_a25_xpc3_autopilot.py \
    --expected-head "$FROZEN_R2_HEAD" \
    --run "$RUN" --branch "$BRANCH" --preflight-only

# All science/source/code recovery SHAs are pushed. Launch once.
nohup env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$WT" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    "$PY" -u scripts/track_r/xp_a25_xpc3_autopilot.py \
    --expected-head "$FROZEN_R2_HEAD" \
    --run "$RUN" --branch "$BRANCH" \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "A20_XPC3_R2_REGISTERED_HEAD=$FROZEN_R2_HEAD"
echo "A20_XPC3_R2_SUPERVISOR_PID=$PID"
echo "A20_XPC3_R2_RUN=$RUN"
echo "A20_XPC3_R2_LOG=$LOG"
echo "A20_XPC3_R2_ONLY_PHYSICAL_GPUS=1,2,3"
echo "A20_XPC3_R2_FROZEN_SOURCE_UNCHANGED=298b88cdd8f4880fe35fd727c6d7c874dabcd9d09da55cd40a469788d5d19b27"
echo "A20_XPC3_R2_OLD_A26_AND_R1_A27_UNTOUCHED"
