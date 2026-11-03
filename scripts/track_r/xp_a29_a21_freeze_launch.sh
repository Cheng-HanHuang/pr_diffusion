#!/usr/bin/env bash
# A21 prereg: ONE PAC block after branch fetch. Freeze/commit/push before
# pilot GPU work; independently start XP-I2 CPU scoring concurrently.
set -euo pipefail
[[ $# == 2 ]] || { echo "Usage: bash $0 SOURCE_HEAD BRANCH" >&2;exit 8; }
SOURCE_HEAD="$1";BRANCH="$2"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
DPY="$ROOT/conda-envs/daps/bin/python"
FREEZE="docs/track_r/executors/xp/XP_C3_A21_FREEZE.json"

export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$WT" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
cd "$WT"
test -z "$(git branch --show-current)"
test "$(git rev-parse HEAD)" = "$SOURCE_HEAD"
test -z "$(git status --porcelain)"
git fetch --no-tags origin "$BRANCH"
test "$(git rev-parse FETCH_HEAD)" = "$SOURCE_HEAD"
test ! -e "$FREEZE"

echo "=== Syntax compile all A21 scripts ==="
"$PY" - <<'PY'
import ast
from pathlib import Path
root=Path.cwd()
files=[
"scripts/track_r/xp_a29_a21_settings.py",
"scripts/track_r/xp_a29_a21_gpu_worker.py",
"scripts/track_r/xp_a29_a21_controller.py",
"scripts/track_r/xp_a29_a21_publish_cpu.py",
"scripts/track_r/xp_a29_a21_final_publish.py",
"scripts/track_r/xp_a29_xpi2_cpu_publish.py",
"tests/test_track_r_xp_a29_a21.py",
]
for name in files:
    ast.parse((root/name).read_text(),filename=name)
print("A21_ALL_SEVEN_PYTHON_SYNTAX_PASS",len(files),flush=True)
PY

echo "=== A21 CPU unit tests ==="
"$PY" -m unittest discover -s tests \
    -p 'test_track_r_xp_a29_a21.py' -v
"$PY" -m unittest discover -s tests \
    -p 'test_track_r_xp_a25_xpc3_a20.py' -v
"$PY" -m unittest discover -s tests \
    -p 'test_track_r_xp_a28_xpc3_recovery_r2.py' -v

echo "=== Native DAPS Python import boundary ==="
"$DPY" - <<'PY'
from scripts.track_r.xp_t_daps4k_native_cpu import NATIVE,verify_clean_sparse
from scripts.track_r.xp_t_daps4k_prepare_cpu import IMAGENET_MODEL_SHA
from scripts.track_r.xp_t_daps4k_worker_gpu import (
    load_model_native,model_config,validate_weight_trace)
from scripts.track_r.xp_a25_xpc3_mid_gpu import (
    orient_sm,fix_exact_native_timestep)
from scripts.track_r.xp_a29_a21_gpu_worker import (
    preflight,score_group,evaluate_image,claim,output_once)
from prdiffusion.track_r.prior_score import frozen_alpha_schedule,prior_seed
assert NATIVE.name=="DAPS_XP_NATIVE_4K"
assert len(IMAGENET_MODEL_SHA)==64 and callable(verify_clean_sparse)
assert callable(score_group) and callable(orient_sm)
assert callable(load_model_native) and callable(fix_exact_native_timestep)
assert len(frozen_alpha_schedule())==1000
print("A21_NATIVE_DAPS_PYTHON_IMPORT_PASS",flush=True)
PY

echo "=== Authoritative freeze BEFORE any A21 pilot ==="
"$PY" - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
from scripts.track_r import xp_a29_a21_settings as S
root=Path.cwd()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=root/"docs/track_r/executors/xp/theory_inputs/XP_C3_A20_SOURCE_MANIFEST.json"
r2=Path("/egr/research-pac/huang248/outputs/pr_diffusion/track_r")/S.R2_RUN
final=json.loads((r2/"XP_C3_A20_FINAL_STATUS.json").read_text())
smoke=json.loads((r2/"XP_N"/"000.json").read_text())
assert sha(source)==S.SOURCE_SHA
assert final["status"]=="A20_XPC3_COMPLETE_PANELS_BUDGET_STOP"
assert final["completed_panels"]==[]
assert 0<float(final["charged_gpu_h"])<1
assert smoke["status"]=="A20_XPC3_IMAGE_PRIOR_PASS"
assert smoke["source_manifest_sha256"]==S.SOURCE_SHA
assert len(smoke["rows"])==64
assert all("L000" in r and "L111" in r for r in smoke["rows"])
old=root/"docs/track_r/executors/xp/XP_C3_A20_CODE_FREEZE.json"
spec=json.loads(old.read_text())
assert spec["status"]=="A20_XPC3_OPTION_B_S_MID_GLOBAL2_ONLY"
assert spec["n_candidates"]==13600
assert spec["checkpoint_sha256"]==S.CHECKPOINT_SHA
assert spec["source_manifest_sha256"]==S.SOURCE_SHA
reg=root/"docs/track_r/executors/xp/XP_C3_A21_FREEZE.json"
paths=[
"scripts/track_r/xp_a29_a21_gpu_worker.py",
"scripts/track_r/xp_a29_a21_controller.py",
"scripts/track_r/xp_a29_a21_settings.py",
"scripts/track_r/xp_a29_a21_publish_cpu.py",
"scripts/track_r/xp_a29_a21_final_publish.py",
"scripts/track_r/xp_a29_xpi2_cpu_publish.py",
"tests/test_track_r_xp_a29_a21.py",
]
registry={
"status":"A21_XPC3_FROZEN_BEFORE_PILOT",
"planner_A21_head":"c5aacd7",
"source_sha256":S.SOURCE_SHA,
"original_A20_scoring_freeze_sha256":sha(old),
"R2_implementation_head":S.R2_HEAD,
"R2_final_receipt_sha256":sha(r2/"XP_C3_A20_FINAL_STATUS.json"),
"R2_64_smoke_sha256":sha(r2/"XP_N"/"000.json"),
"prior_science":"sole S_mid global2; 300,400,500,600 x j=0..3; paired original CPU noise; exact native t; margin L000-L111 choice 111 iff >1e-7",
"prior_sha256":sha(root/"prdiffusion/track_r/prior_score.py"),
"native_checkpoint_path":"/egr/research-pac/huang248/models/imagenet256.pt",
"model_sha256":S.CHECKPOINT_SHA,
"batch_options":list(S.BATCH_OPTIONS),
"allowed_gpus":list(S.GPU_IDS),
"max_per_gpu":S.MAX_WORKERS_PER_GPU,
"pilot_max_L_diff":S.EQUIV_MAX_LOSS_ABS,
"pilot_min_abs_margin_for_choice":S.EQUIV_CHOICE_EXCLUDE_SMALL_MARGIN,
"pilot_max_seconds":S.PILOT_MAX_WALL_SECONDS,
"pilot_max_gpu_occupancy_h":S.PILOT_MAX_OCCUPANCY_GPU_H,
"gpu_occupancy_cap_h":S.CAP_OCCUPANCY_GPU_H,
"project_gpu_hard_cap":S.PROJECT_CAP_GPU_H,
"headroom_multiplier":S.MEMORY_HEADROOM_FACTOR,
"required_extra_mib":S.MEMORY_ADMISSION_EXTRA_MIB,
"yield_if_free_below_mib":S.YIELD_BELOW_FREE_MIB,
"main_text_deadline_unix":S.MAIN_TEXT_DEADLINE,
"hard_deadline_unix":S.HARD_STOP,
"panel_order":list(S.PANELS),
"panel_sizes":S.SIZES,
"skip_before_first_chunk_if_unaffordable":True,
"one_time_image_chunks_atomic_claims":True,
"batch_per_single_t_only":True,
"no_precision_or_cuda_flag_or_compilation_change":True,
"equivalence_failure_fallback_original_R2":True,
"gpu_hour_unit":"per-GPU union of XP-C3 process lifetimes; summed over physical GPUs",
"prior_R2_charged_gpu_h":final["charged_gpu_h"],
"code_sha256":{name:sha(root/name) for name in paths},
}
for key,p in (
 ("worker_sha256",paths[0]),("controller_sha256",paths[1]),
 ("settings_sha256",paths[2])):
    registry[key]=registry["code_sha256"][p]
assert not reg.exists()
reg.write_text(json.dumps(registry,indent=2,sort_keys=True,allow_nan=False)+"\n")
print("A21_REGISTRATION_STAGED",reg)
print("A21_SOURCE_SHA",registry["source_sha256"])
print("A21_R2_PRIOR_CHARGED_GPU_H",registry["prior_R2_charged_gpu_h"])
print("A21_PRE_PILOT_CODE_HASHES",registry["code_sha256"])
PY

git add -- "$FREEZE"
test "$(git diff --cached --name-only)" = "$FREEZE"
git commit -m "freeze(track-r): A21 56 occupancy GPU-h, fixed-t B8/B32 pilot, 4-GPU worker and XP-I2 CPU contracts"
FROZEN_HEAD="$(git rev-parse HEAD)"
git push origin "HEAD:refs/heads/$BRANCH"
git fetch --no-tags origin "$BRANCH"
test "$(git rev-parse FETCH_HEAD)" = "$FROZEN_HEAD"
test -z "$(git status --porcelain)"
echo "A21_FROZEN_AND_PUSHED_HEAD=$FROZEN_HEAD"

echo "=== Verified A21 controller CPU preflight ==="
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="$BASE/xp_a25_xpc3_prod_a21_$STAMP"
LOG="$BASE/XP_C3_A21_SUPERVISOR_$STAMP.log"
"$PY" -u scripts/track_r/xp_a29_a21_controller.py \
    --expected-head "$FROZEN_HEAD" --run "$RUN" --preflight-only

echo "=== Preflight XP-I2 native bags and D1 source before launch ==="
I2RUN="$BASE/xp_a29_xpi2_score_$STAMP"
I2LOG="$BASE/XP_I2_A15_5_CPU_$STAMP.log"
"$PY" -u scripts/track_r/xp_a29_xpi2_cpu_publish.py \
    --expected-head "$FROZEN_HEAD" --run "$I2RUN" --preflight-only
echo "=== Start XP-I2 CPU A15.5 independently ==="
nohup env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$WT" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    "$PY" -u scripts/track_r/xp_a29_xpi2_cpu_publish.py \
    --expected-head "$FROZEN_HEAD" --run "$I2RUN" \
    > "$I2LOG" 2>&1 < /dev/null &
I2PID=$!
echo "A21_XPI2_CPU_PID=$I2PID"
echo "A21_XPI2_CPU_LOG=$I2LOG"

echo "=== Start registered A21 GPU pilot + controller ==="
nohup env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$WT" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    "$PY" -u scripts/track_r/xp_a29_a21_controller.py \
    --expected-head "$FROZEN_HEAD" --run "$RUN" \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "A21_XPC3_GPU_CONTROLLER_PID=$PID"
echo "A21_XPC3_GPU_RUN=$RUN"
echo "A21_XPC3_GPU_LOG=$LOG"
echo "A21_ORIGINAL_A26_R1_A27_R2_A28_UNTOUCHED"
echo "A21_SCIENCE_MANIFEST_IMMUTABLE=298b88cdd8f4880fe35fd727c6d7c874dabcd9d09da55cd40a469788d5d19b27"
