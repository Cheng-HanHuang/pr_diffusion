#!/usr/bin/env bash
# One-shot CP2 root provenance recovery after incorrectly validating G1a
# input-producer SHA instead of its successful GPU execution SHA.
# No root rewrite, no GT access before complete preflight, no GPU generation.
set -euo pipefail
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$ROOT/pr_diffusion_track_r_xp_a13"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
HEAD="$1"
OLD="538d94ecc7f79c605eed82855e918b80e97b190b"
MARK="$BASE/XP_A15_CP23_EXECUTION_HEAD_RECOVERY_APPLIED.txt"
[[ "$HEAD" =~ ^[0-9a-f]{40}$ ]] || { echo STOP_A15_HEAD_SHA40; exit 8; }
test -x "$PY"
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test ! -e "$MARK"
test -f "$BASE/XP_A15_CP23_LATEST_LOG.txt"
test -f "$BASE/XP_A15_CP23_LATEST_SESSION.txt"
test -f "$BASE/XP_A15_CP23_REGISTRY_RECOVERY_APPLIED.txt"
PREV_LOG=$(cat "$BASE/XP_A15_CP23_LATEST_LOG.txt")
PREV_SESSION=$(cat "$BASE/XP_A15_CP23_LATEST_SESSION.txt")
case "$PREV_LOG" in "$BASE"/XP_A15_CP23_*.log) ;; *) echo STOP_A15_PREVIOUS_LOG_PATH;exit 8;; esac
[[ "$PREV_SESSION" =~ ^prdiff_a15_cp23_[0-9]{8}T[0-9]{6}Z$ ]] || {
 echo STOP_A15_PREVIOUS_SESSION_NAME;exit 8;
}
test -f "$PREV_LOG"
grep -Fq "STOP_R05_ROOT_RECEIPT_NOT_MATCHED XP|FAST|imagenet|ILSVRC2012_val_00049000.JPEG|rep16" "$PREV_LOG"
grep -Fq "ACTION NEEDED: A15_CP2_D4" "$PREV_LOG"
if tmux has-session -t "$PREV_SESSION" 2>/dev/null;then
 echo STOP_A15_PREVIOUS_SESSION_ACTIVE;exit 8
fi
if pgrep -af '[x]p_a15_cp2_d4_cpu.py|[x]p_a15_cp2_audit_cpu.py|[x]p_a15_cp3_score_cpu.py|[x]p_a15_cp3_protocol_cpu.py|[x]p_a15_cp23_cpu_pipeline.sh'; then
 echo STOP_A15_CONCURRENT_SCORING;exit 8
fi
# Last failure occurred in preflight: no CP2 output should exist.
if find "$BASE" -maxdepth 1 -type d -name 'xp_a15_cp2_d4_*' -print -quit | grep -q .;then
 echo STOP_CP2_OUTPUT_ALREADY_EXISTS; exit 8
fi
export CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$WT" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
cd "$WT"
echo "===== VERIFY EXECUTOR SYNTAX AND STRICT PROVENANCE CONTRACT ====="
bash -n scripts/track_r/xp_a15_cp23_cpu_pipeline.sh
bash -n scripts/track_r/xp_a15_cp2_publish.sh
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a15_cp23.py' -v
"$PY" - <<'PY'
import json
from scripts.track_r.xp_a15_cp2_d4_cpu import (
 G1A_RUN,G1A_PREP,A12_G1A_FROZEN_PRODUCER,sha)
r=json.loads((G1A_RUN/"XP_A12_G1A_FINAL_STATUS.json").read_text())
p=G1A_PREP/"XP_A12_G1A_1600_TASKS.json"
m=json.loads(p.read_text())
prep=json.loads((G1A_PREP/"XP_A12_G1A_PREP_RECEIPT.json").read_text())
assert r["status"]=="XP_A12_G1A_1600_GENERATION_COMPLETE_SEALED"
assert m["head"]==prep["head"]==A12_G1A_FROZEN_PRODUCER
assert r["source_head"] != m["head"]
assert sha(p)==prep["production_manifest_sha256"]
print("CP2_G1A_PRODUCER_EXECUTOR_HEAD_DISTINCTION_PASS")
print("G1A_FROZEN_PRODUCER",m["head"])
print("G1A_GPU_EXECUTION_HEAD",r["source_head"])
PY
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$BASE/xp_a15_cp2_d4_$STAMP"
echo "===== COMPLETE CP2 NATIVE-ROOT/REGISTRY PREFLIGHT (CPU ONLY) ====="
"$PY" scripts/track_r/xp_a15_cp2_d4_cpu.py \
 --expected-head "$HEAD" --out "$RUN" --preflight-only
test ! -e "$RUN"
echo "===== START CHECKPOINT-ONLY CP2 THEN CP3 CPU ANALYSIS ====="
SESSION="prdiff_a15_cp23_$STAMP"
LOG="$BASE/XP_A15_CP23_$STAMP.log"
test ! -e "$LOG"
if tmux has-session -t "$SESSION" 2>/dev/null;then
 echo STOP_A15_SESSION_NAME_DUPLICATE;exit 8
fi
printf -v CMD '%q ' bash scripts/track_r/xp_a15_cp23_cpu_pipeline.sh "$HEAD" "$STAMP"
tmux new-session -d -s "$SESSION" \
 "cd '$WT'; env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$WT' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 $CMD > '$LOG' 2>&1"
# Preserve previous log and registry recovery; append separate immutable receipt.
{
 echo "status=A15_EXECUTION_HEAD_PROVENANCE_RECOVERY_LAUNCHED"
 echo "prior_executor_head=$OLD"
 echo "prior_log=$PREV_LOG"
 echo "prior_session=$PREV_SESSION"
 echo "corrected_executor_head=$HEAD"
 echo "new_log=$LOG"
 echo "new_session=$SESSION"
 echo "root_regeneration=false"
 echo "preflight_pass=true"
} > "$MARK"
printf '%s\n' "$SESSION" > "$BASE/XP_A15_CP23_LATEST_SESSION.txt"
printf '%s\n' "$LOG" > "$BASE/XP_A15_CP23_LATEST_LOG.txt"
echo "A15_EXECUTION_HEAD_RECOVERY_LAUNCHED"
echo "SESSION=$SESSION"
echo "LOG=$LOG"
echo "NO_PLANNER_RETURN_UNTIL_AUTHENTICATED_CP2_OR_CP3_CHECKPOINT"
