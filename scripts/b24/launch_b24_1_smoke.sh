#!/usr/bin/env bash
set -euo pipefail

ROOT=/egr/research-pac/huang248
CONTROL="$ROOT/pr_diffusion_b23"
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
BRANCH=codex/b24-bestof4-failure-sweep
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
DAPS="$ROOT/pr_diffusion_b19_solver/external/daps"
SITCOM="$ROOT/external/SITCOM_ODE"
DAPS_GPU=0
SITCOM_GPU=1
MIN_FREE=52096

[[ -x "$PY" ]] || { echo "STOP|missing control python:$PY"; exit 2; }
[[ -d "$REPO/.git" || -f "$REPO/.git" ]] || { echo "STOP|missing B24 worktree:$REPO"; exit 2; }
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || { echo "STOP|B24 worktree dirty"; git -C "$REPO" status --short; exit 3; }

git -C "$CONTROL" fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
LOCAL=$(git -C "$REPO" rev-parse HEAD)
REMOTE=$(git -C "$REPO" rev-parse "origin/$BRANCH")
git -C "$REPO" merge-base --is-ancestor "$LOCAL" "$REMOTE" || {
  echo "STOP|local B24 head is not ancestor of remote|local=$LOCAL|remote=$REMOTE"; exit 4;
}
if [[ "$LOCAL" != "$REMOTE" ]]; then
  git -C "$REPO" merge --ff-only "origin/$BRANCH"
fi
HEAD=$(git -C "$REPO" rev-parse HEAD)
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || { echo "STOP|B24 worktree dirty after fast-forward"; exit 5; }

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m py_compile \
  "$REPO/scripts/b24/run_b24_1_method_smoke.py" \
  "$REPO/scripts/b24/analyze_b24_1_smoke.py"

verify_source() {
  local name="$1" path="$2" exp_head="$3" exp_tree="$4" exp_index="$5" exp_diff="$6"
  local head tree index_digest diff_digest
  [[ -d "$path/.git" || -f "$path/.git" ]] || { echo "STOP|source_missing|name=$name|path=$path"; return 20; }
  head=$(git -C "$path" rev-parse HEAD)
  tree=$(git -C "$path" rev-parse 'HEAD^{tree}')
  index_digest=$(git -C "$path" ls-files -s | sha256sum | awk '{print $1}')
  diff_digest=$(git -C "$path" diff --binary HEAD -- . | sha256sum | awk '{print $1}')
  [[ "$head" == "$exp_head" ]] || { echo "STOP|source_head|name=$name|observed=$head|expected=$exp_head"; return 21; }
  [[ "$tree" == "$exp_tree" ]] || { echo "STOP|source_tree|name=$name|observed=$tree|expected=$exp_tree"; return 22; }
  [[ "$index_digest" == "$exp_index" ]] || { echo "STOP|source_index|name=$name|observed=$index_digest|expected=$exp_index"; return 23; }
  [[ "$diff_digest" == "$exp_diff" ]] || { echo "STOP|source_tracked_diff|name=$name|observed=$diff_digest|expected=$exp_diff"; return 24; }
  echo "SOURCE_READY|name=$name|head=$head|tree=$tree|index_sha256=$index_digest|tracked_diff_sha256=$diff_digest"
}

verify_source DAPS "$DAPS" \
  e7a77d094167084faed19b599b96673b7bb11447 \
  e63f9715e4704d9cd7a43a166559496d9d94e781 \
  d5487cdba570dbaac0c1909e549da361a0a0fc3fed81e5c13f59fa12925876b6 \
  fbb5b42369ecf0d3b9b67f8fc162053bc40ec32aed41dbd92a67e8d81dcfad69
verify_source SITCOM "$SITCOM" \
  275ab67efbd8146bffca20155171ba6be1169c09 \
  80263442e3606824a06dc003504c28da5c59c2c5 \
  3ef63a8a29d0ba65cc642027a57ec102257fd9b387b0e9a5b4aae7f46d6a949f \
  a9f0076d6f852b6898000142c19a09131ffc49ceba0e3d935cd465e85df26e6e

# B23.1 generated the five locked inputs in an earlier corrective run and then
# reused them for the accepted scientific execution. Resolve that retained root
# from the signed B23 execution contract instead of guessing a run timestamp.
B23_INPUT_ROOT=$("$PY" - "$REPO/configs/b23/b23_1a_b_execution.yaml" <<'PY'
import json
import sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(cfg["execution"]["required_reuse_input_root"])
PY
)
B23_INPUTS="$B23_INPUT_ROOT/INPUTS.json"
[[ -f "$B23_INPUTS" ]] || {
  echo "STOP|signed_b23_inputs_missing|root=$B23_INPUT_ROOT|inputs=$B23_INPUTS"; exit 6;
}

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT="$OUTROOT/B24_1_smoke_$STAMP"
mkdir -p "$RUNROOT/logs" "$RUNROOT/pids"
printf '%s\n' "$HEAD" > "$RUNROOT/B24_HEAD.txt"
printf '%s\n' "$RUNROOT" > "$OUTROOT/B24_1_LATEST_RUN.txt"

# Fail closed on the exact five-row B23.1 locked-input identity. CUDA is hidden:
# this validates existing files/tensor hashes only and performs no generation.
CUDA_VISIBLE_DEVICES="" "$PY" "$REPO/scripts/b23/prepare_b23_1_inputs.py" \
  --repo "$REPO" \
  --validate-existing "$B23_INPUT_ROOT" \
  --output-json "$RUNROOT/B23_INPUT_VALIDATION.json" \
  > "$RUNROOT/logs/b23_input_validation.log" 2>&1 || {
    echo "STOP|signed_b23_input_validation_failed|log=$RUNROOT/logs/b23_input_validation.log"
    tail -80 "$RUNROOT/logs/b23_input_validation.log" || true
    exit 7
  }
INPUTS_SHA=$(sha256sum "$B23_INPUTS" | awk '{print $1}')
echo "B23_INPUTS_VALIDATED|root=$B23_INPUT_ROOT|sha256=$INPUTS_SHA"

check_gpu() {
  local gpu="$1" expected="$2"
  local row uuid free
  row=$(nvidia-smi --id="$gpu" --query-gpu=uuid,memory.free --format=csv,noheader,nounits)
  IFS=',' read -r uuid free <<< "$row"
  uuid="${uuid// /}"
  free="${free// /}"
  [[ "$uuid" == "$expected" ]] || { echo "STOP|gpu_uuid|gpu=$gpu|observed=$uuid|expected=$expected"; return 10; }
  (( free >= MIN_FREE )) || { echo "STOP|gpu_free|gpu=$gpu|free_mib=$free|required_mib=$MIN_FREE"; return 11; }
  echo "GPU_READY|gpu=$gpu|uuid=$uuid|free_mib=$free"
}

check_gpu 0 GPU-8c9c6250-7b65-20d8-5c81-d6cb618810c3
check_gpu 1 GPU-883c037a-34d2-48c4-467f-9a352fd8fdff

nohup env PYTHONPATH="$PYTHONPATH" "$PY" "$REPO/scripts/b24/run_b24_1_method_smoke.py" \
  --method DAPS --gpu "$DAPS_GPU" --image-id 65082 --repo "$REPO" --inputs "$B23_INPUTS" --output "$RUNROOT/daps" \
  > "$RUNROOT/logs/daps.log" 2>&1 &
DAPS_PID=$!
printf '%s\n' "$DAPS_PID" > "$RUNROOT/pids/daps.pid"

nohup env PYTHONPATH="$PYTHONPATH" "$PY" "$REPO/scripts/b24/run_b24_1_method_smoke.py" \
  --method SITCOM --gpu "$SITCOM_GPU" --image-id 65082 --repo "$REPO" --inputs "$B23_INPUTS" --output "$RUNROOT/sitcom" \
  > "$RUNROOT/logs/sitcom.log" 2>&1 &
SITCOM_PID=$!
printf '%s\n' "$SITCOM_PID" > "$RUNROOT/pids/sitcom.pid"

cat > "$RUNROOT/LAUNCH.json" <<EOF
{
  "stage": "B24.1",
  "b24_head": "$HEAD",
  "image_id": "65082",
  "accepted_b23_input_root": "$B23_INPUT_ROOT",
  "accepted_b23_inputs_path": "$B23_INPUTS",
  "accepted_b23_inputs_sha256": "$INPUTS_SHA",
  "daps_gpu": $DAPS_GPU,
  "daps_pid": $DAPS_PID,
  "sitcom_gpu": $SITCOM_GPU,
  "sitcom_pid": $SITCOM_PID,
  "hard_ceiling_mib": 52452,
  "normal_target_mib": 48000,
  "minimum_free_before_launch_mib": 52096
}
EOF

echo "B24_1_LAUNCHED|runroot=$RUNROOT|head=$HEAD|daps_gpu=$DAPS_GPU|daps_pid=$DAPS_PID|sitcom_gpu=$SITCOM_GPU|sitcom_pid=$SITCOM_PID"
echo "STATUS|bash $REPO/scripts/b24/status_b24_1_smoke.sh $RUNROOT"
