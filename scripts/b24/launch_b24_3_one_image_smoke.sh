#!/usr/bin/env bash
set -euo pipefail

ROOT=/egr/research-pac/huang248
CONTROL="$ROOT/pr_diffusion_b23"
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
BRANCH=codex/b24-bestof4-failure-sweep
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
DAPS_PY="$ROOT/conda-envs/daps/bin/python"
ROLE_PTR="$OUTROOT/B24_METHOD_STAGE_LATEST_FREEZE.txt"
PILOT_SHA=124d3759e4fd540d2e870618dde59ff73d02cbb798d773a785296da5b140e98a
INITIAL_SMOKE_MIN_FREE_MIB=52096
GPU="${B24_GPU:-0}"

case "$GPU" in
  0) EXPECTED_UUID=GPU-8c9c6250-7b65-20d8-5c81-d6cb618810c3 ;;
  1) EXPECTED_UUID=GPU-883c037a-34d2-48c4-467f-9a352fd8fdff ;;
  2) EXPECTED_UUID=GPU-c381c0f4-1dbc-004f-7d3a-1d7f7794dffe ;;
  3) EXPECTED_UUID=GPU-7d65c050-d7e8-5a6b-ee38-1d72d7a5696a ;;
  *) echo "STOP|invalid_B24_GPU:$GPU"; exit 2 ;;
esac

[[ -x "$PY" ]] || { echo "STOP|missing_python:$PY"; exit 2; }
[[ -x "$DAPS_PY" ]] || { echo "STOP|missing_daps_python:$DAPS_PY"; exit 2; }
[[ -f "$ROLE_PTR" ]] || { echo "STOP|missing_role_pointer:$ROLE_PTR"; exit 2; }
ROLEDIR=$(cat "$ROLE_PTR")
PILOT="$ROLEDIR/B24_METHOD_PILOT16.csv"
SUMMARY="$ROLEDIR/B24_METHOD_ROLE_FREEZE_SUMMARY.json"
[[ -f "$PILOT" ]] || { echo "STOP|missing_pilot:$PILOT"; exit 2; }
[[ -f "$SUMMARY" ]] || { echo "STOP|missing_role_summary:$SUMMARY"; exit 2; }
OBS_SHA=$(sha256sum "$PILOT" | awk '{print $1}')
[[ "$OBS_SHA" == "$PILOT_SHA" ]] || { echo "STOP|pilot_sha_mismatch:$OBS_SHA"; exit 2; }

UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
[[ "$UUID" == "$EXPECTED_UUID" ]] || { echo "STOP|gpu_uuid_mismatch|gpu=$GPU|observed=$UUID|expected=$EXPECTED_UUID"; exit 2; }
FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')
[[ "$FREE" =~ ^[0-9]+$ ]] || { echo "STOP|cannot_parse_gpu_free:$FREE"; exit 2; }
if (( FREE < INITIAL_SMOKE_MIN_FREE_MIB )); then
  echo "STOP|one_image_smoke_needs_initial_free_mib=$INITIAL_SMOKE_MIN_FREE_MIB|gpu=$GPU|observed_free_mib=$FREE"
  exit 3
fi

[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  echo "STOP|B24_worktree_dirty"; git -C "$REPO" status --short; exit 4;
}
git -C "$CONTROL" fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
LOCAL=$(git -C "$REPO" rev-parse HEAD)
REMOTE=$(git -C "$REPO" rev-parse "origin/$BRANCH")
git -C "$REPO" merge-base --is-ancestor "$LOCAL" "$REMOTE" || {
  echo "STOP|local_not_ancestor|local=$LOCAL|remote=$REMOTE"; exit 4;
}
if [[ "$LOCAL" != "$REMOTE" ]]; then git -C "$REPO" merge --ff-only "origin/$BRANCH"; fi
HEAD=$(git -C "$REPO" rev-parse HEAD)

cd "$REPO"
export PYTHONPATH="$REPO:$REPO/scripts/b22${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
CUDA_VISIBLE_DEVICES="" "$PY" -m py_compile scripts/b24/run_b24_3_np_branching.py scripts/b24/test_b24_3_np_branching.py
CUDA_VISIBLE_DEVICES="" "$PY" scripts/b24/test_b24_3_np_branching.py

echo "B24_3_ZERO_GPU_TESTS_PASS|head=$HEAD"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$OUTROOT/B24_3_one_image_smoke_${STAMP}"
[[ ! -e "$RUN" ]] || { echo "STOP|run_exists:$RUN"; exit 5; }
mkdir -p "$RUN"

"$PY" - "$PILOT" "$RUN/role_row.csv" > "$RUN/role.env" <<'PY'
import csv, pathlib, sys
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
with src.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
if len(rows)!=16: raise SystemExit(f"expected 16 pilot rows, got {len(rows)}")
row=rows[0]
if row['method_role']!='DEVELOPMENT' or row['pilot16']!='TRUE': raise SystemExit('first pilot row identity mismatch')
with out.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(row)); w.writeheader(); w.writerow(row)
print(f"IMAGE_ID={row['image_id']}")
print(f"MEASUREMENT_SEED={row['dev_measurement_seed']}")
print(f"CLASS_LABEL={row['class_label']}")
PY
# role.env is generated from frozen numeric/alphanumeric fields only.
# shellcheck disable=SC1090
source "$RUN/role.env"

cat > "$RUN/LAUNCH_IDENTITY.txt" <<EOF
head=$HEAD
physical_gpu=$GPU
gpu_uuid=$UUID
initial_free_mib=$FREE
initial_smoke_min_free_mib=$INITIAL_SMOKE_MIN_FREE_MIB
pilot_csv=$PILOT
pilot_csv_sha256=$PILOT_SHA
image_id=$IMAGE_ID
class_label=$CLASS_LABEL
development_measurement_seed=$MEASUREMENT_SEED
EOF

CUDA_VISIBLE_DEVICES="$GPU" "$DAPS_PY" scripts/b24/generate_b24_locked_input.py \
  --image-id "$IMAGE_ID" \
  --measurement-seed "$MEASUREMENT_SEED" \
  --output "$RUN/input" \
  > "$RUN/input_generation.log" 2>&1

[[ -f "$RUN/input/input_manifest.json" ]] || { echo "STOP|input_generation_missing_manifest"; exit 6; }

nohup env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$PYTHONPATH" \
  PYTHONDONTWRITEBYTECODE=1 \
  "$PY" scripts/b24/run_b24_3_np_branching.py \
    --input-manifest "$RUN/input/input_manifest.json" \
    --role-row "$RUN/role_row.csv" \
    --output-root "$RUN/methods" \
    --physical-gpu "$GPU" \
  > "$RUN/run.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$RUN/runner.pid"
printf '%s\n' "$RUN" > "$OUTROOT/B24_3_ONE_IMAGE_LATEST_RUN.txt"

echo "B24_3_ONE_IMAGE_LAUNCHED|run=$RUN|pid=$PID|gpu=$GPU|uuid=$UUID|image=$IMAGE_ID|class=$CLASS_LABEL|head=$HEAD"
echo "STATUS|bash $REPO/scripts/b24/status_b24_3_one_image_smoke.sh"
