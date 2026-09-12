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
GPU=2
EXPECTED_UUID=GPU-c381c0f4-1dbc-004f-7d3a-1d7f7794dffe
MIN_FREE=52096
FROZEN_MAN64_FILE_SHA=23841774c7364b980e3964baed1605aaa8b6eace0370fd765a7369640ea962d7
export TORCH_HOME="$ROOT/models/torch_cache"

RUNROOT=${1:-"$OUTROOT/B24_2_64_20260826T040303Z"}
MANIFEST="$RUNROOT/B24_2_baseline_64.json"
SHARDROOT="$RUNROOT/shard2"

[[ -x "$PY" ]] || { echo "STOP|missing_python:$PY"; exit 2; }
[[ -d "$RUNROOT" ]] || { echo "STOP|missing_runroot:$RUNROOT"; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "STOP|missing_manifest:$MANIFEST"; exit 2; }
[[ -d "$SHARDROOT" ]] || { echo "STOP|missing_existing_shard2:$SHARDROOT"; exit 2; }
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || { echo "STOP|B24_worktree_dirty"; git -C "$REPO" status --short; exit 3; }

git -C "$CONTROL" fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
LOCAL=$(git -C "$REPO" rev-parse HEAD)
REMOTE=$(git -C "$REPO" rev-parse "origin/$BRANCH")
git -C "$REPO" merge-base --is-ancestor "$LOCAL" "$REMOTE" || {
  echo "STOP|local_not_ancestor|local=$LOCAL|remote=$REMOTE"; exit 4;
}
if [[ "$LOCAL" != "$REMOTE" ]]; then
  git -C "$REPO" merge --ff-only "origin/$BRANCH"
fi
HEAD=$(git -C "$REPO" rev-parse HEAD)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

"$PY" - <<'PY'
from pathlib import Path
import prdiffusion.b24_protocol as p
expected = Path("/egr/research-pac/huang248/pr_diffusion_b24/prdiffusion/b24_protocol.py").resolve()
observed = Path(p.__file__).resolve()
assert observed == expected, (observed, expected)
print(f"B24_IMPORT_READY|protocol={observed}")
PY
"$PY" -m py_compile scripts/b24/generate_b24_locked_input.py scripts/b24/run_b24_2_shard.py
"$PY" -m unittest discover -s tests/b24 -p 'test_*.py'
echo "B24_TESTS_READY|suite=tests/b24"

MAN_FILE_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')
[[ "$MAN_FILE_SHA" == "$FROZEN_MAN64_FILE_SHA" ]] || {
  echo "STOP|manifest_sha|observed=$MAN_FILE_SHA|expected=$FROZEN_MAN64_FILE_SHA"; exit 10;
}

verify_source() {
  local name="$1" path="$2" exp_head="$3" exp_tree="$4" exp_index="$5" exp_diff="$6"
  local head tree index_digest diff_digest
  head=$(git -C "$path" rev-parse HEAD)
  tree=$(git -C "$path" rev-parse 'HEAD^{tree}')
  index_digest=$(git -C "$path" ls-files -s | sha256sum | awk '{print $1}')
  diff_digest=$(git -C "$path" diff --binary HEAD -- . | sha256sum | awk '{print $1}')
  [[ "$head" == "$exp_head" ]] || { echo "STOP|source_head|name=$name"; return 21; }
  [[ "$tree" == "$exp_tree" ]] || { echo "STOP|source_tree|name=$name"; return 22; }
  [[ "$index_digest" == "$exp_index" ]] || { echo "STOP|source_index|name=$name"; return 23; }
  [[ "$diff_digest" == "$exp_diff" ]] || { echo "STOP|source_diff|name=$name"; return 24; }
  echo "SOURCE_READY|name=$name|head=$head|tree=$tree"
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

# The other three shards are accepted and must remain untouched.
"$PY" - "$RUNROOT" "$MAN_FILE_SHA" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); sha=sys.argv[2]
for shard in (0,1,3):
    p=root/f"shard{shard}/SHARD_COMPLETE.json"
    if not p.is_file(): raise SystemExit(f"STOP|missing_completed_shard={shard}|path={p}")
    v=json.loads(p.read_text())
    assert v["status"] == "PASS"
    assert v["completed"] == 16 and v["row_count"] == 16
    assert v["manifest_file_sha256"] == sha
print("OTHER_SHARDS_FROZEN|shards=0,1,3|completed=48")
PY

# The previous shard-2 process must be dead before reusing its output root.
OLD_PID_FILE="$RUNROOT/pids/gpu2.pid"
if [[ -f "$OLD_PID_FILE" ]]; then
  OLD_PID=$(cat "$OLD_PID_FILE")
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "STOP|old_gpu2_pid_still_running|pid=$OLD_PID"; exit 30
  fi
else
  OLD_PID=-
fi

ROW_STATE=$("$PY" - "$SHARDROOT" "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]); manifest=json.loads(Path(sys.argv[2]).read_text())
rows=[r for r in manifest["rows"] if int(r["shard_id"]) == 2]
complete=[]
for r in rows:
    d=root/f"row{int(r['row_index']):03d}_{str(r['image_id']).zfill(5)}"
    p=d/"IMAGE_COMPLETE.json"
    if p.is_file():
        v=json.loads(p.read_text())
        assert v["status"] == "PASS"
        assert v["row_index"] == int(r["row_index"])
        assert v["image_id"] == str(r["image_id"]).zfill(5)
        assert v["measurement_seed"] == int(r["measurement_seed"])
        assert v["daps_solver_seeds"] == [int(x) for x in r["daps_solver_seeds"]]
        assert v["sitcom_solver_seeds"] == [int(x) for x in r["sitcom_solver_seeds"]]
        complete.append(int(r["row_index"]))
print(",".join(map(str, complete)))
PY
)
COUNT=$(awk -F',' '{if ($0=="") print 0; else print NF}' <<< "$ROW_STATE")
if (( COUNT >= 16 )); then
  echo "SHARD2_ALREADY_COMPLETE|rows=$ROW_STATE"
  exit 0
fi
echo "SHARD2_RESUME_READY|reusable_count=$COUNT|rows=$ROW_STATE|remaining=$((16-COUNT))"

GPU_ROW=$(nvidia-smi --id="$GPU" --query-gpu=uuid,memory.free --format=csv,noheader,nounits)
IFS=',' read -r UUID FREE <<< "$GPU_ROW"; UUID="${UUID// /}"; FREE="${FREE// /}"
[[ "$UUID" == "$EXPECTED_UUID" ]] || { echo "STOP|gpu_uuid|observed=$UUID|expected=$EXPECTED_UUID"; exit 31; }
(( FREE >= MIN_FREE )) || { echo "STOP|gpu_free|free_mib=$FREE|required_mib=$MIN_FREE"; exit 32; }
echo "GPU_READY|gpu=2|uuid=$UUID|free_mib=$FREE"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUNROOT/logs" "$RUNROOT/pids" "$RUNROOT/resume"
if [[ -f "$RUNROOT/logs/gpu2.log" ]]; then
  mv "$RUNROOT/logs/gpu2.log" "$RUNROOT/logs/gpu2.pre_resume_$STAMP.log"
fi
if [[ -f "$OLD_PID_FILE" ]]; then
  mv "$OLD_PID_FILE" "$RUNROOT/pids/gpu2.pre_resume_$STAMP.pid"
fi
cat > "$RUNROOT/resume/SHARD2_RESUME_$STAMP.json" <<EOF
{
  "stage": "B24.2_64",
  "reason": "sub1000_ffhq_lookup_duplicate_path_bug",
  "runroot": "$RUNROOT",
  "manifest_file_sha256": "$MAN_FILE_SHA",
  "original_launch_head": "f5c4ae65ad5a18285bce8b8428c098616807b26f",
  "resume_head": "$HEAD",
  "gpu_id": 2,
  "gpu_uuid": "$UUID",
  "old_pid": "$OLD_PID",
  "reusable_completed_before_resume": $COUNT,
  "reusable_row_indices": "$ROW_STATE"
}
EOF

nohup env PYTHONPATH="$PYTHONPATH" PYTHONDONTWRITEBYTECODE=1 TORCH_HOME="$TORCH_HOME" \
  "$PY" "$REPO/scripts/b24/run_b24_2_shard.py" \
    --manifest "$MANIFEST" --shard 2 --gpu 2 --repo "$REPO" \
    --output-root "$SHARDROOT" --resume \
  > "$RUNROOT/logs/gpu2.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$RUNROOT/pids/gpu2.pid"

echo "B24_2_64_SHARD2_RESUMED|runroot=$RUNROOT|pid=$PID|reused=$COUNT|remaining=$((16-COUNT))|head=$HEAD"
echo "STATUS|bash $REPO/scripts/b24/status_b24_2_64.sh $RUNROOT"
