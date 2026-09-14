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
PARENT_POINTER="$OUTROOT/B24_2_2048_LATEST_RUN.txt"
BASELINE_MIN_FREE_MIB=10240
export B24_BASELINE_MIN_FREE_MIB="$BASELINE_MIN_FREE_MIB"
export TORCH_HOME="$ROOT/models/torch_cache"
mkdir -p "$TORCH_HOME"

[[ -x "$PY" ]] || { echo "STOP|missing_python:$PY"; exit 2; }
[[ -f "$PARENT_POINTER" ]] || { echo "STOP|missing_parent2048_pointer:$PARENT_POINTER"; exit 2; }
PARENT2048=$(cat "$PARENT_POINTER")
PARENT_MANIFEST="$PARENT2048/B24_2_baseline_2048.json"
[[ -d "$PARENT2048" ]] || { echo "STOP|missing_parent2048:$PARENT2048"; exit 2; }
[[ -f "$PARENT_MANIFEST" ]] || { echo "STOP|missing_parent2048_manifest:$PARENT_MANIFEST"; exit 2; }
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  echo "STOP|B24_worktree_dirty"; git -C "$REPO" status --short; exit 3;
}

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
obs=Path(p.__file__).resolve()
exp=Path('/egr/research-pac/huang248/pr_diffusion_b24/prdiffusion/b24_protocol.py').resolve()
assert obs == exp, (obs, exp)
print(f'B24_IMPORT_READY|protocol={obs}')
PY

"$PY" -m py_compile \
  scripts/b24/render_b24_baseline_manifest.py \
  scripts/b24/run_b24_2_6144_extension_shard.py \
  scripts/b24/generate_b24_locked_input.py \
  scripts/b24/evaluate_b24_baseline_image.py
"$PY" -m unittest discover -s tests/b24 -p 'test_*.py'
echo "B24_TESTS_READY|suite=tests/b24"

verify_source() {
  local name="$1" path="$2" exp_head="$3" exp_tree="$4" exp_index="$5" exp_diff="$6"
  local head tree index_digest diff_digest
  head=$(git -C "$path" rev-parse HEAD)
  tree=$(git -C "$path" rev-parse 'HEAD^{tree}')
  index_digest=$(git -C "$path" ls-files -s | sha256sum | awk '{print $1}')
  diff_digest=$(git -C "$path" diff --binary HEAD -- . | sha256sum | awk '{print $1}')
  [[ "$head" == "$exp_head" ]] || { echo "STOP|source_head|name=$name|observed=$head|expected=$exp_head"; return 21; }
  [[ "$tree" == "$exp_tree" ]] || { echo "STOP|source_tree|name=$name|observed=$tree|expected=$exp_tree"; return 22; }
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

PARENT_SHA=$(sha256sum "$PARENT_MANIFEST" | awk '{print $1}')
"$PY" - "$PARENT2048" "$PARENT_MANIFEST" "$PARENT_SHA" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); mp=Path(sys.argv[2]); sha=sys.argv[3]
v=json.loads(mp.read_text())
assert len(v['rows']) == 2048
assert [int(r['row_index']) for r in v['rows']] == list(range(2048))
for g in range(4):
    sp=root/f'shard{g}/SHARD_COMPLETE.json'
    assert sp.is_file(), sp
    s=json.loads(sp.read_text())
    assert s.get('stage') == 'B24.2_2048_EXTENSION', (g,s)
    assert s.get('status') == 'PASS', (g,s)
    assert int(s.get('completed',-1)) == 448, (g,s)
    assert s.get('manifest_file_sha256') == sha, (g,s.get('manifest_file_sha256'),sha)
    actual=sum(1 for _ in (root/f'shard{g}').rglob('IMAGE_COMPLETE.json'))
    assert actual == 448, (g,actual)
print(f'PARENT2048_READY|path={root}|manifest_sha256={sha}|extension_rows=1792|shards=4x448')
PY

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT="$OUTROOT/B24_2_6144_extension_$STAMP"
mkdir -p "$RUNROOT/logs" "$RUNROOT/pids"
MANIFEST="$RUNROOT/B24_2_baseline_6144.json"
"$PY" scripts/b24/render_b24_baseline_manifest.py --count 6144 --out "$MANIFEST" \
  >"$RUNROOT/logs/render_manifest.log" 2>&1
MAN_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')

"$PY" - "$PARENT_MANIFEST" "$MANIFEST" <<'PY'
import json,sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text())
q=json.loads(Path(sys.argv[2]).read_text())
assert len(p['rows']) == 2048 and len(q['rows']) == 6144
assert q['rows'][:2048] == p['rows']
assert [int(r['row_index']) for r in q['rows']] == list(range(6144))
for g in range(4):
    rows=[r for r in q['rows'] if int(r['gpu_id'])==g and int(r['row_index'])>=2048]
    assert len(rows)==1024, (g,len(rows))
print('PREFIX_READY|parent=2048|target=6144|new_rows=4096|rows_per_gpu=1024')
PY

echo "MANIFEST_READY|count=6144|file_sha256=$MAN_SHA|parent2048_sha256=$PARENT_SHA"

UUIDS=(
  GPU-8c9c6250-7b65-20d8-5c81-d6cb618810c3
  GPU-883c037a-34d2-48c4-467f-9a352fd8fdff
  GPU-c381c0f4-1dbc-004f-7d3a-1d7f7794dffe
  GPU-7d65c050-d7e8-5a6b-ee38-1d72d7a5696a
)
for GPU in 0 1 2 3; do
  ROW=$(nvidia-smi --id="$GPU" --query-gpu=uuid,memory.free --format=csv,noheader,nounits)
  IFS=',' read -r U F <<< "$ROW"; U="${U// /}"; F="${F// /}"
  [[ "$U" == "${UUIDS[$GPU]}" ]] || {
    echo "STOP|gpu_uuid|gpu=$GPU|observed=$U|expected=${UUIDS[$GPU]}"; exit 30;
  }
  echo "GPU_OBSERVED|gpu=$GPU|uuid=$U|free_mib=$F|runtime_fit_gate_mib=$BASELINE_MIN_FREE_MIB"
done

cat >"$RUNROOT/LAUNCH.json" <<EOF
{
  "stage": "B24.2_6144_EXTENSION",
  "b24_head": "$HEAD",
  "parent2048_runroot": "$PARENT2048",
  "parent2048_manifest_path": "$PARENT_MANIFEST",
  "parent2048_manifest_file_sha256": "$PARENT_SHA",
  "manifest_path": "$MANIFEST",
  "manifest_file_sha256": "$MAN_SHA",
  "target_count": 6144,
  "new_row_range": [2048, 6143],
  "new_rows": 4096,
  "rows_per_gpu": 1024,
  "gating_collection_classes": ["B", "C"],
  "d_retained_but_not_gating": true,
  "per_method_concurrency": 4,
  "baseline_min_free_mib": 10240,
  "hard_ceiling_mib": 52452,
  "gpu_fit_behavior": "WAIT_AND_RETRY_CALIBRATED_GATE",
  "parent_requirement": "ALL_2048_SHARDS_PASS_BEFORE_LAUNCH"
}
EOF

for GPU in 0 1 2 3; do
  nohup env \
    PYTHONPATH="$PYTHONPATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME="$TORCH_HOME" \
    B24_BASELINE_MIN_FREE_MIB="$BASELINE_MIN_FREE_MIB" \
    "$PY" scripts/b24/run_b24_2_6144_extension_shard.py \
      --manifest "$MANIFEST" \
      --parent-runroot "$PARENT2048" \
      --shard "$GPU" --gpu "$GPU" --repo "$REPO" \
      --output-root "$RUNROOT/shard${GPU}" \
    >"$RUNROOT/logs/gpu${GPU}.log" 2>&1 &
  PID=$!
  printf '%s\n' "$PID" >"$RUNROOT/pids/gpu${GPU}.pid"
done
printf '%s\n' "$RUNROOT" >"$OUTROOT/B24_2_6144_LATEST_RUN.txt"

echo "B24_2_6144_LAUNCHED|runroot=$RUNROOT|head=$HEAD|manifest_sha256=$MAN_SHA|new_rows=4096|rows_per_gpu=1024|gate_mib=$BASELINE_MIN_FREE_MIB"
echo "STATUS|bash $REPO/scripts/b24/status_b24_2_6144.sh $RUNROOT"
echo "RESUME_IF_NEEDED|bash $REPO/scripts/b24/resume_b24_2_6144.sh $RUNROOT"
