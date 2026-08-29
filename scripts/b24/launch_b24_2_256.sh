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
PARENT64="$OUTROOT/B24_2_64_20260826T040303Z"
PARENT_MANIFEST="$PARENT64/B24_2_baseline_64.json"
PARENT_MANIFEST_SHA=23841774c7364b980e3964baed1605aaa8b6eace0370fd765a7369640ea962d7
export TORCH_HOME="$ROOT/models/torch_cache"
mkdir -p "$TORCH_HOME"

[[ -x "$PY" ]] || { echo "STOP|missing_python:$PY"; exit 2; }
[[ -d "$PARENT64" ]] || { echo "STOP|missing_parent64:$PARENT64"; exit 2; }
[[ -f "$PARENT_MANIFEST" ]] || { echo "STOP|missing_parent_manifest:$PARENT_MANIFEST"; exit 2; }
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || { echo "STOP|B24_worktree_dirty"; git -C "$REPO" status --short; exit 3; }

git -C "$CONTROL" fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
LOCAL=$(git -C "$REPO" rev-parse HEAD)
REMOTE=$(git -C "$REPO" rev-parse "origin/$BRANCH")
git -C "$REPO" merge-base --is-ancestor "$LOCAL" "$REMOTE" || { echo "STOP|local_not_ancestor|local=$LOCAL|remote=$REMOTE"; exit 4; }
if [[ "$LOCAL" != "$REMOTE" ]]; then git -C "$REPO" merge --ff-only "origin/$BRANCH"; fi
HEAD=$(git -C "$REPO" rev-parse HEAD)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

"$PY" - <<'PY'
from pathlib import Path
import prdiffusion.b24_protocol as p
obs=Path(p.__file__).resolve(); exp=Path('/egr/research-pac/huang248/pr_diffusion_b24/prdiffusion/b24_protocol.py').resolve()
assert obs==exp,(obs,exp)
print(f'B24_IMPORT_READY|protocol={obs}')
PY
"$PY" -m py_compile \
  scripts/b24/render_b24_baseline_manifest.py \
  scripts/b24/generate_b24_locked_input.py \
  scripts/b24/evaluate_b24_baseline_image.py \
  scripts/b24/run_b24_2_256_extension_shard.py \
  scripts/b24/analyze_b24_2_64.py \
  scripts/b24/analyze_b24_2_256.py
"$PY" -m unittest discover -s tests/b24 -p 'test_*.py'
echo "B24_TESTS_READY|suite=tests/b24"

ACTUAL_PARENT_SHA=$(sha256sum "$PARENT_MANIFEST" | awk '{print $1}')
[[ "$ACTUAL_PARENT_SHA" == "$PARENT_MANIFEST_SHA" ]] || { echo "STOP|parent64_manifest_sha|observed=$ACTUAL_PARENT_SHA|expected=$PARENT_MANIFEST_SHA"; exit 10; }

# Re-analyze the immutable parent checkpoint directly from PAC completions.
"$PY" scripts/b24/analyze_b24_2_64.py --runroot "$PARENT64" >/tmp/b24_64_reanalysis.$$.txt
cat /tmp/b24_64_reanalysis.$$.txt
"$PY" - "$PARENT64/B24_2_64_SUMMARY.json" <<'PY'
import json,sys
from pathlib import Path
v=json.loads(Path(sys.argv[1]).read_text())
assert v['status']=='PASS' and v['n_images']==64
assert v['class_counts']=={'A':57,'B':2,'C':3,'D':2}, v['class_counts']
print('PARENT64_VERIFIED|counts=A57,B2,C3,D2')
PY
rm -f /tmp/b24_64_reanalysis.$$.txt

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

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT="$OUTROOT/B24_2_256_extension_$STAMP"
mkdir -p "$RUNROOT/logs" "$RUNROOT/pids"
MANIFEST="$RUNROOT/B24_2_baseline_256.json"
"$PY" scripts/b24/render_b24_baseline_manifest.py --count 256 --out "$MANIFEST" >"$RUNROOT/logs/render_manifest.log" 2>&1
MAN_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')

"$PY" - "$PARENT_MANIFEST" "$MANIFEST" <<'PY'
import json,sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text()); q=json.loads(Path(sys.argv[2]).read_text())
assert len(p['rows'])==64 and len(q['rows'])==256
assert q['rows'][:64]==p['rows']
assert [int(r['row_index']) for r in q['rows']]==list(range(256))
print('PREFIX_READY|parent=64|target=256|new_rows=192')
PY

echo "MANIFEST_READY|count=256|file_sha256=$MAN_SHA|parent64_sha256=$PARENT_MANIFEST_SHA"

# UUID is a hard identity gate. Free memory is observational here: each long
# worker waits for >=52,096 MiB immediately before its own GPU stages rather
# than aborting the entire shard when another PAC job temporarily occupies it.
UUIDS=(
  GPU-8c9c6250-7b65-20d8-5c81-d6cb618810c3
  GPU-883c037a-34d2-48c4-467f-9a352fd8fdff
  GPU-c381c0f4-1dbc-004f-7d3a-1d7f7794dffe
  GPU-7d65c050-d7e8-5a6b-ee38-1d72d7a5696a
)
for GPU in 0 1 2 3; do
  ROW=$(nvidia-smi --id="$GPU" --query-gpu=uuid,memory.free --format=csv,noheader,nounits)
  IFS=',' read -r U F <<< "$ROW"; U="${U// /}"; F="${F// /}"
  [[ "$U" == "${UUIDS[$GPU]}" ]] || { echo "STOP|gpu_uuid|gpu=$GPU|observed=$U|expected=${UUIDS[$GPU]}"; exit 30; }
  echo "GPU_OBSERVED|gpu=$GPU|uuid=$U|free_mib=$F|runtime_fit_gate_mib=52096"
done

cat >"$RUNROOT/LAUNCH.json" <<EOF
{
  "stage": "B24.2_256_EXTENSION",
  "b24_head": "$HEAD",
  "parent64_runroot": "$PARENT64",
  "parent64_manifest_file_sha256": "$PARENT_MANIFEST_SHA",
  "manifest_path": "$MANIFEST",
  "manifest_file_sha256": "$MAN_SHA",
  "target_count": 256,
  "new_row_range": [64,255],
  "new_rows": 192,
  "gating_collection_classes": ["B","C"],
  "d_retained_but_not_gating": true,
  "per_method_concurrency": 4,
  "runtime_fit_gate_mib": 52096,
  "normal_target_mib": 48000,
  "hard_ceiling_mib": 52452,
  "gpu_fit_behavior": "WAIT_AND_RETRY_PRE_GROUP_GATE"
}
EOF

for GPU in 0 1 2 3; do
  nohup env PYTHONPATH="$PYTHONPATH" PYTHONDONTWRITEBYTECODE=1 TORCH_HOME="$TORCH_HOME" \
    "$PY" scripts/b24/run_b24_2_256_extension_shard.py \
      --manifest "$MANIFEST" --parent-manifest "$PARENT_MANIFEST" \
      --shard "$GPU" --gpu "$GPU" --repo "$REPO" \
      --output-root "$RUNROOT/shard${GPU}" \
    >"$RUNROOT/logs/gpu${GPU}.log" 2>&1 &
  PID=$!; printf '%s\n' "$PID" >"$RUNROOT/pids/gpu${GPU}.pid"
done
printf '%s\n' "$RUNROOT" >"$OUTROOT/B24_2_256_LATEST_RUN.txt"

echo "B24_2_256_LAUNCHED|runroot=$RUNROOT|head=$HEAD|manifest_sha256=$MAN_SHA|new_rows=192|rows_per_gpu=48|gating=B,C"
echo "STATUS|bash $REPO/scripts/b24/status_b24_2_256.sh $RUNROOT"
