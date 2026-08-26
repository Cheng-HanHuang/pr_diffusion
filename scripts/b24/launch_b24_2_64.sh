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
MIN_FREE=52096
FROZEN_MAN64_FILE_SHA=23841774c7364b980e3964baed1605aaa8b6eace0370fd765a7369640ea962d7
export TORCH_HOME="$ROOT/models/torch_cache"
mkdir -p "$TORCH_HOME"

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

# ``python -m`` places the current working directory first on sys.path.  The
# launcher is commonly invoked from the sibling B23 worktree, whose valid
# ``prdiffusion`` package would otherwise shadow B24's package and make the
# B24-only b24_protocol submodule appear missing.  Pin cwd and verify identity.
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
"$PY" - "$REPO" <<'PY'
import pathlib
import sys
root = pathlib.Path(sys.argv[1]).resolve()
import prdiffusion
import prdiffusion.b24_protocol as protocol
pkg = pathlib.Path(prdiffusion.__file__).resolve()
mod = pathlib.Path(protocol.__file__).resolve()
expected_pkg = root / "prdiffusion" / "__init__.py"
expected_mod = root / "prdiffusion" / "b24_protocol.py"
if pkg != expected_pkg or mod != expected_mod:
    raise SystemExit(
        f"STOP|B24_IMPORT_IDENTITY|package={pkg}|protocol={mod}|"
        f"expected_package={expected_pkg}|expected_protocol={expected_mod}"
    )
print(f"B24_IMPORT_READY|package={pkg}|protocol={mod}")
PY

"$PY" -m py_compile \
  "$REPO/scripts/b24/render_b24_baseline_manifest.py" \
  "$REPO/scripts/b24/generate_b24_locked_input.py" \
  "$REPO/scripts/b24/evaluate_b24_baseline_image.py" \
  "$REPO/scripts/b24/run_b24_2_shard.py" \
  "$REPO/scripts/b24/analyze_b24_2_64.py"
"$PY" -m unittest discover -s tests/b24 -p 'test_*.py' >/dev/null

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
  [[ "$index_digest" == "$exp_index" ]] || { echo "STOP|source_index|name=$name|observed=$index_digest|expected=$exp_index"; return 23; }
  [[ "$diff_digest" == "$exp_diff" ]] || { echo "STOP|source_tracked_diff|name=$name|observed=$diff_digest|expected=$exp_diff"; return 24; }
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

# Require the completed B24.1 gate that validated concurrency=4 for both methods.
B24_1_LATEST="$OUTROOT/B24_1_LATEST_RUN.txt"
[[ -f "$B24_1_LATEST" ]] || { echo "STOP|missing B24.1 latest-run pointer"; exit 30; }
B24_1_ROOT=$(cat "$B24_1_LATEST")
B24_1_SUMMARY="$B24_1_ROOT/B24_1_SUMMARY.json"
[[ -f "$B24_1_SUMMARY" ]] || { echo "STOP|missing B24.1 summary:$B24_1_SUMMARY"; exit 31; }
"$PY" - "$B24_1_SUMMARY" <<'PY'
import json, sys
from pathlib import Path
v=json.loads(Path(sys.argv[1]).read_text())
assert v["overall_pass"] is True
assert v["b24_2_64_gate_recommendation"] == "PASS_TO_64_BASELINE"
for m in ("DAPS","SITCOM"):
    x=v["methods"][m]
    assert x["planned_concurrency"] == 4
    assert x["exact_terminal_hash_equivalence"] is True
    assert x["memory_pass"] is True
print("B24_1_GATE_VERIFIED|summary="+sys.argv[1])
PY
B24_1_SHA=$(sha256sum "$B24_1_SUMMARY" | awk '{print $1}')

check_gpu() {
  local gpu="$1" expected="$2"
  local row uuid free
  row=$(nvidia-smi --id="$gpu" --query-gpu=uuid,memory.free --format=csv,noheader,nounits)
  IFS=',' read -r uuid free <<< "$row"; uuid="${uuid// /}"; free="${free// /}"
  [[ "$uuid" == "$expected" ]] || { echo "STOP|gpu_uuid|gpu=$gpu|observed=$uuid|expected=$expected"; return 10; }
  (( free >= MIN_FREE )) || { echo "STOP|gpu_free|gpu=$gpu|free_mib=$free|required_mib=$MIN_FREE"; return 11; }
  echo "GPU_READY|gpu=$gpu|uuid=$uuid|free_mib=$free"
}
check_gpu 0 GPU-8c9c6250-7b65-20d8-5c81-d6cb618810c3
check_gpu 1 GPU-883c037a-34d2-48c4-467f-9a352fd8fdff
check_gpu 2 GPU-c381c0f4-1dbc-004f-7d3a-1d7f7794dffe
check_gpu 3 GPU-7d65c050-d7e8-5a6b-ee38-1d72d7a5696a

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT="$OUTROOT/B24_2_64_$STAMP"
mkdir -p "$RUNROOT/logs" "$RUNROOT/pids"
MANIFEST="$RUNROOT/B24_2_baseline_64.json"
"$PY" "$REPO/scripts/b24/render_b24_baseline_manifest.py" \
  --exposure "$REPO/manifests/b24/PRE_B24_EXPOSURE.csv" \
  --count 64 --out "$MANIFEST" \
  > "$RUNROOT/logs/render_manifest.log" 2>&1
MAN_FILE_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')
[[ "$MAN_FILE_SHA" == "$FROZEN_MAN64_FILE_SHA" ]] || {
  echo "STOP|64_manifest_file_sha|observed=$MAN_FILE_SHA|expected=$FROZEN_MAN64_FILE_SHA"; exit 40;
}
echo "MANIFEST_READY|count=64|file_sha256=$MAN_FILE_SHA"

# Probe the reporting metric dependency before generating any fresh B24.2 input.
CUDA_VISIBLE_DEVICES=0 "$PY" - <<'PY' > "$RUNROOT/logs/lpips_probe.log" 2>&1
import torch, lpips
m=lpips.LPIPS(net="alex").to("cuda:0").eval()
x=torch.zeros((1,3,256,256),device="cuda:0")
with torch.no_grad(): y=float(m(x,x).mean().cpu())
assert abs(y) < 1e-5
print("LPIPS_PROBE_PASS", y)
PY
echo "LPIPS_READY|model=alex|torch_home=$TORCH_HOME"

cat > "$RUNROOT/LAUNCH.json" <<EOF
{
  "stage": "B24.2_64",
  "b24_head": "$HEAD",
  "manifest_path": "$MANIFEST",
  "manifest_file_sha256": "$MAN_FILE_SHA",
  "b24_1_summary_path": "$B24_1_SUMMARY",
  "b24_1_summary_sha256": "$B24_1_SHA",
  "per_method_concurrency": 4,
  "evaluation_representation": "CANONICAL_SAVED_RGB_8BIT_RAW_ORIENTATION_V1",
  "good25_threshold_db": 25.0,
  "hard_ceiling_mib": 52452,
  "normal_target_mib": 48000,
  "minimum_free_before_launch_mib": 52096
}
EOF

for GPU in 0 1 2 3; do
  nohup env PYTHONPATH="$PYTHONPATH" PYTHONDONTWRITEBYTECODE=1 TORCH_HOME="$TORCH_HOME" \
    "$PY" "$REPO/scripts/b24/run_b24_2_shard.py" \
      --manifest "$MANIFEST" --shard "$GPU" --gpu "$GPU" --repo "$REPO" \
      --output-root "$RUNROOT/shard${GPU}" \
    > "$RUNROOT/logs/gpu${GPU}.log" 2>&1 &
  PID=$!
  printf '%s\n' "$PID" > "$RUNROOT/pids/gpu${GPU}.pid"
done
printf '%s\n' "$RUNROOT" > "$OUTROOT/B24_2_64_LATEST_RUN.txt"

echo "B24_2_64_LAUNCHED|runroot=$RUNROOT|head=$HEAD|manifest_sha256=$MAN_FILE_SHA|gpus=0,1,2,3|rows_per_gpu=16|concurrency_per_method=4"
echo "STATUS|bash $REPO/scripts/b24/status_b24_2_64.sh $RUNROOT"
