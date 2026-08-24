#!/usr/bin/env bash
# Authorized B23.1A/B only. Stops before B23.2, large panels, B24, or adaptive schedules.
set -euo pipefail

REPO=""
OUTPUT_ROOT=""
EXPECTED_HEAD=""
GPUS="0 1 2 3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --expected-head) EXPECTED_HEAD="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    *) echo "[STOP] unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$REPO" && -n "$OUTPUT_ROOT" && -n "$EXPECTED_HEAD" ]] || {
  echo "[STOP] --repo, --output-root, and --expected-head are required" >&2; exit 2;
}
REPO=$(cd "$REPO" && pwd)
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT=$(cd "$OUTPUT_ROOT" && pwd)
read -r -a GPU_ARRAY <<< "$GPUS"
[[ ${#GPU_ARRAY[@]} -eq 4 ]] || { echo "[STOP] exactly four authorized GPU indices are required" >&2; exit 2; }
[[ $(printf '%s\n' "${GPU_ARRAY[@]}" | sort -u | wc -l) -eq 4 ]] || {
  echo "[STOP] authorized GPU indices must be unique" >&2; exit 2;
}

BRANCH=$(git -C "$REPO" branch --show-current)
HEAD=$(git -C "$REPO" rev-parse HEAD)
[[ "$BRANCH" == "codex/b23-execution" ]] || { echo "[STOP] wrong branch: $BRANCH" >&2; exit 3; }
[[ "$HEAD" == "$EXPECTED_HEAD" ]] || { echo "[STOP] wrong pre-run head: $HEAD" >&2; exit 3; }
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || { echo "[STOP] dirty pre-run worktree" >&2; exit 3; }

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT="$OUTPUT_ROOT/B23_1_run_$STAMP"
CAPSULE="$OUTPUT_ROOT/B23_1_return_$STAMP"
STATUS="$RUN_ROOT/STATUS.tsv"
mkdir -p "$RUN_ROOT/logs"
printf 'step\tstatus\trc\n' > "$STATUS"

run_step() {
  local name="$1"; shift
  local log="$RUN_ROOT/logs/${name}.log" rc=0
  if "$@" >"$log" 2>&1; then
    printf '%s\tPASS\t0\n' "$name" >> "$STATUS"
  else
    rc=$?
    printf '%s\tFAIL\t%s\n' "$name" "$rc" >> "$STATUS"
    echo "[STOP] $name failed rc=$rc log=$log" >&2
    return "$rc"
  fi
}

DAPS_PY="/egr/research-pac/huang248/conda-envs/daps/bin/python"
for python_bin in \
  "$DAPS_PY" \
  "/egr/research-pac/huang248/conda-envs/prdiff_ffhq/bin/python" \
  "/egr/research-pac/huang248/conda-envs/sitcom_ode_bw/bin/python"; do
  [[ -x "$python_bin" ]] || { echo "[STOP] missing frozen Python: $python_bin" >&2; exit 4; }
done

run_step prerun_validate env CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO" \
  "$DAPS_PY" "$REPO/scripts/b23/validate_b23_1_prerun.py" \
  --repo "$REPO" --expected-head "$EXPECTED_HEAD" --pac \
  --output-json "$RUN_ROOT/PRERUN_VALIDATION.json"

run_step prepare_inputs env CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" PYTHONPATH="$REPO" \
  "$DAPS_PY" "$REPO/scripts/b23/prepare_b23_1_inputs.py" \
  --repo "$REPO" --output-root "$RUN_ROOT/inputs" --device cuda:0

parent_key() { printf '%s' "$1" | tr '[:upper:]-' '[:lower:]_'; }
# Three interleaved native cycles put every parent measurement next to the
# Fresh1 reference on the same pinned GPU while preserving native-before-wrapper.
for repeat in 0 1 2; do
  for parent in Fresh1 LF-v1 NP-1 SITCOM-1; do
    key=$(parent_key "$parent")
    base="$RUN_ROOT/replay/$key"
    out="$base/native_$repeat"
    run_step "replay_${key}_native_$repeat" env CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" PYTHONPATH="$REPO" \
      "$DAPS_PY" "$REPO/scripts/b23/run_b23_1_parent.py" \
      --repo "$REPO" --inputs "$RUN_ROOT/inputs" --output "$out" \
      --parent "$parent" --mode native --repeat-index "$repeat" --split B23.1-SMOKE-1 --row-id 0
  done
done
for parent in Fresh1 LF-v1 NP-1 SITCOM-1; do
  key=$(parent_key "$parent")
  base="$RUN_ROOT/replay/$key"
  native_args=()
  for repeat in 0 1 2; do
    native_args+=(--native-dir "$base/native_$repeat")
  done
  run_step "freeze_${key}" env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO" \
    "$DAPS_PY" "$REPO/scripts/b23/b23_1_replay_evidence.py" freeze \
    "${native_args[@]}" --output "$base/TOLERANCE_FREEZE.json"
  run_step "replay_${key}_wrapper" env CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" PYTHONPATH="$REPO" \
    "$DAPS_PY" "$REPO/scripts/b23/run_b23_1_parent.py" \
    --repo "$REPO" --inputs "$RUN_ROOT/inputs" --output "$base/wrapper" \
    --parent "$parent" --mode wrapper --repeat-index 0 --split B23.1-SMOKE-1 --row-id 0
  run_step "analyze_${key}" env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO" \
    "$DAPS_PY" "$REPO/scripts/b23/b23_1_replay_evidence.py" analyze \
    "${native_args[@]}" --freeze "$base/TOLERANCE_FREEZE.json" \
    --wrapper-dir "$base/wrapper" --output "$base/REPLAY_REPORT.json"
done

run_step compute_microbench env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO" \
  "$DAPS_PY" "$REPO/scripts/b23/calibrate_b23_1_compute.py" \
  --run-root "$RUN_ROOT" --output "$RUN_ROOT/compute"

pids=()
for row in 0 1 2 3; do
  gpu="${GPU_ARRAY[$row]}"
  (
    set -euo pipefail
    for parent in Fresh1 LF-v1 NP-1 SITCOM-1; do
      key=$(parent_key "$parent")
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO" \
        "$DAPS_PY" "$REPO/scripts/b23/run_b23_1_parent.py" \
        --repo "$REPO" --inputs "$RUN_ROOT/inputs" \
        --output "$RUN_ROOT/smoke/row${row}/$key" \
        --parent "$parent" --mode smoke --repeat-index 0 --split B23.1-SMOKE-4 --row-id "$row"
    done
  ) >"$RUN_ROOT/logs/smoke_row${row}.log" 2>&1 &
  pids+=("$!")
done
smoke_rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then smoke_rc=1; fi
done
if [[ "$smoke_rc" -ne 0 ]]; then
  printf 'four_image_smoke\tFAIL\t1\n' >> "$STATUS"
  echo "[STOP] four-image smoke failed" >&2
  exit 5
fi
printf 'four_image_smoke\tPASS\t0\n' >> "$STATUS"

report_args=()
for parent in Fresh1 LF-v1 NP-1 SITCOM-1; do
  key=$(parent_key "$parent")
  report_args+=(--replay-report "$RUN_ROOT/replay/$key/REPLAY_REPORT.json")
done
run_step donor_classification env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO" \
  "$DAPS_PY" "$REPO/scripts/b23/classify_b23_1_donors.py" \
  "${report_args[@]}" --run-root "$RUN_ROOT" --output "$RUN_ROOT/DONOR_COMPATIBILITY.json"

cp "$STATUS" "$RUN_ROOT/FINAL_STATUS.tsv"
run_step package env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO" \
  "$DAPS_PY" "$REPO/scripts/b23/package_b23_1_return.py" \
  --repo "$REPO" --run-root "$RUN_ROOT" --capsule "$CAPSULE"

echo "[B23.1] status=RETURN_READY_PENDING_PLANNER_REVIEW"
echo "[B23.1] pre_run_commit=$EXPECTED_HEAD"
echo "[B23.1] run_root=$RUN_ROOT"
echo "[B23.1] capsule=$CAPSULE"
echo "[B23.1] archive=${CAPSULE}.tar.gz"
echo "[B23.1] checksum=${CAPSULE}.tar.gz.sha256"
echo "[B23.1] gpu_work_performed=YES_B23.1A_B_ONLY"
echo "[B23.1] b23_2_authorized=NO"
