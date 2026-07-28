#!/usr/bin/env bash
# B22.2: strict four-GPU full-policy smoke, then automatic full-panel launch.
# Intended for nohup. Verbose output is redirected into the run root.

set -u

REPO_ROOT=${B22_REPO_ROOT:-/egr/research-pac/huang248/pr_diffusion_b22}
CONFIG=${B22_CONFIG:-$REPO_ROOT/configs/b22/b22_2_overnight.json}
OUTPUT_ROOT=${B22_OUTPUT_ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/b22_baselines}
DAPS_PY=${B22_DAPS_PY:-/egr/research-pac/huang248/conda-envs/daps/bin/python}
SITCOM_PY=${B22_SITCOM_PY:-/egr/research-pac/huang248/conda-envs/sitcom_ode_bw/bin/python}
NP_PY=${B22_NP_PY:-/egr/research-pac/huang248/conda-envs/prdiff_ffhq/bin/python}

GPU_S0=${1:-0}
GPU_S1=${2:-1}
GPU_N0=${3:-2}
GPU_N1=${4:-3}
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT=${5:-$OUTPUT_ROOT/B22_2_overnight_$STAMP}
LOG_ROOT=$RUN_ROOT/logs
STATUS=$RUN_ROOT/status.tsv

if [[ $(printf '%s\n' "$GPU_S0" "$GPU_S1" "$GPU_N0" "$GPU_N1" | sort -u | wc -l) -ne 4 ]]; then
  echo "STOP: B22.2 requires four distinct physical GPU indices." >&2
  exit 1
fi

mkdir -p "$LOG_ROOT"
printf 'step\tstatus\tdetail\n' > "$STATUS"
echo "run_root=$RUN_ROOT"
echo "repo_root=$REPO_ROOT"
echo "gpu_assignment=sitcom:$GPU_S0,$GPU_S1 np:$GPU_N0,$GPU_N1"

record() {
  local step=$1 status=$2 detail=$3
  printf '%s\t%s\t%s\n' "$step" "$status" "$detail" >> "$STATUS"
  printf '[%-4s] %s — %s\n' "$status" "$step" "$detail"
}

run_logged() {
  local step=$1 logfile=$2 rc
  shift 2
  "$@" >"$logfile" 2>&1
  rc=$?
  if [[ $rc -eq 0 ]]; then
    record "$step" OK "$(basename "$logfile")"
  else
    record "$step" FAIL "$(basename "$logfile") rc=$rc"
  fi
  return "$rc"
}

wait_worker() {
  local step=$1 pid=$2 logfile=$3 rc
  wait "$pid"
  rc=$?
  if [[ $rc -eq 0 ]]; then
    record "$step" OK "$(basename "$logfile")"
  else
    record "$step" FAIL "$(basename "$logfile") rc=$rc"
  fi
  return "$rc"
}

require_stage_plan() {
  local stage=$1 path
  for path in \
    "$RUN_ROOT/$stage/manifest.json" \
    "$RUN_ROOT/$stage/config_snapshot.json" \
    "$RUN_ROOT/$stage/plan.json" \
    "$RUN_ROOT/$stage/shards/sitcom_shard0.json" \
    "$RUN_ROOT/$stage/shards/sitcom_shard1.json" \
    "$RUN_ROOT/$stage/shards/np_shard0.json" \
    "$RUN_ROOT/$stage/shards/np_shard1.json"
  do
    if [[ ! -s "$path" ]]; then
      record "$stage-prepare-contract" FAIL "missing or empty: $path"
      return 1
    fi
  done
  record "$stage-prepare-contract" OK "manifest, plan, config, and four shards present"
}

archive_compact() {
  local suffix=$1
  local archive="${RUN_ROOT}_${suffix}.tar.gz"
  local archive_log="${RUN_ROOT}_${suffix}_archive.log"
  tar \
    --exclude='*.pt' \
    --exclude='*.png' \
    --exclude='*.tmp' \
    -C "$(dirname "$RUN_ROOT")" \
    -czf "$archive" \
    "$(basename "$RUN_ROOT")" \
    >"$archive_log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    record "archive-$suffix" OK "$archive"
  else
    record "archive-$suffix" FAIL "archive rc=$rc log=$archive_log"
  fi
  return "$rc"
}

export PYTHONPATH="$REPO_ROOT/scripts/b22${PYTHONPATH:+:$PYTHONPATH}"

{
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo_root=$REPO_ROOT"
  echo "repo_head=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "repo_branch=$(git -C "$REPO_ROOT" branch --show-current)"
  echo "run_root=$RUN_ROOT"
  echo "sitcom_gpus=$GPU_S0,$GPU_S1"
  echo "np_gpus=$GPU_N0,$GPU_N1"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true
} > "$RUN_ROOT/launch_metadata.txt" 2>&1

if ! run_logged smoke-prepare "$LOG_ROOT/smoke_prepare.log" \
  "$DAPS_PY" -u "$REPO_ROOT/scripts/b22/prepare_b22_2_panel.py" \
    --config "$CONFIG" --repo_root "$REPO_ROOT" --run_root "$RUN_ROOT" --stage smoke
then
  archive_compact smoke_prepare_failed || true
  exit 1
fi
if ! require_stage_plan smoke; then
  archive_compact smoke_prepare_contract_failed || true
  exit 1
fi

CUDA_VISIBLE_DEVICES="$GPU_S0" "$SITCOM_PY" -u "$REPO_ROOT/scripts/b22/run_b22_2_sitcom_worker.py" \
  --config "$CONFIG" --repo_root "$REPO_ROOT" --run_root "$RUN_ROOT" --stage smoke --shard 0 \
  >"$LOG_ROOT/smoke_sitcom_shard0.log" 2>&1 & P_S0=$!
CUDA_VISIBLE_DEVICES="$GPU_S1" "$SITCOM_PY" -u "$REPO_ROOT/scripts/b22/run_b22_2_sitcom_worker.py" \
  --config "$CONFIG" --repo_root "$REPO_ROOT" --run_root "$RUN_ROOT" --stage smoke --shard 1 \
  >"$LOG_ROOT/smoke_sitcom_shard1.log" 2>&1 & P_S1=$!
CUDA_VISIBLE_DEVICES="$GPU_N0" "$NP_PY" -u "$REPO_ROOT/scripts/b22/run_b22_2_np_worker.py" \
  --config "$CONFIG" --repo_root "$REPO_ROOT" --run_root "$RUN_ROOT" --stage smoke --shard 0 \
  >"$LOG_ROOT/smoke_np_shard0.log" 2>&1 & P_N0=$!
CUDA_VISIBLE_DEVICES="$GPU_N1" "$NP_PY" -u "$REPO_ROOT/scripts/b22/run_b22_2_np_worker.py" \
  --config "$CONFIG" --repo_root "$REPO_ROOT" --run_root "$RUN_ROOT" --stage smoke --shard 1 \
  >"$LOG_ROOT/smoke_np_shard1.log" 2>&1 & P_N1=$!

SMOKE_OK=1
wait_worker smoke-sitcom-0 "$P_S0" "$LOG_ROOT/smoke_sitcom_shard0.log" || SMOKE_OK=0
wait_worker smoke-sitcom-1 "$P_S1" "$LOG_ROOT/smoke_sitcom_shard1.log" || SMOKE_OK=0
wait_worker smoke-np-0 "$P_N0" "$LOG_ROOT/smoke_np_shard0.log" || SMOKE_OK=0
wait_worker smoke-np-1 "$P_N1" "$LOG_ROOT/smoke_np_shard1.log" || SMOKE_OK=0
if [[ $SMOKE_OK -ne 1 ]]; then
  record smoke-gate FAIL "one or more GPU smoke workers failed; full launch blocked"
  archive_compact smoke_failed || true
  exit 1
fi

if ! run_logged smoke-validate "$LOG_ROOT/smoke_validate.log" \
  "$DAPS_PY" -u "$REPO_ROOT/scripts/b22/validate_b22_2_stage.py" \
    --config "$CONFIG" --run_root "$RUN_ROOT" --stage smoke
then
  record smoke-gate FAIL "validator failed; full launch blocked"
  archive_compact smoke_validation_failed || true
  exit 1
fi
record smoke-gate PASS "all full-policy smoke checks passed; starting 100-image run"

if ! run_logged full-prepare "$LOG_ROOT/full_prepare.log" \
  "$DAPS_PY" -u "$REPO_ROOT/scripts/b22/prepare_b22_2_panel.py" \
    --config "$CONFIG" --repo_root "$REPO_ROOT" --run_root "$RUN_ROOT" --stage full
then
  archive_compact full_prepare_failed || true
  exit 1
fi
if ! require_stage_plan full; then
  archive_compact full_prepare_contract_failed || true
  exit 1
fi

CUDA_VISIBLE_DEVICES="$GPU_S0" "$SITCOM_PY" -u "$REPO_ROOT/scripts/b22/run_b22_2_sitcom_worker.py" \
  --config "$CONFIG" --repo_root "$REPO_ROOT" --run_root "$RUN_ROOT" --stage full --shard 0 \
  >"$LOG_ROOT/full_sitcom_shard0.log" 2>&1 & F_S0=$!
CUDA_VISIBLE_DEVICES="$GPU_S1" "$SITCOM_PY" -u "$REPO_ROOT/scripts/b22/run_b22_2_sitcom_worker.py" \
  --config "$CONFIG" --repo_root "$REPO_ROOT" --run_root "$RUN_ROOT" --stage full --shard 1 \
  >"$LOG_ROOT/full_sitcom_shard1.log" 2>&1 & F_S1=$!
CUDA_VISIBLE_DEVICES="$GPU_N0" "$NP_PY" -u "$REPO_ROOT/scripts/b22/run_b22_2_np_worker.py" \
  --config "$CONFIG" --repo_root "$REPO_ROOT" --run_root "$RUN_ROOT" --stage full --shard 0 \
  >"$LOG_ROOT/full_np_shard0.log" 2>&1 & F_N0=$!
CUDA_VISIBLE_DEVICES="$GPU_N1" "$NP_PY" -u "$REPO_ROOT/scripts/b22/run_b22_2_np_worker.py" \
  --config "$CONFIG" --repo_root "$REPO_ROOT" --run_root "$RUN_ROOT" --stage full --shard 1 \
  >"$LOG_ROOT/full_np_shard1.log" 2>&1 & F_N1=$!

{
  echo "full_sitcom_shard0_pid=$F_S0"
  echo "full_sitcom_shard1_pid=$F_S1"
  echo "full_np_shard0_pid=$F_N0"
  echo "full_np_shard1_pid=$F_N1"
} > "$RUN_ROOT/full_worker_pids.txt"
record full-launch OK "workers started: $F_S0,$F_S1,$F_N0,$F_N1"

FULL_OK=1
wait_worker full-sitcom-0 "$F_S0" "$LOG_ROOT/full_sitcom_shard0.log" || FULL_OK=0
wait_worker full-sitcom-1 "$F_S1" "$LOG_ROOT/full_sitcom_shard1.log" || FULL_OK=0
wait_worker full-np-0 "$F_N0" "$LOG_ROOT/full_np_shard0.log" || FULL_OK=0
wait_worker full-np-1 "$F_N1" "$LOG_ROOT/full_np_shard1.log" || FULL_OK=0
if [[ $FULL_OK -ne 1 ]]; then
  record full-gate FAIL "one or more full workers failed; preserve run root for resume"
  archive_compact full_worker_failed || true
  exit 1
fi

if ! run_logged full-validate "$LOG_ROOT/full_validate.log" \
  "$DAPS_PY" -u "$REPO_ROOT/scripts/b22/validate_b22_2_stage.py" \
    --config "$CONFIG" --run_root "$RUN_ROOT" --stage full
then
  record full-gate FAIL "full validator failed"
  archive_compact full_validation_failed || true
  exit 1
fi
record full-gate PASS "100-image paired baseline artifacts complete; scientific review pending"

cat > "$RUN_ROOT/FINAL_STATUS.txt" <<EOF
B22.2 overnight execution: COMPLETE
smoke validation: PASS
full validation: PASS
full scientific sign-off: PENDING EXECUTION-LEAD REVIEW
run_root=$RUN_ROOT
repo_head=$(git -C "$REPO_ROOT" rev-parse HEAD)
EOF

if archive_compact complete; then
  echo
  echo "B22.2 overnight run completed."
  echo "run_root=$RUN_ROOT"
  echo "return_archive=${RUN_ROOT}_complete.tar.gz"
else
  echo "B22.2 computation and validation completed, but compact archive creation failed."
  echo "run_root=$RUN_ROOT"
  exit 1
fi
