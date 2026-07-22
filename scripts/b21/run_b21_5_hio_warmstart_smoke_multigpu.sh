#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
DAPS="$REPO/external/daps"
B19_BASE=${B19_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver}
B21_BASE=${B21_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver}
PYTHON_BIN=${PYTHON_BIN:-/egr/research-pac/huang248/conda-envs/daps/bin/python}
IMAGE=${IMAGE:-00046}
MEAS_SEED=${MEAS_SEED:-5001}
CASE_IDS=${CASE_IDS:-"0 1 2"}
GPUS=${GPUS:-"0 1 2"}
BASE_SEED0=${BASE_SEED0:-7300}
HIO_SEED0=${HIO_SEED0:-8300}
WARM_NOISE_SEED0=${WARM_NOISE_SEED0:-9300}
ANN_STEPS=${ANN_STEPS:-400}
DIFF_STEPS=${DIFF_STEPS:-5}
INJECT_STEP=${INJECT_STEP:-200}
HIO_ITERS=${HIO_ITERS:-240}
HIO_BETA=${HIO_BETA:-0.9}
HIO_ER_EVERY=${HIO_ER_EVERY:-20}
HIO_FINAL_ER=${HIO_FINAL_ER:-10}
B21_FORCE=${B21_FORCE:-0}
OUT=${OUT:-$B21_BASE/B21_5_hio_warmstart_smoke_${IMAGE}_3case_step${INJECT_STEP}}
LOGDIR="$OUT/logs"
CASE_ROOT="$OUT/cases"
DONE="$OUT/done.tsv"
FAIL="$OUT/fail.tsv"
GENERATOR="$REPO/scripts/b21/generate_b21_5_hio_state.py"
ANALYZER="$REPO/scripts/b19/analyze_daps_exact_final_loss_selector.py"
CHECKER="$REPO/scripts/b21/check_b21_5_hio_warmstart_smoke.py"
DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024

cd "$REPO"
mkdir -p "$OUT" "$LOGDIR" "$CASE_ROOT"

[[ -x "$PYTHON_BIN" ]] || { echo "[fatal] PYTHON_BIN not executable: $PYTHON_BIN" >&2; exit 2; }
for file in "$GENERATOR" "$ANALYZER" "$CHECKER"; do
  [[ -f "$file" ]] || { echo "[fatal] missing $file" >&2; exit 2; }
done
grep -q 'B21.3 continuation patch' "$DAPS/sampler.py" || {
  echo "[fatal] validated continuation patch is not applied" >&2
  echo "        Run: PYTHON_BIN=$PYTHON_BIN bash scripts/b21/apply_b21_3_continuation_patch.sh" >&2
  exit 2
}

"$PYTHON_BIN" - <<'PY'
import sys, torch, yaml, pandas, PIL
print("python", sys.executable)
print("torch", torch.__version__)
print("yaml", yaml.__version__)
print("pandas", pandas.__version__)
print("PIL", PIL.__version__)
PY

IMAGE=$(printf '%05d' "$((10#$IMAGE))")
MEAS_PATH="$B19_BASE/measurements/ffhq${IMAGE}_phase_noise005_meas${MEAS_SEED}.pt"
GT_PATH="$DATA_ROOT/00000/${IMAGE}.png"
DATA_NAME="b21-ffhq-${IMAGE}"
[[ -f "$MEAS_PATH" ]] || { echo "[fatal] missing locked measurement: $MEAS_PATH" >&2; exit 2; }
[[ -f "$GT_PATH" ]] || { echo "[fatal] missing image: $GT_PATH" >&2; exit 2; }

mkdir -p "$DAPS/dataset/$DATA_NAME" "$DAPS/configs/data"
ln -sfn "$GT_PATH" "$DAPS/dataset/$DATA_NAME/${IMAGE}.png"
cat > "$DAPS/configs/data/${DATA_NAME}.yaml" <<YAML
name: image
root: 'dataset/${DATA_NAME}'
resolution: 256
start_id: 0
end_id: 1
YAML

read -r -a GPU_ARR <<< "$GPUS"
read -r -a CASE_ARR <<< "$CASE_IDS"
[[ ${#GPU_ARR[@]} -ge 1 ]] || { echo "[fatal] GPUS is empty" >&2; exit 2; }
[[ ${#CASE_ARR[@]} -ge 1 ]] || { echo "[fatal] CASE_IDS is empty" >&2; exit 2; }

: > "$DONE"
: > "$FAIL"
{
  echo "timestamp=$(date -Is)"
  echo "repo=$REPO"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "image=$IMAGE"
  echo "measurement_path=$MEAS_PATH"
  echo "case_ids=$CASE_IDS"
  echo "gpus=$GPUS"
  echo "base_seed0=$BASE_SEED0"
  echo "hio_seed0=$HIO_SEED0"
  echo "warm_noise_seed0=$WARM_NOISE_SEED0"
  echo "ann_steps=$ANN_STEPS"
  echo "diff_steps=$DIFF_STEPS"
  echo "inject_step=$INJECT_STEP"
  echo "hio_iters=$HIO_ITERS"
  echo "hio_beta=$HIO_BETA"
  echo "hio_er_every=$HIO_ER_EVERY"
  echo "hio_final_er=$HIO_FINAL_ER"
} > "$OUT/launch_env.txt"

run_daps() {
  local case_dir="$1"
  local candidate="$2"
  local hydra_seed="$3"
  local gpu="$4"
  shift 4

  local save_dir="$case_dir/daps_results"
  local metric_dir="$case_dir/metrics"
  local log_dir="$case_dir/logs"
  local timing_file="$case_dir/timings.tsv"
  local sample="$save_dir/$candidate/samples/00000_run0000.png"
  local csv="$metric_dir/${candidate}.csv"
  local log="$log_dir/${candidate}.log"
  mkdir -p "$save_dir" "$metric_dir" "$log_dir"

  if [[ "$B21_FORCE" != "1" && -s "$sample" && -s "$csv" ]]; then
    echo "[skip][gpu=$gpu] $candidate"
    return 0
  fi
  if [[ "$B21_FORCE" == "1" ]]; then
    rm -rf "$save_dir/$candidate"
    rm -f "$csv" "$log"
  fi

  local start end elapsed
  start=$(date +%s)
  echo "[run][gpu=$gpu] candidate=$candidate seed=$hydra_seed"
  (
    cd "$DAPS"
    env \
      -u B21_CONT_ENABLE \
      -u B21_CONT_STATE_PATH \
      -u B21_CONT_NOISE_SEED \
      -u B21_SAVE_STATE_STEPS \
      -u B21_START_NOISE_SEED \
      -u B21_SOURCE_SEED \
      B20_LF_ENABLE=0 \
      B20_LF_ALPHA=0.0 \
      B21_MEASUREMENT_PATH="$MEAS_PATH" \
      "$@" \
      CUDA_VISIBLE_DEVICES="$gpu" \
      "$PYTHON_BIN" posterior_sample.py \
        +data="$DATA_NAME" \
        +measurement_path="$MEAS_PATH" \
        +model=ffhq256ddpm \
        +sampler=edm_daps \
        +task=phase_retrieval \
        task_group=pixel \
        batch_size=1 \
        data.start_id=0 \
        data.end_id=1 \
        gpu=0 \
        seed="$hydra_seed" \
        num_runs=1 \
        name="$candidate" \
        save_dir="$save_dir" \
        sampler.diffusion_scheduler_config.num_steps="$DIFF_STEPS" \
        sampler.annealing_scheduler_config.num_steps="$ANN_STEPS" \
        save_samples=true \
        save_traj=false \
        save_traj_raw_data=false \
        save_traj_video=false
  ) > "$log" 2>&1

  [[ -s "$sample" ]] || { echo "[fatal] missing sample: $sample" >&2; tail -80 "$log" >&2; return 3; }
  "$PYTHON_BIN" "$ANALYZER" \
    --daps_root "$DAPS" \
    --samples_dir "$save_dir/$candidate/samples" \
    --measurement_path "$MEAS_PATH" \
    --out_csv "$csv" \
    >> "$log" 2>&1
  [[ -s "$csv" ]] || { echo "[fatal] missing metric CSV: $csv" >&2; return 3; }

  end=$(date +%s)
  elapsed=$((end - start))
  touch "$timing_file"
  awk -F'\t' -v c="$candidate" '$1 != c' "$timing_file" > "${timing_file}.tmp" || true
  printf "%s\t%d\n" "$candidate" "$elapsed" >> "${timing_file}.tmp"
  mv "${timing_file}.tmp" "$timing_file"
  echo "[done][gpu=$gpu] candidate=$candidate elapsed=${elapsed}s"
}

run_case() {
  local case_id="$1"
  local gpu="$2"
  local base_seed=$((BASE_SEED0 + case_id))
  local hio_seed=$((HIO_SEED0 + case_id))
  local warm_seed=$((WARM_NOISE_SEED0 + case_id))
  local case_dir="$CASE_ROOT/case$(printf '%02d' "$case_id")"
  local hio_dir="$case_dir/hio"
  local state="$hio_dir/hio_state.pt"
  local hio_png="$hio_dir/hio_raw.png"
  local hio_json="$hio_dir/hio_summary.json"
  local hio_log="$case_dir/logs/hio_generate.log"
  local timing_file="$case_dir/timings.tsv"
  mkdir -p "$hio_dir" "$case_dir/logs"

  cat > "$case_dir/case_env.txt" <<EOF
case_id=$case_id
image_id=$IMAGE
gpu=$gpu
base_seed=$base_seed
hio_seed=$hio_seed
warm_noise_seed=$warm_seed
inject_step=$INJECT_STEP
EOF

  run_daps "$case_dir" base_full "$base_seed" "$gpu" \
    B21_START_NOISE_SEED="$base_seed" \
    B21_SOURCE_SEED="$base_seed"

  if [[ "$B21_FORCE" == "1" || ! -s "$state" || ! -s "$hio_json" || ! -s "$hio_png" ]]; then
    local start end elapsed
    start=$(date +%s)
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$GENERATOR" \
      --measurement-path "$MEAS_PATH" \
      --out-state "$state" \
      --out-png "$hio_png" \
      --out-json "$hio_json" \
      --seed "$hio_seed" \
      --iterations "$HIO_ITERS" \
      --beta "$HIO_BETA" \
      --er-every "$HIO_ER_EVERY" \
      --final-er "$HIO_FINAL_ER" \
      --resolution 256 \
      --inject-step "$INJECT_STEP" \
      --device cuda:0 \
      > "$hio_log" 2>&1
    end=$(date +%s)
    elapsed=$((end - start))
    touch "$timing_file"
    awk -F'\t' '$1 != "hio_generate"' "$timing_file" > "${timing_file}.tmp" || true
    printf "hio_generate\t%d\n" "$elapsed" >> "${timing_file}.tmp"
    mv "${timing_file}.tmp" "$timing_file"
  else
    echo "[skip][gpu=$gpu] hio_generate"
  fi

  [[ -s "$state" ]] || { echo "[fatal] missing HIO state: $state" >&2; return 4; }
  run_daps "$case_dir" hio_warm "$warm_seed" "$gpu" \
    B21_CONT_ENABLE=1 \
    B21_CONT_STATE_PATH="$state" \
    B21_CONT_NOISE_SEED="$warm_seed"
}

pids=()
for index in "${!CASE_ARR[@]}"; do
  case_id="${CASE_ARR[$index]}"
  gpu="${GPU_ARR[$((index % ${#GPU_ARR[@]}))]}"
  (
    if run_case "$case_id" "$gpu"; then
      printf "%s\t%s\tOK\n" "$case_id" "$gpu" >> "$DONE"
    else
      status=$?
      printf "%s\t%s\tFAIL\t%s\n" "$case_id" "$gpu" "$status" >> "$FAIL"
      exit "$status"
    fi
  ) > "$LOGDIR/worker_case$(printf '%02d' "$case_id")_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
  echo "[worker] case=$case_id gpu=$gpu pid=$!"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

completed=$(wc -l < "$DONE")
failed=$(wc -l < "$FAIL")
echo "[workers done] completed=$completed failed=$failed"
if [[ "$status" -ne 0 || "$failed" -ne 0 ]]; then
  cat "$FAIL" >&2
  exit 5
fi

"$PYTHON_BIN" "$CHECKER" \
  --repo "$REPO" \
  --out "$OUT" \
  --image "$IMAGE" \
  --case-ids $CASE_IDS \
  --inject-step "$INJECT_STEP" \
  | tee "$OUT/checker_stdout.txt"

echo "[done] B21.5 HIO warm-start smoke artifacts: $OUT"
