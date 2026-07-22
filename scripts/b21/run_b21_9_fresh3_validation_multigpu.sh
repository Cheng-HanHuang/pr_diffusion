#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
DAPS="$REPO/external/daps"
PYTHON_BIN=${PYTHON_BIN:-/egr/research-pac/huang248/conda-envs/daps/bin/python}
IMAGE_ROOT=${IMAGE_ROOT:-/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024}
IMAGES=${IMAGES:-"63678 68922 67092 64050 69441 67673 66511 64116 63199 63135 65317 68111 65656 64471 60067 63319 64542 66731 63368 62957"}
CASE_IDS=${CASE_IDS:-"0 1 2 3"}
GPUS=${GPUS:-"0 1 2 3"}
MEAS_PANEL_SEED=${MEAS_PANEL_SEED:-5201}
MEAS_TAG=${MEAS_TAG:-5201}
SEED1_0=${SEED1_0:-17000}
SEED2_0=${SEED2_0:-18000}
SEED3_0=${SEED3_0:-19000}
ANN_STEPS=${ANN_STEPS:-400}
DIFF_STEPS=${DIFF_STEPS:-5}
THETA=${THETA:-0.7}
TARGET_GOOD=${TARGET_GOOD:-74}
MAX_ORACLE_GAP=${MAX_ORACLE_GAP:-2}
MIN_INCREMENTAL_GAIN=${MIN_INCREMENTAL_GAIN:-4}
MAX_INCREMENTAL_HARMS=${MAX_INCREMENTAL_HARMS:-1}
MAX_CUMULATIVE_HARMS=${MAX_CUMULATIVE_HARMS:-1}
MIN_POSITIVE_IMAGES=${MIN_POSITIVE_IMAGES:-4}
MAX_NEGATIVE_IMAGES=${MAX_NEGATIVE_IMAGES:-1}
MIN_PER_IMAGE_GOOD=${MIN_PER_IMAGE_GOOD:-2}
B21_FORCE=${B21_FORCE:-0}
B21_SMOKE_ONLY=${B21_SMOKE_ONLY:-0}
B21_BASE=${B21_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver}
OUT=${OUT:-$B21_BASE/B21_9_fresh3_validation_val20x4_meas${MEAS_TAG}}
MEAS_DIR="$OUT/measurements"
CASE_ROOT="$OUT/cases"
LOGDIR="$OUT/logs"
MANIFEST="$OUT/manifest.tsv"
ACTIVE_MANIFEST="$OUT/active_manifest.tsv"
DONE="$OUT/done.tsv"
FAIL="$OUT/fail.tsv"
MEAS_GENERATOR="$REPO/scripts/b21/generate_b21_5_fresh_locked_measurements.py"
METRIC_ANALYZER="$REPO/scripts/b19/analyze_daps_exact_final_loss_selector.py"
FINAL_ANALYZER="$REPO/scripts/b21/analyze_b21_9_fresh3_validation.py"
ANALYSIS="$OUT/analysis_theta${THETA}"
REPORT="$REPO/docs/b21/b21_9_fresh3_validation.md"

cd "$REPO"
mkdir -p "$OUT" "$MEAS_DIR" "$CASE_ROOT" "$LOGDIR"

[[ -x "$PYTHON_BIN" ]] || { echo "[fatal] PYTHON_BIN not executable: $PYTHON_BIN" >&2; exit 2; }
for file in "$MEAS_GENERATOR" "$METRIC_ANALYZER" "$FINAL_ANALYZER"; do
  [[ -f "$file" ]] || { echo "[fatal] missing $file" >&2; exit 2; }
done

"$PYTHON_BIN" - <<'PY'
import sys, torch, torchvision, yaml, pandas, PIL
print("python", sys.executable)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("yaml", yaml.__version__)
print("pandas", pandas.__version__)
print("PIL", PIL.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
PY

read -r -a GPU_ARR <<< "$GPUS"
read -r -a IMAGE_ARR <<< "$IMAGES"
read -r -a CASE_ARR <<< "$CASE_IDS"
[[ ${#GPU_ARR[@]} -ge 1 ]] || { echo "[fatal] GPUS is empty" >&2; exit 2; }
[[ ${#IMAGE_ARR[@]} -eq 20 ]] || { echo "[fatal] expected exactly 20 frozen validation images" >&2; exit 2; }
[[ ${#CASE_ARR[@]} -eq 4 ]] || { echo "[fatal] expected exactly 4 cases per image" >&2; exit 2; }

# Lock the new measurements before any solver output is generated.
MEAS_ARGS=(
  --repo "$REPO"
  --image-root "$IMAGE_ROOT"
  --out-dir "$MEAS_DIR"
  --images "$IMAGES"
  --panel-seed "$MEAS_PANEL_SEED"
  --measurement-tag "$MEAS_TAG"
  --resolution 256
  --oversample 2.0
  --sigma 0.05
  --device cuda:0
)
if [[ "$B21_FORCE" == "1" ]]; then
  MEAS_ARGS+=(--force)
fi
CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" "$PYTHON_BIN" "$MEAS_GENERATOR" "${MEAS_ARGS[@]}" \
  | tee "$OUT/measurement_generation.log"

# Create exact one-image data configs.
for raw_image in "${IMAGE_ARR[@]}"; do
  image=$(printf '%05d' "$((10#$raw_image))")
  folder=$(printf '%05d' "$((10#$image / 1000 * 1000))")
  gt="$IMAGE_ROOT/$folder/${image}.png"
  if [[ ! -f "$gt" ]]; then
    gt=$(find "$IMAGE_ROOT" -type f -name "${image}.png" -print -quit)
  fi
  meas="$MEAS_DIR/ffhq${image}_phase_noise005_meas${MEAS_TAG}.pt"
  data_name="b21-fresh3-val-ffhq-${image}"
  [[ -f "$gt" ]] || { echo "[fatal] missing image $image" >&2; exit 2; }
  [[ -f "$meas" ]] || { echo "[fatal] missing measurement $meas" >&2; exit 2; }
  mkdir -p "$DAPS/dataset/$data_name" "$DAPS/configs/data"
  ln -sfn "$gt" "$DAPS/dataset/$data_name/${image}.png"
  cat > "$DAPS/configs/data/${data_name}.yaml" <<YAML
name: image
root: 'dataset/${data_name}'
resolution: 256
start_id: 0
end_id: 1
YAML
done

: > "$DONE"
: > "$FAIL"
printf "job_id\timage_id\tcase_id\tgpu\tseed1\tseed2\tseed3\tmeasurement_path\texecution_order\n" > "$MANIFEST"

job_id=0
for raw_image in "${IMAGE_ARR[@]}"; do
  image=$(printf '%05d' "$((10#$raw_image))")
  measurement_path="$MEAS_DIR/ffhq${image}_phase_noise005_meas${MEAS_TAG}.pt"
  for case_id in "${CASE_ARR[@]}"; do
    gpu="${GPU_ARR[$((job_id % ${#GPU_ARR[@]}))]}"
    seed1=$((SEED1_0 + job_id))
    seed2=$((SEED2_0 + job_id))
    seed3=$((SEED3_0 + job_id))
    case $((job_id % 3)) in
      0) execution_order="base_full,base_extra,base_extra2" ;;
      1) execution_order="base_extra,base_extra2,base_full" ;;
      2) execution_order="base_extra2,base_full,base_extra" ;;
    esac
    printf "%d\t%s\t%d\t%s\t%d\t%d\t%d\t%s\t%s\n" \
      "$job_id" "$image" "$case_id" "$gpu" "$seed1" "$seed2" "$seed3" \
      "$measurement_path" "$execution_order" >> "$MANIFEST"
    job_id=$((job_id + 1))
  done
done

rows=$(($(wc -l < "$MANIFEST") - 1))
[[ "$rows" -eq 80 ]] || { echo "[fatal] expected 80 rows, found $rows" >&2; exit 2; }

if [[ "$B21_SMOKE_ONLY" == "1" ]]; then
  head -n 2 "$MANIFEST" > "$ACTIVE_MANIFEST"
else
  cp "$MANIFEST" "$ACTIVE_MANIFEST"
fi

{
  echo "timestamp=$(date -Is)"
  echo "repo=$REPO"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "images=$IMAGES"
  echo "case_ids=$CASE_IDS"
  echo "gpus=$GPUS"
  echo "measurement_panel_seed=$MEAS_PANEL_SEED"
  echo "measurement_tag=$MEAS_TAG"
  echo "seed1_0=$SEED1_0"
  echo "seed2_0=$SEED2_0"
  echo "seed3_0=$SEED3_0"
  echo "ann_steps=$ANN_STEPS"
  echo "diff_steps=$DIFF_STEPS"
  echo "theta=$THETA"
  echo "target_good=$TARGET_GOOD"
  echo "max_oracle_gap=$MAX_ORACLE_GAP"
  echo "min_incremental_gain=$MIN_INCREMENTAL_GAIN"
  echo "max_incremental_harms=$MAX_INCREMENTAL_HARMS"
  echo "max_cumulative_harms=$MAX_CUMULATIVE_HARMS"
  echo "min_positive_images=$MIN_POSITIVE_IMAGES"
  echo "max_negative_images=$MAX_NEGATIVE_IMAGES"
  echo "min_per_image_good=$MIN_PER_IMAGE_GOOD"
  echo "smoke_only=$B21_SMOKE_ONLY"
  echo "expected_cases=$rows"
  echo "active_cases=$(($(wc -l < "$ACTIVE_MANIFEST") - 1))"
} > "$OUT/launch_env.txt"

run_candidate() {
  local job="$1" image="$2" case_id="$3" gpu="$4" seed="$5" meas="$6" candidate="$7"
  local case_dir="$CASE_ROOT/${image}_case$(printf '%02d' "$case_id")"
  local save_dir="$case_dir/daps_results"
  local metric_dir="$case_dir/metrics"
  local log_dir="$case_dir/logs"
  local timing_file="$case_dir/timings.tsv"
  local sample="$save_dir/$candidate/samples/00000_run0000.png"
  local csv="$metric_dir/${candidate}.csv"
  local log="$log_dir/${candidate}.log"
  local data_name="b21-fresh3-val-ffhq-${image}"
  mkdir -p "$save_dir" "$metric_dir" "$log_dir"

  if [[ "$B21_FORCE" != "1" && -s "$sample" && -s "$csv" ]]; then
    echo "[skip][gpu=$gpu] job=$job image=$image case=$case_id candidate=$candidate"
    return 0
  fi
  if [[ "$B21_FORCE" == "1" ]]; then
    rm -rf "$save_dir/$candidate"
    rm -f "$csv" "$log"
  fi

  local start end elapsed
  start=$(date +%s)
  echo "[run][gpu=$gpu] job=$job image=$image case=$case_id candidate=$candidate seed=$seed"
  (
    cd "$DAPS"
    env \
      -u B21_CONT_ENABLE -u B21_CONT_STATE_PATH -u B21_CONT_NOISE_SEED \
      -u B21_SAVE_STATE_STEPS \
      B21_START_NOISE_SEED="$seed" \
      B21_SOURCE_SEED="$seed" \
      B21_MEASUREMENT_PATH="$meas" \
      B20_LF_ENABLE=0 B20_LF_ALPHA=0.0 \
      CUDA_VISIBLE_DEVICES="$gpu" \
      "$PYTHON_BIN" posterior_sample.py \
        +data="$data_name" +measurement_path="$meas" \
        +model=ffhq256ddpm +sampler=edm_daps +task=phase_retrieval \
        task_group=pixel batch_size=1 data.start_id=0 data.end_id=1 gpu=0 \
        seed="$seed" num_runs=1 name="$candidate" save_dir="$save_dir" \
        sampler.diffusion_scheduler_config.num_steps="$DIFF_STEPS" \
        sampler.annealing_scheduler_config.num_steps="$ANN_STEPS" \
        save_samples=true save_traj=false save_traj_raw_data=false save_traj_video=false
  ) > "$log" 2>&1

  [[ -s "$sample" ]] || { echo "[fatal] missing sample: $sample" >&2; tail -80 "$log" >&2; return 3; }
  "$PYTHON_BIN" "$METRIC_ANALYZER" \
    --daps_root "$DAPS" --samples_dir "$save_dir/$candidate/samples" \
    --measurement_path "$meas" --out_csv "$csv" >> "$log" 2>&1
  [[ -s "$csv" ]] || { echo "[fatal] missing metric CSV: $csv" >&2; return 3; }

  end=$(date +%s)
  elapsed=$((end - start))
  touch "$timing_file"
  awk -F'\t' -v c="$candidate" '$1 != c' "$timing_file" > "${timing_file}.tmp" || true
  printf "%s\t%d\n" "$candidate" "$elapsed" >> "${timing_file}.tmp"
  mv "${timing_file}.tmp" "$timing_file"
  echo "[done][gpu=$gpu] job=$job candidate=$candidate elapsed=${elapsed}s"
}

run_case() {
  local job="$1" image="$2" case_id="$3" gpu="$4" seed1="$5" seed2="$6" seed3="$7" meas="$8" order="$9"
  local item
  IFS=',' read -r -a ORDER_ARR <<< "$order"
  for item in "${ORDER_ARR[@]}"; do
    case "$item" in
      base_full) run_candidate "$job" "$image" "$case_id" "$gpu" "$seed1" "$meas" base_full ;;
      base_extra) run_candidate "$job" "$image" "$case_id" "$gpu" "$seed2" "$meas" base_extra ;;
      base_extra2) run_candidate "$job" "$image" "$case_id" "$gpu" "$seed3" "$meas" base_extra2 ;;
      *) echo "[fatal] unknown candidate in execution order: $item" >&2; return 4 ;;
    esac
  done
}

pids=()
for gpu in "${GPU_ARR[@]}"; do
  (
    set -euo pipefail
    tail -n +2 "$ACTIVE_MANIFEST" | awk -F'\t' -v g="$gpu" '$4 == g {print}' | \
    while IFS=$'\t' read -r job image case_id assigned_gpu seed1 seed2 seed3 meas order; do
      [[ -z "${job:-}" ]] && continue
      echo "[case start] job=$job image=$image case=$case_id gpu=$assigned_gpu order=$order"
      if run_case "$job" "$image" "$case_id" "$assigned_gpu" "$seed1" "$seed2" "$seed3" "$meas" "$order"; then
        printf "%s\t%s\t%s\t%s\tOK\n" "$job" "$image" "$case_id" "$assigned_gpu" >> "$DONE"
        echo "[case done] job=$job image=$image case=$case_id gpu=$assigned_gpu"
      else
        status=$?
        printf "%s\t%s\t%s\t%s\tFAIL\t%s\n" "$job" "$image" "$case_id" "$assigned_gpu" "$status" >> "$FAIL"
        exit "$status"
      fi
    done
  ) > "$LOGDIR/worker_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
  echo "[worker] gpu=$gpu pid=${pids[-1]} log=$LOGDIR/worker_gpu${gpu}.log"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
completed=$(wc -l < "$DONE")
failed=$(wc -l < "$FAIL")
echo "[workers done] completed=$completed failed=$failed"
if [[ "$status" -ne 0 || "$failed" -ne 0 ]]; then
  cat "$FAIL" >&2
  exit 5
fi

if [[ "$B21_SMOKE_ONLY" == "1" ]]; then
  echo "[smoke-only] completed first three-restart validation case; analyzer intentionally skipped"
  exit 0
fi

"$PYTHON_BIN" "$FINAL_ANALYZER" \
  --out "$OUT" \
  --manifest "$MANIFEST" \
  --theta "$THETA" \
  --target-good "$TARGET_GOOD" \
  --max-oracle-gap "$MAX_ORACLE_GAP" \
  --min-incremental-gain "$MIN_INCREMENTAL_GAIN" \
  --max-incremental-harms "$MAX_INCREMENTAL_HARMS" \
  --max-cumulative-harms "$MAX_CUMULATIVE_HARMS" \
  --min-positive-images "$MIN_POSITIVE_IMAGES" \
  --max-negative-images "$MAX_NEGATIVE_IMAGES" \
  --min-per-image-good "$MIN_PER_IMAGE_GOOD" \
  --image-root "$IMAGE_ROOT" \
  --report "$REPORT" \
  | tee "$OUT/analyzer_stdout.txt"

echo "[done] B21.9 Fresh3 validation artifacts: $OUT"
