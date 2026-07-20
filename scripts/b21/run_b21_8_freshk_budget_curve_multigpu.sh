#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
DAPS="$REPO/external/daps"
PYTHON_BIN=${PYTHON_BIN:-/egr/research-pac/huang248/conda-envs/daps/bin/python}
FRESH_OUT=${FRESH_OUT:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_5_three_arm_fresh_val10x8_meas5101}
FRESH_ROWS=${FRESH_ROWS:-$FRESH_OUT/fresh_analysis_theta0.7/fresh_three_arm_rows.csv}
SOURCE_MANIFEST=${SOURCE_MANIFEST:-$FRESH_OUT/manifest.tsv}
IMAGE_ROOT=${IMAGE_ROOT:-/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024}
GPUS=${GPUS:-"0 1 2 3"}
EXTRA2_SEED0=${EXTRA2_SEED0:-15000}
EXTRA3_SEED0=${EXTRA3_SEED0:-16000}
ANN_STEPS=${ANN_STEPS:-400}
DIFF_STEPS=${DIFF_STEPS:-5}
THETA=${THETA:-0.7}
TARGET_GOOD=${TARGET_GOOD:-76}
MAX_ORACLE_GAP=${MAX_ORACLE_GAP:-1}
MAX_CUMULATIVE_HARMS=${MAX_CUMULATIVE_HARMS:-1}
MIN_PER_IMAGE_GOOD=${MIN_PER_IMAGE_GOOD:-6}
B21_FORCE=${B21_FORCE:-0}
B21_SMOKE_ONLY=${B21_SMOKE_ONLY:-0}
OUT=${OUT:-$FRESH_OUT/b21_8_freshk_budget_curve}
LOGDIR="$OUT/logs"
MANIFEST="$OUT/manifest.tsv"
ACTIVE_MANIFEST="$OUT/active_manifest.tsv"
DONE="$OUT/done.tsv"
FAIL="$OUT/fail.tsv"
METRIC_ANALYZER="$REPO/scripts/b19/analyze_daps_exact_final_loss_selector.py"
FINAL_ANALYZER="$REPO/scripts/b21/analyze_b21_8_freshk_budget_curve.py"
ANALYSIS="$OUT/analysis_theta${THETA}"
REPORT="$REPO/docs/b21/b21_8_freshk_budget_curve.md"

cd "$REPO"
mkdir -p "$OUT" "$LOGDIR"

[[ -x "$PYTHON_BIN" ]] || { echo "[fatal] PYTHON_BIN not executable: $PYTHON_BIN" >&2; exit 2; }
for file in "$SOURCE_MANIFEST" "$FRESH_ROWS" "$METRIC_ANALYZER" "$FINAL_ANALYZER"; do
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
[[ ${#GPU_ARR[@]} -ge 1 ]] || { echo "[fatal] GPUS is empty" >&2; exit 2; }

# Recreate the frozen one-image DAPS configs from the source manifest.
tail -n +2 "$SOURCE_MANIFEST" | cut -f2 | sort -u | while read -r raw_image; do
  [[ -z "${raw_image:-}" ]] && continue
  image=$(printf '%05d' "$((10#$raw_image))")
  folder=$(printf '%05d' "$((10#$image / 1000 * 1000))")
  gt="$IMAGE_ROOT/$folder/${image}.png"
  if [[ ! -f "$gt" ]]; then
    gt=$(find "$IMAGE_ROOT" -type f -name "${image}.png" -print -quit)
  fi
  data_name="b21-fresh-ffhq-${image}"
  [[ -f "$gt" ]] || { echo "[fatal] missing image $image" >&2; exit 2; }
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
printf "job_id\timage_id\tcase_id\tgpu\tbase_seed\textra1_seed\textra2_seed\textra3_seed\tmeasurement_path\torder\n" > "$MANIFEST"

while IFS=$'\t' read -r job_id image case_id old_gpu base_seed hio_seed warm_seed inject_step measurement_path; do
  [[ "$job_id" == "job_id" ]] && continue
  [[ -z "${job_id:-}" ]] && continue
  gpu="${GPU_ARR[$((job_id % ${#GPU_ARR[@]}))]}"
  extra1_seed=$((13000 + job_id))
  extra2_seed=$((EXTRA2_SEED0 + job_id))
  extra3_seed=$((EXTRA3_SEED0 + job_id))
  if (( job_id % 2 == 0 )); then
    order="extra2_first"
  else
    order="extra3_first"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$job_id" "$image" "$case_id" "$gpu" "$base_seed" "$extra1_seed" \
    "$extra2_seed" "$extra3_seed" "$measurement_path" "$order" >> "$MANIFEST"
done < "$SOURCE_MANIFEST"

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
  echo "fresh_out=$FRESH_OUT"
  echo "gpus=$GPUS"
  echo "extra2_seed0=$EXTRA2_SEED0"
  echo "extra3_seed0=$EXTRA3_SEED0"
  echo "ann_steps=$ANN_STEPS"
  echo "diff_steps=$DIFF_STEPS"
  echo "theta=$THETA"
  echo "target_good=$TARGET_GOOD"
  echo "max_oracle_gap=$MAX_ORACLE_GAP"
  echo "max_cumulative_harms=$MAX_CUMULATIVE_HARMS"
  echo "min_per_image_good=$MIN_PER_IMAGE_GOOD"
  echo "smoke_only=$B21_SMOKE_ONLY"
  echo "expected_cases=$rows"
  echo "active_cases=$(($(wc -l < "$ACTIVE_MANIFEST") - 1))"
} > "$OUT/launch_env.txt"

run_candidate() {
  local job_id="$1" image="$2" case_id="$3" gpu="$4" seed="$5" meas="$6" candidate="$7"
  local case_dir="$FRESH_OUT/cases/${image}_case$(printf '%02d' "$case_id")"
  local save_dir="$case_dir/daps_results"
  local metric_dir="$case_dir/metrics"
  local log_dir="$case_dir/logs"
  local timing_file="$case_dir/timings.tsv"
  local sample="$save_dir/$candidate/samples/00000_run0000.png"
  local csv="$metric_dir/${candidate}.csv"
  local log="$log_dir/${candidate}.log"
  local data_name="b21-fresh-ffhq-${image}"
  mkdir -p "$save_dir" "$metric_dir" "$log_dir"

  if [[ "$B21_FORCE" != "1" && -s "$sample" && -s "$csv" ]]; then
    echo "[skip][gpu=$gpu] job=$job_id image=$image case=$case_id candidate=$candidate"
    return 0
  fi
  if [[ "$B21_FORCE" == "1" ]]; then
    rm -rf "$save_dir/$candidate"
    rm -f "$csv" "$log"
  fi

  local start end elapsed
  start=$(date +%s)
  echo "[run][gpu=$gpu] job=$job_id image=$image case=$case_id candidate=$candidate seed=$seed"
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
  echo "[done][gpu=$gpu] job=$job_id candidate=$candidate elapsed=${elapsed}s"
}

run_case() {
  local job_id="$1" image="$2" case_id="$3" gpu="$4" base_seed="$5" extra1_seed="$6" extra2_seed="$7" extra3_seed="$8" meas="$9" order="${10}"
  if [[ "$order" == "extra2_first" ]]; then
    run_candidate "$job_id" "$image" "$case_id" "$gpu" "$extra2_seed" "$meas" base_extra2
    run_candidate "$job_id" "$image" "$case_id" "$gpu" "$extra3_seed" "$meas" base_extra3
  else
    run_candidate "$job_id" "$image" "$case_id" "$gpu" "$extra3_seed" "$meas" base_extra3
    run_candidate "$job_id" "$image" "$case_id" "$gpu" "$extra2_seed" "$meas" base_extra2
  fi
}

pids=()
for gpu in "${GPU_ARR[@]}"; do
  (
    set -euo pipefail
    tail -n +2 "$ACTIVE_MANIFEST" | awk -F'\t' -v g="$gpu" '$4 == g {print}' | \
    while IFS=$'\t' read -r job_id image case_id assigned_gpu base_seed extra1_seed extra2_seed extra3_seed meas order; do
      [[ -z "${job_id:-}" ]] && continue
      echo "[case start] job=$job_id image=$image case=$case_id gpu=$assigned_gpu order=$order"
      if run_case "$job_id" "$image" "$case_id" "$assigned_gpu" "$base_seed" "$extra1_seed" "$extra2_seed" "$extra3_seed" "$meas" "$order"; then
        printf "%s\t%s\t%s\t%s\tOK\n" "$job_id" "$image" "$case_id" "$assigned_gpu" >> "$DONE"
        echo "[case done] job=$job_id image=$image case=$case_id gpu=$assigned_gpu"
      else
        status=$?
        printf "%s\t%s\t%s\t%s\tFAIL\t%s\n" "$job_id" "$image" "$case_id" "$assigned_gpu" "$status" >> "$FAIL"
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
  echo "[smoke-only] completed first Fresh3/Fresh4 extension case; analyzer intentionally skipped"
  exit 0
fi

"$PYTHON_BIN" "$FINAL_ANALYZER" \
  --fresh-out "$FRESH_OUT" \
  --fresh-rows "$FRESH_ROWS" \
  --manifest "$MANIFEST" \
  --outdir "$ANALYSIS" \
  --theta "$THETA" \
  --target-good "$TARGET_GOOD" \
  --max-oracle-gap "$MAX_ORACLE_GAP" \
  --max-cumulative-harms "$MAX_CUMULATIVE_HARMS" \
  --min-per-image-good "$MIN_PER_IMAGE_GOOD" \
  --image-root "$IMAGE_ROOT" \
  --report "$REPORT" \
  | tee "$OUT/analyzer_stdout.txt"

echo "[done] B21.8 FreshK budget-curve artifacts: $OUT"
