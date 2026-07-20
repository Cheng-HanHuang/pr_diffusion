#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
DAPS="$REPO/external/daps"
PYTHON_BIN=${PYTHON_BIN:-/egr/research-pac/huang248/conda-envs/daps/bin/python}
FRESH_OUT=${FRESH_OUT:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_5_three_arm_fresh_val10x8_meas5101}
SOURCE_MANIFEST=${SOURCE_MANIFEST:-$FRESH_OUT/manifest.tsv}
IMAGE_ROOT=${IMAGE_ROOT:-/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024}
GPUS=${GPUS:-"0 1 2 3"}
CAL_JOBS=${CAL_JOBS:-"0 8 16 24 32 40 48 56"}
CAL_SEED0=${CAL_SEED0:-14000}
ANN_STEPS=${ANN_STEPS:-400}
DIFF_STEPS=${DIFF_STEPS:-5}
B21_FORCE=${B21_FORCE:-0}
OUT=${OUT:-$FRESH_OUT/b21_7_runtime_calibration}
LOGDIR="$OUT/logs"
CASE_ROOT="$OUT/cases"
MANIFEST="$OUT/manifest.tsv"
PAIRS="$OUT/runtime_pairs.tsv"
DONE="$OUT/done.tsv"
FAIL="$OUT/fail.tsv"
ANALYZER="$REPO/scripts/b21/analyze_b21_7_runtime_calibration.py"
REPORT="$REPO/docs/b21/b21_7_runtime_calibration.md"
ANALYSIS="$OUT/analysis"

cd "$REPO"
mkdir -p "$OUT" "$LOGDIR" "$CASE_ROOT"

[[ -x "$PYTHON_BIN" ]] || { echo "[fatal] PYTHON_BIN not executable: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$SOURCE_MANIFEST" ]] || { echo "[fatal] missing source manifest: $SOURCE_MANIFEST" >&2; exit 2; }
[[ -f "$ANALYZER" ]] || { echo "[fatal] missing analyzer: $ANALYZER" >&2; exit 2; }

grep -q 'B20.11 experimental low-frequency measurement guidance' "$DAPS/sampler.py" || {
  echo "[fatal] B20 LF patch missing from external/daps/sampler.py" >&2; exit 2;
}

"$PYTHON_BIN" - <<'PY'
import sys, torch, yaml, pandas, PIL, torchvision
print("python", sys.executable)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("yaml", yaml.__version__)
print("pandas", pandas.__version__)
print("PIL", PIL.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
PY

read -r -a GPU_ARR <<< "$GPUS"
read -r -a JOB_ARR <<< "$CAL_JOBS"
[[ ${#GPU_ARR[@]} -eq 4 ]] || { echo "[fatal] expected exactly four GPUs" >&2; exit 2; }
[[ ${#JOB_ARR[@]} -eq 8 ]] || { echo "[fatal] expected exactly eight calibration jobs" >&2; exit 2; }

# Recreate exact one-image DAPS configs for the selected frozen cases.
: > "$MANIFEST"
printf "slot\tjob_id\timage_id\tcase_id\tgpu\torder\tseed\tmeasurement_path\n" >> "$MANIFEST"
slot=0
for wanted in "${JOB_ARR[@]}"; do
  line=$(awk -F'\t' -v j="$wanted" 'NR>1 && $1==j {print; exit}' "$SOURCE_MANIFEST")
  [[ -n "$line" ]] || { echo "[fatal] job $wanted not found in $SOURCE_MANIFEST" >&2; exit 2; }
  IFS=$'\t' read -r job_id image case_id old_gpu base_seed hio_seed warm_seed inject_step measurement_path <<< "$line"
  image=$(printf '%05d' "$((10#$image))")
  gpu="${GPU_ARR[$((slot % 4))]}"
  seed=$((CAL_SEED0 + slot))
  if (( slot % 2 == 0 )); then order=base_first; else order=lf_first; fi
  printf "%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\n" \
    "$slot" "$job_id" "$image" "$case_id" "$gpu" "$order" "$seed" "$measurement_path" >> "$MANIFEST"

  folder=$(printf '%05d' "$((10#$image / 1000 * 1000))")
  gt="$IMAGE_ROOT/$folder/${image}.png"
  if [[ ! -f "$gt" ]]; then gt=$(find "$IMAGE_ROOT" -type f -name "${image}.png" -print -quit); fi
  [[ -f "$gt" ]] || { echo "[fatal] missing GT image $image" >&2; exit 2; }
  [[ -f "$measurement_path" ]] || { echo "[fatal] missing measurement $measurement_path" >&2; exit 2; }
  data_name="b21-fresh-ffhq-${image}"
  mkdir -p "$DAPS/dataset/$data_name" "$DAPS/configs/data"
  ln -sfn "$gt" "$DAPS/dataset/$data_name/${image}.png"
  cat > "$DAPS/configs/data/${data_name}.yaml" <<YAML
name: image
root: 'dataset/${data_name}'
resolution: 256
start_id: 0
end_id: 1
YAML
  slot=$((slot + 1))
done

: > "$DONE"
: > "$FAIL"
printf "job_id\timage_id\tcase_id\tgpu\torder\tseed\tbase_seconds\tlf_seconds\n" > "$PAIRS"

{
  echo "timestamp=$(date -Is)"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "gpus=$GPUS"
  echo "cal_jobs=$CAL_JOBS"
  echo "cal_seed0=$CAL_SEED0"
  echo "ann_steps=$ANN_STEPS"
  echo "diff_steps=$DIFF_STEPS"
  echo "force=$B21_FORCE"
} > "$OUT/launch_env.txt"

run_candidate() {
  local case_dir="$1" image="$2" candidate="$3" seed="$4" gpu="$5" meas="$6" lf_enable="$7"
  local save_dir="$case_dir/daps_results"
  local log_dir="$case_dir/logs"
  local sample="$save_dir/$candidate/samples/00000_run0000.png"
  local log="$log_dir/${candidate}.log"
  local timing="$case_dir/${candidate}.seconds"
  local data_name="b21-fresh-ffhq-${image}"
  mkdir -p "$save_dir" "$log_dir"

  if [[ "$B21_FORCE" != "1" && -s "$sample" && -s "$timing" ]]; then
    cat "$timing"
    return 0
  fi
  if [[ "$B21_FORCE" == "1" ]]; then
    rm -rf "$save_dir/$candidate"
    rm -f "$log" "$timing"
  fi

  local start end elapsed
  start=$(date +%s)
  (
    cd "$DAPS"
    env \
      -u B21_CONT_ENABLE -u B21_CONT_STATE_PATH -u B21_CONT_NOISE_SEED \
      -u B21_SAVE_STATE_STEPS \
      B21_START_NOISE_SEED="$seed" \
      B21_SOURCE_SEED="$seed" \
      B21_MEASUREMENT_PATH="$meas" \
      B20_LF_ENABLE="$lf_enable" \
      B20_LF_ALPHA="$([[ "$lf_enable" == "1" ]] && echo 0.50 || echo 0.0)" \
      B20_LF_FRAC=0.35 \
      B20_LF_RADIUS_FRAC=0.12 \
      B20_LF_VERBOSE=1 \
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
  end=$(date +%s)
  elapsed=$((end - start))
  [[ -s "$sample" ]] || { echo "[fatal] missing sample $sample" >&2; tail -80 "$log" >&2; return 3; }
  printf "%d\n" "$elapsed" > "$timing"
  printf "%d\n" "$elapsed"
}

run_pair() {
  local job_id="$1" image="$2" case_id="$3" gpu="$4" order="$5" seed="$6" meas="$7"
  local case_dir="$CASE_ROOT/job$(printf '%03d' "$job_id")_${image}_case$(printf '%02d' "$case_id")"
  mkdir -p "$case_dir"
  local base_seconds lf_seconds
  if [[ "$order" == "base_first" ]]; then
    base_seconds=$(run_candidate "$case_dir" "$image" timing_base "$seed" "$gpu" "$meas" 0)
    lf_seconds=$(run_candidate "$case_dir" "$image" timing_lf "$seed" "$gpu" "$meas" 1)
  else
    lf_seconds=$(run_candidate "$case_dir" "$image" timing_lf "$seed" "$gpu" "$meas" 1)
    base_seconds=$(run_candidate "$case_dir" "$image" timing_base "$seed" "$gpu" "$meas" 0)
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$job_id" "$image" "$case_id" "$gpu" "$order" "$seed" "$base_seconds" "$lf_seconds" >> "$PAIRS"
}

pids=()
for gpu in "${GPU_ARR[@]}"; do
  (
    set -euo pipefail
    tail -n +2 "$MANIFEST" | awk -F'\t' -v g="$gpu" '$5==g {print}' | \
    while IFS=$'\t' read -r slot job_id image case_id assigned_gpu order seed meas; do
      [[ -z "${job_id:-}" ]] && continue
      echo "[pair start] slot=$slot job=$job_id image=$image gpu=$assigned_gpu order=$order"
      if run_pair "$job_id" "$image" "$case_id" "$assigned_gpu" "$order" "$seed" "$meas"; then
        printf "%s\t%s\t%s\t%s\tOK\n" "$job_id" "$image" "$case_id" "$assigned_gpu" >> "$DONE"
        echo "[pair done] slot=$slot job=$job_id image=$image gpu=$assigned_gpu"
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
if [[ "$status" -ne 0 || "$failed" -ne 0 || "$completed" -ne 8 ]]; then
  cat "$FAIL" >&2
  exit 5
fi

"$PYTHON_BIN" "$ANALYZER" \
  --pairs "$PAIRS" \
  --fresh-out "$FRESH_OUT" \
  --outdir "$ANALYSIS" \
  --report "$REPORT" \
  | tee "$OUT/analyzer_stdout.txt"

echo "[done] B21.7 runtime calibration artifacts: $OUT"
