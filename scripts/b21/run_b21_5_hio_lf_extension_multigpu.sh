#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
DAPS="$REPO/external/daps"
B19_BASE=${B19_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver}
B21_BASE=${B21_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver}
PYTHON_BIN=${PYTHON_BIN:-/egr/research-pac/huang248/conda-envs/daps/bin/python}
HIO_OUT=${HIO_OUT:-$B21_BASE/B21_5_hio_warmstart_pilot_5img_8seed_step200}
OUT=${OUT:-$B21_BASE/B21_5_hio_lf_extension_5img_8seed}
GPUS=${GPUS:-"0 1 2 3"}
MEAS_SEED=${MEAS_SEED:-5001}
ANN_STEPS=${ANN_STEPS:-400}
DIFF_STEPS=${DIFF_STEPS:-5}
THETA=${THETA:-0.7}
LF_ALPHA=${LF_ALPHA:-0.50}
LF_FRAC=${LF_FRAC:-0.35}
LF_RADIUS_FRAC=${LF_RADIUS_FRAC:-0.12}
B21_FORCE=${B21_FORCE:-0}
TUNED_IMAGE=${TUNED_IMAGE:-00046}

HIO_MANIFEST="$HIO_OUT/manifest.tsv"
HIO_ROWS="$HIO_OUT/hio_warmstart_pilot_rows.csv"
LOGDIR="$OUT/logs"
CASE_ROOT="$OUT/cases"
MANIFEST="$OUT/manifest.tsv"
DONE="$OUT/done.tsv"
FAIL="$OUT/fail.tsv"
ANALYZER="$REPO/scripts/b19/analyze_daps_exact_final_loss_selector.py"
PORTFOLIO_ANALYZER="$REPO/scripts/b21/analyze_b21_5_three_arm_portfolio.py"
DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024

cd "$REPO"
mkdir -p "$OUT" "$LOGDIR" "$CASE_ROOT"

[[ -x "$PYTHON_BIN" ]] || { echo "[fatal] PYTHON_BIN not executable: $PYTHON_BIN" >&2; exit 2; }
for file in "$HIO_MANIFEST" "$HIO_ROWS" "$ANALYZER" "$PORTFOLIO_ANALYZER"; do
  [[ -f "$file" ]] || { echo "[fatal] missing $file" >&2; exit 2; }
done
grep -q 'B20.11 experimental low-frequency measurement guidance' "$DAPS/sampler.py" || {
  echo "[fatal] B20 LF patch is not present in external/daps/sampler.py" >&2
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

read -r -a GPU_ARR <<< "$GPUS"
[[ ${#GPU_ARR[@]} -ge 1 ]] || { echo "[fatal] GPUS is empty" >&2; exit 2; }

# Prepare the exact image configs referenced by the frozen HIO manifest.
tail -n +2 "$HIO_MANIFEST" | cut -f2 | sort -u | while read -r raw_image; do
  [[ -z "${raw_image:-}" ]] && continue
  image=$(printf '%05d' "$((10#$raw_image))")
  gt="$DATA_ROOT/00000/${image}.png"
  meas="$B19_BASE/measurements/ffhq${image}_phase_noise005_meas${MEAS_SEED}.pt"
  data_name="b21-ffhq-${image}"
  [[ -f "$gt" ]] || { echo "[fatal] missing image: $gt" >&2; exit 2; }
  [[ -f "$meas" ]] || { echo "[fatal] missing measurement: $meas" >&2; exit 2; }
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
printf "job_id\timage_id\tcase_id\tgpu\tbase_seed\n" > "$MANIFEST"

idx=0
tail -n +2 "$HIO_MANIFEST" | while IFS=$'\t' read -r job_id image case_id old_gpu base_seed hio_seed warm_seed; do
  [[ -z "${job_id:-}" ]] && continue
  gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
  printf "%s\t%s\t%s\t%s\t%s\n" "$job_id" "$image" "$case_id" "$gpu" "$base_seed" >> "$MANIFEST"
  idx=$((idx + 1))
done

{
  echo "timestamp=$(date -Is)"
  echo "repo=$REPO"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "hio_out=$HIO_OUT"
  echo "hio_manifest=$HIO_MANIFEST"
  echo "hio_rows=$HIO_ROWS"
  echo "gpus=$GPUS"
  echo "measurement_seed=$MEAS_SEED"
  echo "ann_steps=$ANN_STEPS"
  echo "diff_steps=$DIFF_STEPS"
  echo "theta=$THETA"
  echo "lf_alpha=$LF_ALPHA"
  echo "lf_frac=$LF_FRAC"
  echo "lf_radius_frac=$LF_RADIUS_FRAC"
  echo "expected_cases=$(($(wc -l < "$MANIFEST") - 1))"
} > "$OUT/launch_env.txt"

run_lf() {
  local job_id="$1" image="$2" case_id="$3" gpu="$4" base_seed="$5"
  local case_dir="$CASE_ROOT/${image}_case$(printf '%02d' "$case_id")"
  local save_dir="$case_dir/daps_results"
  local metric_dir="$case_dir/metrics"
  local log_dir="$case_dir/logs"
  local timing_file="$case_dir/timings.tsv"
  local candidate="lf050"
  local sample="$save_dir/$candidate/samples/00000_run0000.png"
  local csv="$metric_dir/${candidate}.csv"
  local log="$log_dir/${candidate}.log"
  local meas="$B19_BASE/measurements/ffhq${image}_phase_noise005_meas${MEAS_SEED}.pt"
  local data_name="b21-ffhq-${image}"
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
  echo "[run][gpu=$gpu] job=$job_id image=$image case=$case_id seed=$base_seed candidate=$candidate"
  (
    cd "$DAPS"
    env \
      -u B21_CONT_ENABLE -u B21_CONT_STATE_PATH -u B21_CONT_NOISE_SEED \
      -u B21_SAVE_STATE_STEPS \
      B21_START_NOISE_SEED="$base_seed" \
      B21_SOURCE_SEED="$base_seed" \
      B21_MEASUREMENT_PATH="$meas" \
      B20_LF_ENABLE=1 \
      B20_LF_ALPHA="$LF_ALPHA" \
      B20_LF_FRAC="$LF_FRAC" \
      B20_LF_RADIUS_FRAC="$LF_RADIUS_FRAC" \
      B20_LF_VERBOSE=1 \
      CUDA_VISIBLE_DEVICES="$gpu" \
      "$PYTHON_BIN" posterior_sample.py \
        +data="$data_name" +measurement_path="$meas" \
        +model=ffhq256ddpm +sampler=edm_daps +task=phase_retrieval \
        task_group=pixel batch_size=1 data.start_id=0 data.end_id=1 gpu=0 \
        seed="$base_seed" num_runs=1 name="$candidate" save_dir="$save_dir" \
        sampler.diffusion_scheduler_config.num_steps="$DIFF_STEPS" \
        sampler.annealing_scheduler_config.num_steps="$ANN_STEPS" \
        save_samples=true save_traj=false save_traj_raw_data=false save_traj_video=false
  ) > "$log" 2>&1

  [[ -s "$sample" ]] || { echo "[fatal] missing sample: $sample" >&2; tail -80 "$log" >&2; return 3; }
  "$PYTHON_BIN" "$ANALYZER" \
    --daps_root "$DAPS" \
    --samples_dir "$save_dir/$candidate/samples" \
    --measurement_path "$meas" \
    --out_csv "$csv" >> "$log" 2>&1
  [[ -s "$csv" ]] || { echo "[fatal] missing metric CSV: $csv" >&2; return 3; }

  end=$(date +%s)
  elapsed=$((end - start))
  printf "lf050\t%d\n" "$elapsed" > "$timing_file"
  echo "[done][gpu=$gpu] job=$job_id image=$image case=$case_id elapsed=${elapsed}s"
}

pids=()
for gpu in "${GPU_ARR[@]}"; do
  (
    set -euo pipefail
    tail -n +2 "$MANIFEST" | awk -F'\t' -v g="$gpu" '$4 == g {print}' | \
    while IFS=$'\t' read -r job_id image case_id assigned_gpu base_seed; do
      [[ -z "${job_id:-}" ]] && continue
      echo "[case start] job=$job_id image=$image case=$case_id gpu=$assigned_gpu"
      if run_lf "$job_id" "$image" "$case_id" "$assigned_gpu" "$base_seed"; then
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

"$PYTHON_BIN" "$PORTFOLIO_ANALYZER" \
  --hio-rows "$HIO_ROWS" \
  --lf-out "$OUT" \
  --outdir "$OUT/three_arm_analysis_theta${THETA}" \
  --theta "$THETA" \
  --tuned-image "$TUNED_IMAGE" \
  --report "$REPO/docs/b21/b21_5_three_arm_portfolio.md" \
  | tee "$OUT/analyzer_stdout.txt"

echo "[done] B21.5 LF extension and three-arm analysis: $OUT"
