#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
DAPS="$REPO/external/daps"
B21_BASE=${B21_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver}
PYTHON_BIN=${PYTHON_BIN:-/egr/research-pac/huang248/conda-envs/daps/bin/python}
IMAGE_ROOT=${IMAGE_ROOT:-/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024}
IMAGES=${IMAGES:-"68263 63803 66452 63282 66892 69293 68924 65808 62802 65960"}
CASE_IDS=${CASE_IDS:-"0 1 2 3 4 5 6 7"}
GPUS=${GPUS:-"0 1 2 3"}
MEAS_PANEL_SEED=${MEAS_PANEL_SEED:-5101}
MEAS_TAG=${MEAS_TAG:-5101}
BASE_SEED0=${BASE_SEED0:-10000}
HIO_SEED0=${HIO_SEED0:-11000}
WARM_NOISE_SEED0=${WARM_NOISE_SEED0:-12000}
ANN_STEPS=${ANN_STEPS:-400}
DIFF_STEPS=${DIFF_STEPS:-5}
INJECT_STEP=${INJECT_STEP:-200}
THETA=${THETA:-0.7}
LF_ALPHA=${LF_ALPHA:-0.50}
LF_FRAC=${LF_FRAC:-0.35}
LF_RADIUS_FRAC=${LF_RADIUS_FRAC:-0.12}
HIO_ITERS=${HIO_ITERS:-240}
HIO_BETA=${HIO_BETA:-0.9}
HIO_ER_EVERY=${HIO_ER_EVERY:-20}
HIO_FINAL_ER=${HIO_FINAL_ER:-10}
B21_FORCE=${B21_FORCE:-0}
B21_SMOKE_ONLY=${B21_SMOKE_ONLY:-0}
OUT=${OUT:-$B21_BASE/B21_5_three_arm_fresh_val10x8_meas${MEAS_TAG}}
MEAS_DIR="$OUT/measurements"
CASE_ROOT="$OUT/cases"
LOGDIR="$OUT/logs"
MANIFEST="$OUT/manifest.tsv"
ACTIVE_MANIFEST="$OUT/active_manifest.tsv"
DONE="$OUT/done.tsv"
FAIL="$OUT/fail.tsv"
MEAS_GENERATOR="$REPO/scripts/b21/generate_b21_5_fresh_locked_measurements.py"
HIO_GENERATOR="$REPO/scripts/b21/generate_b21_5_hio_state.py"
METRIC_ANALYZER="$REPO/scripts/b19/analyze_daps_exact_final_loss_selector.py"
FINAL_ANALYZER="$REPO/scripts/b21/analyze_b21_5_three_arm_fresh_validation.py"
REPORT="$REPO/docs/b21/b21_5_three_arm_fresh_validation.md"

cd "$REPO"
mkdir -p "$OUT" "$MEAS_DIR" "$CASE_ROOT" "$LOGDIR"

[[ -x "$PYTHON_BIN" ]] || { echo "[fatal] PYTHON_BIN not executable: $PYTHON_BIN" >&2; exit 2; }
for file in "$MEAS_GENERATOR" "$HIO_GENERATOR" "$METRIC_ANALYZER" "$FINAL_ANALYZER"; do
  [[ -f "$file" ]] || { echo "[fatal] missing $file" >&2; exit 2; }
done
grep -q 'B20.11 experimental low-frequency measurement guidance' "$DAPS/sampler.py" || {
  echo "[fatal] B20 LF patch missing from external/daps/sampler.py" >&2; exit 2;
}
grep -q 'B21.3 continuation patch' "$DAPS/sampler.py" || {
  echo "[fatal] B21.3 continuation patch missing from external/daps/sampler.py" >&2; exit 2;
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
read -r -a IMAGE_ARR <<< "$IMAGES"
read -r -a CASE_ARR <<< "$CASE_IDS"
[[ ${#GPU_ARR[@]} -ge 1 ]] || { echo "[fatal] GPUS is empty" >&2; exit 2; }
[[ ${#IMAGE_ARR[@]} -eq 10 ]] || { echo "[fatal] expected exactly ten frozen fresh images" >&2; exit 2; }
[[ ${#CASE_ARR[@]} -eq 8 ]] || { echo "[fatal] expected exactly eight cases per image" >&2; exit 2; }

# Freeze new locked measurements before any solver run.  Existing files are
# reused byte-for-byte unless B21_FORCE=1.
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

# Prepare exact one-image DAPS configs.
for raw_image in "${IMAGE_ARR[@]}"; do
  image=$(printf '%05d' "$((10#$raw_image))")
  folder=$(printf '%05d' "$((10#$image / 1000 * 1000))")
  gt="$IMAGE_ROOT/$folder/${image}.png"
  if [[ ! -f "$gt" ]]; then
    gt=$(find "$IMAGE_ROOT" -type f -name "${image}.png" -print -quit)
  fi
  meas="$MEAS_DIR/ffhq${image}_phase_noise005_meas${MEAS_TAG}.pt"
  data_name="b21-fresh-ffhq-${image}"
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
printf "job_id\timage_id\tcase_id\tgpu\tbase_seed\thio_seed\twarm_noise_seed\tinject_step\tmeasurement_path\n" > "$MANIFEST"

job_id=0
for raw_image in "${IMAGE_ARR[@]}"; do
  image=$(printf '%05d' "$((10#$raw_image))")
  measurement_path="$MEAS_DIR/ffhq${image}_phase_noise005_meas${MEAS_TAG}.pt"
  for case_id in "${CASE_ARR[@]}"; do
    gpu="${GPU_ARR[$((job_id % ${#GPU_ARR[@]}))]}"
    base_seed=$((BASE_SEED0 + job_id))
    hio_seed=$((HIO_SEED0 + job_id))
    warm_seed=$((WARM_NOISE_SEED0 + job_id))
    printf "%d\t%s\t%d\t%s\t%d\t%d\t%d\t%d\t%s\n" \
      "$job_id" "$image" "$case_id" "$gpu" "$base_seed" "$hio_seed" "$warm_seed" \
      "$INJECT_STEP" "$measurement_path" >> "$MANIFEST"
    job_id=$((job_id + 1))
  done
done

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
  echo "base_seed0=$BASE_SEED0"
  echo "hio_seed0=$HIO_SEED0"
  echo "warm_noise_seed0=$WARM_NOISE_SEED0"
  echo "ann_steps=$ANN_STEPS"
  echo "diff_steps=$DIFF_STEPS"
  echo "inject_step=$INJECT_STEP"
  echo "theta=$THETA"
  echo "lf_alpha=$LF_ALPHA"
  echo "lf_frac=$LF_FRAC"
  echo "lf_radius_frac=$LF_RADIUS_FRAC"
  echo "hio_iters=$HIO_ITERS"
  echo "hio_beta=$HIO_BETA"
  echo "hio_er_every=$HIO_ER_EVERY"
  echo "hio_final_er=$HIO_FINAL_ER"
  echo "smoke_only=$B21_SMOKE_ONLY"
  echo "expected_full_cases=$job_id"
  echo "active_cases=$(($(wc -l < "$ACTIVE_MANIFEST") - 1))"
} > "$OUT/launch_env.txt"

run_daps() {
  local case_dir="$1" image="$2" candidate="$3" hydra_seed="$4" gpu="$5" meas="$6"
  shift 6
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
    echo "[skip][gpu=$gpu] image=$image candidate=$candidate"
    return 0
  fi
  if [[ "$B21_FORCE" == "1" ]]; then
    rm -rf "$save_dir/$candidate"
    rm -f "$csv" "$log"
  fi

  local start end elapsed
  start=$(date +%s)
  echo "[run][gpu=$gpu] image=$image candidate=$candidate seed=$hydra_seed"
  (
    cd "$DAPS"
    env \
      -u B21_CONT_ENABLE -u B21_CONT_STATE_PATH -u B21_CONT_NOISE_SEED \
      -u B21_SAVE_STATE_STEPS -u B21_START_NOISE_SEED -u B21_SOURCE_SEED \
      B21_MEASUREMENT_PATH="$meas" "$@" CUDA_VISIBLE_DEVICES="$gpu" \
      "$PYTHON_BIN" posterior_sample.py \
        +data="$data_name" +measurement_path="$meas" \
        +model=ffhq256ddpm +sampler=edm_daps +task=phase_retrieval \
        task_group=pixel batch_size=1 data.start_id=0 data.end_id=1 gpu=0 \
        seed="$hydra_seed" num_runs=1 name="$candidate" save_dir="$save_dir" \
        sampler.diffusion_scheduler_config.num_steps="$DIFF_STEPS" \
        sampler.annealing_scheduler_config.num_steps="$ANN_STEPS" \
        save_samples=true save_traj=false save_traj_raw_data=false save_traj_video=false
  ) > "$log" 2>&1

  [[ -s "$sample" ]] || { echo "[fatal] missing sample: $sample" >&2; tail -80 "$log" >&2; return 3; }
  "$PYTHON_BIN" "$METRIC_ANALYZER" \
    --daps_root "$DAPS" --samples_dir "$save_dir/$candidate/samples" \
    --measurement_path "$meas" --out_csv "$csv" >> "$log" 2>&1
  [[ -s "$csv" ]] || { echo "[fatal] missing metric CSV: $csv" >&2; return 3; }

  end=$(date +%s); elapsed=$((end - start)); touch "$timing_file"
  awk -F'\t' -v c="$candidate" '$1 != c' "$timing_file" > "${timing_file}.tmp" || true
  printf "%s\t%d\n" "$candidate" "$elapsed" >> "${timing_file}.tmp"
  mv "${timing_file}.tmp" "$timing_file"
  echo "[done][gpu=$gpu] image=$image candidate=$candidate elapsed=${elapsed}s"
}

run_case() {
  local job_id="$1" image="$2" case_id="$3" gpu="$4" base_seed="$5" hio_seed="$6" warm_seed="$7" inject_step="$8" meas="$9"
  local case_dir="$CASE_ROOT/${image}_case$(printf '%02d' "$case_id")"
  local hio_dir="$case_dir/hio"
  local state="$hio_dir/hio_state.pt"
  local hio_png="$hio_dir/hio_raw.png"
  local hio_json="$hio_dir/hio_summary.json"
  local hio_log="$case_dir/logs/hio_generate.log"
  local timing_file="$case_dir/timings.tsv"
  mkdir -p "$hio_dir" "$case_dir/logs"

  cat > "$case_dir/case_env.txt" <<EOF
job_id=$job_id
image_id=$image
case_id=$case_id
gpu=$gpu
base_seed=$base_seed
hio_seed=$hio_seed
warm_noise_seed=$warm_seed
inject_step=$inject_step
measurement_path=$meas
EOF

  run_daps "$case_dir" "$image" base_full "$base_seed" "$gpu" "$meas" \
    B20_LF_ENABLE=0 B20_LF_ALPHA=0.0 \
    B21_START_NOISE_SEED="$base_seed" B21_SOURCE_SEED="$base_seed"

  run_daps "$case_dir" "$image" lf050 "$base_seed" "$gpu" "$meas" \
    B21_START_NOISE_SEED="$base_seed" B21_SOURCE_SEED="$base_seed" \
    B20_LF_ENABLE=1 B20_LF_ALPHA="$LF_ALPHA" B20_LF_FRAC="$LF_FRAC" \
    B20_LF_RADIUS_FRAC="$LF_RADIUS_FRAC" B20_LF_VERBOSE=1

  if [[ "$B21_FORCE" == "1" || ! -s "$state" || ! -s "$hio_json" || ! -s "$hio_png" ]]; then
    local start end elapsed
    start=$(date +%s)
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$HIO_GENERATOR" \
      --measurement-path "$meas" --out-state "$state" --out-png "$hio_png" \
      --out-json "$hio_json" --seed "$hio_seed" --iterations "$HIO_ITERS" \
      --beta "$HIO_BETA" --er-every "$HIO_ER_EVERY" --final-er "$HIO_FINAL_ER" \
      --resolution 256 --inject-step "$inject_step" --device cuda:0 > "$hio_log" 2>&1
    end=$(date +%s); elapsed=$((end - start)); touch "$timing_file"
    awk -F'\t' '$1 != "hio_generate"' "$timing_file" > "${timing_file}.tmp" || true
    printf "hio_generate\t%d\n" "$elapsed" >> "${timing_file}.tmp"
    mv "${timing_file}.tmp" "$timing_file"
  else
    echo "[skip][gpu=$gpu] image=$image hio_generate"
  fi

  [[ -s "$state" ]] || { echo "[fatal] missing HIO state: $state" >&2; return 4; }
  run_daps "$case_dir" "$image" hio_warm "$warm_seed" "$gpu" "$meas" \
    B20_LF_ENABLE=0 B20_LF_ALPHA=0.0 \
    B21_CONT_ENABLE=1 B21_CONT_STATE_PATH="$state" B21_CONT_NOISE_SEED="$warm_seed"
}

pids=()
for gpu in "${GPU_ARR[@]}"; do
  (
    set -euo pipefail
    tail -n +2 "$ACTIVE_MANIFEST" | awk -F'\t' -v g="$gpu" '$4 == g {print}' | \
    while IFS=$'\t' read -r jid image case_id assigned_gpu base_seed hio_seed warm_seed inject_step meas; do
      [[ -z "${jid:-}" ]] && continue
      echo "[case start] job=$jid image=$image case=$case_id gpu=$assigned_gpu"
      if run_case "$jid" "$image" "$case_id" "$assigned_gpu" "$base_seed" "$hio_seed" "$warm_seed" "$inject_step" "$meas"; then
        printf "%s\t%s\t%s\t%s\tOK\n" "$jid" "$image" "$case_id" "$assigned_gpu" >> "$DONE"
        echo "[case done] job=$jid image=$image case=$case_id gpu=$assigned_gpu"
      else
        status=$?
        printf "%s\t%s\t%s\t%s\tFAIL\t%s\n" "$jid" "$image" "$case_id" "$assigned_gpu" "$status" >> "$FAIL"
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
  echo "[smoke-only] completed first fresh three-arm case; analyzer intentionally skipped"
  exit 0
fi

"$PYTHON_BIN" "$FINAL_ANALYZER" \
  --repo "$REPO" --out "$OUT" --manifest "$MANIFEST" --theta "$THETA" \
  --report "$REPORT" | tee "$OUT/analyzer_stdout.txt"

echo "[done] B21.5 fresh three-arm validation artifacts: $OUT"
