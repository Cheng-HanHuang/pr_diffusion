#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
DAPS="$REPO/external/daps"
B19_BASE=${B19_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver}
B21_BASE=${B21_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver}
PYTHON_BIN=${PYTHON_BIN:-/egr/research-pac/huang248/conda-envs/daps/bin/python}
IMAGES=${IMAGES:-"00046 00171 00224 00746 00971"}
CASE_IDS=${CASE_IDS:-"0 1 2 3 4 5 6 7"}
GPUS=${GPUS:-"0 1 2 3"}
MEAS_SEED=${MEAS_SEED:-5001}
BASE_SEED0=${BASE_SEED0:-7400}
HIO_SEED0=${HIO_SEED0:-8400}
WARM_NOISE_SEED0=${WARM_NOISE_SEED0:-9400}
ANN_STEPS=${ANN_STEPS:-400}
DIFF_STEPS=${DIFF_STEPS:-5}
INJECT_STEP=${INJECT_STEP:-200}
HIO_ITERS=${HIO_ITERS:-240}
HIO_BETA=${HIO_BETA:-0.9}
HIO_ER_EVERY=${HIO_ER_EVERY:-20}
HIO_FINAL_ER=${HIO_FINAL_ER:-10}
B21_FORCE=${B21_FORCE:-0}
OUT=${OUT:-$B21_BASE/B21_5_hio_warmstart_pilot_5img_8seed_step${INJECT_STEP}}
LOGDIR="$OUT/logs"
CASE_ROOT="$OUT/cases"
MANIFEST="$OUT/manifest.tsv"
DONE="$OUT/done.tsv"
FAIL="$OUT/fail.tsv"
GENERATOR="$REPO/scripts/b21/generate_b21_5_hio_state.py"
ANALYZER="$REPO/scripts/b19/analyze_daps_exact_final_loss_selector.py"
PILOT_ANALYZER="$REPO/scripts/b21/analyze_b21_5_hio_warmstart_pilot.py"
DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024

cd "$REPO"
mkdir -p "$OUT" "$LOGDIR" "$CASE_ROOT"

[[ -x "$PYTHON_BIN" ]] || { echo "[fatal] PYTHON_BIN not executable: $PYTHON_BIN" >&2; exit 2; }
for file in "$GENERATOR" "$ANALYZER" "$PILOT_ANALYZER"; do
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

read -r -a GPU_ARR <<< "$GPUS"
read -r -a IMAGE_ARR <<< "$IMAGES"
read -r -a CASE_ARR <<< "$CASE_IDS"
[[ ${#GPU_ARR[@]} -ge 1 ]] || { echo "[fatal] GPUS is empty" >&2; exit 2; }
[[ ${#IMAGE_ARR[@]} -eq 5 ]] || { echo "[fatal] expected exactly five images" >&2; exit 2; }
[[ ${#CASE_ARR[@]} -ge 1 ]] || { echo "[fatal] CASE_IDS is empty" >&2; exit 2; }

# Prepare one-image configs before workers start.
for raw_image in "${IMAGE_ARR[@]}"; do
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
printf "job_id\timage_id\tcase_id\tgpu\tbase_seed\thio_seed\twarm_noise_seed\n" > "$MANIFEST"

jid=0
for raw_image in "${IMAGE_ARR[@]}"; do
  image=$(printf '%05d' "$((10#$raw_image))")
  for case_id in "${CASE_ARR[@]}"; do
    gpu="${GPU_ARR[$((jid % ${#GPU_ARR[@]}))]}"
    base_seed=$((BASE_SEED0 + jid))
    hio_seed=$((HIO_SEED0 + jid))
    warm_seed=$((WARM_NOISE_SEED0 + jid))
    printf "%d\t%s\t%d\t%s\t%d\t%d\t%d\n" \
      "$jid" "$image" "$case_id" "$gpu" "$base_seed" "$hio_seed" "$warm_seed" >> "$MANIFEST"
    jid=$((jid + 1))
  done
done

{
  echo "timestamp=$(date -Is)"
  echo "repo=$REPO"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "images=$IMAGES"
  echo "case_ids=$CASE_IDS"
  echo "gpus=$GPUS"
  echo "measurement_seed=$MEAS_SEED"
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
  echo "expected_cases=$jid"
} > "$OUT/launch_env.txt"

echo "[manifest] $MANIFEST"
echo "[cases] $jid"
echo "[gpus] $GPUS"
echo "[out] $OUT"

run_daps() {
  local case_dir="$1" image="$2" candidate="$3" hydra_seed="$4" gpu="$5"
  shift 5
  local save_dir="$case_dir/daps_results"
  local metric_dir="$case_dir/metrics"
  local log_dir="$case_dir/logs"
  local timing_file="$case_dir/timings.tsv"
  local sample="$save_dir/$candidate/samples/00000_run0000.png"
  local csv="$metric_dir/${candidate}.csv"
  local log="$log_dir/${candidate}.log"
  local meas="$B19_BASE/measurements/ffhq${image}_phase_noise005_meas${MEAS_SEED}.pt"
  local data_name="b21-ffhq-${image}"
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
      B20_LF_ENABLE=0 B20_LF_ALPHA=0.0 B21_MEASUREMENT_PATH="$meas" \
      "$@" CUDA_VISIBLE_DEVICES="$gpu" \
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
  "$PYTHON_BIN" "$ANALYZER" \
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
  local job_id="$1" image="$2" case_id="$3" gpu="$4" base_seed="$5" hio_seed="$6" warm_seed="$7"
  local case_dir="$CASE_ROOT/${image}_case$(printf '%02d' "$case_id")"
  local hio_dir="$case_dir/hio"
  local state="$hio_dir/hio_state.pt"
  local hio_png="$hio_dir/hio_raw.png"
  local hio_json="$hio_dir/hio_summary.json"
  local hio_log="$case_dir/logs/hio_generate.log"
  local timing_file="$case_dir/timings.tsv"
  local meas="$B19_BASE/measurements/ffhq${image}_phase_noise005_meas${MEAS_SEED}.pt"
  mkdir -p "$hio_dir" "$case_dir/logs"

  cat > "$case_dir/case_env.txt" <<EOF
job_id=$job_id
image_id=$image
case_id=$case_id
gpu=$gpu
base_seed=$base_seed
hio_seed=$hio_seed
warm_noise_seed=$warm_seed
inject_step=$INJECT_STEP
EOF

  run_daps "$case_dir" "$image" base_full "$base_seed" "$gpu" \
    B21_START_NOISE_SEED="$base_seed" B21_SOURCE_SEED="$base_seed"

  if [[ "$B21_FORCE" == "1" || ! -s "$state" || ! -s "$hio_json" || ! -s "$hio_png" ]]; then
    local start end elapsed
    start=$(date +%s)
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$GENERATOR" \
      --measurement-path "$meas" --out-state "$state" --out-png "$hio_png" \
      --out-json "$hio_json" --seed "$hio_seed" --iterations "$HIO_ITERS" \
      --beta "$HIO_BETA" --er-every "$HIO_ER_EVERY" --final-er "$HIO_FINAL_ER" \
      --resolution 256 --inject-step "$INJECT_STEP" --device cuda:0 > "$hio_log" 2>&1
    end=$(date +%s); elapsed=$((end - start)); touch "$timing_file"
    awk -F'\t' '$1 != "hio_generate"' "$timing_file" > "${timing_file}.tmp" || true
    printf "hio_generate\t%d\n" "$elapsed" >> "${timing_file}.tmp"
    mv "${timing_file}.tmp" "$timing_file"
  else
    echo "[skip][gpu=$gpu] image=$image hio_generate"
  fi

  [[ -s "$state" ]] || { echo "[fatal] missing HIO state: $state" >&2; return 4; }
  run_daps "$case_dir" "$image" hio_warm "$warm_seed" "$gpu" \
    B21_CONT_ENABLE=1 B21_CONT_STATE_PATH="$state" B21_CONT_NOISE_SEED="$warm_seed"
}

pids=()
for gpu in "${GPU_ARR[@]}"; do
  (
    set -euo pipefail
    tail -n +2 "$MANIFEST" | awk -F'\t' -v g="$gpu" '$4 == g {print}' | \
    while IFS=$'\t' read -r job_id image case_id assigned_gpu base_seed hio_seed warm_seed; do
      echo "[case start] job=$job_id image=$image case=$case_id gpu=$assigned_gpu"
      if run_case "$job_id" "$image" "$case_id" "$assigned_gpu" "$base_seed" "$hio_seed" "$warm_seed"; then
        printf "%s\t%s\t%s\t%s\tOK\n" "$job_id" "$image" "$case_id" "$assigned_gpu" >> "$DONE"
        echo "[case done] job=$job_id image=$image case=$case_id gpu=$assigned_gpu"
      else
        status=$?
        printf "%s\t%s\t%s\t%s\tFAIL\t%s\n" "$job_id" "$image" "$case_id" "$assigned_gpu" "$status" >> "$FAIL"
        echo "[case fail] job=$job_id image=$image case=$case_id gpu=$assigned_gpu status=$status"
      fi
    done
  ) > "$LOGDIR/worker_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
  echo "[worker] gpu=$gpu pid=$! log=$LOGDIR/worker_gpu${gpu}.log"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
completed=$(wc -l < "$DONE"); failed=$(wc -l < "$FAIL")
echo "[workers done] completed=$completed failed=$failed"
if [[ "$status" -ne 0 || "$failed" -ne 0 || "$completed" -ne "$jid" ]]; then
  cat "$FAIL" >&2 || true
  exit 5
fi

"$PYTHON_BIN" "$PILOT_ANALYZER" \
  --repo "$REPO" --out "$OUT" --manifest "$MANIFEST" --tuned-image 00046 \
  --inject-step "$INJECT_STEP" | tee "$OUT/analyzer_stdout.txt"

echo "[done] B21.5 HIO warm-start pilot artifacts: $OUT"
