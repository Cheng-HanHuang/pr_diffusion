#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
DAPS="$REPO/external/daps"
PYTHON_BIN=${PYTHON_BIN:-/egr/research-pac/huang248/conda-envs/daps/bin/python}
IMAGE_ROOT=${IMAGE_ROOT:-/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024}
GPUS=${GPUS:-"0 1 2 3"}
PANEL_SEED=${PANEL_SEED:-5401}
MEAS_TAG=${MEAS_TAG:-5401}
SEED1_BASE=${SEED1_BASE:-22000}
SEED2_BASE=${SEED2_BASE:-23000}
ANN_STEPS=${ANN_STEPS:-400}
DIFF_STEPS=${DIFF_STEPS:-5}
THETA=${THETA:-0.7}
B21_FORCE=${B21_FORCE:-0}
B21_SMOKE_ONLY=${B21_SMOKE_ONLY:-0}
B21_BASE=${B21_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver}
OUT=${OUT:-$B21_BASE/B21_11_fresh2_final_val100_meas${MEAS_TAG}}

PANEL_DIR="$OUT/panel"
PANEL_MANIFEST="$PANEL_DIR/panel_manifest.tsv"
PANEL_CHECKSUM="$PANEL_DIR/panel_manifest.sha256"
MEAS_DIR="$OUT/measurements"
MEAS_MANIFEST="$MEAS_DIR/fresh_measurement_manifest_meas${MEAS_TAG}.json"
CASE_ROOT="$OUT/cases"
LOGDIR="$OUT/logs"
MANIFEST="$OUT/manifest.tsv"
ACTIVE_MANIFEST="$OUT/active_manifest.tsv"
DONE="$OUT/done.tsv"
FAIL="$OUT/fail.tsv"
ANALYSIS="$OUT/analysis_theta0.7"
REPORT="$REPO/docs/b21/b21_11_fresh2_final_benchmark.md"

PANEL_BUILDER="$REPO/scripts/b21/build_b21_11_panel.py"
MEAS_GENERATOR="$REPO/scripts/b21/generate_b21_5_fresh_locked_measurements.py"
METRIC_ANALYZER="$REPO/scripts/b19/analyze_daps_exact_final_loss_selector.py"
FINAL_ANALYZER="$REPO/scripts/b21/analyze_b21_11_fresh2_final_benchmark.py"

cd "$REPO"
mkdir -p "$OUT" "$PANEL_DIR" "$MEAS_DIR" "$CASE_ROOT" "$LOGDIR" "$ANALYSIS"

[[ -x "$PYTHON_BIN" ]] || { echo "[fatal] PYTHON_BIN not executable: $PYTHON_BIN" >&2; exit 2; }
for file in "$PANEL_BUILDER" "$MEAS_GENERATOR" "$METRIC_ANALYZER" "$FINAL_ANALYZER"; do
  [[ -f "$file" ]] || { echo "[fatal] missing $file" >&2; exit 2; }
done
[[ "$PANEL_SEED" == "5401" ]] || { echo "[fatal] PANEL_SEED is frozen to 5401" >&2; exit 2; }
[[ "$MEAS_TAG" == "5401" ]] || { echo "[fatal] MEAS_TAG is frozen to 5401" >&2; exit 2; }
[[ "$SEED1_BASE" == "22000" ]] || { echo "[fatal] SEED1_BASE is frozen to 22000" >&2; exit 2; }
[[ "$SEED2_BASE" == "23000" ]] || { echo "[fatal] SEED2_BASE is frozen to 23000" >&2; exit 2; }
[[ "$ANN_STEPS" == "400" ]] || { echo "[fatal] ANN_STEPS is frozen to 400" >&2; exit 2; }
[[ "$DIFF_STEPS" == "5" ]] || { echo "[fatal] DIFF_STEPS is frozen to 5" >&2; exit 2; }
[[ "$THETA" == "0.7" ]] || { echo "[fatal] THETA is frozen to 0.7" >&2; exit 2; }

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

PANEL_ARGS=(
  --image-root "$IMAGE_ROOT"
  --outdir "$PANEL_DIR"
  --panel-seed "$PANEL_SEED"
  --count 100
  --seed1-base "$SEED1_BASE"
  --seed2-base "$SEED2_BASE"
)
if [[ "$B21_FORCE" == "1" ]]; then
  PANEL_ARGS+=(--force)
fi
"$PYTHON_BIN" "$PANEL_BUILDER" "${PANEL_ARGS[@]}" | tee "$OUT/panel_generation.log"

(
  cd "$PANEL_DIR"
  sha256sum -c "$(basename "$PANEL_CHECKSUM")"
)

mapfile -t IMAGE_ARR < <(tail -n +2 "$PANEL_MANIFEST" | cut -f2)
[[ ${#IMAGE_ARR[@]} -eq 100 ]] || { echo "[fatal] expected 100 frozen panel images" >&2; exit 2; }
IMAGES="${IMAGE_ARR[*]}"

MEAS_ARGS=(
  --repo "$REPO"
  --image-root "$IMAGE_ROOT"
  --out-dir "$MEAS_DIR"
  --images "$IMAGES"
  --panel-seed "$PANEL_SEED"
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
[[ -s "$MEAS_MANIFEST" ]] || { echo "[fatal] missing measurement manifest: $MEAS_MANIFEST" >&2; exit 2; }

for raw_image in "${IMAGE_ARR[@]}"; do
  image=$(printf '%05d' "$((10#$raw_image))")
  source=$(awk -F'\t' -v id="$image" 'NR>1 && $2==id {print $4}' "$PANEL_MANIFEST")
  meas="$MEAS_DIR/ffhq${image}_phase_noise005_meas${MEAS_TAG}.pt"
  data_name="b21-fresh2-final-ffhq-${image}"
  [[ -f "$source" ]] || { echo "[fatal] missing image $source" >&2; exit 2; }
  [[ -f "$meas" ]] || { echo "[fatal] missing measurement $meas" >&2; exit 2; }
  mkdir -p "$DAPS/dataset/$data_name" "$DAPS/configs/data"
  ln -sfn "$source" "$DAPS/dataset/$data_name/${image}.png"
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
printf "row_id\timage_id\tgpu\tseed1\tseed2\tmeasurement_path\texecution_order\n" > "$MANIFEST"

while IFS=$'\t' read -r row_id image digest source seed1 seed2; do
  [[ "$row_id" == "row_id" ]] && continue
  gpu="${GPU_ARR[$((row_id % ${#GPU_ARR[@]}))]}"
  measurement_path="$MEAS_DIR/ffhq${image}_phase_noise005_meas${MEAS_TAG}.pt"
  if (( row_id % 2 == 0 )); then
    execution_order="base_full,base_extra"
  else
    execution_order="base_extra,base_full"
  fi
  printf "%d\t%s\t%s\t%d\t%d\t%s\t%s\n" \
    "$row_id" "$image" "$gpu" "$seed1" "$seed2" "$measurement_path" "$execution_order" \
    >> "$MANIFEST"
done < "$PANEL_MANIFEST"

rows=$(($(wc -l < "$MANIFEST") - 1))
[[ "$rows" -eq 100 ]] || { echo "[fatal] expected 100 execution rows, found $rows" >&2; exit 2; }

if [[ "$B21_SMOKE_ONLY" == "1" ]]; then
  head -n 2 "$MANIFEST" > "$ACTIVE_MANIFEST"
else
  cp "$MANIFEST" "$ACTIVE_MANIFEST"
fi

{
  echo "timestamp=$(date -Is)"
  echo "repo=$REPO"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "panel_seed=$PANEL_SEED"
  echo "panel_manifest_sha256=$(cut -d' ' -f1 "$PANEL_CHECKSUM")"
  echo "measurement_tag=$MEAS_TAG"
  echo "seed1_base=$SEED1_BASE"
  echo "seed2_base=$SEED2_BASE"
  echo "ann_steps=$ANN_STEPS"
  echo "diff_steps=$DIFF_STEPS"
  echo "theta=$THETA"
  echo "gpus=$GPUS"
  echo "smoke_only=$B21_SMOKE_ONLY"
  echo "expected_rows=$rows"
  echo "active_rows=$(($(wc -l < "$ACTIVE_MANIFEST") - 1))"
} > "$OUT/launch_env.txt"

timing_has_candidate() {
  local timing_file="$1" candidate="$2"
  [[ -s "$timing_file" ]] && awk -F'\t' -v c="$candidate" '$1==c {found=1} END {exit !found}' "$timing_file"
}

run_candidate() {
  local row_id="$1" image="$2" gpu="$3" seed="$4" meas="$5" candidate="$6"
  local case_dir="$CASE_ROOT/${image}_row$(printf '%03d' "$row_id")"
  local save_dir="$case_dir/daps_results"
  local metric_dir="$case_dir/metrics"
  local log_dir="$case_dir/logs"
  local timing_file="$case_dir/timings.tsv"
  local sample="$save_dir/$candidate/samples/00000_run0000.png"
  local csv="$metric_dir/${candidate}.csv"
  local log="$log_dir/${candidate}.log"
  local data_name="b21-fresh2-final-ffhq-${image}"
  mkdir -p "$save_dir" "$metric_dir" "$log_dir"

  if [[ "$B21_FORCE" != "1" && -s "$sample" && -s "$csv" ]] \
      && timing_has_candidate "$timing_file" "$candidate"; then
    echo "[skip][gpu=$gpu] row=$row_id image=$image candidate=$candidate"
    return 0
  fi
  if [[ "$B21_FORCE" == "1" ]]; then
    rm -rf "$save_dir/$candidate"
    rm -f "$csv" "$log"
  fi

  local start end elapsed
  start=$(date +%s)
  echo "[run][gpu=$gpu] row=$row_id image=$image candidate=$candidate seed=$seed"
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

  [[ -s "$sample" ]] || {
    echo "[fatal] missing sample: $sample" >&2
    tail -80 "$log" >&2
    return 3
  }

  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$METRIC_ANALYZER" \
    --daps_root "$DAPS" \
    --samples_dir "$save_dir/$candidate/samples" \
    --measurement_path "$meas" \
    --out_csv "$csv" >> "$log" 2>&1
  [[ -s "$csv" ]] || { echo "[fatal] missing metric CSV: $csv" >&2; return 3; }

  end=$(date +%s)
  elapsed=$((end - start))
  touch "$timing_file"
  awk -F'\t' -v c="$candidate" '$1 != c' "$timing_file" > "${timing_file}.tmp" || true
  printf "%s\t%d\n" "$candidate" "$elapsed" >> "${timing_file}.tmp"
  mv "${timing_file}.tmp" "$timing_file"
  echo "[done][gpu=$gpu] row=$row_id candidate=$candidate elapsed=${elapsed}s"
}

run_row() {
  local row_id="$1" image="$2" gpu="$3" seed1="$4" seed2="$5" meas="$6" order="$7"
  local item
  IFS=',' read -r -a ORDER_ARR <<< "$order"
  for item in "${ORDER_ARR[@]}"; do
    case "$item" in
      base_full) run_candidate "$row_id" "$image" "$gpu" "$seed1" "$meas" base_full ;;
      base_extra) run_candidate "$row_id" "$image" "$gpu" "$seed2" "$meas" base_extra ;;
      *) echo "[fatal] unknown candidate in execution order: $item" >&2; return 4 ;;
    esac
  done
}

pids=()
for worker_gpu in "${GPU_ARR[@]}"; do
  (
    set -uo pipefail
    while IFS=$'\t' read -r row_id image gpu seed1 seed2 meas order; do
      [[ "$row_id" == "row_id" ]] && continue
      [[ "$gpu" == "$worker_gpu" ]] || continue
      if run_row "$row_id" "$image" "$gpu" "$seed1" "$seed2" "$meas" "$order"; then
        printf "%s\t%s\t%s\n" "$row_id" "$image" "$gpu" >> "$DONE"
      else
        rc=$?
        printf "%s\t%s\t%s\t%s\n" "$row_id" "$image" "$gpu" "$rc" >> "$FAIL"
      fi
    done < "$ACTIVE_MANIFEST"
  ) > "$LOGDIR/worker_gpu${worker_gpu}.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "[worker] gpu=$worker_gpu pid=$pid log=$LOGDIR/worker_gpu${worker_gpu}.log"
done

worker_failure=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    worker_failure=1
  fi
done

completed=$(wc -l < "$DONE")
failed=$(wc -l < "$FAIL")
echo "[workers done] completed=$completed failed=$failed"

if [[ "$worker_failure" -ne 0 ]]; then
  echo "[fatal] at least one worker shell failed" >&2
  exit 4
fi
if [[ "$failed" -ne 0 ]]; then
  echo "[fatal] candidate rows failed; inspect $FAIL and worker logs" >&2
  cat "$FAIL" >&2
  exit 4
fi

if [[ "$B21_SMOKE_ONLY" == "1" ]]; then
  [[ "$completed" -eq 1 ]] || { echo "[fatal] smoke expected one completed row" >&2; exit 4; }
  echo "[smoke-only] completed first frozen Fresh2 benchmark row; final analyzer intentionally skipped"
  exit 0
fi

[[ "$completed" -eq 100 ]] || { echo "[fatal] expected 100 completed rows, got $completed" >&2; exit 4; }

"$PYTHON_BIN" "$FINAL_ANALYZER" \
  --out "$OUT" \
  --manifest "$MANIFEST" \
  --panel-manifest "$PANEL_MANIFEST" \
  --panel-checksum "$PANEL_CHECKSUM" \
  --measurement-manifest "$MEAS_MANIFEST" \
  --theta "$THETA" \
  --image-root "$IMAGE_ROOT" \
  --report "$REPORT" \
  2>&1 | tee "$ANALYSIS/analyzer_stdout.txt"

echo "[done] B21.11 final benchmark artifacts: $ANALYSIS"
