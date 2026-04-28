#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT}"
IMAGE_LIST_FILE="${IMAGE_LIST_FILE:?Set IMAGE_LIST_FILE}"
GUIDED_MODEL_PATH="${GUIDED_MODEL_PATH:?Set GUIDED_MODEL_PATH}"
GUIDED_DIFFUSION_DIR="${GUIDED_DIFFUSION_DIR:?Set GUIDED_DIFFUSION_DIR}"
OUTDIR="${OUTDIR:?Set OUTDIR}"

RADII="${RADII:-0.2 0.4 0.6 full}"
PROJ_RADII="${PROJ_RADII:-$RADII}"
PROJ_STARTS="${PROJ_STARTS:-200 400 600 800}"
CANDIDATES="${CANDIDATES:-5,1 5,3 8,1 8,2}"

SEEDS="${SEEDS:-100}"
MAX_IMAGES="${MAX_IMAGES:-10}"
NP_STEPS="${NP_STEPS:-1000}"
ALIGNMENTS="${ALIGNMENTS:-rot180}"

mkdir -p "$OUTDIR"

radius_value() {
  if [[ "$1" == "full" ]]; then
    echo "0.72"
  else
    echo "$1"
  fi
}

for SCORE_R in $RADII; do
  for PROJ_R in $PROJ_RADII; do
    SCORE_VAL="$(radius_value "$SCORE_R")"
    PROJ_VAL="$(radius_value "$PROJ_R")"

    for START in $PROJ_STARTS; do
      for KH in $CANDIDATES; do
        SOFT_K="${KH%,*}"
        HARD_K="${KH#*,}"

        RUN_OUTDIR="${OUTDIR}/score_${SCORE_R}_proj_${PROJ_R}_start_${START}_soft_${SOFT_K}_hard_${HARD_K}"
        mkdir -p "$RUN_OUTDIR"

        echo
        echo "[tuning] score=${SCORE_R} proj=${PROJ_R} start=${START} soft=${SOFT_K} hard=${HARD_K}"

        "$PYTHON_BIN" scripts/pr_external_difffpr_np_guided_benchmark.py \
          --data_root "$DATA_ROOT" \
          --image_list_file "$IMAGE_LIST_FILE" \
          --outdir "$RUN_OUTDIR" \
          --guided_model_path "$GUIDED_MODEL_PATH" \
          --guided_diffusion_dir "$GUIDED_DIFFUSION_DIR" \
          --guided_preset difffpr_ffhq_10m \
          --variants np_canonical \
          --seeds "$SEEDS" \
          --np_steps "$NP_STEPS" \
          --late_start "$START" \
          --fixed_k 5 \
          --soft_candidates "$SOFT_K" \
          --hard_candidates "$HARD_K" \
          --score_radius "$SCORE_VAL" \
          --proj_radius "$PROJ_VAL" \
          --oversample_values 2 \
          --measurement_noise_values 0.05 \
          --alignments "$ALIGNMENTS" \
          --clip_noisy_magnitude \
          --skip_lpips \
          --fast_eval \
          --max_images "$MAX_IMAGES" \
          --log_every 100
      done
    done
  done
done
