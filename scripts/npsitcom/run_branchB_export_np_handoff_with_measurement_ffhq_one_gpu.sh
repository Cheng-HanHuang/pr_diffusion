#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${1:-0}
TAG=${2:-handoff_with_measurement_smoke}

OUT=${OUT:-/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610}
DATA_ROOT=${DATA_ROOT:-/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024}
SPLIT=${SPLIT:-$OUT/splits/ffhq_available25.txt}
GUIDED_MODEL=${GUIDED_MODEL:-/egr/research-pac/huang248/models/ffhq_10m.pt}
GUIDED_DIFFUSION_DIR=${GUIDED_DIFFUSION_DIR:-/egr/research-pac/huang248/external/DiffFPR}

SEEDS=${SEEDS:-100}
NOISES=${NOISES:-0.05}
MAX_IMAGES=${MAX_IMAGES:-1}
HANDOFF_SIGMAS=${HANDOFF_SIGMAS:-20,10,5,2,1}

RUN_OUT=$OUT/branchB_handoff/$TAG
mkdir -p "$RUN_OUT"

CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/npsitcom/export_np_handoff_states_with_measurement.py \
  --data_root "$DATA_ROOT" \
  --image_list_file "$SPLIT" \
  --outdir "$RUN_OUT" \
  --guided_model_path "$GUIDED_MODEL" \
  --guided_diffusion_dir "$GUIDED_DIFFUSION_DIR" \
  --guided_preset difffpr_ffhq_10m \
  --seeds "$SEEDS" \
  --noise_values "$NOISES" \
  --max_images "$MAX_IMAGES" \
  --handoff_sigmas "$HANDOFF_SIGMAS" \
  --clip_noisy_magnitude
