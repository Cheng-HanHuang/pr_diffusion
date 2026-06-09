#!/usr/bin/env bash
set -euo pipefail

# ImageNet pilot using the same guided checkpoint interface only if the selected
# guided preset/checkpoint supports it.  By default this is a smoke/pilot runner;
# use official ImageNet-capable models if available in your environment.

GPU_ID="${1:-0}"
TAG="${2:-imagenet25_s100_103_np_refine}"

REPO="${REPO:-/egr/research-pac/huang248/pr_diffusion_repo}"
OUT_ROOT="${OUT_ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260608}"
DATA_ROOT="${DATA_ROOT:-/egr/research-pac/huang248/data/imagenet/imagenet256_val}"
IMAGE_LIST_FILE="${IMAGE_LIST_FILE:-$OUT_ROOT/splits/imagenet_available25.txt}"
GUIDED_MODEL_PATH="${GUIDED_MODEL_PATH:-/egr/research-pac/huang248/models/ffhq_10m.pt}"
GUIDED_DIFFUSION_DIR="${GUIDED_DIFFUSION_DIR:-/egr/research-pac/huang248/external/DiffFPR}"
GUIDED_PRESET="${GUIDED_PRESET:-difffpr_ffhq_10m}"
SEEDS="${SEEDS:-100,101,102,103}"
NOISES="${NOISES:-0,0.01,0.05}"
MAX_IMAGES="${MAX_IMAGES:-25}"
REFINE_STEPS="${REFINE_STEPS:-100}"
SKIP_LPIPS="${SKIP_LPIPS:-1}"

cd "$REPO"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate prdiff_ffhq
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

bash scripts/phase_retrieval_20260608/prepare_phase_retrieval_20260608.sh
OUTDIR="$OUT_ROOT/np_refine_imagenet/$TAG"
mkdir -p "$OUTDIR"

LPIPS_FLAG=()
if [[ "$SKIP_LPIPS" == "1" ]]; then
  LPIPS_FLAG=(--skip_lpips)
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/pr_external_difffpr_np_refine_benchmark.py \
  --data_root "$DATA_ROOT" \
  --image_list_file "$IMAGE_LIST_FILE" \
  --outdir "$OUTDIR" \
  --guided_model_path "$GUIDED_MODEL_PATH" \
  --guided_diffusion_dir "$GUIDED_DIFFUSION_DIR" \
  --guided_preset "$GUIDED_PRESET" \
  --seeds "$SEEDS" \
  --np_steps 1000 \
  --late_start 300 \
  --soft_candidates 5 \
  --hard_candidates 1 \
  --score_radius 0.6 \
  --proj_radius 0.2 \
  --s2_lambda 0.01 \
  --oversample_values 2 \
  --measurement_noise_values "$NOISES" \
  --clip_noisy_magnitude \
  --alignments raw,rot180,resolve \
  --max_images "$MAX_IMAGES" \
  --refine_steps "$REFINE_STEPS" \
  --refine_lr 0.01 \
  --refine_anchor_weights 0,0.01,0.05 \
  --refine_mag_radius full \
  --log_every 100 \
  "${LPIPS_FLAG[@]}"
