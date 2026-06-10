#!/usr/bin/env bash
set -euo pipefail

# Official SITCOM-ODE baseline launcher for FFHQ split images.
# It creates a SITCOM-compatible image folder, then runs posterior_sample.py with
# Hydra overrides.  Run from pr_diffusion_repo.

GPU_ID=${1:-0}
TAG=${2:-sitcom_ffhq25_s100_103_noise005}

OUT=${OUT:-/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610}
SITCOM_ROOT=${SITCOM_ROOT:-/egr/research-pac/huang248/external/SITCOM_ODE}
DATA_ROOT=${DATA_ROOT:-/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024}
SPLIT=${SPLIT:-$OUT/splits/ffhq_available25.txt}

NOISE=${NOISE:-0.05}
NUM_RUNS=${NUM_RUNS:-4}
MAX_IMAGES=${MAX_IMAGES:-25}
BATCH_SIZE=${BATCH_SIZE:-10}
ANNEAL_STEPS=${ANNEAL_STEPS:-200}
DIFF_STEPS=${DIFF_STEPS:-5}
SEED=${SEED:-43}

RUN_ROOT=$OUT/sitcom_official/$TAG
SITCOM_DATA=$RUN_ROOT/sitcom_images
RESULTS=$RUN_ROOT/results
mkdir -p "$RUN_ROOT" "$RESULTS"

python scripts/npsitcom/make_sitcom_image_folder.py \
  --data_root "$DATA_ROOT" \
  --split_file "$SPLIT" \
  --outdir "$SITCOM_DATA"

cd "$SITCOM_ROOT"
python posterior_sample.py \
  +data=demo \
  +model=ffhq256ddpm \
  +task=phase_retrieval \
  +sampler=edm_daps \
  save_dir="$RESULTS" \
  num_runs="$NUM_RUNS" \
  sampler.diffusion_scheduler_config.num_steps="$DIFF_STEPS" \
  sampler.annealing_scheduler_config.num_steps="$ANNEAL_STEPS" \
  batch_size="$BATCH_SIZE" \
  data.root="$SITCOM_DATA" \
  data.start_id=0 \
  data.end_id="$MAX_IMAGES" \
  task.operator.sigma="$NOISE" \
  task.operator.oversample=2.0 \
  name="$TAG" \
  gpu="$GPU_ID" \
  seed="$SEED" \
  save_samples=True \
  save_traj=False

echo "SITCOM results: $RESULTS/$TAG"
echo "Image manifest: $SITCOM_DATA/manifest.csv"
