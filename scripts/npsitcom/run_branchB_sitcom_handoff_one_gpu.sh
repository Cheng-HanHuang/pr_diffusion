#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${1:-0}
TAG=${2:-handoff_sitcom_smoke}

OUT=${OUT:-/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610}
PATCHED_ROOT=${PATCHED_ROOT:-$OUT/sitcom_ode_handoff_patch}
HANDOFF_MANIFEST=${HANDOFF_MANIFEST:-$OUT/branchB_handoff/handoff_smoke/handoff_manifest.csv}
IMAGE_MANIFEST=${IMAGE_MANIFEST:-$OUT/sitcom_official/sitcom_ffhq25_s4_noise005/sitcom_images/manifest.csv}
SITCOM_IMAGE_ROOT=${SITCOM_IMAGE_ROOT:-$OUT/sitcom_official/sitcom_ffhq25_s4_noise005/sitcom_images}
RUN_OUT=$OUT/branchB_sitcom_handoff/$TAG

MAX_ROWS=${MAX_ROWS:-8}
NOISE=${NOISE:-0.05}
BATCH_SIZE=${BATCH_SIZE:-1}
ANNEAL_STEPS=${ANNEAL_STEPS:-200}
DIFF_STEPS=${DIFF_STEPS:-5}
SEED=${SEED:-43}

if [[ ! -f "$PATCHED_ROOT/npsitcom_handoff_sample.py" ]]; then
  echo "Missing patched runner. Run scripts/npsitcom/sitcom_patch/apply_sitcom_handoff_patch.sh first." >&2
  exit 2
fi
if [[ ! -f "$HANDOFF_MANIFEST" ]]; then
  echo "Missing HANDOFF_MANIFEST=$HANDOFF_MANIFEST" >&2
  exit 2
fi
if [[ ! -d "$SITCOM_IMAGE_ROOT" ]]; then
  echo "Missing SITCOM_IMAGE_ROOT=$SITCOM_IMAGE_ROOT" >&2
  exit 2
fi

mkdir -p "$RUN_OUT"
cd "$PATCHED_ROOT"
python npsitcom_handoff_sample.py \
  +data=demo \
  +model=ffhq256ddpm \
  +task=phase_retrieval \
  +sampler=edm_daps \
  +handoff_manifest="$HANDOFF_MANIFEST" \
  +handoff_outdir="$RUN_OUT" \
  +handoff_image_manifest="$IMAGE_MANIFEST" \
  +handoff_max_rows="$MAX_ROWS" \
  save_dir="$RUN_OUT" \
  name="$TAG" \
  gpu="$GPU_ID" \
  seed="$SEED" \
  batch_size="$BATCH_SIZE" \
  data.root="$SITCOM_IMAGE_ROOT" \
  data.start_id=0 \
  data.end_id=25 \
  task.operator.sigma="$NOISE" \
  task.operator.oversample=2.0 \
  sampler.diffusion_scheduler_config.num_steps="$DIFF_STEPS" \
  sampler.annealing_scheduler_config.num_steps="$ANNEAL_STEPS" \
  save_samples=False \
  save_traj=False
