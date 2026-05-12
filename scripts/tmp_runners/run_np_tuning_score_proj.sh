#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?GPU id required}"
SCORE_SET="${2:?score radius required}"
PROJ_SET="${3:?proj radius required}"
TAG="${4:?tag required}"

cd /egr/research-pac/huang248/pr_diffusion_repo
source ~/miniforge3/etc/profile.d/conda.sh
conda activate prdiff_ffhq

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

export DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024
export IMAGE_LIST_FILE=/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/splits/ffhq_available25.txt
export GUIDED_MODEL_PATH=/egr/research-pac/huang248/models/ffhq_10m.pt
export GUIDED_DIFFUSION_DIR=/egr/research-pac/huang248/external/DiffFPR

export OUTDIR=/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411/np_ffhq_tuning_triage8w_${TAG}

export RADII="${SCORE_SET}"
export PROJ_RADII="${PROJ_SET}"
export PROJ_STARTS="300 500 700"
export CANDIDATES="5,1 8,1"
export SEEDS="100"
export MAX_IMAGES=5
export NP_STEPS=1000
export ALIGNMENTS="rot180"

CUDA_VISIBLE_DEVICES="${GPU_ID}" bash scripts/run_ffhq25_np_tuning_fast.sh
