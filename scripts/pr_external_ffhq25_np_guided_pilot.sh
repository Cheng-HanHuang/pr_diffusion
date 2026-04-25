#!/usr/bin/env bash
set -euo pipefail

# FFHQ pilot matrix for guided-diffusion backend (canonical NP first).
#
# Sweeps:
#   oversample ratio: 0,2,4
#   measurement noise sigma_y: 0,0.01,0.05,0.1
#
# Default reconstruction count = 4 (seeds 100..103), matching the pilot spec.
# For methods with 3 reconstructions (e.g. DiffFPR), consume first three seeds.

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-}"
IMAGE_LIST_FILE="${IMAGE_LIST_FILE:-}"
OUTDIR="${OUTDIR:-./outputs/external_ffhq25_guided_pilot}"
GUIDED_MODEL_PATH="${GUIDED_MODEL_PATH:-}"
GUIDED_DIFFUSION_DIR="${GUIDED_DIFFUSION_DIR:-}"
GUIDED_PRESET="${GUIDED_PRESET:-difffpr_ffhq_10m}"
VARIANTS="${VARIANTS:-np_canonical}"
SEEDS="${SEEDS:-100,101,102,103}"
NP_STEPS="${NP_STEPS:-1000}"
LATE_START="${LATE_START:-400}"
FIXED_K="${FIXED_K:-5}"
RADIUS="${RADIUS:-0.5}"
OVERSAMPLE_VALUES="${OVERSAMPLE_VALUES:-0,2,4}"
NOISE_VALUES="${NOISE_VALUES:-0,0.01,0.05,0.1}"
ALIGNMENTS="${ALIGNMENTS:-raw,rot180,resolve}"
PSNR_THRESHOLD="${PSNR_THRESHOLD:-20.0}"

if [[ -z "${DATA_ROOT}" || -z "${IMAGE_LIST_FILE}" || -z "${GUIDED_MODEL_PATH}" ]]; then
  echo "ERROR: DATA_ROOT, IMAGE_LIST_FILE, and GUIDED_MODEL_PATH must be set." >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}" scripts/pr_external_difffpr_np_guided_benchmark.py
  --data_root "${DATA_ROOT}"
  --image_list_file "${IMAGE_LIST_FILE}"
  --outdir "${OUTDIR}"
  --guided_model_path "${GUIDED_MODEL_PATH}"
  --guided_preset "${GUIDED_PRESET}"
  --variants "${VARIANTS}"
  --seeds "${SEEDS}"
  --np_steps "${NP_STEPS}"
  --late_start "${LATE_START}"
  --fixed_k "${FIXED_K}"
  --radius "${RADIUS}"
  --oversample_values "${OVERSAMPLE_VALUES}"
  --measurement_noise_values "${NOISE_VALUES}"
  --alignments "${ALIGNMENTS}"
  --psnr_threshold "${PSNR_THRESHOLD}"
  --clip_noisy_magnitude
  --log_every 100
)

if [[ -n "${GUIDED_DIFFUSION_DIR}" ]]; then
  cmd+=(--guided_diffusion_dir "${GUIDED_DIFFUSION_DIR}")
fi

printf '[pilot] command:\n%s\n' "${cmd[*]}"
"${cmd[@]}"
