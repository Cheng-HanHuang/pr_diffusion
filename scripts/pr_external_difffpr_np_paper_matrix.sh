#!/usr/bin/env bash
set -euo pipefail

# Run NP external benchmarks under DiffFPR Table-2-style settings for FFHQ + ImageNet.
#
# This wrapper launches both NP variants:
#   - np_canonical
#   - np_fixedk_lateproj
#
# and sweeps the paper noise levels:
#   sigma_y in {0.00, 0.01, 0.05}
#
# Environment variables:
#   PYTHON_BIN       python executable (default: python)
#   FFHQ_DATA_ROOT   required path to FFHQ images
#   IMAGENET_DATA_ROOT required path to ImageNet images
#   FFHQ_LIST_FILE   optional image list for FFHQ
#   IMAGENET_LIST_FILE optional image list for ImageNet
#   OUT_ROOT         output directory root (default: ./outputs/external_np_benchmark)
#   SEEDS            seed list (default: 100..119)
#   VARIANTS         solver list (default: np_canonical,np_fixedk_lateproj)
#   NP_STEPS         default 1000
#   LATE_START       default 400
#   FIXED_K          default 5

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-./outputs/external_np_benchmark}"
SEEDS="${SEEDS:-100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119}"
VARIANTS="${VARIANTS:-np_canonical,np_fixedk_lateproj}"
NP_STEPS="${NP_STEPS:-1000}"
LATE_START="${LATE_START:-400}"
FIXED_K="${FIXED_K:-5}"
NOISE_LEVELS=("0.00" "0.01" "0.05")

run_dataset() {
  local dataset="$1"
  local data_root="$2"
  local list_file="${3:-}"
  if [[ -z "${data_root}" ]]; then
    echo "ERROR: missing data_root for ${dataset}" >&2
    return 1
  fi
  for sigma in "${NOISE_LEVELS[@]}"; do
    echo "[paper-matrix] dataset=${dataset} sigma_y=${sigma}"
    cmd=(
      "${PYTHON_BIN}" scripts/pr_external_difffpr_np_benchmark.py
      --paper_preset "${dataset}"
      --data_root "${data_root}"
      --outdir "${OUT_ROOT}/${dataset}/sigma_${sigma}"
      --variants "${VARIANTS}"
      --seeds "${SEEDS}"
      --np_steps "${NP_STEPS}"
      --late_start "${LATE_START}"
      --fixed_k "${FIXED_K}"
      --measurement_noise_std "${sigma}"
      --clip_noisy_magnitude
      --log_every 100
    )
    if [[ -n "${list_file}" ]]; then
      cmd+=(--image_list_file "${list_file}")
    fi
    "${cmd[@]}"
  done
}

run_dataset "ffhq" "${FFHQ_DATA_ROOT:-}" "${FFHQ_LIST_FILE:-}"
run_dataset "imagenet" "${IMAGENET_DATA_ROOT:-}" "${IMAGENET_LIST_FILE:-}"

