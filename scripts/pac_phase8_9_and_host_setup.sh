#!/usr/bin/env bash
set -euo pipefail

# Phase 8/9 + second-host setup runbook for PAC.
#
# Usage examples:
#   bash scripts/pac_phase8_9_and_host_setup.sh print_plan
#   bash scripts/pac_phase8_9_and_host_setup.sh phase8_pilot
#   bash scripts/pac_phase8_9_and_host_setup.sh phase8_full
#   bash scripts/pac_phase8_9_and_host_setup.sh phase9_pilot
#   bash scripts/pac_phase8_9_and_host_setup.sh setup_hosts
#
# Notes:
# - Baseline/unmasked + always-masked SITCOM runs work with current scripts/pr_canonical_compare.py.
# - Late-mask schedules for Phase 8/9 expect a dedicated runner (default: scripts/pr_phase8_9_schedule.py).
#

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/egr/research-pac/huang248/data/celeba_hq_256_stage}"
SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/splits}"
OUT_ROOT="${OUT_ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411}"
MODEL_ID="${MODEL_ID:-google/ddpm-celebahq-256}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Frozen defaults from docs/archive/phase8_plus_experiment_plan_initial.md + docs/progress_report.md
SEEDS_PILOT="${SEEDS_PILOT:-100,101,102,103,104}"
SEEDS_FULL="${SEEDS_FULL:-100,101,102,103,104,105,106,107,108,109}"
R_PRIMARY="${R_PRIMARY:-0.5}"
R_SECONDARY="${R_SECONDARY:-0.2}"

SITCOM_STEPS="${SITCOM_STEPS:-20}"
SITCOM_INNER_STEPS="${SITCOM_INNER_STEPS:-20}"
SITCOM_LR="${SITCOM_LR:-0.02}"
SITCOM_LAM="${SITCOM_LAM:-0.1}"
SITCOM_ETA_SCALE="${SITCOM_ETA_SCALE:-1.0}"
SITCOM_INIT_SCALE="${SITCOM_INIT_SCALE:-1.0}"

NP_STEPS="${NP_STEPS:-1000}"
NP_SOFT="${NP_SOFT:-5}"
NP_HARD="${NP_HARD:-1}"
NP_PROJ_START="${NP_PROJ_START:-400}"

# Expected new Phase 8/9 runner for late-mask schedule variants.
PHASE89_DRIVER="${PHASE89_DRIVER:-${REPO_ROOT}/scripts/pr_phase8_9_schedule.py}"

run_sitcom_baseline_pair() {
  local split_file="$1"
  local seeds="$2"
  local out_tag="$3"

  ${PYTHON_BIN} "${REPO_ROOT}/scripts/pr_canonical_compare.py" \
    --data_root "${DATA_ROOT}" \
    --image_list_file "${split_file}" \
    --outdir "${OUT_ROOT}/${out_tag}/sitcom_unmasked" \
    --model_id "${MODEL_ID}" \
    --seeds "${seeds}" \
    --radii "${R_PRIMARY}" \
    --sitcom_variant unmasked \
    --sitcom_steps "${SITCOM_STEPS}" \
    --sitcom_inner_steps "${SITCOM_INNER_STEPS}" \
    --sitcom_lr "${SITCOM_LR}" \
    --sitcom_lam "${SITCOM_LAM}" \
    --sitcom_eta_scale "${SITCOM_ETA_SCALE}" \
    --sitcom_init_scale "${SITCOM_INIT_SCALE}" \
    --np_steps "${NP_STEPS}" \
    --np_num_candidates_soft "${NP_SOFT}" \
    --np_num_candidates_hard "${NP_HARD}" \
    --np_proj_start "${NP_PROJ_START}"

  ${PYTHON_BIN} "${REPO_ROOT}/scripts/pr_canonical_compare.py" \
    --data_root "${DATA_ROOT}" \
    --image_list_file "${split_file}" \
    --outdir "${OUT_ROOT}/${out_tag}/sitcom_masked_all_r05" \
    --model_id "${MODEL_ID}" \
    --seeds "${seeds}" \
    --radii "${R_PRIMARY}" \
    --sitcom_variant masked \
    --sitcom_steps "${SITCOM_STEPS}" \
    --sitcom_inner_steps "${SITCOM_INNER_STEPS}" \
    --sitcom_lr "${SITCOM_LR}" \
    --sitcom_lam "${SITCOM_LAM}" \
    --sitcom_eta_scale "${SITCOM_ETA_SCALE}" \
    --sitcom_init_scale "${SITCOM_INIT_SCALE}" \
    --np_steps "${NP_STEPS}" \
    --np_num_candidates_soft "${NP_SOFT}" \
    --np_num_candidates_hard "${NP_HARD}" \
    --np_proj_start "${NP_PROJ_START}"
}

run_phase8_late_mask_variants() {
  local split_file="$1"
  local seeds="$2"
  local out_tag="$3"

  if [[ ! -f "${PHASE89_DRIVER}" ]]; then
    cat <<MSG
[Phase 8/9 late-mask driver missing]
Expected file: ${PHASE89_DRIVER}

Use/add a dedicated runner with args like:
  --mask_mode late --mask_start <int> --mask_radius <float>

Template commands:
  ${PYTHON_BIN} ${REPO_ROOT}/scripts/pr_phase8_9_schedule.py \\
    --data_root ${DATA_ROOT} --image_list_file ${split_file} \\
    --outdir ${OUT_ROOT}/${out_tag}/sitcom_late_mask_start400_r05 \\
    --model_id ${MODEL_ID} --seeds ${seeds} --sitcom_steps ${SITCOM_STEPS} \\
    --sitcom_inner_steps ${SITCOM_INNER_STEPS} --sitcom_lr ${SITCOM_LR} \\
    --sitcom_lam ${SITCOM_LAM} --sitcom_eta_scale ${SITCOM_ETA_SCALE} \\
    --sitcom_init_scale ${SITCOM_INIT_SCALE} --mask_mode late --mask_start 400 --mask_radius 0.5

  ${PYTHON_BIN} ${REPO_ROOT}/scripts/pr_phase8_9_schedule.py \\
    --data_root ${DATA_ROOT} --image_list_file ${split_file} \\
    --outdir ${OUT_ROOT}/${out_tag}/sitcom_late_mask_start400_r02 \\
    --model_id ${MODEL_ID} --seeds ${seeds} --sitcom_steps ${SITCOM_STEPS} \\
    --sitcom_inner_steps ${SITCOM_INNER_STEPS} --sitcom_lr ${SITCOM_LR} \\
    --sitcom_lam ${SITCOM_LAM} --sitcom_eta_scale ${SITCOM_ETA_SCALE} \\
    --sitcom_init_scale ${SITCOM_INIT_SCALE} --mask_mode late --mask_start 400 --mask_radius 0.2
MSG
    return 0
  fi

  ${PYTHON_BIN} "${PHASE89_DRIVER}" \
    --data_root "${DATA_ROOT}" \
    --image_list_file "${split_file}" \
    --outdir "${OUT_ROOT}/${out_tag}/sitcom_late_mask_start400_r05" \
    --model_id "${MODEL_ID}" \
    --seeds "${seeds}" \
    --sitcom_steps "${SITCOM_STEPS}" \
    --sitcom_inner_steps "${SITCOM_INNER_STEPS}" \
    --sitcom_lr "${SITCOM_LR}" \
    --sitcom_lam "${SITCOM_LAM}" \
    --sitcom_eta_scale "${SITCOM_ETA_SCALE}" \
    --sitcom_init_scale "${SITCOM_INIT_SCALE}" \
    --mask_mode late --mask_start 400 --mask_radius "${R_PRIMARY}"

  ${PYTHON_BIN} "${PHASE89_DRIVER}" \
    --data_root "${DATA_ROOT}" \
    --image_list_file "${split_file}" \
    --outdir "${OUT_ROOT}/${out_tag}/sitcom_late_mask_start400_r02" \
    --model_id "${MODEL_ID}" \
    --seeds "${seeds}" \
    --sitcom_steps "${SITCOM_STEPS}" \
    --sitcom_inner_steps "${SITCOM_INNER_STEPS}" \
    --sitcom_lr "${SITCOM_LR}" \
    --sitcom_lam "${SITCOM_LAM}" \
    --sitcom_eta_scale "${SITCOM_ETA_SCALE}" \
    --sitcom_init_scale "${SITCOM_INIT_SCALE}" \
    --mask_mode late --mask_start 400 --mask_radius "${R_SECONDARY}"
}

phase8_pilot() {
  run_sitcom_baseline_pair "${SPLIT_DIR}/validation_10.txt" "${SEEDS_PILOT}" "phase8_pilot"
  run_phase8_late_mask_variants "${SPLIT_DIR}/validation_10.txt" "${SEEDS_PILOT}" "phase8_pilot"
}

phase8_full() {
  run_sitcom_baseline_pair "${SPLIT_DIR}/validation_25.txt" "${SEEDS_FULL}" "phase8_full"
  run_phase8_late_mask_variants "${SPLIT_DIR}/validation_25.txt" "${SEEDS_FULL}" "phase8_full"
}

phase9_pilot() {
  if [[ ! -f "${PHASE89_DRIVER}" ]]; then
    echo "Missing ${PHASE89_DRIVER}; cannot run Phase 9 sweep automatically."
    return 0
  fi

  for start in 200 400 600; do
    ${PYTHON_BIN} "${PHASE89_DRIVER}" \
      --data_root "${DATA_ROOT}" \
      --image_list_file "${SPLIT_DIR}/validation_10.txt" \
      --outdir "${OUT_ROOT}/phase9_pilot/sitcom_late_mask_start${start}_r05" \
      --model_id "${MODEL_ID}" \
      --seeds "${SEEDS_PILOT}" \
      --sitcom_steps "${SITCOM_STEPS}" \
      --sitcom_inner_steps "${SITCOM_INNER_STEPS}" \
      --sitcom_lr "${SITCOM_LR}" \
      --sitcom_lam "${SITCOM_LAM}" \
      --sitcom_eta_scale "${SITCOM_ETA_SCALE}" \
      --sitcom_init_scale "${SITCOM_INIT_SCALE}" \
      --mask_mode late --mask_start "${start}" --mask_radius "${R_PRIMARY}"
  done
}

setup_hosts() {
  local host_root="${HOST_ROOT:-${OUT_ROOT}/external_hosts}"
  mkdir -p "${host_root}"
  cd "${host_root}"

  # Primary second host (per docs/archive/phase10_second_host_literature_review_initial.md)
  [[ -d DiffFPR ]] || git clone https://github.com/Chilie/DiffFPR.git

  # Practical comparison host from the addendum.
  [[ -d RED-diff ]] || git clone https://github.com/NVlabs/RED-diff.git

  # Optional transferable fallback pattern.
  [[ -d DiffPIR ]] || git clone https://github.com/yuanzhi-zhu/DiffPIR.git

  # Quick smoke checks (no long training/runs).
  if command -v conda >/dev/null 2>&1; then
    echo "Conda detected. Create isolated envs per host as needed (recommended)."
  fi

  echo "Running lightweight repo-level checks..."
  (cd DiffFPR && git rev-parse --short HEAD && ${PYTHON_BIN} -c "import os; print('DiffFPR_OK', os.getcwd())")
  (cd RED-diff && git rev-parse --short HEAD && ${PYTHON_BIN} -c "import os; print('REDDIFF_OK', os.getcwd())")
  (cd DiffPIR && git rev-parse --short HEAD && ${PYTHON_BIN} -c "import os; print('DIFFPIR_OK', os.getcwd())")

  cat <<'MSG'

Second-host alignment checklist (match current in-repo comparisons):
1) Keep model backbone family aligned to DDPM/CelebA-HQ-256 where possible.
2) Evaluate at r=0.5 primary and r=0.2 secondary.
3) Keep seed protocol consistent (pilot: 5 seeds, full: 10 seeds).
4) Compare three projection modes for each host:
   - no lowfreq projection
   - always lowfreq projection
   - late lowfreq projection (start=400)
5) Report same metrics as current repo (PSNR, full-mag L2, lowfreq-mag L2, runtime).

MSG
}

print_plan() {
  cat <<MSG
Phase 8 pilot (validation_10, 5 seeds):
  bash scripts/pac_phase8_9_and_host_setup.sh phase8_pilot

Phase 8 full (validation_25, 10 seeds):
  bash scripts/pac_phase8_9_and_host_setup.sh phase8_full

Phase 9 pilot sweep (mask_start in 200/400/600):
  bash scripts/pac_phase8_9_and_host_setup.sh phase9_pilot

Setup second hosts (DiffFPR + RED-diff + DiffPIR fallback):
  bash scripts/pac_phase8_9_and_host_setup.sh setup_hosts
MSG
}

cmd="${1:-print_plan}"
case "${cmd}" in
  print_plan|phase8_pilot|phase8_full|phase9_pilot|setup_hosts)
    "${cmd}"
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    exit 2
    ;;
esac
