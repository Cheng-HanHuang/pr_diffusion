#!/usr/bin/env bash
set -euo pipefail

# Phase 10-12 execution runbook (before second-host integration).
#
# Usage:
#   bash scripts/pac_phase10_12_before_second_host.sh print_plan
#   bash scripts/pac_phase10_12_before_second_host.sh phase10_pilot
#   bash scripts/pac_phase10_12_before_second_host.sh phase10_full
#   bash scripts/pac_phase10_12_before_second_host.sh phase11_pilot
#   bash scripts/pac_phase10_12_before_second_host.sh phase11_full
#   bash scripts/pac_phase10_12_before_second_host.sh phase12_pilot
#   bash scripts/pac_phase10_12_before_second_host.sh phase12_full

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/egr/research-pac/huang248/data/celeba_hq_256_stage}"
SPLIT_DIR="${SPLIT_DIR:-${REPO_ROOT}/splits}"
OUT_ROOT="${OUT_ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260418}"
MODEL_ID="${MODEL_ID:-google/ddpm-celebahq-256}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SEEDS_PILOT="${SEEDS_PILOT:-100,101,102,103,104}"
SEEDS_FULL="${SEEDS_FULL:-100,101,102,103,104,105,106,107,108,109}"
R_PRIMARY="${R_PRIMARY:-0.5}"

NP_STEPS="${NP_STEPS:-1000}"
NP_FIXED_K="${NP_FIXED_K:-5}"
NP_CANONICAL_SOFT="${NP_CANONICAL_SOFT:-5}"
NP_CANONICAL_HARD="${NP_CANONICAL_HARD:-1}"
NP_LATE_START="${NP_LATE_START:-400}"

SITCOM_STEPS="${SITCOM_STEPS:-20}"
SITCOM_INNER_STEPS="${SITCOM_INNER_STEPS:-20}"
SITCOM_LR="${SITCOM_LR:-0.02}"
SITCOM_LAM="${SITCOM_LAM:-0.1}"
SITCOM_ETA_SCALE="${SITCOM_ETA_SCALE:-1.0}"
SITCOM_INIT_SCALE="${SITCOM_INIT_SCALE:-1.0}"

PHASE1011_DRIVER="${PHASE1011_DRIVER:-${REPO_ROOT}/scripts/pr_phase10_11_np_grid.py}"
PHASE12_DRIVER="${PHASE12_DRIVER:-${REPO_ROOT}/scripts/pr_phase8_9_schedule.py}"
PHASE12_HYBRID_DRIVER="${PHASE12_HYBRID_DRIVER:-${REPO_ROOT}/scripts/pr_phase12_hybrid_ladder.py}"

phase10_run() {
  local split_file="$1"
  local seeds="$2"
  local out_tag="$3"

  ${PYTHON_BIN} "${PHASE1011_DRIVER}" \
    --phase phase10 \
    --data_root "${DATA_ROOT}" \
    --image_list_file "${split_file}" \
    --outdir "${OUT_ROOT}/${out_tag}/phase10" \
    --model_id "${MODEL_ID}" \
    --seeds "${seeds}" \
    --radius "${R_PRIMARY}" \
    --np_steps "${NP_STEPS}" \
    --fixed_k "${NP_FIXED_K}" \
    --canonical_soft "${NP_CANONICAL_SOFT}" \
    --canonical_hard "${NP_CANONICAL_HARD}" \
    --late_start "${NP_LATE_START}"
}

phase11_run() {
  local split_file="$1"
  local seeds="$2"
  local out_tag="$3"

  ${PYTHON_BIN} "${PHASE1011_DRIVER}" \
    --phase phase11 \
    --data_root "${DATA_ROOT}" \
    --image_list_file "${split_file}" \
    --outdir "${OUT_ROOT}/${out_tag}/phase11" \
    --model_id "${MODEL_ID}" \
    --seeds "${seeds}" \
    --radius "${R_PRIMARY}" \
    --np_steps "${NP_STEPS}" \
    --fixed_k "${NP_FIXED_K}" \
    --canonical_soft "${NP_CANONICAL_SOFT}" \
    --canonical_hard "${NP_CANONICAL_HARD}" \
    --late_start "${NP_LATE_START}"
}

phase12_one_method() {
  local split_file="$1"
  local seeds="$2"
  local out_dir="$3"
  local method="$4"

  case "${method}" in
    sitcom_unmasked)
      ${PYTHON_BIN} "${PHASE12_DRIVER}" \
        --data_root "${DATA_ROOT}" \
        --image_list_file "${split_file}" \
        --outdir "${out_dir}/${method}" \
        --model_id "${MODEL_ID}" \
        --seeds "${seeds}" \
        --sitcom_steps "${SITCOM_STEPS}" \
        --sitcom_inner_steps "${SITCOM_INNER_STEPS}" \
        --sitcom_lr "${SITCOM_LR}" \
        --sitcom_lam "${SITCOM_LAM}" \
        --sitcom_eta_scale "${SITCOM_ETA_SCALE}" \
        --sitcom_init_scale "${SITCOM_INIT_SCALE}" \
        --mask_mode unmasked \
        --mask_radius "${R_PRIMARY}" \
        --mask_start "${NP_LATE_START}" \
        --metrics_radius "${R_PRIMARY}"
      ;;
    sitcom_weak_then_strong)
      ${PYTHON_BIN} "${PHASE12_DRIVER}" \
        --data_root "${DATA_ROOT}" \
        --image_list_file "${split_file}" \
        --outdir "${out_dir}/${method}" \
        --model_id "${MODEL_ID}" \
        --seeds "${seeds}" \
        --sitcom_steps "${SITCOM_STEPS}" \
        --sitcom_inner_steps "${SITCOM_INNER_STEPS}" \
        --sitcom_lr "${SITCOM_LR}" \
        --sitcom_lam "${SITCOM_LAM}" \
        --sitcom_eta_scale "${SITCOM_ETA_SCALE}" \
        --sitcom_init_scale "${SITCOM_INIT_SCALE}" \
        --mask_mode weighted \
        --mask_radius "${R_PRIMARY}" \
        --mask_start "${NP_LATE_START}" \
        --early_meas_weight 0.25 \
        --late_meas_weight 1.0 \
        --metrics_radius "${R_PRIMARY}"
      ;;
    sitcom_hard_from_start_masked)
      ${PYTHON_BIN} "${PHASE12_DRIVER}" \
        --data_root "${DATA_ROOT}" \
        --image_list_file "${split_file}" \
        --outdir "${out_dir}/${method}" \
        --model_id "${MODEL_ID}" \
        --seeds "${seeds}" \
        --sitcom_steps "${SITCOM_STEPS}" \
        --sitcom_inner_steps "${SITCOM_INNER_STEPS}" \
        --sitcom_lr "${SITCOM_LR}" \
        --sitcom_lam "${SITCOM_LAM}" \
        --sitcom_eta_scale "${SITCOM_ETA_SCALE}" \
        --sitcom_init_scale "${SITCOM_INIT_SCALE}" \
        --mask_mode masked \
        --mask_radius "${R_PRIMARY}" \
        --mask_start "${NP_LATE_START}" \
        --metrics_radius "${R_PRIMARY}"
      ;;
    sitcom_late_mask_proxy)
      ${PYTHON_BIN} "${PHASE12_DRIVER}" \
        --data_root "${DATA_ROOT}" \
        --image_list_file "${split_file}" \
        --outdir "${out_dir}/${method}" \
        --model_id "${MODEL_ID}" \
        --seeds "${seeds}" \
        --sitcom_steps "${SITCOM_STEPS}" \
        --sitcom_inner_steps "${SITCOM_INNER_STEPS}" \
        --sitcom_lr "${SITCOM_LR}" \
        --sitcom_lam "${SITCOM_LAM}" \
        --sitcom_eta_scale "${SITCOM_ETA_SCALE}" \
        --sitcom_init_scale "${SITCOM_INIT_SCALE}" \
        --mask_mode late \
        --mask_radius "${R_PRIMARY}" \
        --mask_start "${NP_LATE_START}" \
        --metrics_radius "${R_PRIMARY}"
      ;;
    *)
      echo "Unknown Phase 12 method: ${method}" >&2
      return 1
      ;;
  esac
}

phase12_run() {
  local split_file="$1"
  local seeds="$2"
  local out_tag="$3"
  local out_dir="${OUT_ROOT}/${out_tag}/phase12"

  phase12_one_method "${split_file}" "${seeds}" "${out_dir}" sitcom_unmasked
  phase12_one_method "${split_file}" "${seeds}" "${out_dir}" sitcom_weak_then_strong
  phase12_one_method "${split_file}" "${seeds}" "${out_dir}" sitcom_hard_from_start_masked
  phase12_one_method "${split_file}" "${seeds}" "${out_dir}" sitcom_late_mask_proxy

  ${PYTHON_BIN} "${PHASE12_HYBRID_DRIVER}" \
    --data_root "${DATA_ROOT}" \
    --image_list_file "${split_file}" \
    --outdir "${out_dir}/np_to_sitcom_hybrid" \
    --model_id "${MODEL_ID}" \
    --seeds "${seeds}" \
    --radius "${R_PRIMARY}" \
    --num_steps "${NP_STEPS}" \
    --sitcom_inner_steps "${SITCOM_INNER_STEPS}" \
    --sitcom_lr "${SITCOM_LR}" \
    --sitcom_lam "${SITCOM_LAM}" \
    --methods "np_to_sitcom_400,np_to_sitcom_600,np_to_sitcom_masked_400,np_to_sitcom_masked_600"
}

print_plan() {
  cat <<PLAN
Phase 10/11/12 runbook (before second host)

Pilot defaults:
  split: ${SPLIT_DIR}/validation_10.txt
  seeds: ${SEEDS_PILOT}
  radius: ${R_PRIMARY}

Full defaults:
  split: ${SPLIT_DIR}/validation_25.txt
  seeds: ${SEEDS_FULL}
  radius: ${R_PRIMARY}

Phase 10 (NP decoupling):
  - np_canonical
  - np_fixedk_lateproj
  - np_fixedk_alwaysproj
  - np_fixedk_noproj
  - np_candidate_switch_only
  - np_projection_only_switch

Phase 11 (hard early vs deferred hard):
  - hard_from_start
  - hard_late
  - hard_never
  - soft_only
  - soft_then_hard

Phase 12 (SITCOM surrogate):
  - sitcom_unmasked
  - sitcom_weak_then_strong (weighted schedule)
  - sitcom_hard_from_start_masked
  - sitcom_late_mask_proxy
  - np_to_sitcom_400
  - np_to_sitcom_600
  - np_to_sitcom_masked_400
  - np_to_sitcom_masked_600
PLAN
}

cmd="${1:-print_plan}"
case "${cmd}" in
  print_plan)
    print_plan
    ;;
  phase10_pilot)
    phase10_run "${SPLIT_DIR}/validation_10.txt" "${SEEDS_PILOT}" "phase10_pilot"
    ;;
  phase10_full)
    phase10_run "${SPLIT_DIR}/validation_25.txt" "${SEEDS_FULL}" "phase10_full"
    ;;
  phase11_pilot)
    phase11_run "${SPLIT_DIR}/validation_10.txt" "${SEEDS_PILOT}" "phase11_pilot"
    ;;
  phase11_full)
    phase11_run "${SPLIT_DIR}/validation_25.txt" "${SEEDS_FULL}" "phase11_full"
    ;;
  phase12_pilot)
    phase12_run "${SPLIT_DIR}/validation_10.txt" "${SEEDS_PILOT}" "phase12_pilot"
    ;;
  phase12_full)
    phase12_run "${SPLIT_DIR}/validation_25.txt" "${SEEDS_FULL}" "phase12_full"
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    exit 1
    ;;
esac
