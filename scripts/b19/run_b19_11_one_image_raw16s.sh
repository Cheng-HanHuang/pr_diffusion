#!/usr/bin/env bash
set -euo pipefail

ID="${1:?usage: run_b19_11_one_image_raw16s.sh IMAGE_ID PHYSICAL_GPU [SEED]}"
PHYS_GPU="${2:?usage: run_b19_11_one_image_raw16s.sh IMAGE_ID PHYSICAL_GPU [SEED]}"
SEED="${3:-3000}"

REPO=/egr/research-pac/huang248/pr_diffusion_b19_solver
BASE=/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver
MEAS="$BASE/measurements/ffhq${ID}_phase_noise005_meas${SEED}.pt"

NAME="b19_daps_${ID}_meas${SEED}_16S_rawtraj"
SAVE_DIR="results/b19_11_hard_mixed_raw16S"
ROOT="$REPO/external/daps/$SAVE_DIR/$NAME"
METRICS="$ROOT/metrics.json"

if [ ! -f "$MEAS" ]; then
  echo "[error] missing measurement $MEAS"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -f "$METRICS" ]; then
  echo "[run] image=$ID seed=$SEED physical_gpu=$PHYS_GPU"
  cd "$REPO/external/daps"
  conda activate daps

  CUDA_VISIBLE_DEVICES="$PHYS_GPU" python posterior_sample.py \
    +data=b19-ffhq-${ID} \
    +model=ffhq256ddpm \
    +task=phase_retrieval \
    +sampler=edm_daps \
    +measurement_path="$MEAS" \
    task_group=pixel \
    save_dir="$SAVE_DIR" \
    num_runs=16 \
    sampler.diffusion_scheduler_config.num_steps=5 \
    sampler.annealing_scheduler_config.num_steps=200 \
    batch_size=1 \
    data.start_id=0 data.end_id=1 \
    save_traj=true \
    save_traj_raw_data=true \
    name="$NAME" \
    gpu=0 2>&1 | tee "$BASE/B19_11_daps16S_${ID}_meas${SEED}_rawtraj.log"
else
  echo "[skip run] metrics exists $METRICS"
fi

cd "$REPO"
conda activate daps

echo "[analyze rawtraj] $ID"
python scripts/b19/analyze_daps_rawtraj_early_features.py \
  --daps_root external/daps \
  --raw_dir "$ROOT/trajectory/raw" \
  --measurement_path "$MEAS" \
  --metrics_json "$METRICS" \
  --out_step_csv "$BASE/B19_11_daps_${ID}_meas${SEED}_16S_rawtraj_step_features.csv" \
  --out_window_csv "$BASE/B19_11_daps_${ID}_meas${SEED}_16S_rawtraj_window_features.csv"

echo "[final exact selector] $ID"
python scripts/b19/analyze_daps_exact_final_loss_selector.py \
  --daps_root external/daps \
  --samples_dir "$ROOT/samples" \
  --measurement_path "$MEAS" \
  --metrics_json "$METRICS" \
  --out_csv "$BASE/B19_11_daps16S_${ID}_meas${SEED}_exact_final_loss_selector.csv"
