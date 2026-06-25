#!/usr/bin/env bash
set -euo pipefail

ID="${1:?usage: run_b19_16b_one_image_raw6s_freshtraj.sh IMAGE_ID PHYSICAL_GPU [MEAS_SEED] [RUN_SEED]}"
PHYS_GPU="${2:?usage: run_b19_16b_one_image_raw6s_freshtraj.sh IMAGE_ID PHYSICAL_GPU [MEAS_SEED] [RUN_SEED]}"
MEAS_SEED="${3:-3000}"
RUN_SEED="${4:-4100}"

REPO=/egr/research-pac/huang248/pr_diffusion_b19_solver
BASE=/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver
DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024

MEAS="$BASE/measurements/ffhq${ID}_phase_noise005_meas${MEAS_SEED}.pt"

NAME="b19_daps_${ID}_meas${MEAS_SEED}_runseed${RUN_SEED}_6S_rawtraj"
SAVE_DIR="results/b19_16B_full25_raw6S_meas${MEAS_SEED}_runseed${RUN_SEED}"
ROOT="$REPO/external/daps/$SAVE_DIR/$NAME"
METRICS="$ROOT/metrics.json"

if [ ! -f "$MEAS" ]; then
  echo "[error] missing measurement $MEAS"
  exit 1
fi

mkdir -p "$REPO/external/daps/dataset/b19-ffhq-${ID}"
ln -sfn "$DATA_ROOT/00000/${ID}.png" "$REPO/external/daps/dataset/b19-ffhq-${ID}/${ID}.png"

mkdir -p "$REPO/external/daps/configs/data"
cat > "$REPO/external/daps/configs/data/b19-ffhq-${ID}.yaml" <<YAML
name: image
root: 'dataset/b19-ffhq-${ID}'
resolution: 256
start_id: 0
end_id: 1
YAML

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -f "$METRICS" ]; then
  echo "[run] image=$ID meas_seed=$MEAS_SEED run_seed=$RUN_SEED physical_gpu=$PHYS_GPU"
  cd "$REPO/external/daps"
  conda activate daps

  set +e
  CUDA_VISIBLE_DEVICES="$PHYS_GPU" python posterior_sample.py \
    +data=b19-ffhq-${ID} \
    +model=ffhq256ddpm \
    +task=phase_retrieval \
    +sampler=edm_daps \
    +measurement_path="$MEAS" \
    task_group=pixel \
    save_dir="$SAVE_DIR" \
    num_runs=6 \
    sampler.diffusion_scheduler_config.num_steps=5 \
    sampler.annealing_scheduler_config.num_steps=200 \
    batch_size=1 \
    data.start_id=0 data.end_id=1 \
    save_traj=true \
    save_traj_raw_data=true \
    name="$NAME" \
    seed="$RUN_SEED" \
    gpu=0 2>&1 | tee "$BASE/B19_16B_daps6S_${ID}_meas${MEAS_SEED}_runseed${RUN_SEED}_rawtraj.log"
  status=${PIPESTATUS[0]}

  if [ "$status" -ne 0 ]; then
    echo "[retry] top-level seed= failed; retrying with +seed="
    CUDA_VISIBLE_DEVICES="$PHYS_GPU" python posterior_sample.py \
      +data=b19-ffhq-${ID} \
      +model=ffhq256ddpm \
      +task=phase_retrieval \
      +sampler=edm_daps \
      +measurement_path="$MEAS" \
      task_group=pixel \
      save_dir="$SAVE_DIR" \
      num_runs=6 \
      sampler.diffusion_scheduler_config.num_steps=5 \
      sampler.annealing_scheduler_config.num_steps=200 \
      batch_size=1 \
      data.start_id=0 data.end_id=1 \
      save_traj=true \
      save_traj_raw_data=true \
      name="$NAME" \
      +seed="$RUN_SEED" \
      gpu=0 2>&1 | tee -a "$BASE/B19_16B_daps6S_${ID}_meas${MEAS_SEED}_runseed${RUN_SEED}_rawtraj.log"
    status=${PIPESTATUS[0]}
  fi

  set -e
  if [ "$status" -ne 0 ]; then
    echo "[error] DAPS run failed"
    exit "$status"
  fi
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
  --out_step_csv "$BASE/B19_16B_daps_${ID}_meas${MEAS_SEED}_runseed${RUN_SEED}_6S_rawtraj_step_features.csv" \
  --out_window_csv "$BASE/B19_16B_daps_${ID}_meas${MEAS_SEED}_runseed${RUN_SEED}_6S_rawtraj_window_features.csv"

echo "[final exact selector] $ID"
python scripts/b19/analyze_daps_exact_final_loss_selector.py \
  --daps_root external/daps \
  --samples_dir "$ROOT/samples" \
  --measurement_path "$MEAS" \
  --metrics_json "$METRICS" \
  --out_csv "$BASE/B19_16B_daps6S_${ID}_meas${MEAS_SEED}_runseed${RUN_SEED}_exact_final_loss_selector.csv"
