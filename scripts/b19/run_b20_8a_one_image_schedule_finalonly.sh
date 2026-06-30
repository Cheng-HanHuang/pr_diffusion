#!/usr/bin/env bash
set -euo pipefail

ID="${1:?usage: run_b20_8a_one_image_schedule_finalonly.sh IMAGE_ID PHYSICAL_GPU [MEAS_SEED] [RUN_SEED] [ANN_STEPS] [DIFF_STEPS]}"
PHYS_GPU="${2:?usage: run_b20_8a_one_image_schedule_finalonly.sh IMAGE_ID PHYSICAL_GPU [MEAS_SEED] [RUN_SEED] [ANN_STEPS] [DIFF_STEPS]}"
MEAS_SEED="${3:-5001}"
RUN_SEED="${4:-4400}"
ANN_STEPS="${5:-400}"
DIFF_STEPS="${6:-5}"
NUM_RUNS="${NUM_RUNS:-16}"

REPO=/egr/research-pac/huang248/pr_diffusion_b19_solver
BASE=/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver
DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024

MEAS="$BASE/measurements/ffhq${ID}_phase_noise005_meas${MEAS_SEED}.pt"

TAG="ann${ANN_STEPS}_diff${DIFF_STEPS}"
NAME="b20_8a_${TAG}_${ID}_meas${MEAS_SEED}_runseed${RUN_SEED}_${NUM_RUNS}S"
SAVE_DIR="results/b20_8a_${TAG}_${NUM_RUNS}S_meas${MEAS_SEED}_runseed${RUN_SEED}"
ROOT="$REPO/external/daps/$SAVE_DIR/$NAME"
METRICS="$ROOT/metrics.json"

EXACT_OUT="$BASE/B20_8A_${TAG}_daps${NUM_RUNS}S_${ID}_meas${MEAS_SEED}_runseed${RUN_SEED}_exact_final_loss_selector.csv"
RUN_LOG="$BASE/B20_8A_${TAG}_daps${NUM_RUNS}S_${ID}_meas${MEAS_SEED}_runseed${RUN_SEED}.log"
EXACT_LOG="$BASE/B20_8A_${TAG}_daps${NUM_RUNS}S_${ID}_meas${MEAS_SEED}_runseed${RUN_SEED}_exact_selector.log"

if [ ! -f "$MEAS" ]; then
  echo "[error] missing measurement $MEAS"
  exit 1
fi

mkdir -p "$REPO/external/daps/dataset/b20-ffhq-${ID}"
ln -sfn "$DATA_ROOT/00000/${ID}.png" "$REPO/external/daps/dataset/b20-ffhq-${ID}/${ID}.png"

mkdir -p "$REPO/external/daps/configs/data"
cat > "$REPO/external/daps/configs/data/b20-ffhq-${ID}.yaml" <<YAML
name: image
root: 'dataset/b20-ffhq-${ID}'
resolution: 256
start_id: 0
end_id: 1
YAML

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -f "$METRICS" ]; then
  echo "[run] image=$ID meas_seed=$MEAS_SEED run_seed=$RUN_SEED num_runs=$NUM_RUNS ann=$ANN_STEPS diff=$DIFF_STEPS gpu=$PHYS_GPU"
  cd "$REPO/external/daps"
  conda activate daps

  set +e
  CUDA_VISIBLE_DEVICES="$PHYS_GPU" python posterior_sample.py \
    +data=b20-ffhq-${ID} \
    +model=ffhq256ddpm \
    +task=phase_retrieval \
    +sampler=edm_daps \
    +measurement_path="$MEAS" \
    task_group=pixel \
    save_dir="$SAVE_DIR" \
    num_runs="$NUM_RUNS" \
    sampler.diffusion_scheduler_config.num_steps="$DIFF_STEPS" \
    sampler.annealing_scheduler_config.num_steps="$ANN_STEPS" \
    batch_size=1 \
    data.start_id=0 data.end_id=1 \
    save_traj=false \
    save_traj_raw_data=false \
    name="$NAME" \
    seed="$RUN_SEED" \
    gpu=0 2>&1 | tee "$RUN_LOG"
  status=${PIPESTATUS[0]}

  if [ "$status" -ne 0 ]; then
    echo "[retry] seed= failed; retrying +seed="
    CUDA_VISIBLE_DEVICES="$PHYS_GPU" python posterior_sample.py \
      +data=b20-ffhq-${ID} \
      +model=ffhq256ddpm \
      +task=phase_retrieval \
      +sampler=edm_daps \
      +measurement_path="$MEAS" \
      task_group=pixel \
      save_dir="$SAVE_DIR" \
      num_runs="$NUM_RUNS" \
      sampler.diffusion_scheduler_config.num_steps="$DIFF_STEPS" \
      sampler.annealing_scheduler_config.num_steps="$ANN_STEPS" \
      batch_size=1 \
      data.start_id=0 data.end_id=1 \
      save_traj=false \
      save_traj_raw_data=false \
      name="$NAME" \
      +seed="$RUN_SEED" \
      gpu=0 2>&1 | tee -a "$RUN_LOG"
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

if [ ! -f "$EXACT_OUT" ]; then
  python scripts/b19/analyze_daps_exact_final_loss_selector.py \
    --daps_root external/daps \
    --samples_dir "$ROOT/samples" \
    --measurement_path "$MEAS" \
    --metrics_json "$METRICS" \
    --out_csv "$EXACT_OUT" > "$EXACT_LOG" 2>&1
fi

echo "[done] $EXACT_OUT"
