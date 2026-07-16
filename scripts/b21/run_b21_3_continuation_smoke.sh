#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
DAPS="$REPO/external/daps"
B19_BASE=${B19_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver}
B21_BASE=${B21_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver}
PYTHON_BIN=${PYTHON_BIN:-/egr/research-pac/huang248/conda-envs/daps/bin/python}
GPU=${GPU:-${1:-0}}
IMAGE=${IMAGE:-00046}
MEAS_SEED=${MEAS_SEED:-5001}
RUN_SEED=${RUN_SEED:-6900}
ANN_STEPS=${ANN_STEPS:-400}
DIFF_STEPS=${DIFF_STEPS:-5}
SPLIT_STEP=${SPLIT_STEP:-200}
BRANCH_SEEDS=${BRANCH_SEEDS:-"7900 7901 7902"}
B21_FORCE=${B21_FORCE:-0}
OUT=${OUT:-$B21_BASE/B21_3_continuation_smoke_${IMAGE}_seed${RUN_SEED}_split${SPLIT_STEP}}
SAVE_DIR="$OUT/daps_results"
LOGDIR="$OUT/logs"
METRICDIR="$OUT/metrics"
MEAS_PATH="$B19_BASE/measurements/ffhq${IMAGE}_phase_noise005_meas${MEAS_SEED}.pt"
DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024
DATA_NAME="b21-ffhq-${IMAGE}"
ANALYZER="$REPO/scripts/b19/analyze_daps_exact_final_loss_selector.py"
CHECKER="$REPO/scripts/b21/check_b21_3_continuation_smoke.py"

cd "$REPO"
mkdir -p "$OUT" "$SAVE_DIR" "$LOGDIR" "$METRICDIR"

[[ -x "$PYTHON_BIN" ]] || { echo "[fatal] PYTHON_BIN not executable: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$MEAS_PATH" ]] || { echo "[fatal] missing locked measurement: $MEAS_PATH" >&2; exit 2; }
[[ -f "$ANALYZER" ]] || { echo "[fatal] missing analyzer: $ANALYZER" >&2; exit 2; }
[[ -f "$CHECKER" ]] || { echo "[fatal] missing checker: $CHECKER" >&2; exit 2; }
grep -q 'B21.3 continuation patch' "$DAPS/sampler.py" || {
  echo "[fatal] B21.3 DAPS patch is not applied. Run:" >&2
  echo "  bash scripts/b21/apply_b21_3_continuation_patch.sh" >&2
  exit 2
}

"$PYTHON_BIN" - <<'PY'
import sys, torch, yaml
print("python", sys.executable)
print("torch", torch.__version__)
print("yaml", yaml.__version__)
PY

# Audited one-image dataset config.  This never modifies the locked measurement.
IMAGE=$(printf '%05d' "$((10#$IMAGE))")
mkdir -p "$DAPS/dataset/$DATA_NAME" "$DAPS/configs/data"
ln -sfn "$DATA_ROOT/00000/${IMAGE}.png" "$DAPS/dataset/$DATA_NAME/${IMAGE}.png"
cat > "$DAPS/configs/data/${DATA_NAME}.yaml" <<YAML
name: image
root: 'dataset/${DATA_NAME}'
resolution: 256
start_id: 0
end_id: 1
YAML

cat > "$OUT/launch_env.txt" <<EOF
timestamp=$(date -Is)
repo=$REPO
git_head=$(git rev-parse HEAD 2>/dev/null || true)
python=$PYTHON_BIN
gpu=$GPU
image=$IMAGE
measurement_seed=$MEAS_SEED
measurement_path=$MEAS_PATH
run_seed=$RUN_SEED
ann_steps=$ANN_STEPS
diff_steps=$DIFF_STEPS
split_step=$SPLIT_STEP
branch_seeds=$BRANCH_SEEDS
EOF

sample_path() {
  local case_name="$1"
  echo "$SAVE_DIR/$case_name/samples/00000_run0000.png"
}

metric_path() {
  local case_name="$1"
  echo "$METRICDIR/${case_name}.csv"
}

run_case() {
  local case_name="$1"
  shift
  local sample csv log
  sample=$(sample_path "$case_name")
  csv=$(metric_path "$case_name")
  log="$LOGDIR/${case_name}.log"

  if [[ "$B21_FORCE" != "1" && -s "$sample" && -s "$csv" ]]; then
    echo "[skip] $case_name"
    return 0
  fi
  if [[ "$B21_FORCE" == "1" ]]; then
    rm -rf "$SAVE_DIR/$case_name"
    rm -f "$csv" "$log"
  fi

  echo "[run] case=$case_name gpu=$GPU log=$log"
  (
    cd "$DAPS"
    env \
      -u B21_CONT_ENABLE \
      -u B21_CONT_STATE_PATH \
      -u B21_CONT_NOISE_SEED \
      -u B21_SAVE_STATE_STEPS \
      -u B21_START_NOISE_SEED \
      -u B21_SOURCE_SEED \
      B20_LF_ENABLE=0 \
      B20_LF_ALPHA=0.0 \
      B21_MEASUREMENT_PATH="$MEAS_PATH" \
      "$@" \
      CUDA_VISIBLE_DEVICES="$GPU" \
      "$PYTHON_BIN" posterior_sample.py \
        +data="$DATA_NAME" \
        +measurement_path="$MEAS_PATH" \
        +model=ffhq256ddpm \
        +sampler=edm_daps \
        +task=phase_retrieval \
        task_group=pixel \
        batch_size=1 \
        data.start_id=0 \
        data.end_id=1 \
        gpu=0 \
        seed="$RUN_SEED" \
        num_runs=1 \
        name="$case_name" \
        save_dir="$SAVE_DIR" \
        sampler.diffusion_scheduler_config.num_steps="$DIFF_STEPS" \
        sampler.annealing_scheduler_config.num_steps="$ANN_STEPS" \
        save_samples=true \
        save_traj=false \
        save_traj_raw_data=false \
        save_traj_video=false
  ) > "$log" 2>&1

  [[ -s "$sample" ]] || { echo "[fatal] missing sample after $case_name: $sample" >&2; tail -80 "$log" >&2; exit 3; }

  "$PYTHON_BIN" "$ANALYZER" \
    --daps_root "$DAPS" \
    --samples_dir "$SAVE_DIR/$case_name/samples" \
    --measurement_path "$MEAS_PATH" \
    --out_csv "$csv" \
    >> "$log" 2>&1
  [[ -s "$csv" ]] || { echo "[fatal] missing metric CSV after $case_name: $csv" >&2; exit 3; }
  echo "[done] $case_name"
}

# C0: a completely default-off full run must still complete.
run_case default_full

# Source full run.  The explicit start-noise seed is smoke-only and lets the
# synthetic step-0 payload reproduce the same initial xt exactly.
run_case source_full \
  B21_START_NOISE_SEED="$RUN_SEED" \
  B21_SOURCE_SEED="$RUN_SEED" \
  B21_SAVE_STATE_STEPS="0,$SPLIT_STEP"

STATE_DIR="$SAVE_DIR/source_full/continuation_states/run0000"
STATE0="$STATE_DIR/step0000.pt"
STATE_SPLIT="$STATE_DIR/step$(printf '%04d' "$SPLIT_STEP").pt"
[[ -s "$STATE0" ]] || { echo "[fatal] missing step-0 state: $STATE0" >&2; exit 4; }
[[ -s "$STATE_SPLIT" ]] || { echo "[fatal] missing split state: $STATE_SPLIT" >&2; exit 4; }

# Exact continuation-from-0 reproducibility check.
run_case cont0_same_seed \
  B21_CONT_ENABLE=1 \
  B21_CONT_STATE_PATH="$STATE0" \
  B21_CONT_NOISE_SEED="$RUN_SEED"

# Three independent late continuations from the same step-SPLIT x0y payload.
for branch_seed in $BRANCH_SEEDS; do
  run_case "branch_seed${branch_seed}" \
    B21_CONT_ENABLE=1 \
    B21_CONT_STATE_PATH="$STATE_SPLIT" \
    B21_CONT_NOISE_SEED="$branch_seed"
done

"$PYTHON_BIN" "$CHECKER" \
  --out "$OUT" \
  --repo "$REPO" \
  --image "$IMAGE" \
  --run-seed "$RUN_SEED" \
  --split-step "$SPLIT_STEP" \
  --branch-seeds $BRANCH_SEEDS \
  | tee "$OUT/checker_stdout.txt"

echo "[done] B21.3 continuation smoke artifacts: $OUT"
