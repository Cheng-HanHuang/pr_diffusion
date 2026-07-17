#!/usr/bin/env bash
set -euo pipefail

# B21.3 equal-cost mini-pilot.
#
# For each (image, parent seed), both policies share the same source_full
# candidate (ann400).  The remaining 400-step budget is allocated as:
#
#   Fresh2:    source_full + one independent ann400 fresh_extra
#   Branch3:   source_full + two ann200 continuations from step 200
#
# Therefore each policy costs 800 annealing transitions and the comparison is
# paired at the case level.

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
DAPS="$REPO/external/daps"
B19_BASE=${B19_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver}
B21_BASE=${B21_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver}
PYTHON_BIN=${PYTHON_BIN:-/egr/research-pac/huang248/conda-envs/daps/bin/python}
IMAGES=${IMAGES:-"00046 00171 00746"}
PARENT_SEEDS=${PARENT_SEEDS:-"7000 7001 7002 7003 7004 7005 7006 7007"}
GPUS=${GPUS:-"0 1 2 3"}
MEAS_SEED=${MEAS_SEED:-5001}
ANN_STEPS=${ANN_STEPS:-400}
DIFF_STEPS=${DIFF_STEPS:-5}
SPLIT_STEP=${SPLIT_STEP:-200}
FRESH_OFFSET=${FRESH_OFFSET:-10000}
BRANCH_A_OFFSET=${BRANCH_A_OFFSET:-20000}
BRANCH_B_OFFSET=${BRANCH_B_OFFSET:-30000}
B21_FORCE=${B21_FORCE:-0}
OUT=${OUT:-$B21_BASE/B21_3_branch_vs_fresh_pilot_3img_8seed_split${SPLIT_STEP}}
LOGDIR="$OUT/logs"
CASE_ROOT="$OUT/cases"
MANIFEST="$OUT/manifest.tsv"
DONE="$OUT/done.tsv"
FAIL="$OUT/fail.tsv"
ANALYZER="$REPO/scripts/b19/analyze_daps_exact_final_loss_selector.py"
PILOT_ANALYZER="$REPO/scripts/b21/analyze_b21_3_branch_vs_fresh_pilot.py"
DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024

cd "$REPO"
mkdir -p "$OUT" "$LOGDIR" "$CASE_ROOT"

[[ -x "$PYTHON_BIN" ]] || { echo "[fatal] PYTHON_BIN not executable: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$ANALYZER" ]] || { echo "[fatal] missing analyzer: $ANALYZER" >&2; exit 2; }
[[ -f "$PILOT_ANALYZER" ]] || { echo "[fatal] missing pilot analyzer: $PILOT_ANALYZER" >&2; exit 2; }
grep -q 'B21.3 continuation patch' "$DAPS/sampler.py" || {
  echo "[fatal] B21.3 DAPS continuation patch is not applied" >&2
  echo "        Run: PYTHON_BIN=$PYTHON_BIN bash scripts/b21/apply_b21_3_continuation_patch.sh" >&2
  exit 2
}

"$PYTHON_BIN" - <<'PY'
import sys, torch, yaml, pandas, PIL
print("python", sys.executable)
print("torch", torch.__version__)
print("yaml", yaml.__version__)
print("pandas", pandas.__version__)
print("PIL", PIL.__version__)
PY

read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
[[ "$NGPU" -ge 1 ]] || { echo "[fatal] no GPUs specified" >&2; exit 2; }

# Prepare all one-image configs before concurrent workers begin.
for image in $IMAGES; do
  image=$(printf '%05d' "$((10#$image))")
  data_name="b21-ffhq-${image}"
  gt="$DATA_ROOT/00000/${image}.png"
  meas="$B19_BASE/measurements/ffhq${image}_phase_noise005_meas${MEAS_SEED}.pt"
  [[ -f "$gt" ]] || { echo "[fatal] missing image: $gt" >&2; exit 2; }
  [[ -f "$meas" ]] || { echo "[fatal] missing measurement: $meas" >&2; exit 2; }
  mkdir -p "$DAPS/dataset/$data_name" "$DAPS/configs/data"
  ln -sfn "$gt" "$DAPS/dataset/$data_name/${image}.png"
  cat > "$DAPS/configs/data/${data_name}.yaml" <<YAML
name: image
root: 'dataset/${data_name}'
resolution: 256
start_id: 0
end_id: 1
YAML
done

{
  echo "timestamp=$(date -Is)"
  echo "repo=$REPO"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "python=$PYTHON_BIN"
  echo "images=$IMAGES"
  echo "parent_seeds=$PARENT_SEEDS"
  echo "gpus=$GPUS"
  echo "measurement_seed=$MEAS_SEED"
  echo "ann_steps=$ANN_STEPS"
  echo "diff_steps=$DIFF_STEPS"
  echo "split_step=$SPLIT_STEP"
  echo "fresh_offset=$FRESH_OFFSET"
  echo "branch_a_offset=$BRANCH_A_OFFSET"
  echo "branch_b_offset=$BRANCH_B_OFFSET"
  echo "policy_fresh_cost=$((ANN_STEPS * 2))"
  echo "policy_branch_cost=$((ANN_STEPS + 2 * (ANN_STEPS - SPLIT_STEP)))"
  echo "git_status_short:"
  git status --short || true
} > "$OUT/launch_env.txt"

: > "$MANIFEST"
: > "$DONE"
: > "$FAIL"
printf "job_id\timage_id\tparent_seed\tgpu\tsource_seed\tfresh_seed\tbranch_a_seed\tbranch_b_seed\n" > "$MANIFEST"

jid=0
for image in $IMAGES; do
  image=$(printf '%05d' "$((10#$image))")
  for parent in $PARENT_SEEDS; do
    gpu="${GPU_ARR[$((jid % NGPU))]}"
    fresh=$((parent + FRESH_OFFSET))
    branch_a=$((parent + BRANCH_A_OFFSET))
    branch_b=$((parent + BRANCH_B_OFFSET))
    printf "%d\t%s\t%d\t%s\t%d\t%d\t%d\t%d\n" \
      "$jid" "$image" "$parent" "$gpu" "$parent" "$fresh" "$branch_a" "$branch_b" \
      >> "$MANIFEST"
    jid=$((jid + 1))
  done
done

echo "[manifest] $MANIFEST"
echo "[cases] $jid"
echo "[gpus] $GPUS"
echo "[out] $OUT"

run_candidate() {
  local case_dir="$1"
  local image="$2"
  local candidate="$3"
  local hydra_seed="$4"
  local gpu="$5"
  shift 5

  local save_dir="$case_dir/daps_results"
  local metric_dir="$case_dir/metrics"
  local log_dir="$case_dir/logs"
  local timing_file="$case_dir/timings.tsv"
  local sample="$save_dir/$candidate/samples/00000_run0000.png"
  local csv="$metric_dir/${candidate}.csv"
  local log="$log_dir/${candidate}.log"
  local meas="$B19_BASE/measurements/ffhq${image}_phase_noise005_meas${MEAS_SEED}.pt"
  local data_name="b21-ffhq-${image}"

  mkdir -p "$save_dir" "$metric_dir" "$log_dir"
  if [[ "$B21_FORCE" != "1" && -s "$sample" && -s "$csv" ]]; then
    echo "[skip][gpu=$gpu] image=$image candidate=$candidate"
    return 0
  fi
  if [[ "$B21_FORCE" == "1" ]]; then
    rm -rf "$save_dir/$candidate"
    rm -f "$csv" "$log"
  fi

  local start end elapsed
  start=$(date +%s)
  echo "[run][gpu=$gpu] image=$image candidate=$candidate seed=$hydra_seed log=$log"
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
      B21_MEASUREMENT_PATH="$meas" \
      "$@" \
      CUDA_VISIBLE_DEVICES="$gpu" \
      "$PYTHON_BIN" posterior_sample.py \
        +data="$data_name" \
        +measurement_path="$meas" \
        +model=ffhq256ddpm \
        +sampler=edm_daps \
        +task=phase_retrieval \
        task_group=pixel \
        batch_size=1 \
        data.start_id=0 \
        data.end_id=1 \
        gpu=0 \
        seed="$hydra_seed" \
        num_runs=1 \
        name="$candidate" \
        save_dir="$save_dir" \
        sampler.diffusion_scheduler_config.num_steps="$DIFF_STEPS" \
        sampler.annealing_scheduler_config.num_steps="$ANN_STEPS" \
        save_samples=true \
        save_traj=false \
        save_traj_raw_data=false \
        save_traj_video=false
  ) > "$log" 2>&1

  [[ -s "$sample" ]] || {
    echo "[fatal][gpu=$gpu] missing sample: $sample" >&2
    tail -80 "$log" >&2 || true
    return 3
  }

  "$PYTHON_BIN" "$ANALYZER" \
    --daps_root "$DAPS" \
    --samples_dir "$save_dir/$candidate/samples" \
    --measurement_path "$meas" \
    --out_csv "$csv" \
    >> "$log" 2>&1
  [[ -s "$csv" ]] || { echo "[fatal][gpu=$gpu] missing metric CSV: $csv" >&2; return 3; }

  end=$(date +%s)
  elapsed=$((end - start))
  touch "$timing_file"
  awk -F'\t' -v c="$candidate" '$1 != c' "$timing_file" > "${timing_file}.tmp" || true
  printf "%s\t%d\n" "$candidate" "$elapsed" >> "${timing_file}.tmp"
  mv "${timing_file}.tmp" "$timing_file"
  echo "[done][gpu=$gpu] image=$image candidate=$candidate elapsed=${elapsed}s"
}

run_case() {
  local job_id="$1"
  local image="$2"
  local parent="$3"
  local gpu="$4"
  local source_seed="$5"
  local fresh_seed="$6"
  local branch_a_seed="$7"
  local branch_b_seed="$8"

  local case_dir="$CASE_ROOT/${image}_parent${parent}"
  local state="$case_dir/daps_results/source_full/continuation_states/run0000/step$(printf '%04d' "$SPLIT_STEP").pt"
  mkdir -p "$case_dir"

  cat > "$case_dir/case_env.txt" <<EOF
job_id=$job_id
image_id=$image
parent_seed=$parent
gpu=$gpu
source_seed=$source_seed
fresh_seed=$fresh_seed
branch_a_seed=$branch_a_seed
branch_b_seed=$branch_b_seed
split_step=$SPLIT_STEP
EOF

  # Shared candidate for both policies.  Save the clean x0y state at split step.
  run_candidate "$case_dir" "$image" source_full "$source_seed" "$gpu" \
    B21_START_NOISE_SEED="$source_seed" \
    B21_SOURCE_SEED="$source_seed" \
    B21_SAVE_STATE_STEPS="$SPLIT_STEP"

  [[ -s "$state" ]] || { echo "[fatal][gpu=$gpu] missing split state: $state" >&2; return 4; }

  # Fresh2 receives one additional full independent trajectory.
  run_candidate "$case_dir" "$image" fresh_extra "$fresh_seed" "$gpu" \
    B21_START_NOISE_SEED="$fresh_seed" \
    B21_SOURCE_SEED="$fresh_seed"

  # Branch3 receives two independent late continuations from the shared state.
  run_candidate "$case_dir" "$image" branch_a "$branch_a_seed" "$gpu" \
    B21_CONT_ENABLE=1 \
    B21_CONT_STATE_PATH="$state" \
    B21_CONT_NOISE_SEED="$branch_a_seed"

  run_candidate "$case_dir" "$image" branch_b "$branch_b_seed" "$gpu" \
    B21_CONT_ENABLE=1 \
    B21_CONT_STATE_PATH="$state" \
    B21_CONT_NOISE_SEED="$branch_b_seed"
}

pids=()
for gpu in "${GPU_ARR[@]}"; do
  (
    set -euo pipefail
    tail -n +2 "$MANIFEST" | awk -F'\t' -v g="$gpu" '$4 == g {print}' | \
    while IFS=$'\t' read -r job_id image parent assigned_gpu source fresh branch_a branch_b; do
      echo "[case start] job=$job_id image=$image parent=$parent gpu=$assigned_gpu"
      if run_case "$job_id" "$image" "$parent" "$assigned_gpu" "$source" "$fresh" "$branch_a" "$branch_b"; then
        printf "%s\t%s\t%s\t%s\tOK\n" "$job_id" "$image" "$parent" "$assigned_gpu" >> "$DONE"
        echo "[case done] job=$job_id image=$image parent=$parent gpu=$assigned_gpu"
      else
        status=$?
        printf "%s\t%s\t%s\t%s\tFAIL\t%s\n" "$job_id" "$image" "$parent" "$assigned_gpu" "$status" >> "$FAIL"
        echo "[case fail] job=$job_id image=$image parent=$parent gpu=$assigned_gpu status=$status"
      fi
    done
  ) > "$LOGDIR/worker_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
  echo "[worker] gpu=$gpu pid=${pids[-1]} log=$LOGDIR/worker_gpu${gpu}.log"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

fail_count=$(wc -l < "$FAIL")
done_count=$(wc -l < "$DONE")
echo "[workers done] completed=$done_count failed=$fail_count"
if [[ "$status" != "0" || "$fail_count" != "0" ]]; then
  echo "[fatal] one or more cases failed; inspect $LOGDIR/worker_gpu*.log and $FAIL" >&2
  exit 5
fi

"$PYTHON_BIN" "$PILOT_ANALYZER" \
  --out "$OUT" \
  --repo "$REPO" \
  --ann-steps "$ANN_STEPS" \
  --split-step "$SPLIT_STEP" \
  | tee "$OUT/analyzer_stdout.txt"

echo "[done] B21.3 branch-vs-fresh pilot artifacts: $OUT"
