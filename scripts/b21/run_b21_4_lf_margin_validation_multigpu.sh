#!/usr/bin/env bash
set -euo pipefail

# B21.4 validation: matched base vs LF candidate generation, multi-GPU.
# Runs base and LF arms as separate candidates, then applies the clean-free
# margin selector:
#   select LF iff exact_loss_lf < exact_loss_base - THETA

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
B19_BASE=${B19_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver}
B21_BASE=${B21_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver}
OUT=${OUT:-$B21_BASE/B21_4_lf_margin_validation}
LOGDIR=${LOGDIR:-$OUT/logs}
RUNNER=${RUNNER:-$REPO/scripts/b19/run_b20_12a_one_image_portfolio_finalonly.sh}
IMAGES=${IMAGES:-"00171 00480 00746 00971"}
MEAS_SEED=${MEAS_SEED:-5001}
SEEDS=${SEEDS:-"6816 6817 6818 6819 6820 6821 6822 6823 6824 6825 6826 6827 6828 6829 6830 6831 6832 6833 6834 6835 6836 6837 6838 6839 6840 6841 6842 6843 6844 6845 6846 6847"}
GPUS=${GPUS:-"0 1 2 3"}
ANN_STEPS=${ANN_STEPS:-400}
DIFF_STEPS=${DIFF_STEPS:-5}
LF_VARIANT=${LF_VARIANT:-lf050}
BASE_VARIANT=${BASE_VARIANT:-base}
THETA=${THETA:-0.5}
B21_FORCE=${B21_FORCE:-0}
B21_SMOKE_ONLY=${B21_SMOKE_ONLY:-0}

cd "$REPO"
mkdir -p "$OUT" "$LOGDIR" "$OUT/joblists"

if [[ ! -x "$RUNNER" ]]; then
  echo "[fatal] runner missing or not executable: $RUNNER" >&2
  echo "        Expected local B20.12A runner from earlier B21.2/B21.4 work." >&2
  exit 2
fi

read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
if [[ "$NGPU" -lt 1 ]]; then
  echo "[fatal] no GPUs specified" >&2
  exit 2
fi

{
  echo "timestamp=$(date -Is)"
  echo "repo=$REPO"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "git_status_short:"
  git status --short || true
  echo "B19_BASE=$B19_BASE"
  echo "B21_BASE=$B21_BASE"
  echo "OUT=$OUT"
  echo "RUNNER=$RUNNER"
  echo "IMAGES=$IMAGES"
  echo "MEAS_SEED=$MEAS_SEED"
  echo "SEEDS=$SEEDS"
  echo "GPUS=$GPUS"
  echo "ANN_STEPS=$ANN_STEPS"
  echo "DIFF_STEPS=$DIFF_STEPS"
  echo "LF_VARIANT=$LF_VARIANT"
  echo "BASE_VARIANT=$BASE_VARIANT"
  echo "THETA=$THETA"
} > "$OUT/launch_env.txt"

expected_csv() {
  local image="$1"
  local seed="$2"
  local variant="$3"
  echo "$B19_BASE/B20_12A_${image}_ann${ANN_STEPS}_diff${DIFF_STEPS}_${variant}_daps1S_meas${MEAS_SEED}_runseed${seed}_exact_final_loss_selector.csv"
}

run_one() {
  local image="$1"
  local seed="$2"
  local variant="$3"
  local gpu="$4"
  local log="$LOGDIR/B21_4_val_${image}_meas${MEAS_SEED}_seed${seed}_ann${ANN_STEPS}_diff${DIFF_STEPS}_${variant}_gpu${gpu}.log"
  local csv
  csv="$(expected_csv "$image" "$seed" "$variant")"
  if [[ "$B21_FORCE" != "1" && -s "$csv" ]]; then
    echo "[skip][gpu=$gpu] existing $csv"
    return 0
  fi
  echo "[run][gpu=$gpu] image=$image seed=$seed variant=$variant log=$log"
  NUM_RUNS=1 bash "$RUNNER" "$image" "$gpu" "$MEAS_SEED" "$seed" "$ANN_STEPS" "$DIFF_STEPS" "$variant" \
    > "$log" 2>&1
  if [[ ! -s "$csv" ]]; then
    echo "[fatal][gpu=$gpu] expected CSV missing after run: $csv" >&2
    echo "[fatal][gpu=$gpu] see log: $log" >&2
    exit 3
  fi
}

collect_and_report() {
  local images_arg seeds_arg
  images_arg="$(echo "$IMAGES" | tr ' ' ',')"
  seeds_arg="${SEEDS// /,}"
  python scripts/b21/collect_b21_4_lf_gate_pilot.py \
    --b19_base "$B19_BASE" \
    --outdir "$OUT" \
    --images "$images_arg" \
    --meas_seed "$MEAS_SEED" \
    --seeds "$seeds_arg" \
    --lf_variant "$LF_VARIANT" \
    --ann_steps "$ANN_STEPS" \
    --diff_steps "$DIFF_STEPS" \
    --report_path "$REPO/docs/b21/b21_4_lf_gate_validation_raw.md"

  python scripts/b21/apply_b21_4_margin_gate.py \
    --input_pairs "$OUT/b21_4_lf_gate_pilot_pairs.csv" \
    --outdir "$OUT" \
    --theta "$THETA" \
    --report_path "$REPO/docs/b21/b21_4_margin_validation.md"
}

# Smoke test the first matched pair on the first requested GPU.
first_image=$(echo "$IMAGES" | awk '{print $1}')
first_seed=$(echo "$SEEDS" | awk '{print $1}')
first_gpu="${GPU_ARR[0]}"
run_one "$first_image" "$first_seed" "$BASE_VARIANT" "$first_gpu"
run_one "$first_image" "$first_seed" "$LF_VARIANT" "$first_gpu"

if [[ "$B21_SMOKE_ONLY" == "1" ]]; then
  echo "[smoke-only] stopping after first matched pair"
  IMAGES="$first_image" SEEDS="$first_seed" collect_and_report
  exit 0
fi

# Build disjoint per-GPU job lists.  The first smoke-tested pair may be skipped
# naturally by run_one if it already exists.
rm -f "$OUT"/joblists/gpu_*.tsv
for gpu in "${GPU_ARR[@]}"; do
  : > "$OUT/joblists/gpu_${gpu}.tsv"
done

idx=0
for image in $IMAGES; do
  for seed in $SEEDS; do
    for variant in "$BASE_VARIANT" "$LF_VARIANT"; do
      gpu="${GPU_ARR[$((idx % NGPU))]}"
      printf "%s\t%s\t%s\t%s\n" "$image" "$seed" "$variant" "$gpu" >> "$OUT/joblists/gpu_${gpu}.tsv"
      idx=$((idx + 1))
    done
  done
done
cp "$OUT/joblists"/gpu_*.tsv "$OUT/" 2>/dev/null || true

pids=()
for gpu in "${GPU_ARR[@]}"; do
  (
    set -euo pipefail
    while IFS=$'\t' read -r image seed variant assigned_gpu; do
      [[ -z "${image:-}" ]] && continue
      run_one "$image" "$seed" "$variant" "$assigned_gpu"
    done < "$OUT/joblists/gpu_${gpu}.tsv"
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
if [[ "$status" != "0" ]]; then
  echo "[fatal] one or more GPU workers failed; inspect $LOGDIR/worker_gpu*.log" >&2
  exit 4
fi

collect_and_report

echo "[done] raw gate report: $REPO/docs/b21/b21_4_lf_gate_validation_raw.md"
echo "[done] margin report: $REPO/docs/b21/b21_4_margin_validation.md"
echo "[done] artifacts: $OUT"
