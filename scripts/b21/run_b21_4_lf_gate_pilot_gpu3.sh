#!/usr/bin/env bash
set -euo pipefail

# B21.4 pilot: matched base vs LF candidate generation, GPU 3 only.
# Selection is done afterwards by exact final measurement loss.

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
B19_BASE=${B19_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver}
B21_BASE=${B21_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver}
OUT=${OUT:-$B21_BASE/B21_4_lf_gate_pilot}
LOGDIR=${LOGDIR:-$OUT/logs}
RUNNER=${RUNNER:-$REPO/scripts/b19/run_b20_12a_one_image_portfolio_finalonly.sh}
IMAGES=${IMAGES:-"00171 00480 00746 00971"}
MEAS_SEED=${MEAS_SEED:-5001}
SEEDS=${SEEDS:-"6800 6801 6802 6803 6804 6805 6806 6807 6808 6809 6810 6811 6812 6813 6814 6815"}
GPU=${GPU:-3}
ANN_STEPS=${ANN_STEPS:-400}
DIFF_STEPS=${DIFF_STEPS:-5}
LF_VARIANT=${LF_VARIANT:-lf050}
BASE_VARIANT=${BASE_VARIANT:-base}
B21_FORCE=${B21_FORCE:-0}
B21_SMOKE_ONLY=${B21_SMOKE_ONLY:-0}

cd "$REPO"
mkdir -p "$OUT" "$LOGDIR"

if [[ "$GPU" != "3" ]]; then
  echo "[fatal] This wrapper is intended for GPU=3 only. Got GPU=$GPU" >&2
  exit 2
fi
if [[ ! -x "$RUNNER" ]]; then
  echo "[fatal] runner missing or not executable: $RUNNER" >&2
  echo "        Expected local B20.12A runner from earlier B21.2 recovery." >&2
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
  echo "GPU=$GPU"
  echo "ANN_STEPS=$ANN_STEPS"
  echo "DIFF_STEPS=$DIFF_STEPS"
  echo "LF_VARIANT=$LF_VARIANT"
  echo "BASE_VARIANT=$BASE_VARIANT"
} > "$OUT/launch_env.txt"

python - <<PY > "$OUT/job_manifest.csv"
images = """$IMAGES""".split()
seeds = [int(x) for x in """$SEEDS""".split()]
print("image_id,meas_seed,seed,variant,ann_steps,diff_steps,gpu")
for image in images:
    for seed in seeds:
        for variant in ["$BASE_VARIANT", "$LF_VARIANT"]:
            print(f"{image},$MEAS_SEED,{seed},{variant},$ANN_STEPS,$DIFF_STEPS,$GPU")
PY

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
  local log="$LOGDIR/B21_4_${image}_meas${MEAS_SEED}_seed${seed}_ann${ANN_STEPS}_diff${DIFF_STEPS}_${variant}_gpu${GPU}.log"
  local csv
  csv="$(expected_csv "$image" "$seed" "$variant")"
  if [[ "$B21_FORCE" != "1" && -s "$csv" ]]; then
    echo "[skip] existing $csv"
    return 0
  fi
  echo "[run] image=$image seed=$seed variant=$variant gpu=$GPU log=$log"
  NUM_RUNS=1 bash "$RUNNER" "$image" "$GPU" "$MEAS_SEED" "$seed" "$ANN_STEPS" "$DIFF_STEPS" "$variant" \
    > "$log" 2>&1
  if [[ ! -s "$csv" ]]; then
    echo "[fatal] expected CSV missing after run: $csv" >&2
    echo "[fatal] see log: $log" >&2
    exit 3
  fi
}

# Smoke test first job pair before launching the rest.
first_image=$(echo "$IMAGES" | awk '{print $1}')
first_seed=$(echo "$SEEDS" | awk '{print $1}')
run_one "$first_image" "$first_seed" "$BASE_VARIANT"
run_one "$first_image" "$first_seed" "$LF_VARIANT"

if [[ "$B21_SMOKE_ONLY" == "1" ]]; then
  echo "[smoke-only] stopping after first matched pair"
  python scripts/b21/collect_b21_4_lf_gate_pilot.py \
    --b19_base "$B19_BASE" \
    --outdir "$OUT" \
    --images "$first_image" \
    --meas_seed "$MEAS_SEED" \
    --seeds "$first_seed" \
    --lf_variant "$LF_VARIANT" \
    --ann_steps "$ANN_STEPS" \
    --diff_steps "$DIFF_STEPS" \
    --report_path "$REPO/docs/b21/b21_4_lf_gate_pilot.md"
  exit 0
fi

for image in $IMAGES; do
  for seed in $SEEDS; do
    run_one "$image" "$seed" "$BASE_VARIANT"
    run_one "$image" "$seed" "$LF_VARIANT"
  done
done

python scripts/b21/collect_b21_4_lf_gate_pilot.py \
  --b19_base "$B19_BASE" \
  --outdir "$OUT" \
  --images "$(echo "$IMAGES" | tr ' ' ',')" \
  --meas_seed "$MEAS_SEED" \
  --seeds "${SEEDS// /,}" \
  --lf_variant "$LF_VARIANT" \
  --ann_steps "$ANN_STEPS" \
  --diff_steps "$DIFF_STEPS" \
  --report_path "$REPO/docs/b21/b21_4_lf_gate_pilot.md"

echo "[done] report: $REPO/docs/b21/b21_4_lf_gate_pilot.md"
echo "[done] artifacts: $OUT"
