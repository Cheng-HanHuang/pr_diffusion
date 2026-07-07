#!/usr/bin/env bash
set -euo pipefail

# B21.2 small audited candidate-recovery rerun.
# Purpose: regenerate a compact set of final candidate PNGs with explicit
# sample-path logging so selector-v2 prior/symmetry scoring can proceed.
# This is NOT a full FFHQ100 rerun.

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
B19_BASE=${B19_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver}
B21_BASE=${B21_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver}
OUT=${OUT:-$B21_BASE/B21_2_candidate_recovery}
LOGDIR=${LOGDIR:-$OUT/logs}

IMAGES=(${IMAGES:-00136 00154 00253 00480 00971})
MEAS_SEED=${MEAS_SEED:-5001}
SEEDS=(${SEEDS:-6700 6701 6702 6703 6704 6705})
# variant:ann_steps:diff_steps. Keep this compact and plan-aligned.
ARMS=(${ARMS:-base:400:5 lf025:350:5 lf050:400:5})
GPUS=(${GPUS:-1 2 3})
B21_FORCE=${B21_FORCE:-0}
B21_SMOKE_ONLY=${B21_SMOKE_ONLY:-0}

RUNNER="$REPO/scripts/b19/run_b20_12a_one_image_portfolio_finalonly.sh"
if [[ ! -f "$RUNNER" ]]; then
  echo "[error] missing required local runner: $RUNNER" >&2
  echo "This wrapper intentionally reuses the inspected B20.12A final-only runner." >&2
  exit 2
fi

cd "$REPO"
mkdir -p "$OUT" "$LOGDIR"

echo "[info] repo=$REPO"
echo "[info] out=$OUT"
echo "[info] images=${IMAGES[*]} meas_seed=$MEAS_SEED seeds=${SEEDS[*]} arms=${ARMS[*]} gpus=${GPUS[*]} smoke=$B21_SMOKE_ONLY force=$B21_FORCE"

git rev-parse HEAD > "$OUT/git_head.txt" || true
git status --short > "$OUT/git_status_short.txt" || true
{
  echo "REPO=$REPO"
  echo "B19_BASE=$B19_BASE"
  echo "B21_BASE=$B21_BASE"
  echo "OUT=$OUT"
  echo "IMAGES=${IMAGES[*]}"
  echo "MEAS_SEED=$MEAS_SEED"
  echo "SEEDS=${SEEDS[*]}"
  echo "ARMS=${ARMS[*]}"
  echo "GPUS=${GPUS[*]}"
  echo "B21_FORCE=$B21_FORCE"
  echo "B21_SMOKE_ONLY=$B21_SMOKE_ONLY"
} > "$OUT/launch_env.txt"

# Run-time locked measurement audit: path, file SHA, and tensor stats.
python - <<'PY' "$B19_BASE" "$OUT" "$MEAS_SEED" "${IMAGES[@]}"
import csv, hashlib, json, sys
from pathlib import Path
import torch

base = Path(sys.argv[1])
out = Path(sys.argv[2])
meas_seed = int(sys.argv[3])
images = [f"{int(x):05d}" for x in sys.argv[4:]]
rows = []
for image_id in images:
    p = base / "measurements" / f"ffhq{image_id}_phase_noise005_meas{meas_seed}.pt"
    rec = {"image_id": image_id, "meas_seed": meas_seed, "measurement_path": str(p), "exists": p.exists()}
    if p.exists():
        data = p.read_bytes()
        rec["file_sha256"] = hashlib.sha256(data).hexdigest()
        obj = torch.load(p, map_location="cpu")
        y = obj.get("measurement", obj if torch.is_tensor(obj) else None)
        if y is not None:
            y = y.detach().cpu().float()
            flat = y.reshape(-1)
            rec.update({
                "shape": "x".join(str(v) for v in y.shape),
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max()),
                "first4": ",".join(f"{float(v):.8g}" for v in flat[:4]),
            })
    rows.append(rec)
with (out / "measurement_manifest.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
    w.writeheader(); w.writerows(rows)
print(f"[write] {out / 'measurement_manifest.csv'}")
PY

MANIFEST="$OUT/job_manifest.csv"
echo "image_id,meas_seed,run_seed,variant,ann_steps,diff_steps,gpu,exact_csv,log_path,skipped_existing" > "$MANIFEST"

idx=0
launched=0
for ID0 in "${IMAGES[@]}"; do
  ID=$(printf "%05d" "$((10#$ID0))")
  for SEED in "${SEEDS[@]}"; do
    for ARM in "${ARMS[@]}"; do
      IFS=: read -r VARIANT ANN_STEPS DIFF_STEPS <<< "$ARM"
      TAG="ann${ANN_STEPS}_diff${DIFF_STEPS}_${VARIANT}"
      EXACT_CSV="$B19_BASE/B20_12A_${ID}_${TAG}_daps1S_meas${MEAS_SEED}_runseed${SEED}_exact_final_loss_selector.csv"
      GPU="${GPUS[$((idx % ${#GPUS[@]}))]}"
      LOG="$LOGDIR/B21_2_${ID}_meas${MEAS_SEED}_seed${SEED}_${TAG}_gpu${GPU}.log"
      SKIP=0
      if [[ -f "$EXACT_CSV" && "$B21_FORCE" != "1" ]]; then
        SKIP=1
        echo "[skip existing] image=$ID meas=$MEAS_SEED seed=$SEED arm=$TAG"
      else
        echo "[launch] image=$ID meas=$MEAS_SEED seed=$SEED arm=$TAG gpu=$GPU"
        (
          cd "$REPO"
          NUM_RUNS=1 DATA_PREFIX=b19 bash "$RUNNER" "$ID" "$GPU" "$MEAS_SEED" "$SEED" "$ANN_STEPS" "$DIFF_STEPS" "$VARIANT"
        ) > "$LOG" 2>&1 &
        launched=$((launched + 1))
      fi
      echo "$ID,$MEAS_SEED,$SEED,$VARIANT,$ANN_STEPS,$DIFF_STEPS,$GPU,$EXACT_CSV,$LOG,$SKIP" >> "$MANIFEST"
      idx=$((idx + 1))
      if [[ "$B21_SMOKE_ONLY" == "1" ]]; then
        wait
        echo "[smoke done] launched=$launched"
        python scripts/b21/collect_b21_2_candidate_recovery.py \
          --repo "$REPO" \
          --b19_base "$B19_BASE" \
          --outdir "$OUT" \
          --images "$ID" \
          --seeds "$SEED" \
          --meas_seed "$MEAS_SEED" \
          --arms "$ARM" \
          --report_path "$REPO/docs/b21/b21_2_candidate_recovery.md"
        exit 0
      fi
      if (( idx % ${#GPUS[@]} == 0 )); then
        wait
      fi
    done
  done
done
wait

echo "[done] launched=$launched"
python scripts/b21/collect_b21_2_candidate_recovery.py \
  --repo "$REPO" \
  --b19_base "$B19_BASE" \
  --outdir "$OUT" \
  --images "${IMAGES[*]}" \
  --seeds "${SEEDS[*]}" \
  --meas_seed "$MEAS_SEED" \
  --arms "${ARMS[*]}" \
  --report_path "$REPO/docs/b21/b21_2_candidate_recovery.md"

echo "B21.2 candidate recovery artifacts: $OUT"
echo "B21.2 candidate recovery report: $REPO/docs/b21/b21_2_candidate_recovery.md"
