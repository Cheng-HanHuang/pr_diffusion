#!/usr/bin/env bash
set -euo pipefail

# Four-GPU hard-image sweep.  Each process uses best-of-4 through a different
# seed group.  This estimates whether refinement improves hard-image p_x.
# Usage:
#   bash scripts/phase_retrieval_20260608/run_np_refine_hard_ffhq_four_gpu.sh
# Optional:
#   GPUS="0 1 2 3" NOISES="0.05" REFINE_STEPS=150

REPO="${REPO:-/egr/research-pac/huang248/pr_diffusion_repo}"
OUT_ROOT="${OUT_ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260608}"
GPUS_STR="${GPUS:-0 1 2 3}"
NOISES="${NOISES:-0,0.01,0.05,0.08,0.10}"
REFINE_STEPS="${REFINE_STEPS:-100}"
IMAGE_LIST_FILE="${IMAGE_LIST_FILE:-$OUT_ROOT/splits/ffhq_hard9_from_available25.txt}"
MAX_IMAGES="${MAX_IMAGES:-9}"

cd "$REPO"
bash scripts/phase_retrieval_20260608/prepare_phase_retrieval_20260608.sh
mkdir -p "$OUT_ROOT/logs"

seed_groups=("100,101,102,103" "104,105,106,107" "108,109,110,111" "148,149,150,151")
read -r -a gpus <<< "$GPUS_STR"

for i in "${!seed_groups[@]}"; do
  gpu="${gpus[$((i % ${#gpus[@]}))]}"
  seeds="${seed_groups[$i]}"
  tag="hard9_seeds_${seeds//,/_}"
  log="$OUT_ROOT/logs/np_refine_${tag}.log"
  echo "[launch] gpu=$gpu seeds=$seeds log=$log"
  SEEDS="$seeds" NOISES="$NOISES" IMAGE_LIST_FILE="$IMAGE_LIST_FILE" MAX_IMAGES="$MAX_IMAGES" REFINE_STEPS="$REFINE_STEPS" \
    bash scripts/phase_retrieval_20260608/run_np_refine_ffhq_one_gpu.sh "$gpu" "$tag" \
    > "$log" 2>&1 &
done

wait
echo "[done] all hard-image refinement jobs finished."
python scripts/phase_retrieval_20260608/summarize_phase_retrieval_20260608.py --root "$OUT_ROOT"
