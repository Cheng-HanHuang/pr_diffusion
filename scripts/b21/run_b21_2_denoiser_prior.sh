#!/usr/bin/env bash
set -euo pipefail

# B21.2 selector-v2: denoiser-prior clean-free tie-breaker.
# GPU required. Uses the absolute FFHQ checkpoint discovered on PAC by default.

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_b19_solver}
B21_BASE=${B21_BASE:-/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver}
INPUT=${INPUT:-$B21_BASE/B21_2_candidate_recovery/candidate_recovery_rows.csv}
OUT=${OUT:-$B21_BASE/B21_2_denoiser_prior}
CKPT=${CKPT:-/egr/research-pac/huang248/models/ffhq_10m.pt}
GPU=${GPU:-1}
BATCH_SIZE=${BATCH_SIZE:-8}
SIGMAS=${SIGMAS:-0.05,0.10}
DRAWS=${DRAWS:-2}
NOISE_SEED=${NOISE_SEED:-91021}
ETAS=${ETAS:-0,0.005,0.01,0.02,0.05,0.10}

cd "$REPO"
mkdir -p "$OUT"

python scripts/b21/score_b21_2_denoiser_prior.py \
  --input_csv "$INPUT" \
  --outdir "$OUT" \
  --repo "$REPO" \
  --daps_root "$REPO/external/daps" \
  --checkpoint_path "$CKPT" \
  --gpu "$GPU" \
  --batch_size "$BATCH_SIZE" \
  --sigmas "$SIGMAS" \
  --draws "$DRAWS" \
  --noise_seed "$NOISE_SEED" \
  --etas "$ETAS" \
  --report_path "$REPO/docs/b21/b21_2_denoiser_prior.md"

echo "B21.2 denoiser-prior report: $REPO/docs/b21/b21_2_denoiser_prior.md"
echo "B21.2 denoiser-prior artifacts: $OUT"
