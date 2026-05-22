#!/usr/bin/env bash
set -euo pipefail

# Four-GPU launcher for the near-term FFHQ NP experiments:
#   0. A1 scheduled S2, lambda0=0.005, linear decay to proj_start
#   1. A1 scheduled S2, lambda0=0.01,  linear decay to proj_start
#   2. A1 S2, lambda0=0.01, pre-projection-only
#   3. A3 memory bank, memory_k=1, original LF score
#
# Usage from repo root:
#   bash scripts/launch_ffhq_nearterm4_4gpu.sh
#
# Optional overrides:
#   OUTROOT=... SEEDS=100,101 MAX_IMAGES=25 bash scripts/launch_ffhq_nearterm4_4gpu.sh

REPO=${REPO:-/egr/research-pac/huang248/pr_diffusion_repo}
ROOT=${ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411}

DATA_ROOT=${DATA_ROOT:-/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024}
if [[ ! -d "$DATA_ROOT" ]]; then
  DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset
fi

SPLIT=${SPLIT:-$ROOT/splits/ffhq_available25.txt}
GUIDED_MODEL=${GUIDED_MODEL:-/egr/research-pac/huang248/models/ffhq_10m.pt}
GUIDED_DIFFUSION_DIR=${GUIDED_DIFFUSION_DIR:-/egr/research-pac/huang248/external/DiffFPR}
OUTROOT=${OUTROOT:-$ROOT/np_ffhq_nearterm_4gpu}

SEEDS=${SEEDS:-100,101}
MAX_IMAGES=${MAX_IMAGES:-25}
SIGMA=${SIGMA:-0.05}
NP_STEPS=${NP_STEPS:-1000}
LATE_START=${LATE_START:-300}
SOFT=${SOFT:-5}
HARD=${HARD:-1}
SCORE_RADIUS=${SCORE_RADIUS:-0.6}
PROJ_RADIUS=${PROJ_RADIUS:-0.2}
OVERSAMPLE=${OVERSAMPLE:-2}
ALIGNMENTS=${ALIGNMENTS:-raw,rot180,resolve}
SKIP_LPIPS=${SKIP_LPIPS:-0}
FAST_EVAL=${FAST_EVAL:-0}

cd "$REPO"
mkdir -p "$OUTROOT/logs"

COMMON_ARGS=(
  --data_root "$DATA_ROOT"
  --image_list_file "$SPLIT"
  --guided_model_path "$GUIDED_MODEL"
  --guided_diffusion_dir "$GUIDED_DIFFUSION_DIR"
  --variants np_canonical
  --seeds "$SEEDS"
  --np_steps "$NP_STEPS"
  --late_start "$LATE_START"
  --soft_candidates "$SOFT"
  --hard_candidates "$HARD"
  --score_radius "$SCORE_RADIUS"
  --proj_radius "$PROJ_RADIUS"
  --oversample_values "$OVERSAMPLE"
  --measurement_noise_values "$SIGMA"
  --clip_noisy_magnitude
  --alignments "$ALIGNMENTS"
  --max_images "$MAX_IMAGES"
  --log_every 100
)

if [[ "$SKIP_LPIPS" == "1" ]]; then
  COMMON_ARGS+=(--skip_lpips)
fi
if [[ "$FAST_EVAL" == "1" ]]; then
  COMMON_ARGS+=(--fast_eval)
fi

launch_one() {
  local gpu="$1"
  local tag="$2"
  shift 2
  local log="$OUTROOT/logs/${tag}.log"
  echo "[launch] GPU=$gpu tag=$tag log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python scripts/pr_external_difffpr_np_guided_nearterm4.py \
    "${COMMON_ARGS[@]}" \
    --outdir "$OUTROOT/$tag" \
    "$@" \
    > "$log" 2>&1 &
  echo "  pid=$!"
}

launch_one 0 a1_decay_s2_lam0005 \
  --score_mode prev_l2 \
  --score_reg_lambda 0.005 \
  --score_reg_lambda_schedule linear_decay_to_proj_start \
  --noise_memory_k 0

launch_one 1 a1_decay_s2_lam001 \
  --score_mode prev_l2 \
  --score_reg_lambda 0.01 \
  --score_reg_lambda_schedule linear_decay_to_proj_start \
  --noise_memory_k 0

launch_one 2 a1_preproj_s2_lam001 \
  --score_mode prev_l2 \
  --score_reg_lambda 0.01 \
  --score_reg_lambda_schedule pre_projection_only \
  --noise_memory_k 0

launch_one 3 a3_memory_k1_lf \
  --score_mode lf \
  --score_reg_lambda 0.0 \
  --score_reg_lambda_schedule constant \
  --noise_memory_k 1

echo
cat <<EOF
Launched four near-term jobs.

Output root:
  $OUTROOT

Monitor:
  tail -f $OUTROOT/logs/*.log
  watch -n 5 nvidia-smi

Analyze after completion:
  python scripts/analyze_ffhq_nearterm4_nopandas.py --root $OUTROOT
EOF
