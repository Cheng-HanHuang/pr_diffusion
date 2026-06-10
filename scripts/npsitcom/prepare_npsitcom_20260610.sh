#!/usr/bin/env bash
set -euo pipefail

OUT=${OUT:-/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610}
OLD=${OLD:-/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411}
SITCOM_ROOT=${SITCOM_ROOT:-/egr/research-pac/huang248/external/SITCOM_ODE}
NP_REPO=${NP_REPO:-/egr/research-pac/huang248/pr_diffusion_repo}

mkdir -p "$OUT"/{splits,logs,branchA_mix,branchB_handoff,manifests}

if [[ -f "$OLD/splits/ffhq_available25.txt" ]]; then
  cp "$OLD/splits/ffhq_available25.txt" "$OUT/splits/ffhq_available25.txt"
fi
if [[ -f "$OLD/splits/imagenet_available25.txt" ]]; then
  cp "$OLD/splits/imagenet_available25.txt" "$OUT/splits/imagenet_available25.txt"
fi

cat > "$OUT/manifests/paths.env" <<EOF
OUT=$OUT
NP_REPO=$NP_REPO
SITCOM_ROOT=$SITCOM_ROOT
FFHQ_DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024
FFHQ_SPLIT=$OUT/splits/ffhq_available25.txt
GUIDED_MODEL=/egr/research-pac/huang248/models/ffhq_10m.pt
GUIDED_DIFFUSION_DIR=/egr/research-pac/huang248/external/DiffFPR
EOF

echo "Prepared $OUT"
echo "SITCOM_ROOT=$SITCOM_ROOT"
if [[ ! -d "$SITCOM_ROOT" ]]; then
  echo "WARNING: SITCOM_ROOT does not exist: $SITCOM_ROOT" >&2
fi
if [[ ! -f "$OUT/splits/ffhq_available25.txt" ]]; then
  echo "WARNING: missing FFHQ split: $OUT/splits/ffhq_available25.txt" >&2
fi
