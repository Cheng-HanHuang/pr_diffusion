#!/usr/bin/env bash
# A17.3 CPU-only: reproduce native/E0, add all720 face-chosen metrics,
# publish sanitized evaluator-only table (no local GT/native tensor paths).
set -euo pipefail
[[ $# == 1 ]] || { echo "usage: bash $0 PINNED_HEAD" >&2; exit 8; }
HEAD="$1"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$BASE/xp_a19_xpi_chosen_$STAMP"
PUB="$ROOT/pr_diffusion_track_r_xpi_chosen_pub_$STAMP"
BRANCH="codex/prdiff-track-r-xpi-chosen-$STAMP"
REL="docs/track_r/executors/xp/returns/XP_A19_XPI_CHOSEN_$STAMP"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$WT" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
cd "$WT"
"$PY" scripts/track_r/xp_a19_xpi_chosen_cpu.py \
  --expected-head "$HEAD" --out "$OUT" --preflight-only
"$PY" scripts/track_r/xp_a19_xpi_chosen_cpu.py \
  --expected-head "$HEAD" --out "$OUT"
test ! -e "$PUB"
git -C "$WT" worktree add -b "$BRANCH" "$PUB" "$HEAD"
mkdir -p "$PUB/$REL"
cp "$OUT/A17_XPI_PER_PARTICLE_CHOSEN_METRICS.csv" "$PUB/$REL/"
cp "$OUT/A17_XPI_CHOSEN_RECEIPT.json" "$PUB/$REL/"
cat > "$PUB/$REL/README.md" <<'EOF'
# A17.3 XP-I PC-DAPS per-particle metrics with frozen face decisions
720 rows / 180 images. Previous A16 raw columns unchanged.
PSNR_chosen, SSIM_chosen and VGG_LPIPS_chosen use 111-orientation
when frozen face_choice is 111; otherwise original raw-orientation.
GT only for evaluation; face choices frozen GT-free.
E0 rotated precisely 7 selected particles. No GPU or matched-NFE claim.
EOF
(cd "$PUB/$REL" && sha256sum A17_XPI_PER_PARTICLE_CHOSEN_METRICS.csv A17_XPI_CHOSEN_RECEIPT.json README.md > SHA256SUMS && sha256sum -c SHA256SUMS)
git -C "$PUB" add "$REL"
git -C "$PUB" -c core.whitespace=cr-at-eol diff --cached --check
git -C "$PUB" commit -m "results(track-r): A17 XP-I per-particle raw/chosen metrics all720 seven E0 rotations"
GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=10" \
  git -C "$PUB" push git@github.com:Cheng-HanHuang/pr_diffusion.git "HEAD:refs/heads/$BRANCH"
LOCAL=$(git -C "$PUB" rev-parse HEAD)
REMOTE=$(git -C "$PUB" ls-remote git@github.com:Cheng-HanHuang/pr_diffusion.git "refs/heads/$BRANCH" | cut -f1)
test -n "$REMOTE" && test "$LOCAL" = "$REMOTE"
echo "A19_XPI_CHOSEN_REMOTE_CHECKPOINT_PUBLISHED_AND_VERIFIED=$LOCAL"
echo "A19_XPI_CHOSEN_REMOTE_BRANCH=$BRANCH"
