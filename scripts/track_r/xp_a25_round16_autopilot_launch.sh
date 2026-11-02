#!/usr/bin/env bash
# Round 16: publish Nov4 frozen CPU results FIRST (commit/push), then
# prepare full XP-C3 source & code SHA freeze (commit/push), then let the
# 8h GPU controller start only after freeze and the shared XP-I2 lock.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: bash $0 PINNED_HEAD REMOTE_BRANCH" >&2;exit 8; }
HEAD="$1"; BRANCH="$2"
ROOT=/egr/research-pac/huang248
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$WT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
cd "$WT"
test "$(git rev-parse HEAD)" = "$HEAD"
test -z "$(git status --porcelain)"
test "$(git branch --show-current)" = ""
test "$(git rev-parse "origin/$BRANCH")" = "$HEAD"

"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a25_nov4_publish.py' -v
"$PY" scripts/track_r/xp_a25_nov4_publish_cpu.py \
   --expected-head "$HEAD" --preflight-only
"$PY" scripts/track_r/xp_a25_nov4_publish_cpu.py \
   --expected-head "$HEAD"

PUB="docs/track_r/executors/xp/returns/XP_A25_NOV4_20261102"
THEORY="docs/track_r/executors/xp/theory_inputs/XP_A25_FAST_FACE_CLIP_11520.csv"
test -f "$PUB/XP_A25_NOV4_PUBLICATION_RECEIPT.json"
test -f "$THEORY"
test "$(find "$PUB/png_256" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 15
git add -- "$PUB" "$THEORY"
test "$(git diff --cached --name-only | wc -l)" -eq 20
git commit -m "results(track-r): A20 publish 15 256px panels and 11520 face E1/v1 coverage"
PUBLISHED="$(git rev-parse HEAD)"
git push origin "HEAD:refs/heads/$BRANCH"
test -z "$(git status --porcelain)"
git fetch --no-tags origin "$BRANCH"
test "$(git rev-parse FETCH_HEAD)" = "$PUBLISHED"
echo "A20_NOV4_PUBLISHED_COMMIT=$PUBLISHED"

# The next sublaunch executes 13,600 source-SHA checks with no GT scoring,
# commits the exact manifest + scorer SHA, pushes both, then launches GPU.
bash scripts/track_r/xp_a25_xpc3_freeze_and_launch.sh "$PUBLISHED" "$BRANCH"
