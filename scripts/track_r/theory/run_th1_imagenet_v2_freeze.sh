#!/usr/bin/env bash
# Controlled ImageNet FAST TH1 v2 freeze launcher (prereg A6.6, A7.4).
# Mirrors run_th1_a6_freeze.sh. Only the authorized phase-1 theory input
# (XP_D1_P1_K4_A8_STYLE_PER_IMAGE.csv) is read.
# Export PY and IN (absolute paths); optional WORK directory.
set -euo pipefail
: "$PY"
: "$IN"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
WORK="$(printenv WORK || true)"
if [[ -z "$WORK" ]]; then WORK="/tmp/track_r_imagenet_v2_freeze"; fi
mkdir -p "$WORK"
BRANCH="theory/prdiff-track-r-th-20261026"
head="$(git -C "$ROOT" rev-parse HEAD)"
actual="$(git -C "$ROOT" branch --show-current)"
if [[ "$actual" != "$BRANCH" ]]; then
    echo "STOP_NOT_THEORY_BRANCH: $actual" >&2
    exit 3
fi
if [[ -n "$(git -C "$ROOT" status --porcelain)" ]]; then
    echo "STOP_DIRTY_WORKTREE" >&2
    exit 3
fi
DRAFT="$WORK/TH1_IMAGENET_V2_DRAFT.json"
OUTPUT="$ROOT/docs/track_r/theory/TH1_PREDICTIONS_v2.json"
if [[ -e "$OUTPUT" ]]; then
    echo "STOP_ALREADY_FROZEN: $OUTPUT" >&2
    exit 3
fi
"$PY" "$ROOT/scripts/track_r/theory/fit_th1_imagenet_v2.py" \
    --input "$IN" --output "$DRAFT" \
    --point-reps 350 --bootstrap 60 --boot-reps 3
"$PY" "$ROOT/scripts/track_r/theory/check_th1_imagenet_v2_contract.py" "$DRAFT"
"$PY" "$ROOT/scripts/track_r/theory/seal_th1_imagenet_v2.py" \
    --input "$IN" --draft "$DRAFT" --output "$OUTPUT" \
    --code-commit "$head" \
    --point-reps 350 --bootstrap 60 --boot-reps 3
sha256sum "$OUTPUT"
git -C "$ROOT" add -- docs/track_r/theory/TH1_PREDICTIONS_v2.json
git -C "$ROOT" commit -m "theory: freeze ImageNet FAST TH1 v2 before phase-2 unblinding"
echo "FREEZE_COMMIT=$(git -C "$ROOT" rev-parse HEAD)"
echo "FROZEN_FILE_SHA256=$(sha256sum "$OUTPUT" | cut -d' ' -f1)"
echo "FREEZE_TIME_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "NEXT: push the freeze commit and register SHA/UTC"
