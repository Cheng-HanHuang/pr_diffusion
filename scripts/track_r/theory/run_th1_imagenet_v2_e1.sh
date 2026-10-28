#!/usr/bin/env bash
# Optional registered v2 E1 addendum, using ONLY phase-1 D1 G1/G2 decisions.
# Required env: PY, IN, CLIP, CLIP_SHA256. Optional WORK.
# Runs ONLY on a clean theory branch. No XP return access or phase-2 data.
# Publication deadline: Oct 29 2026 12:01 America/Detroit.
set -euo pipefail
: "$PY"
: "$IN"
: "$CLIP"
: "$CLIP_SHA256"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
WORK="$(printenv WORK || true)"
if [[ -z "$WORK" ]]; then WORK="/tmp/track_r_v2_e1"; fi
mkdir -p "$WORK"
BRANCH="theory/prdiff-track-r-th-20261026"
if [[ "$(git -C "$ROOT" branch --show-current)" != "$BRANCH" ]]; then
  echo "STOP_NOT_THEORY_BRANCH" >&2; exit 3
fi
if [[ -n "$(git -C "$ROOT" status --porcelain)" ]]; then
  echo "STOP_DIRTY_WORKTREE" >&2; exit 3
fi
HEAD="$(git -C "$ROOT" rev-parse HEAD)"
BASE="$ROOT/docs/track_r/theory/TH1_PREDICTIONS_v2.json"
DRAFT="$WORK/TH1_V2_E1_DRAFT.json"
SEALED="$ROOT/docs/track_r/theory/TH1_PREDICTIONS_v2_E1.json"
if [[ -e "$SEALED" ]]; then
  echo "STOP_ADDENDUM_ALREADY_EXISTS" >&2; exit 3
fi
"$PY" "$ROOT/scripts/track_r/theory/fit_th1_imagenet_v2_e1.py" \
  --input "$IN" --clip "$CLIP" --clip-sha256 "$CLIP_SHA256" \
  --base-v2 "$BASE" --output "$DRAFT" \
  --point-reps 350 --bootstrap 60 --boot-reps 3
"$PY" "$ROOT/scripts/track_r/theory/check_th1_imagenet_v2_e1_contract.py" \
  --draft "$DRAFT" --base-v2 "$BASE"
"$PY" "$ROOT/scripts/track_r/theory/seal_th1_imagenet_v2_e1.py" \
  --input "$IN" --clip "$CLIP" --clip-sha256 "$CLIP_SHA256" \
  --base-v2 "$BASE" --draft "$DRAFT" --output "$SEALED" \
  --code-commit "$HEAD" --point-reps 350 --bootstrap 60 --boot-reps 3
git -C "$ROOT" add -- docs/track_r/theory/TH1_PREDICTIONS_v2_E1.json
git -C "$ROOT" commit -m "theory: freeze optional ImageNet D1 v2 E1 phase-1 addendum"
echo "E1_COMMIT=$(git -C "$ROOT" rev-parse HEAD)"
echo "E1_SHA256=$(sha256sum "$SEALED" | cut -d' ' -f1)"
echo "E1_COMMIT_TIME_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "NEXT: PUSH then register v2 E1 with planner; do not open D1 phase-2 beforehand"
