#!/usr/bin/env bash
# A12 optional ImageNet v4 freeze. Only released pilot runs 0--15.
# Export: PY, PREFIX4, PILOT, PILOT_SHA256, CLIP, CLIP_SHA256.
# Optional WORK path; must be empty/new. Optional explicit column args
# must be passed to fit manually if the executor's actual schema differs.
set -euo pipefail
: "$PY"; : "$PREFIX4"; : "$PILOT"; : "$PILOT_SHA256"; : "$CLIP"; : "$CLIP_SHA256"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
WORK="$(printenv WORK || true)"
if [[ -z "$WORK" ]]; then WORK="/tmp/prdiff_theory_v4_a12"; fi
mkdir -p "$WORK"
if [[ "$(git -C "$ROOT" branch --show-current)" != "theory/prdiff-track-r-th-20261026" ]]; then
 echo STOP_NOT_THEORY_BRANCH >&2; exit 3
fi
if [[ -n "$(git -C "$ROOT" status --porcelain)" ]]; then
 echo STOP_DIRTY_TREE >&2; exit 3
fi
HEAD="$(git -C "$ROOT" rev-parse HEAD)"
DRAFT="$WORK/TH1_V4_DRAFT.json"
OUT="$ROOT/docs/track_r/theory/TH1_PREDICTIONS_v4.json"
if [[ -e "$DRAFT" || -e "$OUT" ]]; then echo STOP_EXISTING_DRAFT_OR_FREEZE >&2; exit 3; fi
"$PY" "$ROOT/scripts/track_r/theory/fit_th1_imagenet_v4.py" \
 --prefix4 "$PREFIX4" --pilot "$PILOT" --pilot-sha256 "$PILOT_SHA256" \
 --clip "$CLIP" --clip-sha256 "$CLIP_SHA256" \
 --output "$DRAFT" --point-reps 250 --bootstrap 60 --boot-reps 3
"$PY" "$ROOT/scripts/track_r/theory/check_th1_imagenet_v4_contract.py" "$DRAFT"
"$PY" "$ROOT/scripts/track_r/theory/seal_th1_imagenet_v4.py" \
 --prefix4 "$PREFIX4" --pilot "$PILOT" --pilot-sha256 "$PILOT_SHA256" \
 --clip "$CLIP" --clip-sha256 "$CLIP_SHA256" \
 --draft "$DRAFT" --output "$OUT" --code-commit "$HEAD"
git -C "$ROOT" add -- docs/track_r/theory/TH1_PREDICTIONS_v4.json
git -C "$ROOT" commit -m 'theory: freeze separately registered ImageNet D4 v4 forecast'
echo "V4_COMMIT=$(git -C "$ROOT" rev-parse HEAD)"
echo "V4_SHA256=$(sha256sum "$OUT" | cut -d' ' -f1)"
echo "V4_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo 'NEXT: push and planner-register BEFORE any D4 outcome unblinding'
