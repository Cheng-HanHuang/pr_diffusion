#!/usr/bin/env bash
# XP-N v5 controlled production: no held 4..63 outcomes opened.
# Define PY, PREFIX4, PREFIX4_SHA256, CLIP4, CLIP4_SHA256, optional WORK.
set -euo pipefail
: "${PY:?}" "${PREFIX4:?}" "${PREFIX4_SHA256:?}" "${CLIP4:?}" "${CLIP4_SHA256:?}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
[[ "$(git -C "$ROOT" branch --show-current)" == 'theory/prdiff-track-r-th-20261026' ]] || { echo 'STOP_WRONG_BRANCH'; exit 3; }
[[ -z "$(git -C "$ROOT" status --porcelain)" ]] || { echo 'STOP_DIRTY_WORKTREE'; exit 3; }
WORK="${WORK:-/tmp/prdiff_xpn_v5}"
mkdir -p "$WORK"
DRAFT="$WORK/XP_N_V5_DRAFT.json"
FREEZE="$ROOT/docs/track_r/theory/TH1_PREDICTIONS_v5_XP_N.json"
[[ ! -e "$DRAFT" && ! -e "$FREEZE" ]] || { echo 'STOP_EXISTING_DRAFT_OR_FREEZE'; exit 3; }
CODE_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
"$PY" "$ROOT/scripts/track_r/theory/fit_th1_imagenet_v5.py" \
  --prefix4 "$PREFIX4" --prefix4-sha256 "$PREFIX4_SHA256" \
  --clip4 "$CLIP4" --clip4-sha256 "$CLIP4_SHA256" \
  --output "$DRAFT" --point-reps 250 --bootstrap 60 --boot-reps 3
"$PY" "$ROOT/scripts/track_r/theory/check_th1_imagenet_v5_contract.py" "$DRAFT"
"$PY" "$ROOT/scripts/track_r/theory/seal_th1_imagenet_v5.py" \
  --draft "$DRAFT" --output "$FREEZE" --code-commit "$CODE_COMMIT" \
  --prefix4 "$PREFIX4" --prefix4-sha256 "$PREFIX4_SHA256" \
  --clip4 "$CLIP4" --clip4-sha256 "$CLIP4_SHA256"
git -C "$ROOT" add -- docs/track_r/theory/TH1_PREDICTIONS_v5_XP_N.json
git -C "$ROOT" commit -m 'theory: freeze prospective image-disjoint XP-N v5'
echo "XP_N_V5_COMMIT=$(git -C "$ROOT" rev-parse HEAD)"
echo "XP_N_V5_SHA256=$(sha256sum "$FREEZE" | cut -d' ' -f1)"
echo 'NEXT: push/register commit+SHA+UTC BEFORE opening any XP-N run 4..63 outcomes'
