#!/usr/bin/env bash
# A10 held B16/F8 v3, safe controlled production launcher.
# Export PY, A8, A8B, FAST64 as absolute AUTHORIZED input paths.
# Optional WORK points to a writable scratch directory.
# The driver reads no target solver post-prefix runs and no xp/returns.
set -euo pipefail
: "$PY"
: "$A8"
: "$A8B"
: "$FAST64"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
WORK="$(printenv WORK || true)"
if [[ -z "$WORK" ]]; then WORK="/tmp/track_r_v3_b16_f8"; fi
mkdir -p "$WORK"
if [[ "$(git -C "$ROOT" branch --show-current)" != "theory/prdiff-track-r-th-20261026" ]]; then
 echo "STOP_WRONG_BRANCH" >&2; exit 3
fi
if [[ -n "$(git -C "$ROOT" status --porcelain)" ]]; then
 echo "STOP_DIRTY_TREE" >&2; exit 3
fi
HEAD="$(git -C "$ROOT" rev-parse HEAD)"
DRAFT="$WORK/TH1_V3_DRAFT.json"
OUTPUT="$ROOT/docs/track_r/theory/TH1_PREDICTIONS_v3.json"
if [[ -e "$DRAFT" || -e "$OUTPUT" ]]; then
 echo "STOP_EXISTING_DRAFT_OR_FREEZE; choose fresh WORK" >&2; exit 3
fi
"$PY" "$ROOT/scripts/track_r/theory/fit_th1_fast_difficulty_v3.py" \
 --a8 "$A8" --a8b "$A8B" --fast64 "$FAST64" --output "$DRAFT" \
 --point-reps 250 --bootstrap 60 --boot-reps 3
"$PY" "$ROOT/scripts/track_r/theory/check_th1_fast_difficulty_v3.py" "$DRAFT"
"$PY" "$ROOT/scripts/track_r/theory/seal_th1_fast_difficulty_v3.py" \
 --a8 "$A8" --a8b "$A8B" --fast64 "$FAST64" \
 --draft "$DRAFT" --output "$OUTPUT" --code-commit "$HEAD" \
 --point-reps 250 --bootstrap 60 --boot-reps 3
git -C "$ROOT" add -- docs/track_r/theory/TH1_PREDICTIONS_v3.json
git -C "$ROOT" commit -m "theory: freeze separately registered FAST-informed B16/F8 v3"
echo "TH1_V3_COMMIT=$(git -C "$ROOT" rev-parse HEAD)"
echo "TH1_V3_SHA256=$(sha256sum "$OUTPUT" | cut -d' ' -f1)"
echo "TH1_V3_FREEZE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "NEXT: push + planner registration before any held B16/F8 outcome is opened"
