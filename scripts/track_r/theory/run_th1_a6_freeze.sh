#!/usr/bin/env bash
# Controlled A6 FFHQ TH1 v1 freeze launcher.
# Only the three authorized K<=4 input files are permitted.
# Export PY, A8, A8B and TAU as absolute paths; optional WORK directory.
set -euo pipefail
: "$PY"
: "$A8"
: "$A8B"
: "$TAU"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
WORK="$(printenv WORK || true)"
if [[ -z "$WORK" ]]; then WORK="/tmp/track_r_a6_freeze"; fi
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
DRAFT="$WORK/TH1_A6_DRAFT.json"
OUTPUT="$ROOT/docs/track_r/theory/TH1_PREDICTIONS_v1.json"
if [[ -e "$OUTPUT" ]]; then
    echo "STOP_ALREADY_FROZEN: $OUTPUT" >&2
    exit 3
fi
"$PY" "$ROOT/scripts/track_r/theory/fit_th1_prefix_v1.py" \
    --a8 "$A8" --output "$DRAFT" \
    --point-reps 250 --bootstrap 60 --boot-reps 3
"$PY" "$ROOT/scripts/track_r/theory/check_a6_panel_contract.py" "$DRAFT"
"$PY" "$ROOT/scripts/track_r/theory/seal_th1_a6_v1.py" \
    --a8 "$A8" --a8b "$A8B" --tau "$TAU" \
    --draft "$DRAFT" --output "$OUTPUT" \
    --code-commit "$head"
sha256sum "$OUTPUT"
git -C "$ROOT" add -- docs/track_r/theory/TH1_PREDICTIONS_v1.json
git -C "$ROOT" commit -m "theory: freeze A6 TH1 v1 before XP K>4 unblinding"
echo "FREEZE_COMMIT=$(git -C "$ROOT" rev-parse HEAD)"
echo "FROZEN_FILE_SHA256=$(sha256sum "$OUTPUT" | cut -d' ' -f1)"
echo "FREEZE_TIME_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "NEXT: push the freeze commit and register SHA/UTC with Cheng"
echo "DO NOT UNBLIND XP RESULTS UNTIL PLANNER REGISTRATION"
