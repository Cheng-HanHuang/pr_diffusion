#!/usr/bin/env bash
set -euo pipefail
ROOT=/egr/research-pac/huang248/outputs/pr_diffusion/b24
LATEST="$ROOT/LATEST_CONTROL_RUN.txt"
if [ ! -f "$LATEST" ]; then
  echo "B24_STOP|NO_CONTROL_RUN"
  exit 0
fi
RUNROOT=$(cat "$LATEST")
STOP="$RUNROOT/STOP_REQUESTED.json"
TMP="$STOP.tmp.$$"
printf '{"requested_at_utc":"%s","mechanism":"cooperative_no_kill"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$TMP"
mv "$TMP" "$STOP"
echo "B24_STOP|REQUESTED|$STOP"
echo "No process was killed. Future scientific workers must check this file between atomic image units."
