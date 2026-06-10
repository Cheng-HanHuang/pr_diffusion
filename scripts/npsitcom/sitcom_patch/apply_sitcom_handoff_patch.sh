#!/usr/bin/env bash
set -euo pipefail

SITCOM_ROOT=${SITCOM_ROOT:-/egr/research-pac/huang248/external/SITCOM_ODE}
PATCHED_ROOT=${PATCHED_ROOT:-/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610/sitcom_ode_handoff_patch}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ! -d "$SITCOM_ROOT" ]]; then
  echo "Missing SITCOM_ROOT=$SITCOM_ROOT" >&2
  exit 2
fi

mkdir -p "$(dirname "$PATCHED_ROOT")"
rsync -a --delete --exclude .git "$SITCOM_ROOT"/ "$PATCHED_ROOT"/
cp "$SCRIPT_DIR/npsitcom_handoff_sample.py" "$PATCHED_ROOT/npsitcom_handoff_sample.py"
chmod +x "$PATCHED_ROOT/npsitcom_handoff_sample.py"

echo "Patched SITCOM copy: $PATCHED_ROOT"
echo "Runner: $PATCHED_ROOT/npsitcom_handoff_sample.py"
