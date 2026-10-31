#!/usr/bin/env bash
# One-shot A16 XP-I CPU publication, then wait-gated XP-N GPU autopilot.
# Other A16 CPU/GPU lanes remain separate and are NOT silently launched.
set -euo pipefail
[[ $# -eq 1 ]] || {
  echo "usage: bash $0 EXPECTED_HEAD" >&2
  exit 8
}
HEAD="$1"
ROOT=/egr/research-pac/huang248
WT="$(cd "$(dirname "$0")/../.." && pwd)"
TASKS="$ROOT/outputs/pr_diffusion/track_r/xp_a18_xpn_tasks_20261031T224645Z"
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -f "$TASKS/XP_N_TASKS_PREP_RECEIPT.json"
cd "$WT"
echo "=== A16 PRIORITY: PUBLISH ACCEPTED 720-ROW XP-I METRIC TABLE ==="
bash scripts/track_r/xp_a18_xpi_publish.sh publish "$HEAD"
echo "=== A16 XP-N: SCHEDULE ALL 6400 NATIVE ROOTS WITH V5 CODE-FREEZE GATE ==="
bash scripts/track_r/xp_a18_xpn_launch.sh launch "$HEAD" "$TASKS"
echo "A18_XPI_RETURN_PUBLISHED_XPN_AUTOPILOT_SUBMITTED"
