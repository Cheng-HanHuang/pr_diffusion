#!/usr/bin/env bash
# A17 handoff: publish XP-I chosen CPU metric return, then submit v5
# prefix scoring and prospective frozen-forecast registration in tmux.
# The existing XP-N generator remains intact and independent.
set -euo pipefail
[[ $# == 1 ]] || { echo "usage: bash $0 PINNED_HEAD" >&2; exit 8; }
HEAD="$1"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
TASKS="$BASE/xp_a18_xpn_tasks_20261031T224645Z"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$WT" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -d "$TASKS"
cd "$WT"

echo "=== A17 SOURCE TESTS ==="
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a18_planner_a16.py' -v
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a19_planner_a17.py' -v
bash -n scripts/track_r/xp_a19_xpn_v5_autopilot.sh \
  scripts/track_r/xp_a19_xpi_chosen_publish.sh \
  scripts/track_r/xp_a19_launch_cpu_and_v5.sh

echo "=== PRIORITY NOV4 A17 XP-I FACE CHOSEN METRIC TABLE ==="
bash scripts/track_r/xp_a19_xpi_chosen_publish.sh "$HEAD"

echo "=== LOCATE THE UNIQUE ACTIVE A18 XP-N GENERATION RUN ==="
GEN="$("$PY" - "$BASE" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
valid=[]
for f in root.glob("xp_a18_xpn_prod_*/XP_N_CONTROLLER_STATE.json"):
 doc=json.loads(f.read_text())
 if doc["status"] in ("WAIT_V5_CODE_FREEZE_OR_NOV2_CUTOFF",
                      "V5_PROSPECTIVE_GATE_PASS","A18_XPN_RUNNING",
                      "A18_XPN_ALL6400_NATIVE_GENERATION_COMPLETE_SEALED"):
  valid.append(f.parent)
assert len(valid)==1, f"expected exactly one XP-N controller; found {valid}"
print(valid[0])
PY
)"
echo "A19_XPN_GENERATOR_SOURCE=$GEN"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SESSION="prdiff_a19_xpn_v5_$STAMP"
LOG="$BASE/XP_A19_XPN_V5_AUTOPILOT_$STAMP.log"
test ! -e "$LOG"
echo "=== SUBMIT A17 PREFIX4 THEN PINNED V5 FORECAST AUTO-REGISTRATION ==="
tmux new-session -d -s "$SESSION" -c "$WT" \
 "bash '$WT/scripts/track_r/xp_a19_xpn_v5_autopilot.sh' '$HEAD' '$TASKS' '$GEN' > '$LOG' 2>&1"
echo "A19_V5_SESSION=$SESSION"
echo "A19_V5_LOG=$LOG"
echo "A19_V5_PREFIX_WILL_RUN_WHEN_ALL400_COMPLETE"
echo "A19_XPI_PUBLICATION_AND_V5_AUTOPILOT_SUBMITTED"
