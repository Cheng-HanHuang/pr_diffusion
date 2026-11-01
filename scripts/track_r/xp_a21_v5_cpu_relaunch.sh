#!/usr/bin/env bash
# A21 CPU-only, A17-compliant v5 repair after A20 resumed chunk acceptance.
# This launches a separate detached code worktree. Never modifies the
# running A20 recovery worktree, GPU jobs, original chunk, or forecasts.
# No held run (>=4) scoring; only the fully finished 400-run PREFIX4.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Usage: bash $0 PINNED_A21_HEAD" >&2; exit 8; }
HEAD="$1"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
TASKS="$BASE/xp_a18_xpn_tasks_20261031T224645Z"
GEN="$BASE/xp_a18_xpn_prod_20261031T225748Z"
PHASE=A18_XPN_PILOT_0_3
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$WT" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -f "$TASKS/XP_N_TASKS_PREP_RECEIPT.json"
test -d "$GEN"
echo "=== A21 REFUSE DUPLICATE V5 PUBLICATION ==="
if find "$BASE" -maxdepth 2 -name XP_N_V5_REMOTE_REGISTERED.json -print -quit | grep -q .; then
  echo "STOP_A21_V5_ALREADY_REGISTERED";exit 8
fi
echo "=== VERIFY ALL 400 PREFIX ROOT RECEIPTS (NO HELD RUN READS) ==="
"$PY" - "$GEN" "$TASKS" <<'PY'
import json,sys
from pathlib import Path
gen,tasks=map(Path,sys.argv[1:])
phase="A18_XPN_PILOT_0_3"
p=tasks/"XP_N_PILOT_0_3_TASKS.json"
manifest=json.loads(p.read_text())
assert len(manifest["rows"])==400
reused=0
for st in range(0,400,20):
 d=gen/"production"/phase/f"chunk_{st:05d}_020"
 m=json.loads((d/"R0_5_CHUNK_META.json").read_text())
 c=json.loads((d/"R0_5_CHUNK_COMPLETE.json").read_text())
 assert c["status"]=="R0_5_CHUNK_PASS" and c["count"]==20 and c["start"]==st
 assert c["new"]+c["reused"]==20 and m["uids"]==[
  r["uid"] for r in manifest["rows"][st:st+20]]
 reused+=c["reused"]
print("A21_PREFIX_400_CHUNKS_ACCEPT", "RESUMED_ROOTS",reused)
PY
echo "=== A21 RUN A18/A19 SOURCE REGRESSION TESTS ==="
"$PY" -m unittest discover -s "$WT/tests" -p 'test_track_r_xp_a19_planner_a17.py' -v
bash -n "$WT/scripts/track_r/xp_a19_xpn_v5_autopilot.sh"
echo "=== A21 STRICT CPU PREFLIGHT OF RESUMED PREFIX ==="
PREF="$BASE/xp_a21_prefix4_unused_preflight"
"$PY" "$WT/scripts/track_r/xp_a19_xpn_prefix4_cpu.py" \
  --expected-head "$HEAD" --tasks "$TASKS" --run "$GEN" \
  --out "$PREF" --preflight-only
echo "=== A21 SUBMIT ONLY V5 CPU WORKER IN TMUX ==="
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SESSION="prdiff_a21_xpn_v5_$STAMP"
LOG="$BASE/XP_A21_XPN_V5_$STAMP.log"
test ! -e "$LOG"
tmux new-session -d -s "$SESSION" -c "$WT" \
 "bash '$WT/scripts/track_r/xp_a19_xpn_v5_autopilot.sh' '$HEAD' '$TASKS' '$GEN' > '$LOG' 2>&1"
echo "A21_XPN_V5_TMUX=$SESSION"
echo "A21_XPN_V5_LOG=$LOG"
echo "A21_XPN_V5_CPU_ONLY_SUBMITTED_NO_GPU_INTERRUPTION"
