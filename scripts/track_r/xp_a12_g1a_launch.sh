#!/usr/bin/env bash
# A12 G1a: one command, CPU source+ledger gate -> historical 2-root replay
# -> pixel-exact check -> 4-GPU 1,600-root production -> receipts.
# Never opens/scores XP-D4 scientific outcomes.
set -euo pipefail
[[ $# -eq 4 && "$1" == launch ]] || {
  echo "usage: $0 launch HEAD PREP_DIR VERIFIED_LEDGER_JSON" >&2
  exit 8
}
HEAD="$2";PREP="$3";LEDGER="$4"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$WT" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -f "$PY"
cd "$WT"
echo "===== VALIDATE A12 G1A SOURCE + CONTRACTS ====="
"$PY" - <<'PY'
import ast
from pathlib import Path
for p in ("scripts/track_r/xp_a12_g1a_prepare_cpu.py",
          "scripts/track_r/xp_a12_g1a_fast_gpu.py",
          "scripts/track_r/xp_a12_g1a_autopilot.py",
          "tests/test_track_r_xp_a12_g1a.py",
          "tests/test_track_r_xp_a12_g1a_autopilot.py"):
 ast.parse(Path(p).read_text(),filename=p)
 print("A12_G1A_SOURCE_AST_PASS",p)
PY
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a12_g1a*.py' -v
echo "===== REQUIRE FULL SOURCE-BACKED PROJECT LEDGER ====="
"$PY" - "$LEDGER" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
assert p.is_file(), "STOP_NO_A12_FULL_PROJECT_LEDGER"
v=json.loads(p.read_text())
assert v["status"]=="XP_A12_CUMULATIVE_PROJECT_LEDGER_VERIFIED"
assert v["has_every_stage_since_XP0"] is True
assert len(v["stages"])>=8
print("A12_G1A_PROJECT_LEDGER_PRESENT",len(v["stages"]),"stages")
PY
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="$BASE/xp_a12_g1a_prod_$STAMP"
LOG="$BASE/XP_A12_G1A_AUTOPILOT_$STAMP.log"
SESSION="prdiff_a12_g1a_$STAMP"
test ! -e "$RUN" && test ! -e "$LOG"
echo "===== PRE-GPU INVENTORY ====="
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu \
   --format=csv,noheader,nounits
echo "===== AUTOPILOT IN DETACHED TMUX ====="
tmux new-session -d -s "$SESSION" -c "$WT" \
  "env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$WT' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 '$PY' -u -m scripts.track_r.xp_a12_g1a_autopilot --expected-head '$HEAD' --prep '$PREP' --project-ledger '$LEDGER' --run '$RUN' > '$LOG' 2>&1"
echo "A12_G1A_SESSION=$SESSION"
echo "A12_G1A_LOG=$LOG"
echo "A12_G1A_RUN=$RUN"
echo "A12_G1A_STOP_FILE=$RUN/STOP_G1A"
echo "A12_G1A_AUTOPILOT_REPLAY_THEN_1600_SUBMITTED"
