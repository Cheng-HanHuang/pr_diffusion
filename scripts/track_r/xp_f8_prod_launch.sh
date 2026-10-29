#!/usr/bin/env bash
# A10 F8 920-root generation. STOP_F8 drains owned workers; never touch others.
# All GPU samples and candidate files remain scientifically embargoed.
set -euo pipefail
[[ $# -ge 2 ]] || { echo "usage: launch|execute HEAD [RUN]"; exit 8; }
MODE="$1"
HEAD="$2"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -f "$PY"
cd "$WT"
if [[ "$MODE" == launch ]]; then
  echo "===== STATIC SOURCE VALIDATION ====="
  "$PY" - <<'PY'
import ast
from pathlib import Path
for name in (
 "scripts/track_r/xp_f8_prod_prepare_cpu.py",
 "scripts/track_r/xp_f8_prod_worker_gpu.py",
 "scripts/track_r/xp_f8_prod_controller.py",
 "tests/test_track_r_xp_f8_prod.py"):
  ast.parse(Path(name).read_text(),filename=name)
  print("F8_PROD_SOURCE_SYNTAX_PASS",name)
PY
  "$PY" -m unittest discover -s tests -p 'test_track_r_xp_f8_prod.py' -v
  "$PY" -m unittest discover -s tests -p 'test_track_r_xp_f8_legacy_replay.py' -v
  "$PY" -m unittest discover -s tests -p 'test_track_r_xp_f8_legacy_preflight.py' -v
  echo "===== ORIGINAL NATIVE DAPS SOURCE + REPLAY GATE ====="
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN="$BASE/xp_f8_prod_$STAMP"
  LOG="$BASE/XP_F8_PROD_$STAMP.log"
  SESSION="prdiff_f8_prod_$STAMP"
  test ! -e "$RUN" && test ! -e "$LOG"
  "$PY" -m scripts.track_r.xp_f8_prod_prepare_cpu \
    --expected-head "$HEAD" --out "$RUN"
  echo "===== PRE-ADMISSION GPU CO-TENANT INVENTORY ====="
  nvidia-smi \
    --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
  tmux new-session -d -s "$SESSION" -c "$WT" \
    "env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$WT' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 bash '$WT/scripts/track_r/xp_f8_prod_launch.sh' execute '$HEAD' '$RUN' > '$LOG' 2>&1"
  echo "XP_F8_PROD_RUN=$RUN"
  echo "XP_F8_PROD_LOG=$LOG"
  echo "XP_F8_PROD_SESSION=$SESSION"
  echo "XP_F8_PROD_HEAD=$HEAD"
  echo "STOP_FILE=$RUN/STOP_F8"
  echo "F8_920_ATTEMPTS_STARTING_ON_DIRTY_GPUS=true"
  echo "NO_NEW_F8_OUTCOME_ANALYZED=true"
  exit 0
fi
if [[ "$MODE" == execute ]]; then
  [[ $# -eq 3 ]] || { echo "STOP_F8_EXECUTE_NO_RUN"; exit 8; }
  RUN="$3"
  echo "===== F8 FOUR-ROOT NATIVE GENERATION SMOKE ====="
  "$PY" -u -m scripts.track_r.xp_f8_prod_controller \
    --expected-head "$HEAD" --run "$RUN" \
    --max-completed-this-session 4
  "$PY" - "$RUN" <<'PY'
import json,sys,hashlib
from pathlib import Path
root=Path(sys.argv[1])
state=json.loads((root/"XP_F8_CONTROLLER_STATE.json").read_text())
assert state["status"]=="XP_F8_STAGED_SMOKE_COMPLETE",state["status"]
assert len(state["done"])==4,len(state["done"])
assert state["active"]==[]
for key,x in state["done"].items():
    p=root/"roots"/f"r{int(key):04d}_a{x['attempt']}"/"XP_F8_NEW_ROOT_COMPLETE.json"
    r=json.loads(p.read_text())
    assert hashlib.sha256(p.read_bytes()).hexdigest()==x["receipt_sha256"]
    assert r["status"]=="XP_F8_NEW_ROOT_GENERATION_PASS_EMBARGOED"
    assert r["no_GT_scoring"] and r["no_scientific_outcome_opened"] and r["no_labels_calculated"]
print("F8_FOUR_ROOT_GENERATION_ONLY_SMOKE_PASS",flush=True)
PY
  echo "===== CONTINUE F8 TO 920 OR A10 HARD BUDGET ====="
  "$PY" -u -m scripts.track_r.xp_f8_prod_controller \
    --expected-head "$HEAD" --run "$RUN" --resume \
    --max-completed-this-session 920
  "$PY" - "$RUN" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
last=json.loads((root/"XP_F8_PROD_LAST_STATUS.json").read_text())
print("XP_F8_PROD_FINAL_STATUS",json.dumps(last,sort_keys=True),flush=True)
if last["status"]!="XP_F8_920_GENERATION_COMPLETE_EMBARGOED":
    print("ACTION NEEDED: F8_BUDGET_OR_DEADLINE_PARTIAL",
          last["complete"],"/920",flush=True)
    sys.exit(8)
assert last["complete"]==920
assert last["scientific_outcomes_embargoed"] is True
print("XP_F8_920_GT_FREE_GENERATION_COMPLETE",flush=True)
PY
  exit 0
fi
echo "STOP_UNSUPPORTED_F8_LAUNCH_MODE=$MODE"
exit 8
