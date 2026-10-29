#!/usr/bin/env bash
# B16 primary-v1 secondary-v3 prospective CPU analysis + frozen N2 face CLIP.
# No GPU, no modifications to live F8 checkout.
set -euo pipefail
[[ $# -ge 3 && $# -le 4 ]] || { echo "usage: launch|execute HEAD ADMISSION_DIR [RUN]"; exit 8; }
MODE="$1";HEAD="$2";ADMISSION="$3"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -f "$ADMISSION/B16_V1_V3_CPU_ADMISSION.json"
cd "$WT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="$BASE/xp_b16_v1_v3_pipeline_$STAMP"
SCORE="$BASE/xp_b16_v1_v3_score_$STAMP"
CLIP="$BASE/xp_b16_faceclip_$STAMP"
COMP="$BASE/xp_b16_v1_v3_compare_$STAMP"
LOG="$BASE/XP_B16_V1_V3_PIPELINE_$STAMP.log"
if [[ "$MODE" == launch ]]; then
  echo "===== CPU SOURCE AST + UNIT TEST ====="
  "$PY" - <<'PY'
import ast
from pathlib import Path
for p in (
 "scripts/track_r/xp_b16_v1_v3_admission_cpu.py",
 "scripts/track_r/xp_b16_v1_v3_score_cpu.py",
 "scripts/track_r/xp_b16_v1_v3_faceclip_cpu.py",
 "scripts/track_r/xp_b16_v1_v3_compare_cpu.py",
 "tests/test_track_r_xp_b16_v1_v3_admission.py"):
 ast.parse(Path(p).read_text(),filename=p)
 print("B16_AST_PASS",p)
PY
  "$PY" -m unittest discover -s tests -p 'test_track_r_xp_b16_v1_v3_admission.py' -v
  test ! -e "$RUN" && test ! -e "$SCORE" && test ! -e "$CLIP" && test ! -e "$COMP"
  mkdir "$RUN"
  "$PY" - "$RUN" "$SCORE" "$CLIP" "$COMP" "$HEAD" "$ADMISSION" <<'PY'
import json,sys
from pathlib import Path
run,score,clip,comp=map(str,sys.argv[1:5])
with (Path(run)/"XP_B16_PIPELINE_INPUT.json").open("x") as f:
 json.dump(dict(run=run,score=score,clip=clip,comp=comp,
                head=sys.argv[5],admission=sys.argv[6],
                gpu_requested=False),f,indent=2,sort_keys=True)
PY
  SESSION="prdiff_b16_v1_v3_$STAMP"
  tmux new-session -d -s "$SESSION" -c "$WT" \
    "env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$WT' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 bash '$WT/scripts/track_r/xp_b16_v1_v3_launch.sh' execute '$HEAD' '$ADMISSION' '$RUN' > '$LOG' 2>&1"
  echo "B16_PIPELINE_SESSION=$SESSION"
  echo "B16_PIPELINE_RUN=$RUN"
  echo "B16_PIPELINE_LOG=$LOG"
  echo "B16_CPU_ONLY=true"
  exit 0
fi
if [[ "$MODE" == execute ]]; then
  [[ $# -eq 4 ]] || { echo "STOP_B16_EXECUTE_RUN"; exit 8; }
  RUN="$4"
  "$PY" - "$RUN" "$HEAD" "$ADMISSION" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/"XP_B16_PIPELINE_INPUT.json"
v=json.loads(p.read_text())
assert v["run"]==sys.argv[1]
assert v["head"]==sys.argv[2]
assert v["admission"]==sys.argv[3]
print("B16_PINNED_PIPELINE_INPUT_PASS",flush=True)
PY
  SCORE="$("$PY" - "$RUN" <<'PY'
import json,sys
from pathlib import Path
d=json.loads((Path(sys.argv[1])/"XP_B16_PIPELINE_INPUT.json").read_text())
print(d["score"])
PY
)"
  CLIP="$("$PY" - "$RUN" <<'PY'
import json,sys
from pathlib import Path
d=json.loads((Path(sys.argv[1])/"XP_B16_PIPELINE_INPUT.json").read_text())
print(d["clip"])
PY
)"
  COMP="$("$PY" - "$RUN" <<'PY'
import json,sys
from pathlib import Path
d=json.loads((Path(sys.argv[1])/"XP_B16_PIPELINE_INPUT.json").read_text())
print(d["comp"])
PY
)"
  echo "===== B16 2880 CPU-SCORED ROOTS ====="
  "$PY" -u -m scripts.track_r.xp_b16_v1_v3_score_cpu      --expected-head "$HEAD" --admission "$ADMISSION" --out "$SCORE"
  echo "===== B16 N2 FACE CLIP CPU / 2880 ====="
  "$PY" -u -m scripts.track_r.xp_b16_v1_v3_faceclip_cpu      --expected-head "$HEAD" --score-dir "$SCORE" --out "$CLIP" --batch 8
  echo "===== B16 V1/V3 PRIMARY PROSPECTIVE COMPARISON ====="
  "$PY" -u -m scripts.track_r.xp_b16_v1_v3_compare_cpu      --expected-head "$HEAD" --score-dir "$SCORE" --clip-dir "$CLIP" --out "$COMP"
  echo "===== B16 CPU SCIENTIFIC OUTPUT DIGEST ====="
  "$PY" - "$COMP" <<'PY'
import json,sys,hashlib
from pathlib import Path
p=Path(sys.argv[1])
j=json.loads((p/"XP_B16_V1_V3_PROSPECTIVE_COMPARISON.json").read_text())
assert j["status"]=="XP_B16_V1_PRIMARY_V3_SECONDARY_PROSPECTIVE_COMPARISON_PASS"
assert j["n_forecast_rows"]==288
print("B16_V1_PRIMARY_V3_SECONDARY_SCIENTIFIC_PASS",
      "forecasts",j["n_forecast_rows"],
      "never_GOOD",j["n_never_GOOD"],
      "K16_E0_loss_cases",j["n_K16_raw_selection_loss"],flush=True)
print("B16_COMPARISON_SHA256",
 hashlib.sha256((p/"XP_B16_V1_V3_PROSPECTIVE_COMPARISON.json").read_bytes()).hexdigest(),flush=True)
print("B16_SCIENCE_OUTPUT="+str(p),flush=True)
print("B16_READY_TO_COMMIT_TO_XP_RETURNS",flush=True)
PY
  exit 0
fi
echo "STOP_UNKNOWN_MODE" >&2
exit 8
