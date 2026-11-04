#!/usr/bin/env bash
# One PAC invocation for A22: CPU Tab3, frozen prompts, available native
# G7 diagnostics, recorded XP-K/G7 remaining/CP4 adapter gates.
set -euo pipefail
test "$#" -eq 1 || { echo "usage: bash $0 <EXACT_A22_COMMIT_SHA>" >&2;exit 8; }
HEAD="$1"
ROOT=/egr/research-pac/huang248
WT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE="$ROOT/outputs/pr_diffusion/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
DAPS="$ROOT/conda-envs/daps/bin/python"
VENDOR="$BASE/.vendor/piq_0p8"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$VENDOR:$WT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
cd "$WT"

echo "=== Verify pinned A22 execution; no new science after launch ==="
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$HEAD"
test -n "$HEAD"
test -f "$WT/docs/track_r/executors/xp/theory_inputs/XP_A22_FROZEN_N2_FACE_PROMPTS.json"
test -f "$VENDOR/piq/__init__.py"
if pgrep -u "$(id -un)" -af '[x]p_a30_a22_autopilot.py'; then
  echo "STOP_DUPLICATE_A22_AUTOPILOT"
  exit 8
fi
for PREVIOUS in "$BASE"/xp_a30_a22_autopilot_*/A22_AUTOPILOT_STATE.json; do
  if [ -f "$PREVIOUS" ]; then
    echo "STOP_A22_PRIOR_ATTEMPT_EXISTS=$PREVIOUS"
    echo "Reconcile that state before a new preregistered continuation."
    exit 8
  fi
done

echo "=== Syntax: all A22 scripts, no side effects ==="
"$PY" - <<'PY'
import ast
from pathlib import Path
names=[
 "xp_a30_a22_tab3_e1_cpu.py",
 "xp_a30_a22_tab3_publish_cpu.py",
 "xp_a30_a22_g7_prepare_cpu.py",
 "xp_a30_a22_g7_c2_autopilot.py",
 "xp_a30_a22_g7_c2_report_cpu.py",
 "xp_a30_a22_autopilot.py"]
for name in names:
 p=Path("scripts/track_r")/name
 ast.parse(p.read_text(),filename=str(p))
ast.parse(Path("scripts/track_r/xp_a12_g7_c2_gpu.py").read_text())
for name in ["test_track_r_xp_a30_a22_tab3.py",
             "test_track_r_xp_a30_a22_autopilot.py"]:
 p=Path("tests")/name
 ast.parse(p.read_text(),filename=str(p))
print("A22_PYTHON_AST_8_PASS",flush=True)
PY

echo "=== Genuine CPU unit tests ==="
"$PY" -m unittest discover -s tests \
  -p 'test_track_r_xp_a30_a22_*.py' -v

echo "=== Native library import and metric provenance ==="
"$PY" - <<'PY'
import importlib.metadata
import torch,torchvision,piq
from scripts.track_r.xp_a30_a22_tab3_e1_cpu import g1_select
assert importlib.metadata.version("piq")=="0.8.0"
assert hasattr(piq,"ssim") and hasattr(piq,"LPIPS")
print("A22_CPU_PIQ_0P8_AND_TORCH_PASS",torch.__version__,flush=True)
PY
env PYTHONPATH="$WT" "$DAPS" - <<'PY'
from scripts.track_r.xp_t_daps4k_native_cpu import NATIVE,verify_clean_sparse
from scripts.track_r.xp_a12_g7_c2_gpu import check
from scripts.track_r.xp_t_daps4k_prepare_cpu import IMAGENET_MODEL_SHA
assert callable(verify_clean_sparse) and callable(check)
assert len(IMAGENET_MODEL_SHA)==64
print("A22_G7_NATIVE_IMPORT_PASS",flush=True)
PY

echo "=== All 800 historical A7 ImageNet K4 native SHA checks ==="
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="$BASE/xp_a30_a22_autopilot_$STAMP"
"$PY" -u scripts/track_r/xp_a30_a22_tab3_e1_cpu.py \
 --expected-head "$HEAD" \
 --out "$BASE/xp_a30_a22_tab3_preflight_$STAMP" \
 --preflight-only

echo "=== Start one detached autonomous A22 checkpoint supervisor ==="
LOG="$BASE/XP_A22_AUTOPILOT_SUPERVISOR_$STAMP.log"
test ! -e "$RUN" && test ! -e "$LOG"
nohup env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$VENDOR:$WT" \
 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
 "$PY" -u scripts/track_r/xp_a30_a22_autopilot.py \
 --expected-head "$HEAD" --run "$RUN" \
 > "$LOG" 2>&1 < /dev/null &
echo "A22_AUTOPILOT_PID=$!"
echo "A22_AUTOPILOT_RUN=$RUN"
echo "A22_AUTOPILOT_LOG=$LOG"
echo "A22_A21_FROZEN_CHECKOUT_UNTOUCHED"
echo "A22_A21_10_OPTIONAL_GPU_REPLICATION_EXCLUDED"
echo "A22_XP_K_SCIENCE_ADAPTER_GATE_EXPLICIT"
