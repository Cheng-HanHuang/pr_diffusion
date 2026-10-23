#!/usr/bin/env bash
# Single-entrypoint for full R0-N PAC execution, including optional N2/N3.
# Call: bash scripts/track_r/r0n_all_launch.sh EXACT_GITHUB_SHA
set -u
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
CACHE="$ROOT/outputs/.cache/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
BRANCH=codex/prdiff-track-r-r0n-20261023
EXPECTED="${1:-}"

if [ -z "$EXPECTED" ]; then echo STOP_N0N_MISSING_PIN; exit 3; fi
if [ ! -x "$PY" ]; then echo STOP_R0N_NO_PYTHON; exit 4; fi
if [ -n "$(git -C "$WT" status --porcelain)" ]; then
  echo STOP_R0N_DIRTY_WORKTREE; exit 5
fi
git -C "$WT" fetch origin "$BRANCH" || exit $?
git -C "$WT" checkout --detach "$EXPECTED" || exit $?
if [ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED" ] ||
   [ -n "$(git -C "$WT" status --porcelain)" ]; then
  echo STOP_R0N_BAD_HEAD_OR_DIRTY; exit 6
fi
# Two commits and two branch pushes are part of this unattended handoff.
# Check author identity and permission to push before touching N0 files.
git -C "$WT" var GIT_AUTHOR_IDENT >/dev/null 2>&1 || {
  echo STOP_R0N_GIT_AUTHOR_IDENTITY; exit 6;
}
git -C "$WT" push --dry-run origin "HEAD:refs/heads/$BRANCH" >/dev/null 2>&1 || {
  echo STOP_R0N_GIT_PUSH_PERMISSION_OR_REMOTE_DIVERGENCE; exit 6;
}
echo R0N_GIT_PUSH_DRY_RUN_PASS
mkdir -p "$BASE/logs" "$CACHE/home" "$CACHE/tmp" "$CACHE/config" \
 "$CACHE/data" "$CACHE/state" "$CACHE/torch" "$CACHE/hf" "$CACHE/pip"
export PYTHONPATH="$WT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HOME="$CACHE/home" TMPDIR="$CACHE/tmp"
export XDG_CACHE_HOME="$CACHE" XDG_CONFIG_HOME="$CACHE/config"
export XDG_DATA_HOME="$CACHE/data" XDG_STATE_HOME="$CACHE/state"
export TORCH_HOME="$CACHE/torch" HF_HOME="$CACHE/hf" PIP_CACHE_DIR="$CACHE/pip"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$WT" || exit 7

# Syntax/static preflight before touching Git or PAC result directories.
CUDA_VISIBLE_DEVICES="" "$PY" - <<'PY'
import ast, pathlib
root=pathlib.Path("scripts/track_r")
for name in (
  "r0n_0_import_tables_cpu.py",
  "r0n_1_letter_tables_cpu.py",
  "r0n_2_clip_orientation.py",
  "r0n_3_imagenet_inventory_cpu.py",
  "r0n_all_autopilot.py"):
    ast.parse((root/name).read_text(),filename=str(root/name))
ast.parse(pathlib.Path("prdiffusion/track_r/r0n_letter.py").read_text())
print("R0N_STATIC_PYTHON_SYNTAX_PASS 6 scripts")
PY
RC=$?
if [ "$RC" -ne 0 ]; then echo "STOP_R0N_SYNTAX_RC=$RC";exit "$RC";fi

CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q -p no:cacheprovider tests/track_r
RC=$?
echo "R0N_PREFLIGHT_CPU_TEST_RC=$RC"
if [ "$RC" -ne 0 ]; then echo STOP_R0N_PREFLIGHT_TESTS;exit "$RC";fi

"$PY" - <<'PY'
from pathlib import Path
from scripts.track_r.r0n_0_import_tables_cpu import ZIP_SHA,FILES
from hashlib import sha256
import zipfile
p=Path("/egr/research-pac/huang248/outputs/pr_diffusion/track_r/r0_8_overnight_20261023T082909Z/R0_8_PLANNER_RETURN.zip")
assert p.is_file() and not p.is_symlink()
assert sha256(p.read_bytes()).hexdigest()==ZIP_SHA
with zipfile.ZipFile(p) as z:
 assert z.testzip() is None
 for n,h in FILES.items():
  assert sha256(z.read(n)).hexdigest()==h,n
print("R0N_ORIGINAL_ZIP_SHA_PREFLIGHT_PASS",len(FILES),"TABLES")
PY
RC=$?
if [ "$RC" -ne 0 ]; then echo "STOP_R0N_ZIP_PREFLIGHT_RC=$RC";exit "$RC";fi

nvidia-smi --query-gpu=index,name,memory.used --format=csv || true
GPU=1
# Prefer GPU 1, then any other currently idle physical GPU.
if command -v nvidia-smi >/dev/null 2>&1;then
  mapfile -t MEMS < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  for IDX in 1 0 2 3;do
    USED="${MEMS[$IDX]:-999999}"
    USED="${USED//[[:space:]]/}"
    if [[ "$USED" =~ ^[0-9]+$ ]] && [ "$USED" -le 1024 ];then
      GPU=$IDX
      break
    fi
  done
fi
echo "R0N_N2_PREFERRED_GPU=$GPU (worker rechecks at launch)"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$BASE/r0n_all_$STAMP"
LOG="$BASE/logs/r0n_all_$STAMP.log"
if [ -e "$RUN" ];then echo STOP_R0N_RUN_ALREADY_EXISTS;exit 9;fi
CUDA_VISIBLE_DEVICES="" nohup "$PY" -u scripts/track_r/r0n_all_autopilot.py \
 --expected-head "$EXPECTED" --physical-gpu "$GPU" --out "$RUN" > "$LOG" 2>&1 &
PID=$!
printf '%s\n' "$RUN" > "$BASE/LATEST_R0N_ALL_RUN"
printf '%s\n' "$PID" > "$BASE/logs/r0n_all_$STAMP.pid"
echo "R0N_ALL_PID=$PID"
echo "R0N_ALL_RUN=$RUN"
echo "R0N_ALL_LOG=$LOG"
echo "R0N_ALL_LAUNCHED"
