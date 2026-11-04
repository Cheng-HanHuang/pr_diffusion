#!/usr/bin/env bash
# A37: repair *only* isolated UnivFD Python imports; resume exact A36 CPU run.
# Registered native model/weights/GT score are unchanged. No GPU operations.
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "usage: bash scripts/track_r/xp_a37_univfd_resume.sh <exact-A37-commit>" >&2
  exit 8
fi
HEAD="$1"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
A36="$ROOT/pr_diffusion_track_r_xpn_a36b_univfd"
A36_HEAD=e7fb7e88f442aae863747c38342c5ff5f652a767
RUN="$BASE/xp_a35_g7_remaining_a36_univfd_20261104T185041Z"
VENDOR="$BASE/a36_univfd_vendor_setuptools_80_9_0"
PY="$ROOT/conda-envs/daps/bin/python"
export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$BASE/.pycache_a37_univfd_cpu"
export PYTHONPATH="$VENDOR:$A36"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PIP_CACHE_DIR="$ROOT/cache/pip"
export TMPDIR="$ROOT/tmp"
cd "$WT"

echo "=== Pin A37 repair and immutable A36 scientific executor ==="
test "$(git rev-parse HEAD)" = "$HEAD"
test -z "$(git status --porcelain)"
test "$(git -C "$A36" rev-parse HEAD)" = "$A36_HEAD"
test -z "$(git -C "$A36" status --porcelain)"
test -f "$RUN/A36_UNIVFD_RECOVERY_STATE.json"
test -f "$RUN/UnivFD/logs/ffhq_first_00.log"
test -f "$VENDOR/pkg_resources/__init__.py"
test -f "$A36/scripts/track_r/xp_a36_univfd_recovery_cpu.py"
test -f "$A36/scripts/track_r/xp_a35_g7_worker.py"
bash -n "$WT/scripts/track_r/xp_a37_univfd_resume.sh"

# Prevent two A37 repair supervisors from racing for the same vendor/output.
exec 9>"$BASE/XP_A37_UNIVFD_ENVIRONMENT_RECOVERY.lock"
if ! flock -n 9; then
  echo "STOP_A37_ANOTHER_RECOVERY_RUNNING"
  exit 8
fi

echo "=== Guard against accidental science or GPU resubmission ==="
"$PY" - "$RUN" "$A36_HEAD" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1]);head=sys.argv[2]
s=json.loads((run/"A36_UNIVFD_RECOVERY_STATE.json").read_text())
assert s["code_head"]==head
assert s["status"]=="A36_UNIVFD_CPU_RECOVERY_IN_PROGRESS"
assert s["original_A35_failure_preserved"] is True
assert s["source_GT_manifest_sha256"]=="d517a8ea337c5fed8925f925bf24bae21cea60a566eb29f6795bc31e350abfde"
assert s["previous_G7_cumulative_gpu_h"]==5.971352510419156
assert s["previous_project_cumulative_gpu_h"]==247.80191737622505
assert s["univfd_GPU_h"]==0
assert s["CPU_only"] is True
assert not s["group_receipts"]
assert not list((run/"UnivFD/results").glob("*.json"))
worker=(run/"UnivFD/logs/ffhq_first_00.log").read_text()
assert "ModuleNotFoundError: No module named 'ftfy'" in worker
print("A37_ORIGINAL_FTFY_IMPORT_FAILURE_CONFIRMED")
print("A37_ORIGINAL_A36_SCIENCE_PRESERVED")
PY

echo "=== Isolated official dependency addition, no Conda or model changes ==="
PKG_SHA_BEFORE=$(sha256sum "$VENDOR/pkg_resources/__init__.py" | awk '{print $1}')
# ftfy 6.3.1 imports wcwidth; pin both in the EXISTING isolated A36
# provider so the frozen A36 subprocess (which sets its own PYTHONPATH)
# resolves them without modifying any scientific source.
if ! "$PY" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("ftfy") is not None
                 and importlib.util.find_spec("wcwidth") is not None else 1)
PY
then
  "$PY" -m pip install --no-input --disable-pip-version-check \
     --no-deps --target "$VENDOR" "ftfy==6.3.1" "wcwidth==0.2.13"
fi
# Original bundled CLIP also uses regex and tqdm. Install an isolated,
# pinned copy only when the original daps interpreter lacks the module.
for item in regex tqdm; do
  if ! "$PY" - "$item" <<'PY'
import importlib.util,sys
raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PY
  then
    case "$item" in
      regex) VERSION=2024.11.6;;
      tqdm) VERSION=4.67.1;;
    esac
    "$PY" -m pip install --no-input --disable-pip-version-check \
      --no-deps --target "$VENDOR" "$item==$VERSION"
  fi
done
test "$(sha256sum "$VENDOR/pkg_resources/__init__.py" | awk '{print $1}')" = "$PKG_SHA_BEFORE"

echo "=== Verify exact vendor Python imports and original CLIP dependencies ==="
"$PY" -W ignore::UserWarning - "$VENDOR" <<'PY'
import hashlib,importlib.metadata as md,importlib.util,json,pathlib,sys
vendor=pathlib.Path(sys.argv[1]).resolve()
import pkg_resources,ftfy,wcwidth,regex,tqdm
from pkg_resources import packaging
assert pathlib.Path(pkg_resources.__file__).resolve().is_relative_to(vendor)
assert pathlib.Path(ftfy.__file__).resolve().is_relative_to(vendor)
assert pathlib.Path(wcwidth.__file__).resolve().is_relative_to(vendor)
assert md.version("setuptools")=="80.9.0"
assert md.version("ftfy")=="6.3.1"
assert md.version("wcwidth")=="0.2.13"
assert str(packaging.version.parse("1.2.3"))=="1.2.3"
for m in (ftfy,wcwidth,regex,tqdm):
    print("A37_IMPORT_PASS",m.__name__,m.__file__,flush=True)
PY

echo "=== Original UnivFD full CPU model+8-orbit inference smoke ==="
# Use a separate subprocess. The original model code is not patched, and
# the synthetic black image does not count as a frozen GT experiment.
timeout --signal=TERM --kill-after=15 900 \
"$PY" -W ignore::UserWarning - "$A36" <<'PY'
import json,math,sys
from pathlib import Path
import torch
source=Path(sys.argv[1])
sys.path.insert(0,str(source))
from prdiffusion.track_r.univfd import UnivFD
lock=json.loads((source/"docs/track_r/executors/xp/theory_inputs/XP_A23_G7_REMAINING_SOURCE_LOCK.json").read_text())
e=lock["univfd"]
root=Path(e["source_repo"])
fd=UnivFD(root,root/e["fc_path"],e["clip_path"],device="cpu")
fake=torch.zeros((8,3,256,256),dtype=torch.uint8)
out=fd.p_fake(fake)
assert out.shape==(8,) and torch.isfinite(out).all()
assert bool(((out>=0)&(out<=1)).all())
assert fd.forward_images==8 and fd.forward_calls==1
print("A37_UNIVFD_FULL_MODEL_CPU_8_ORBIT_SMOKE_PASS",
      "fake_prob_mean",float(out.mean()),flush=True)
PY

echo "=== Resume existing A36 run: no new run or scientific commit ==="
cd "$A36"
test "$(git rev-parse HEAD)" = "$A36_HEAD"
test -z "$(git status --porcelain)"
export PYTHONPATH="$VENDOR:$A36"
export PYTHONPYCACHEPREFIX="$BASE/.pycache_a37_a36_resume"
"$PY" -u scripts/track_r/xp_a36_univfd_recovery_cpu.py \
   --expected-head "$A36_HEAD" --run "$RUN" --resume

echo "A37_UNIVFD_ENV_REPAIR_AND_ORIGINAL_A36_RESUME_COMPLETED"
echo "A37_NEW_GPU_HOURS=0"
echo "A37_NO_REPEATED_G7_GPU_METHODS"
