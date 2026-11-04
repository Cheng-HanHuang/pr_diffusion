#!/usr/bin/env bash
# Track R A36 / A23 G7: CPU-only UnivFD dependency fix and independent recovery.
# Original A35 all8 / 2-forward publication commits and failed UnivFD run
# remain immutable. Same pinned UnivFD worker/model and CPU readout/publisher.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: bash scripts/track_r/xp_a36_univfd_recovery_launch.sh <exact-A36-commit>" >&2
  exit 8
fi

HEAD="$1"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/daps/bin/python"
VENDOR="$BASE/a36_univfd_vendor_setuptools_80_9_0"

export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$BASE/.pycache_a36_univfd_cpu"
export PYTHONPATH="$WT"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PIP_CACHE_DIR="$ROOT/cache/pip"
export TMPDIR="$ROOT/tmp"

cd "$WT"
echo "=== Verify isolated pinned A36 source ==="
test "$(git rev-parse HEAD)" = "$HEAD"
test -z "$(git status --porcelain)"
test -f scripts/track_r/xp_a36_univfd_recovery_cpu.py
test -f scripts/track_r/xp_a35_g7_worker.py
test -f scripts/track_r/xp_a35_g7_readout_publish_cpu.py
test -f docs/track_r/executors/xp/theory_inputs/XP_A23_G7_REMAINING_SOURCE_LOCK.json
bash -n scripts/track_r/xp_a36_univfd_recovery_launch.sh
echo "A36_SOURCE_PIN_PASS $HEAD"

echo "=== CPU-only regression tests, before any dependency install ==="
"$PY" -m unittest discover -s tests \
  -p 'test_track_r_xp_a36_univfd_recovery.py' -v

echo "=== Provide official pinned legacy pkg_resources to this run only ==="
# setuptools 80.9.0 still includes pkg_resources. The installed daps env
# apparently has a later setuptools with this legacy module removed.
# Use a target in our project output; do not mutate Conda or external
# UniversalFakeDetect Git checkout.
if [ ! -f "$VENDOR/pkg_resources/__init__.py" ]; then
    mkdir -p "$VENDOR" "$PIP_CACHE_DIR" "$TMPDIR"
    "$PY" -m pip install \
        --no-input --disable-pip-version-check \
        --no-deps --target "$VENDOR" \
        "setuptools==80.9.0"
fi
test -f "$VENDOR/pkg_resources/__init__.py"
export PYTHONPATH="$VENDOR:$WT"

echo "=== Isolated packaging import + exact compiler source smoke ==="
"$PY" -W ignore::UserWarning - <<'PY'
import importlib.metadata as md
import pathlib
import pkg_resources
from pkg_resources import packaging
from scripts.track_r.xp_a36_univfd_recovery_cpu import VENDOR
assert md.version("setuptools") == "80.9.0"
assert pathlib.Path(pkg_resources.__file__).resolve().is_relative_to(VENDOR)
assert str(packaging.version.parse("1.2.3")) == "1.2.3"
print("A36_ISOLATED_PKG_RESOURCES_IMPORT_PASS",
      pkg_resources.__file__,flush=True)
PY

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$BASE/xp_a35_g7_remaining_a36_univfd_$STAMP"

echo "=== Run ONLY frozen UnivFD GT-only CPU stage ==="
echo "A36_UNIVFD_CPU_RUN=$RUN"
echo "A36_REUSE_A35_ALL8_COMMIT=967e73cc7adcc5cc3347d257572bd779ff0a6dea"
echo "A36_REUSE_A35_2FORWARD_COMMIT=6d2afbfd8c405cbf7db4a6f193ba21c885c7eaa4"
echo "A36_PREVIOUS_G7_GPU_H=5.971352510419156"
echo "A36_NO_GPU_ALLOCATION=TRUE"

"$PY" -u scripts/track_r/xp_a36_univfd_recovery_cpu.py \
    --expected-head "$HEAD" --run "$RUN"

echo "A36_UNIVFD_CPU_ONLY_RECOVERY_AUTOPILOT_DONE"
echo "A36_UNIVFD_ORIGINAL_FAILED_RUN_UNMODIFIED"
echo "A36_NO_NEW_GPU_TIME"
echo "A36_CP4_SUPPLEMENT_ONLY"
