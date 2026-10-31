#!/usr/bin/env bash
# One-time, CPU-only, hash-pinned piq bootstrap into an isolated output
# directory. Never touches conda env or changes any solver checkout.
set -euo pipefail
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$ROOT/pr_diffusion_track_r_xp_a13"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
VENDOR="$BASE/.vendor/piq_0p8"
WHEELS="$BASE/.cache/piq_wheels"
WHEEL="$WHEELS/piq-0.8.0-py3-none-any.whl"
WHEEL_SHA=bad83f8e68bad6c09df09c2bdf50d1e8ccf5e852d7b64562317c27ed5eb63159
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
test -x "$PY"
test -d "$WT"
EXPECTED_HEAD="$1"
test -n "$EXPECTED_HEAD"
test "$(git -C "$WT" rev-parse HEAD)" = "$EXPECTED_HEAD"
test -z "$(git -C "$WT" status --porcelain)"
mkdir -p "$VENDOR" "$WHEELS"
if test ! -f "$WHEEL"; then
  echo "===== DOWNLOAD PINNED 106 KB PIQ WHEEL ONLY ====="
  "$PY" -m pip download --disable-pip-version-check --no-input \
     --no-deps --only-binary=:all: --no-cache-dir \
     --dest "$WHEELS" 'piq==0.8.0'
fi
echo "$WHEEL_SHA  $WHEEL" | sha256sum -c -
if test ! -d "$VENDOR/piq"; then
  echo "===== INSTALL PIQ INTO PRIVATE OUTPUT VENDOR, WITHOUT DEPENDENCIES ====="
  "$PY" -m pip install --disable-pip-version-check --no-input \
    --no-deps --no-index --no-compile --no-warn-script-location \
    --target "$VENDOR" "$WHEEL"
fi
export PYTHONPATH="$VENDOR:$WT"
echo "===== VERIFY PINNED PIQ UNDER EXISTING TORCH ====="
"$PY" - <<'PY'
import importlib.metadata
import torch,torchvision,piq
assert importlib.metadata.version("piq")=="0.8.0"
assert hasattr(piq,"LPIPS") and hasattr(piq,"ssim")
print("PIQ_0P8_ISOLATED_IMPORT_PASS",piq.__file__)
print("TORCH",torch.__version__,"TORCHVISION",torchvision.__version__)
print("TORCH_HUB_CHECKPOINT_DIR",torch.hub.get_dir()+"/checkpoints")
PY
echo "===== PASSIVE PRETRAINED WEIGHT CACHE INVENTORY ====="
"$PY" - <<'PY'
import hashlib,torch
from pathlib import Path
p=Path(torch.hub.get_dir())/"checkpoints"
for name in ("lpips_weights.pt","vgg16-397923af.pth"):
 f=p/name
 if f.is_file():
  h=hashlib.sha256()
  with f.open("rb") as stream:
   for chunk in iter(lambda:stream.read(1<<20),b""):h.update(chunk)
  print("PRETRAINED_WEIGHT_CACHED",name,f.stat().st_size,h.hexdigest())
 else:print("PRETRAINED_WEIGHT_MISSING",name,str(f))
PY
echo "===== TEST ACTUAL G4 PIQ PREFLIGHT, NO METRIC WORK ====="
"$PY" "$WT/scripts/track_r/xp_a16_cp3_piq_metrics_cpu.py" \
  --expected-head "$EXPECTED_HEAD" --stage G4 --stamp 20261030T210259Z --preflight-only
echo "CP3_PINNED_PIQ_AND_BOTH_WEIGHT_CACHES_READY"
