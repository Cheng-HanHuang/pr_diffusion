#!/usr/bin/env bash
# Fetch official pretrained metric dependencies into the torch.hub cache.
# No solver environment or original reconstructions are modified.
set -euo pipefail
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$ROOT/pr_diffusion_track_r_xp_a13"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
HEAD="$1"
VENDOR="$BASE/.vendor/piq_0p8"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$VENDOR:$WT"
test -n "$HEAD"
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -d "$VENDOR/piq"
CACHE=$("$PY" - <<'PY'
import torch
print(torch.hub.get_dir()+"/checkpoints")
PY
)
test -n "$CACHE"
mkdir -p "$CACHE"
VGG="$CACHE/vgg16-397923af.pth"
LP="$CACHE/lpips_weights.pt"
VGG_SHA=397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0
VGG_URL=https://download.pytorch.org/models/vgg16-397923af.pth
LP_URL=https://github.com/photosynthesis-team/photosynthesis.metrics/releases/download/v0.4.0/lpips_weights.pt
echo "===== OFFICIAL VGG16 WEIGHTS (ABOUT 553 MB ONLY IF UNCACHED) ====="
if test -f "$VGG"; then
 echo "$VGG_SHA  $VGG" | sha256sum -c -
else
 tmp="$VGG.partial.$$"
 trap 'rm -f "$tmp"' EXIT
 curl --proto '=https' --tlsv1.2 -fL --retry 3 \
   --connect-timeout 15 --max-time 1800 -o "$tmp" "$VGG_URL"
 echo "$VGG_SHA  $tmp" | sha256sum -c -
 mv "$tmp" "$VGG"
 trap - EXIT
fi
echo "===== OFFICIAL PIQ LPIPS CALIBRATION WEIGHTS (6629 BYTES) ====="
if test ! -f "$LP"; then
 tmp="$LP.partial.$$"
 trap 'rm -f "$tmp"' EXIT
 curl --proto '=https' --tlsv1.2 -fL --retry 3 \
   --connect-timeout 15 --max-time 90 -o "$tmp" "$LP_URL"
 test "$(stat -c %s "$tmp")" = 6629
 "$PY" - "$tmp" <<'PY'
from pathlib import Path
import sys,torch
p=Path(sys.argv[1])
weights=torch.load(p,map_location="cpu",weights_only=True)
assert isinstance(weights,(list,tuple)) and len(weights)==5
assert all(isinstance(t,torch.Tensor) and bool(torch.isfinite(t).all())
           for t in weights)
print("LPIPS_OFFICIAL_FIVE_LAYER_WEIGHTS_SCHEMA_PASS",p.stat().st_size)
PY
 mv "$tmp" "$LP"
 trap - EXIT
fi
test "$(stat -c %s "$LP")" = 6629
"$PY" - "$LP" <<'PY'
import sys,torch
x=torch.load(sys.argv[1],map_location="cpu",weights_only=True)
assert isinstance(x,(list,tuple)) and len(x)==5
assert all(isinstance(y,torch.Tensor) and bool(y.isfinite().all()) for y in x)
print("LPIPS_CALIBRATION_WEIGHTS_PASS")
PY
echo "$VGG_SHA  $VGG" | sha256sum -c -
echo "===== METRIC WEIGHT FULL DIGEST RECEIPT ====="
"$PY" - "$VGG" "$LP" <<'PY'
import hashlib,sys
from pathlib import Path
for p in map(Path,sys.argv[1:]):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 print("METRIC_WEIGHT_PIN",p.name,h.hexdigest(),p.stat().st_size)
print("CP3_OFFICIAL_METRIC_WEIGHTS_CACHED_AND_AUTHENTICATED")
PY
