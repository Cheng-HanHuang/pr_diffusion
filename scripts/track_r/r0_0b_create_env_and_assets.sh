#!/usr/bin/env bash
# R0.0b, opt-in only after inventory review. Never modifies historic environments.
# Run with: bash scripts/track_r/r0_0b_create_env_and_assets.sh <pushed_SHA>
# Step functions return explicit codes; no shell-wide abort or shell replacement.
r0b_guard() {
  ROOT=/egr/research-pac/huang248
  WT="$ROOT/pr_diffusion_track_r"
  BASE="$ROOT/outputs/pr_diffusion/track_r"
  CACHE="$ROOT/outputs/.cache/track_r"
  PREFIX="$ROOT/conda-envs/prdiff_track_r"
  SRC="$ROOT/conda-envs/prdiff_ffhq"
  UNIV="$ROOT/external/UniversalFakeDetect"
  CLIP="$ROOT/models/clip/ViT-L-14.pt"
  EXPECTED="$1"
  if [ -z "$EXPECTED" ] || [ ! -d "$WT/.git" ]; then
    echo "ENV_FAILED: missing commit or Track R checkout"; return 3
  fi
  HEAD=$(git -C "$WT" rev-parse HEAD 2>/dev/null)
  DIRTY=$(git -C "$WT" status --porcelain 2>/dev/null)
  if [ "$HEAD" != "$EXPECTED" ] || [ -n "$DIRTY" ]; then
    echo "ENV_FAILED: dirty/mismatched Track R checkout ($HEAD)"; return 3
  fi
  if [ ! -x "$SRC/bin/python" ] || ! command -v conda >/dev/null 2>&1; then
    echo "ENV_FAILED: source environment or conda missing"; return 4
  fi
  mkdir -p "$BASE/logs" "$CACHE/conda_pkgs" "$CACHE/pip" "$CACHE/tmp" "$CACHE/torch" "$CACHE/hf"
  RC=$?
  if [ "$RC" -ne 0 ]; then return "$RC"; fi
  export CONDA_PKGS_DIRS="$CACHE/conda_pkgs" PIP_CACHE_DIR="$CACHE/pip"
  export TMPDIR="$CACHE/tmp" XDG_CACHE_HOME="$CACHE" TORCH_HOME="$CACHE/torch" HF_HOME="$CACHE/hf"
  export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=""
  return 0
}
r0b_env() {
  if [ ! -e "$PREFIX" ]; then
    if [ -L "$PREFIX" ]; then echo "ENV_FAILED: prefix symlink"; return 10; fi
    conda create --yes --prefix "$PREFIX" --clone "$SRC"
    RC=$?
    if [ "$RC" -ne 0 ]; then echo "ENV_FAILED: conda clone rc=$RC"; return "$RC"; fi
  fi
  if [ ! -x "$PREFIX/bin/python" ]; then echo "ENV_FAILED: target python absent"; return 11; fi
  # Install *only* genuinely missing packages inside the isolated prefix.
  R0_INSTALL=$("$PREFIX/bin/python" - <<'PYTHON'
import importlib.util
req={"ftfy":"ftfy","regex":"regex","pandas":"pandas","sklearn":"scikit-learn",
     "PIL":"Pillow","scipy":"scipy","yaml":"PyYAML","pytest":"pytest",
     "torchvision":"torchvision","packaging":"packaging"}
missing=[pip for mod,pip in req.items() if importlib.util.find_spec(mod) is None]
try: import pkg_resources
except ImportError: missing.append("setuptools==80.9.0")
print(" ".join(missing))
PYTHON
)
  RC=$?
  if [ "$RC" -ne 0 ]; then echo "ENV_FAILED: dependency probe rc=$RC"; return "$RC"; fi
  if [ -n "$R0_INSTALL" ]; then
    echo "R0_0B_MISSING_PACKAGES=$R0_INSTALL"
    "$PREFIX/bin/python" -m pip install --no-input --no-deps $R0_INSTALL
    RC=$?
    if [ "$RC" -ne 0 ]; then echo "ENV_FAILED: isolated pip install rc=$RC"; return "$RC"; fi
  fi
  "$PREFIX/bin/python" -m pip check
  RC=$?
  if [ "$RC" -ne 0 ]; then echo "ENV_FAILED: pip dependency check"; return "$RC"; fi
  "$PREFIX/bin/python" -m pip freeze
  "$PREFIX/bin/python" - <<'PYTHON'
import torch
print("R0B_TORCH",torch.__version__,"CUDA",torch.version.cuda,"ARCHES",torch.cuda.get_arch_list())
print("R0B_SM_120", "sm_120" in torch.cuda.get_arch_list())
PYTHON
  RC=$?
  if [ "$RC" -ne 0 ]; then return "$RC"; fi
  return 0
}
r0b_sha() {
  PATH_TO_HASH="$1" EXPECTED_HASH="$2"
  if [ ! -f "$PATH_TO_HASH" ] || [ -L "$PATH_TO_HASH" ]; then return 20; fi
  ACTUAL_HASH=$(sha256sum "$PATH_TO_HASH" | awk '{print $1}')
  if [ "$ACTUAL_HASH" != "$EXPECTED_HASH" ]; then
    echo "ENV_FAILED: sha mismatch $PATH_TO_HASH expected=$EXPECTED_HASH actual=$ACTUAL_HASH"
    return 21
  fi
  return 0
}
r0b_univfd() {
  WANT=030495aea3300a8b54c0ec37ec7fe1dd7e63c619
  if [ ! -e "$UNIV" ]; then
    mkdir -p "$ROOT/external"
    RC=$?
    if [ "$RC" -ne 0 ]; then return "$RC"; fi
    git clone https://github.com/Yuheng-Li/UniversalFakeDetect "$UNIV"
    RC=$?
    if [ "$RC" -ne 0 ]; then echo "ENV_FAILED: UnivFD clone"; return "$RC"; fi
    git -C "$UNIV" checkout --detach "$WANT"
    RC=$?
    if [ "$RC" -ne 0 ]; then echo "ENV_FAILED: UnivFD commit unavailable"; return "$RC"; fi
  fi
  CURRENT=$(git -C "$UNIV" rev-parse HEAD 2>/dev/null)
  DIRTY=$(git -C "$UNIV" status --porcelain 2>/dev/null)
  if [ "$CURRENT" != "$WANT" ] || [ -n "$DIRTY" ]; then
    echo "ENV_FAILED: existing UnivFD HEAD/clean mismatch"; return 22
  fi
  r0b_sha "$UNIV/pretrained_weights/fc_weights.pth" 477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847
  RC=$?
  if [ "$RC" -ne 0 ]; then echo "ENV_FAILED: UnivFD fc_weights absent or wrong; no unpinned download"; return "$RC"; fi
  return 0
}
r0b_clip() {
  WANT=b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836
  if [ ! -e "$CLIP" ]; then
    mkdir -p "$ROOT/models/clip"
    RC=$?
    if [ "$RC" -ne 0 ]; then return "$RC"; fi
    # Explicit target in the project model namespace, never the user cache.
    "$PREFIX/bin/python" - "$WT/configs/track_r/pac_paths_track_r_v1.json" "$CLIP" "$WANT" <<'PYTHON'
import hashlib,json,os,sys,urllib.request
from pathlib import Path
cfg=json.loads(Path(sys.argv[1]).read_text())["models"]["clip_vit_l14"]
target=Path(sys.argv[2]); want=sys.argv[3]; tmp=target.with_suffix(".pt.r0partial")
if cfg["sha256"]!=want or target.exists() or tmp.exists():
    raise RuntimeError("STOP_R0B_CLIP_PIN_OR_EXISTS")
with urllib.request.urlopen(cfg["source_url"],timeout=180) as response, tmp.open("xb") as f:
    h=hashlib.sha256()
    while True:
        block=response.read(4*1024*1024)
        if not block: break
        f.write(block); h.update(block)
if h.hexdigest()!=want:
    tmp.unlink()
    raise RuntimeError("STOP_R0B_CLIP_HASH_MISMATCH")
os.replace(tmp,target)
print("R0B_CLIP_DOWNLOADED_SHA256",h.hexdigest())
PYTHON
    RC=$?
    if [ "$RC" -ne 0 ]; then echo "ENV_FAILED: CLIP download or pin"; return "$RC"; fi
  fi
  r0b_sha "$CLIP" "$WANT"
  return $?
}
r0b_main() {
  r0b_guard "$1"
  RC=$?
  if [ "$RC" -ne 0 ]; then return "$RC"; fi
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  LOG="$BASE/logs/r0_0b_env_assets_$STAMP.log"
  {
    echo "R0B_HEAD=$(git -C "$WT" rev-parse HEAD)"
    echo "R0B_TREE_STATUS_BEGIN"
    git -C "$WT" status --porcelain
    echo "R0B_TREE_STATUS_END"
    r0b_env
    ENV_RC=$?
    echo "R0B_ENV_RC=$ENV_RC"
    if [ "$ENV_RC" -ne 0 ]; then
      echo "ENV_FAILED: isolated environment creation"; echo "R0B_RC=$ENV_RC"
    else
      r0b_univfd
      UNIV_RC=$?
      echo "R0B_UNIV_RC=$UNIV_RC"
      if [ "$UNIV_RC" -ne 0 ]; then
        echo "ENV_FAILED: UnivFD"; echo "R0B_RC=$UNIV_RC"
      else
        r0b_clip
        CLIP_RC=$?
        echo "R0B_CLIP_RC=$CLIP_RC"
        if [ "$CLIP_RC" -eq 0 ]; then
          echo "ENV_READY"; echo "R0B_RC=0"
        else
          echo "ENV_FAILED: CLIP"; echo "R0B_RC=$CLIP_RC"
        fi
      fi
    fi
  } > "$LOG" 2>&1
  RC=$(grep '^R0B_RC=' "$LOG" | tail -n 1 | cut -d= -f2)
  if [ -z "$RC" ]; then RC=99; fi
  echo "R0B_RC=$RC LOG=$LOG"
  grep -E '^(R0B_(ENV|UNIV|CLIP|SM_120|RC)|ENV_READY|ENV_FAILED)' "$LOG" | tail -n 12
  return "$RC"
}
r0b_main "$@"
