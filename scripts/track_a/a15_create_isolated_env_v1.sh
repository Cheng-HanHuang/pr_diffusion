#!/usr/bin/env bash
# A1.5 isolated FAST-DIPS (pixel-only) environment installer. PAC user executes.
# NO set -e / exit. A failed step returns nonzero and preserves partial env.
a15_env_main() {
  ROOT=/egr/research-pac/huang248
  WORK="$ROOT/pr_diffusion_track_a_a15_fastdips"
  PREFIX="$ROOT/conda-envs/fast_dips_a15"
  PY="$PREFIX/bin/python"
  REQS="$WORK/configs/track_a/a15_fastdips_pixel_requirements.txt"
  CACHE="$ROOT/outputs/.cache/fast_dips_a15"
  EXPECTED="$1"
  if [ -z "$EXPECTED" ] || [ ! -d "$WORK/.git" ]; then
    echo "STOP_A15_ENV: expected commit or dedicated source checkout missing"
    return 3
  fi
  ACTUAL=$(git -C "$WORK" rev-parse HEAD 2>/dev/null)
  if [ "$ACTUAL" != "$EXPECTED" ] ||
     [ "$(git -C "$WORK" branch --show-current)" != "codex/prdiff-track-a-a15-fastdips-exposed-20261020" ]; then
    echo "STOP_A15_ENV: A15 source branch/commit drift ($ACTUAL)"
    return 3
  fi
  if [ -n "$(git -C "$WORK" status --porcelain --untracked-files=no)" ]; then
    echo "STOP_A15_ENV: tracked A15 source modified"
    return 3
  fi
  if [ ! -f "$REQS" ] || [ -L "$REQS" ] || [ -e "$PREFIX" ] || [ -L "$PREFIX" ]; then
    echo "STOP_A15_ENV: requirements missing/symlink or environment already exists"
    return 3
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "STOP_A15_ENV: conda unavailable in PATH; send shell conda inventory"
    return 3
  fi

  mkdir -p "$ROOT/conda-envs" "$CACHE/conda_pkgs" "$CACHE/pip" "$CACHE/tmp"
  MD_RC=$?
  if [ "$MD_RC" -ne 0 ]; then return "$MD_RC"; fi
  export CONDA_PKGS_DIRS="$CACHE/conda_pkgs"
  export PIP_CACHE_DIR="$CACHE/pip"
  export TMPDIR="$CACHE/tmp"
  export XDG_CACHE_HOME="$CACHE"
  export PYTHONNOUSERSITE=1
  export PYTHONDONTWRITEBYTECODE=1
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  export CUDA_VISIBLE_DEVICES=""

  echo "=== A15 creating isolated Python3.10 conda prefix ==="
  conda create --yes --prefix "$PREFIX" "python=3.10" pip
  RC=$?
  echo "A15_CONDA_CREATE_RC=$RC"
  if [ "$RC" -ne 0 ]; then return "$RC"; fi
  if [ ! -x "$PY" ]; then
    echo "STOP_A15_ENV: python absent after env create"
    return 5
  fi

  echo "=== A15 installing exact official FAST PyTorch/CUDA wheel pair ==="
  "$PY" -m pip install --no-input --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.3.0" "torchvision==0.18.0"
  RC=$?
  echo "A15_PYTORCH_INSTALL_RC=$RC"
  if [ "$RC" -ne 0 ]; then return "$RC"; fi

  echo "=== A15 installing bounded pixel-only dependencies ==="
  "$PY" -m pip install --no-input -r "$REQS"
  RC=$?
  echo "A15_PIXEL_REQUIREMENTS_RC=$RC"
  if [ "$RC" -ne 0 ]; then return "$RC"; fi

  "$PY" -m pip check
  RC=$?
  echo "A15_PIP_CHECK_RC=$RC"
  if [ "$RC" -ne 0 ]; then return "$RC"; fi

  "$PY" - <<'PYTHON'
import importlib
import sys
import torch
import torchvision
names = ("numpy","scipy","PIL","scipy","packaging","yaml","imageio",
         "tqdm","wandb","setproctitle","torchmetrics","skimage",
         "lpips","prettytable","omegaconf")
for name in names:
    importlib.import_module(name)
print("A15_ISOLATED_IMPORT_PASS")
print("PYTHON:",sys.executable,sys.version.split()[0])
print("TORCH:",torch.__version__,"CUDA_WHEEL:",torch.version.cuda)
print("TORCHVISION:",torchvision.__version__)
PYTHON
  RC=$?
  echo "A15_IMPORT_RC=$RC"
  if [ "$RC" -ne 0 ]; then return "$RC"; fi
  echo "A15_ENV_CREATED_NO_GPU"
  return 0
}
a15_env_main "$@"
