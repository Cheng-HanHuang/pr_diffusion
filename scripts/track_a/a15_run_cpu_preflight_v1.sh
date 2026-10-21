#!/usr/bin/env bash
# A15 source-frozen CPU-only preflight: no CUDA/model inference, no set -e/u/pipefail.
# Executed via bash; errors return nonzero without closing interactive terminal.
a15_preflight_main() {
  ROOT=/egr/research-pac/huang248
  WORK="$ROOT/pr_diffusion_track_a_a15_fastdips"
  FAST="$ROOT/external/FAST_DIPS_A15"
  OUT="$ROOT/outputs/pr_diffusion/track_a_np_paper/a15_fastdips_exposed_20261020/a15_0_cpu_preflight_v1"
  PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
  SHA="a9758b38906b7c7950fa04491f37f6634948db26"
  EXPECTED="$1"

  if [ -z "$EXPECTED" ] || [ ! -d "$WORK/.git" ] || [ ! -x "$PY" ]; then
    echo "STOP_A15_0: expected A15 commit, isolated checkout, or PAC Python unavailable"
    return 3
  fi
  if [ -e "$OUT" ] || [ -L "$OUT" ]; then
    echo "STOP_A15_0: preflight output already occupied (do not overwrite)"
    return 3
  fi
  if [ "$(git -C "$WORK" rev-parse HEAD 2>/dev/null)" != "$EXPECTED" ]; then
    echo "STOP_A15_0: checkout at wrong commit"
    return 3
  fi
  if [ "$(git -C "$WORK" branch --show-current)" != "codex/prdiff-track-a-a15-fastdips-exposed-20261020" ]; then
    echo "STOP_A15_0: wrong branch"
    return 3
  fi
  if [ -n "$(git -C "$WORK" status --porcelain --untracked-files=no)" ]; then
    echo "STOP_A15_0: local tracked source dirty"
    return 3
  fi

  if [ ! -e "$FAST" ] && [ ! -L "$FAST" ]; then
    echo "A15_CLONING_PINNED_FAST_UPSTREAM"
    git clone --no-checkout https://github.com/ququlza/FAST-DIPS.git "$FAST"
    CLONE_RC=$?
    echo "A15_FAST_CLONE_RC=$CLONE_RC"
    if [ "$CLONE_RC" -ne 0 ]; then return "$CLONE_RC"; fi
    git -C "$FAST" checkout --detach "$SHA"
    CO_RC=$?
    echo "A15_FAST_CHECKOUT_RC=$CO_RC"
    if [ "$CO_RC" -ne 0 ]; then return "$CO_RC"; fi
  fi

  if [ "$(git -C "$FAST" rev-parse HEAD 2>/dev/null)" != "$SHA" ]; then
    echo "STOP_A15_0: FAST upstream wrong HEAD"
    return 3
  fi
  export CUDA_VISIBLE_DEVICES=""
  export PYTHONDONTWRITEBYTECODE=1
  # Use AST parser without importing or writing bytecode.
  (cd "$WORK" && "$PY" - <<'PYTHON'
import ast
from pathlib import Path
paths = [
  "scripts/track_a/a15_fastdips_preflight.py",
  "scripts/track_a/a15_fastdips_smoke.py",
  "scripts/track_a/a15_fastdips_report.py",
]
for p in paths:
    ast.parse(Path(p).read_text(encoding="utf8"),filename=p)
print("A15_AST_PARSE_PASS",len(paths))
PYTHON
  )
  AST_RC=$?
  echo "A15_AST_RC=$AST_RC"
  if [ "$AST_RC" -ne 0 ]; then return "$AST_RC"; fi

  (cd "$WORK" && "$PY" -m scripts.track_a.a15_fastdips_preflight \
    --expected-head "$EXPECTED" --out "$OUT")
  RC=$?
  echo "A15_CPU_PREFLIGHT_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "STOP_A15_0: return diagnostic; NO GPU and NO package install"
    return "$RC"
  fi
  if [ ! -f "$OUT/A15_PREFLIGHT_PASS.json" ]; then
    echo "STOP_A15_0: PASS receipt absent"
    return 5
  fi
  echo "A15_CPU_RECEIPT_SHA256"
  sha256sum "$OUT/A15_PREFLIGHT_PASS.json"
  echo "A15_CPU_STAGE0_COMPLETE_NO_GPU"
  return 0
}
a15_preflight_main "$@"
