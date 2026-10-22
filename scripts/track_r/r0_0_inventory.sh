#!/usr/bin/env bash
# R0.0: bounded, read-only inventory. Cheng executes; no GPU compute.
r0_inventory_main() {
  ROOT=/egr/research-pac/huang248
  WT="$ROOT/pr_diffusion_track_r"
  BRANCH=codex/prdiff-track-r-r0-20261021
  EXPECTED="$1"
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  BASE="$ROOT/outputs/pr_diffusion/track_r"
  CACHE="$ROOT/outputs/.cache/track_r"
  OUT="$BASE/r0_0_inventory_$STAMP"
  LOG="$BASE/logs/r0_0_inventory_$STAMP.log"
  PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
  if [ -z "$EXPECTED" ] || [ ! -d "$WT/.git" ] || [ ! -x "$PY" ]; then
    echo "STOP_R0_0: expected SHA, checkout or source Python missing"
    return 3
  fi
  ACTUAL=$(git -C "$WT" rev-parse HEAD 2>/dev/null)
  DIRTY=$(git -C "$WT" status --porcelain 2>/dev/null)
  if [ "$ACTUAL" != "$EXPECTED" ] || [ -n "$DIRTY" ]; then
    echo "STOP_R0_0: HEAD or clean-tree gate failed: actual=$ACTUAL expected=$EXPECTED"
    return 4
  fi
  if [ -e "$OUT" ]; then echo "STOP_R0_0: output already exists"; return 5; fi
  mkdir -p "$BASE" "$BASE/logs" "$CACHE" "$CACHE/torch" "$CACHE/pip" "$CACHE/tmp" "$CACHE/conda_pkgs"
  RC=$?
  if [ "$RC" -ne 0 ]; then echo "STOP_R0_0: cannot create Track R directories"; return "$RC"; fi
  export XDG_CACHE_HOME="$CACHE" TORCH_HOME="$CACHE/torch" PIP_CACHE_DIR="$CACHE/pip"
  export CONDA_PKGS_DIRS="$CACHE/conda_pkgs" TMPDIR="$CACHE/tmp" HF_HOME="$CACHE/hf"
  export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=""
  {
    echo "R0_0_UTC=$STAMP"
    echo "R0_0_BRANCH=$BRANCH"
    echo "R0_0_HEAD=$(git -C "$WT" rev-parse HEAD)"
    echo "R0_0_TREE_STATUS_BEGIN"
    git -C "$WT" status --porcelain
    echo "R0_0_TREE_STATUS_END"
    "$PY" "$WT/scripts/track_r/r0_0_inventory.py" \
      --paths "$WT/configs/track_r/pac_paths_track_r_v1.json" \
      --out "$OUT" --expected-sha "$EXPECTED"
    RC=$?
    echo "R0_0_RC=$RC"
  } > "$LOG" 2>&1
  echo "R0_0_RC=$RC OUT=$OUT LOG=$LOG"
  grep -E '^(R0_0_INVENTORY|STATUS_COUNTS|MISSING |MISMATCH |WARN |R0_0_RC=)' "$LOG" | head -n 38
  return "$RC"
}
r0_inventory_main "$@"
