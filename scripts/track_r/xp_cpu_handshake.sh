#!/usr/bin/env bash
# CPU-only Track R XP handshake: A4.3 historical ladder + official ImageNet100 import.
# No GPU work, no historical-environment modification, no shell replacement or exit.
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
CACHE="$ROOT/outputs/.cache/track_r"
BRANCH=codex/prdiff-track-r-xp-20261026
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
ORIGIN_ID=1pqVO-LYrRRL4bVxUidvy-Eb22edpuFCs
EXPECTED="$1"
TAG="$(date -u +%Y%m%dT%H%M%SZ)"

export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"
export TMPDIR="$CACHE/tmp"
export XDG_CACHE_HOME="$CACHE"
export PIP_CACHE_DIR="$CACHE/pip"
mkdir -p "$BASE" "$CACHE/tmp" "$CACHE/pip" "$CACHE/home"

main() {
  if [ -z "$EXPECTED" ] || [ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED" ] ||
     [ -n "$(git -C "$WT" status --porcelain)" ]; then
    echo "ACTION NEEDED: XP_CPU_DIRTY_OR_WRONG_HEAD"
    return 8
  fi
  echo "RUN_COMMIT=$EXPECTED"
  echo "=== UNIT TESTS (CPU) ==="
  "$PY" -m unittest discover -s "$WT/tests" -p 'test_track_r_xp_*.py' -q
  TEST_RC=$?
  echo "TEST_RC=$TEST_RC"
  if [ "$TEST_RC" -ne 0 ]; then
    echo "ACTION NEEDED: CPU_TESTS"
    return 8
  fi

  echo "=== R0-N.2 A4.3 HISTORICAL LADDER ==="
  A43="$BASE/r0n2_a43_$TAG"
  "$PY" "$WT/scripts/track_r/xp_r0n2_ladder_cpu.py" \
    --expected-head "$EXPECTED" --out "$A43"
  A43_RC=$?
  echo "A43_RC=$A43_RC"
  if [ "$A43_RC" -eq 0 ]; then
    TARGET="$WT/docs/track_r/executors/r0n/returns/R0N2_A43_$TAG"
    mkdir -p "$TARGET"
    if cp "$A43/R0N2_A43_LADDER_AGGREGATES.csv" "$TARGET/" &&
       cp "$A43/R0N2_A43_POSTSELECTION.json" "$TARGET/" &&
       cp "$A43/R0N2_A43_COMPLETE.json" "$TARGET/" &&
       (
         cd "$TARGET" &&
         sha256sum R0N2_A43_LADDER_AGGREGATES.csv \
           R0N2_A43_POSTSELECTION.json \
           R0N2_A43_COMPLETE.json > SHA256SUMS.txt
       ) &&
       git -C "$WT" add "docs/track_r/executors/r0n/returns/R0N2_A43_$TAG" &&
       git -C "$WT" commit -m "data(track-r): R0-N.2 A4.3 frozen historical ladder" &&
       git -C "$WT" push origin "$BRANCH"; then
      echo "A43_PUBLISHED_COMMIT=$(git -C "$WT" rev-parse HEAD)"
    else
      echo "ACTION NEEDED: A43_PUBLISH_FAILURE"
      return 8
    fi
  else
    echo "ACTION NEEDED: A43_NOT_PUBLISHED"
  fi

  echo "=== IMAGENET100 OFFICIAL DAPS DOWNLOAD ==="
  DATA="$ROOT/data/imagenet/test-imagenet"
  if [ -d "$DATA" ]; then
    echo "IMAGENET_TARGET_ALREADY_PRESENT_REINVENTORY_ONLY"
  else
    VENDOR="$CACHE/gdown_vendor"
    mkdir -p "$VENDOR"
    if ! PYTHONPATH="$VENDOR:$WT" "$PY" -c 'import gdown' >/dev/null 2>&1; then
      HOME="$CACHE/home" "$PY" -m pip install \
        --no-input --disable-pip-version-check \
        --target "$VENDOR" "gdown==5.2.0" \
        > "$CACHE/gdown_install_$TAG.log" 2>&1
      INSTALL_RC=$?
      echo "GDOWN_INSTALL_RC=$INSTALL_RC"
    fi
    if PYTHONPATH="$VENDOR:$WT" "$PY" -c 'import gdown' >/dev/null 2>&1; then
      DOWNLOAD="$BASE/xp_imagenet_download_$TAG"
      mkdir -p "$DOWNLOAD"
      ARCHIVE="$DOWNLOAD/test-imagenet.zip"
      HOME="$CACHE/home" PYTHONPATH="$VENDOR:$WT" "$PY" -m gdown \
        "https://drive.google.com/uc?id=$ORIGIN_ID" \
        -O "$ARCHIVE" --quiet > "$DOWNLOAD/gdown.log" 2>&1
      FETCH_RC=$?
      echo "IMAGENET_FETCH_RC=$FETCH_RC"
      if [ "$FETCH_RC" -eq 0 ] && [ -s "$ARCHIVE" ]; then
        NOW="$(git -C "$WT" rev-parse HEAD)"
        if [ -n "$(git -C "$WT" status --porcelain)" ]; then
          echo "ACTION NEEDED: WORKTREE_DIRTY_BEFORE_IMPORT"
        else
          "$PY" "$WT/scripts/track_r/xp_imagenet_stage_cpu.py" \
            --archive "$ARCHIVE" \
            --out "$BASE/xp_0_imagenet_$TAG" \
            --expected-head "$NOW"
          IMPORT_RC=$?
          echo "IMAGENET_IMPORT_RC=$IMPORT_RC"
        fi
      else
        echo "ACTION NEEDED: IMAGENET_DOWNLOAD (see $DOWNLOAD/gdown.log)"
      fi
    else
      echo "ACTION NEEDED: GDOWN_INSTALL (see $CACHE/gdown_install_$TAG.log)"
    fi
  fi

  echo "=== FINAL XP-0 INVENTORY ==="
  if [ -d "$DATA" ] && [ -z "$(git -C "$WT" status --porcelain)" ]; then
    NOW="$(git -C "$WT" rev-parse HEAD)"
    "$PY" "$WT/scripts/track_r/xp_0_inventory_cpu.py" \
      --expected-head "$NOW" --out "$BASE/xp_0_$TAG"
    echo "XP0_RC=$?"
  fi
  echo "=== DIGEST ==="
  "$PY" "$WT/scripts/track_r/xp_status.py" --root "$BASE"
}

main
