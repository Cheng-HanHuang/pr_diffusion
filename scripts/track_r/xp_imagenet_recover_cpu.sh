#!/usr/bin/env bash
# CPU-only retry of an already-fetched official DAPS ImageNet ZIP.
# Deterministic prereg first-100 selection, immutable archive, no GPU usage.
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
CACHE="$ROOT/outputs/.cache/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
BRANCH=codex/prdiff-track-r-xp-20261026
ARCHIVE="$BASE/xp_imagenet_download_20261026T193644Z/test-imagenet.zip"
EXPECTED="$1"
TAG="$(date -u +%Y%m%dT%H%M%SZ)"

export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"
export TMPDIR="$CACHE/tmp"
export XDG_CACHE_HOME="$CACHE"
mkdir -p "$BASE" "$CACHE/tmp" "$CACHE"

main() {
  if [ -z "$EXPECTED" ] ||
     [ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED" ] ||
     [ -n "$(git -C "$WT" status --porcelain)" ]; then
    echo "ACTION NEEDED: SOURCE_HEAD_OR_WORKTREE"
    return 8
  fi

  echo "=== IMPORT REGRESSION TESTS ==="
  "$PY" -m unittest discover -s "$WT/tests" \
    -p 'test_track_r_xp_*.py' -q
  TEST_RC=$?
  echo "TEST_RC=$TEST_RC"
  if [ "$TEST_RC" -ne 0 ]; then
    echo "ACTION NEEDED: IMPORT_UNIT_TEST"
    return 8
  fi
  if [ ! -s "$ARCHIVE" ]; then
    echo "ACTION NEEDED: ORIGINAL_IMAGENET_ARCHIVE_NOT_FOUND"
    echo "EXPECTED_ARCHIVE=$ARCHIVE"
    return 8
  fi
  if [ -e "$ROOT/data/imagenet/test-imagenet" ] ||
     [ -L "$ROOT/data/imagenet/test-imagenet" ]; then
    echo "ACTION NEEDED: IMPORT_TARGET_EXISTS_REINVENTORY"
    return 8
  fi

  echo "=== IMAGESET100 STRICT IMPORT FROM SAVED ZIP ==="
  STAGE="$BASE/xp_0_imagenet_$TAG"
  "$PY" "$WT/scripts/track_r/xp_imagenet_stage_cpu.py" \
    --archive "$ARCHIVE" --expected-head "$EXPECTED" --out "$STAGE"
  IMPORT_RC=$?
  echo "IMAGENET_IMPORT_RC=$IMPORT_RC"
  if [ "$IMPORT_RC" -ne 0 ]; then
    echo "ACTION NEEDED: IMAGENET_IMPORT"
    return 8
  fi

  echo "=== NEW XP-0 INTEGRITY RECHECK ==="
  XP0="$BASE/xp_0_$TAG"
  "$PY" "$WT/scripts/track_r/xp_0_inventory_cpu.py" \
    --expected-head "$EXPECTED" --out "$XP0"
  XP0_RC=$?
  echo "XP0_RC=$XP0_RC"
  if [ "$XP0_RC" -ne 0 ]; then
    echo "ACTION NEEDED: XP0_INVENTORY"
    return 8
  fi

  echo "=== MANIFEST CROSS-VALIDATION ==="
  "$PY" - "$STAGE" "$XP0" <<'PY'
import json,sys
from pathlib import Path
s,x=(Path(z) for z in sys.argv[1:])
a=json.loads((s/"XP_IMAGENET100_IMPORT_RECEIPT.json").read_text())
b=json.loads((x/"XP_0_INVENTORY.json").read_text())
c=json.loads((x/"XP_0_IMAGENET_IMAGES.json").read_text())
assert a["status"]=="XP_IMAGENET100_IMPORTED_PASS", "BAD_IMPORT_STATUS"
assert a["image_count"]==100 and a["archive_image_count"]==1000, "WRONG_SOURCE_COUNT"
assert a["selection_indices"]==[0,99], "WRONG_SELECTION"
assert len(a["images"])==100 and len(c)==100, "INCOMPLETE_MANIFEST"
assert b["status"]=="XP_0_PASS" and b["imagenet_n"]==100 and b["ffhq_n"]==180, "XP0_WRONG_COUNTS"
assert sorted((r["basename"],r["sha256"]) for r in a["images"])==sorted(
    (r["basename"],r["sha256"]) for r in c), "IMAGENET_SHA_MISMATCH"
print("IMAGENET_MANIFEST_CROSSCHECK_PASS 100/100 FIRST_SORTED_OF_1000")
PY
  VERIFY_RC=$?
  echo "MANIFEST_VERIFY_RC=$VERIFY_RC"
  if [ "$VERIFY_RC" -ne 0 ]; then
    echo "ACTION NEEDED: IMAGENET_CROSSCHECK"
    return 8
  fi

  echo "=== PUBLISH TO XP EXECUTOR BRANCH ==="
  TARGET="$WT/docs/track_r/executors/xp/returns/XP0_IMAGENET_$TAG"
  mkdir -p "$TARGET"
  if cp "$STAGE/XP_IMAGENET100_IMPORT_RECEIPT.json" "$TARGET/" &&
     cp "$XP0/XP_0_INVENTORY.json" "$TARGET/" &&
     cp "$XP0/XP_0_IMAGENET_IMAGES.json" "$TARGET/" &&
     (
       cd "$TARGET" &&
       sha256sum XP_IMAGENET100_IMPORT_RECEIPT.json \
         XP_0_INVENTORY.json XP_0_IMAGENET_IMAGES.json > SHA256SUMS.txt
     ) &&
     git -C "$WT" add "docs/track_r/executors/xp/returns/XP0_IMAGENET_$TAG" &&
     git -C "$WT" commit -m "data(track-r): publish SHA-pinned ImageNet first-100-of-1000" &&
     git -C "$WT" push origin "$BRANCH"; then
    echo "XP0_IMAGESET_PUBLISHED_COMMIT=$(git -C "$WT" rev-parse HEAD)"
    echo "OK"
    return 0
  fi
  echo "ACTION NEEDED: IMAGENET_MANIFEST_PUBLISH"
  return 8
}

main
RC=$?
echo "IMAGENET_RECOVERY_RC=$RC"
echo "=== XP STATUS DIGEST ==="
"$PY" "$WT/scripts/track_r/xp_status.py" --root "$BASE"
