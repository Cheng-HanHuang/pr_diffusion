#!/usr/bin/env bash
set -euo pipefail

ROOT=/egr/research-pac/huang248
CONTROL="$ROOT/pr_diffusion_b23"
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
BRANCH=codex/b24-bestof4-failure-sweep
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
ABC_PTR="$OUTROOT/B24_ABC300_LATEST_FREEZE.txt"
C1_PTR="$OUTROOT/B24_C1_SITCOM_WORST100_LATEST_FREEZE.txt"

[[ -x "$PY" ]] || { echo "STOP|missing_python:$PY"; exit 2; }
[[ -f "$ABC_PTR" ]] || { echo "STOP|missing_abc_pointer:$ABC_PTR"; exit 2; }
[[ -f "$C1_PTR" ]] || { echo "STOP|missing_c1_pointer:$C1_PTR"; exit 2; }
FREEZE=$(cat "$ABC_PTR")
C1DIR=$(cat "$C1_PTR")
[[ -d "$FREEZE" ]] || { echo "STOP|missing_freeze_dir:$FREEZE"; exit 2; }
[[ -d "$C1DIR" ]] || { echo "STOP|missing_c1_dir:$C1DIR"; exit 2; }

UNIVERSE="$FREEZE/B24_7424_CASE_UNIVERSE.csv"
ABC300="$FREEZE/B24_ABC300_FROZEN.csv"
C1="$C1DIR/B24_C1_SITCOM_WORST100_FROZEN.csv"
for p in "$UNIVERSE" "$ABC300" "$C1"; do
  [[ -f "$p" ]] || { echo "STOP|missing_input:$p"; exit 2; }
done

export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1

[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  echo "STOP|B24_worktree_dirty"; git -C "$REPO" status --short; exit 3;
}
git -C "$CONTROL" fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
LOCAL=$(git -C "$REPO" rev-parse HEAD)
REMOTE=$(git -C "$REPO" rev-parse "origin/$BRANCH")
git -C "$REPO" merge-base --is-ancestor "$LOCAL" "$REMOTE" || {
  echo "STOP|local_not_ancestor|local=$LOCAL|remote=$REMOTE"; exit 4;
}
if [[ "$LOCAL" != "$REMOTE" ]]; then git -C "$REPO" merge --ff-only "origin/$BRANCH"; fi
HEAD=$(git -C "$REPO" rev-parse HEAD)

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m py_compile scripts/b24/freeze_b24_method_roles.py

OUT="$FREEZE/method_stage"
if [[ -e "$OUT/B24_METHOD_ROLE_FREEZE_SUMMARY.json" ]]; then
  echo "STOP|method_role_freeze_already_exists:$OUT/B24_METHOD_ROLE_FREEZE_SUMMARY.json"
  exit 5
fi
mkdir -p "$OUT"

"$PY" scripts/b24/freeze_b24_method_roles.py \
  --universe-csv "$UNIVERSE" \
  --abc300-csv "$ABC300" \
  --c1-csv "$C1" \
  --out-dir "$OUT" \
  | tee "$OUT/freeze_method_roles.log"

FILES=(
  B24_METHOD_IMAGE_ROLES.csv
  B24_METHOD_IMAGE_ROLES.json
  B24_METHOD_PILOT16.csv
  B24_METHOD_ROLE_FREEZE_SUMMARY.json
)
for f in "${FILES[@]}"; do
  [[ -f "$OUT/$f" ]] || { echo "STOP|missing_output:$OUT/$f"; exit 6; }
done
(
  cd "$OUT"
  sha256sum "${FILES[@]}" > SHA256SUMS.txt
)
printf '%s\n' "$OUT" > "$OUTROOT/B24_METHOD_STAGE_LATEST_FREEZE.txt"

echo "B24_METHOD_ROLE_FREEZE_PASS|out=$OUT|head=$HEAD|gpu_work=0"
echo "RETURN_FILES|$OUT/B24_METHOD_ROLE_FREEZE_SUMMARY.json|$OUT/B24_METHOD_PILOT16.csv|$OUT/SHA256SUMS.txt"
