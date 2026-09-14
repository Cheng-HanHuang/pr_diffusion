#!/usr/bin/env bash
set -euo pipefail

ROOT=/egr/research-pac/huang248
CONTROL="$ROOT/pr_diffusion_b23"
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
BRANCH=codex/b24-bestof4-failure-sweep
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
LATEST_FREEZE="$OUTROOT/B24_ABC300_LATEST_FREEZE.txt"

[[ -x "$PY" ]] || { echo "STOP|missing_python:$PY"; exit 2; }
[[ -f "$LATEST_FREEZE" ]] || { echo "STOP|missing_abc300_freeze_pointer:$LATEST_FREEZE"; exit 2; }
FREEZE=$(cat "$LATEST_FREEZE")
[[ -d "$FREEZE" ]] || { echo "STOP|missing_freeze_dir:$FREEZE"; exit 2; }

UNIVERSE="$FREEZE/B24_7424_CASE_UNIVERSE.csv"
UNIVERSE_SUM="$FREEZE/B24_7424_CASE_FREEZE_SUMMARY.json"
ABC300="$FREEZE/B24_ABC300_FROZEN.csv"
ABC300_SUM="$FREEZE/B24_ABC300_FREEZE_SUMMARY.json"
for p in "$UNIVERSE" "$UNIVERSE_SUM" "$ABC300" "$ABC300_SUM"; do
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
if [[ "$LOCAL" != "$REMOTE" ]]; then
  git -C "$REPO" merge --ff-only "origin/$BRANCH"
fi
HEAD=$(git -C "$REPO" rev-parse HEAD)

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m py_compile scripts/b24/freeze_b24_c1_sitcom_worst100.py

echo "C1_FREEZE_READY|head=$HEAD|gpu_work=0|source_freeze=$FREEZE"

OUT="$FREEZE/c1_sitcom_worst100"
if [[ -e "$OUT/B24_C1_SITCOM_WORST100_FREEZE_SUMMARY.json" ]]; then
  echo "STOP|C1_freeze_already_exists:$OUT/B24_C1_SITCOM_WORST100_FREEZE_SUMMARY.json"
  exit 5
fi
mkdir -p "$OUT"

"$PY" scripts/b24/freeze_b24_c1_sitcom_worst100.py \
  --universe-csv "$UNIVERSE" \
  --universe-summary "$UNIVERSE_SUM" \
  --abc300-csv "$ABC300" \
  --abc300-summary "$ABC300_SUM" \
  --out-dir "$OUT" \
  | tee "$OUT/freeze_c1.log"

FILES=(
  B24_C1_SITCOM_WORST100_FROZEN.csv
  B24_C1_SITCOM_WORST100_FROZEN.json
  B24_C1_SITCOM_WORST100_FREEZE_SUMMARY.json
)
for f in "${FILES[@]}"; do
  [[ -f "$OUT/$f" ]] || { echo "STOP|missing_output:$OUT/$f"; exit 6; }
done
(
  cd "$OUT"
  sha256sum "${FILES[@]}" > SHA256SUMS.txt
)
printf '%s\n' "$OUT" > "$OUTROOT/B24_C1_SITCOM_WORST100_LATEST_FREEZE.txt"

echo "B24_C1_SITCOM_WORST100_FREEZE_PASS|out=$OUT|head=$HEAD"
echo "RETURN_FILES|$OUT/B24_C1_SITCOM_WORST100_FREEZE_SUMMARY.json|$OUT/B24_C1_SITCOM_WORST100_FROZEN.csv|$OUT/SHA256SUMS.txt"
