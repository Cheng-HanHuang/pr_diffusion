#!/usr/bin/env bash
# A8b publication + ONLY preregistered D1 phase-1 K4 CPU quality evaluation
# + C/C2 diagnostic GT input freezing. Does not score sealed K>4.
set -euo pipefail
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BASE="$ROOT/outputs/pr_diffusion/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
BRANCH=codex/prdiff-track-r-xp-20261026
EXPECTED="$1"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"

if [ -z "$EXPECTED" ] ||
   [ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED" ] ||
   [ -n "$(git -C "$WT" status --porcelain)" ]; then
  echo "ACTION NEEDED: A8B_D1_C_CPU_HEAD_OR_DIRTY"
  exit 8
fi

echo "=== CPU-ONLY FOLLOW-UP TESTS ==="
"$PY" -m unittest discover -s "$WT/tests" \
  -p 'test_track_r_xp_cpu_followup.py' -q
"$PY" -m unittest discover -s "$WT/tests" \
  -p 'test_track_r_xp_d1p1_theory_publish.py' -q

echo "=== PUBLISH HISTORICAL A8b NEXT TO A8 ==="
"$PY" "$WT/scripts/track_r/xp_r0n2_a8b_publish_cpu.py" \
  --expected-head "$EXPECTED"
SUBDIR=docs/track_r/executors/r0n/returns/R0N2_A8B_20261027T041156Z
git -C "$WT" add "$SUBDIR"
git -C "$WT" commit -m \
  "data(track-r): publish historical A8b signed rho strings and frozen tau"
git -C "$WT" push origin "$BRANCH"
HEAD="$(git -C "$WT" rev-parse HEAD)"
test -z "$(git -C "$WT" status --porcelain)"
echo "R0N2_A8B_PUBLISHED_COMMIT=$HEAD"

echo "=== D1 PHASE 1 ONLY: ALLOWED GT-CLEAN CPU K4 EVALUATION ==="
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
"$PY" "$WT/scripts/track_r/xp_d1p1_score_cpu.py" \
  --expected-head "$HEAD" \
  --out "$BASE/xp_d1p1_eval_$STAMP"
echo "XP_D1P1_CPU_RESULT_DIR=$BASE/xp_d1p1_eval_$STAMP"

echo "=== CLEAR AND PUBLISH ONLY D1-P1 K4 THEORY STRINGS ==="
"$PY" "$WT/scripts/track_r/xp_d1p1_theory_publish_cpu.py" \
  --expected-head "$HEAD" --eval-dir "$BASE/xp_d1p1_eval_$STAMP"
git -C "$WT" add docs/track_r/executors/xp/theory_inputs
git -C "$WT" commit -m \
  "data(track-r): publish cleared ImageNet D1 phase-1 K4 theory-input strings"
git -C "$WT" push origin "$BRANCH"
HEAD="$(git -C "$WT" rev-parse HEAD)"
test -z "$(git -C "$WT" status --porcelain)"
echo "XP_D1P1_THEORY_INPUT_PUBLISHED_COMMIT=$HEAD"

echo "=== XP-C/C2 CLEAN-GT DIAGNOSTIC INPUT PREPARATION ONLY ==="
"$PY" "$WT/scripts/track_r/xp_c_c2_prepare_cpu.py" \
  --expected-head "$HEAD" \
  --out "$BASE/xp_c_c2_$STAMP"
echo "XP_C_C2_PREP_DIR=$BASE/xp_c_c2_$STAMP"
echo "NO_GPU_WORK_PERFORMED=true"
echo "ONLY_D1_PHASE1_OUTCOMES_EVALUATED=true"
echo "SEALED_K_GT4_EMBARGO=ACTIVE"
echo "A8B_D1P1_C_C2_CPU_WORK_PASS"
