#!/usr/bin/env bash
# A19 XP-I2 self-contained PAC launch (CPU QA, frozen input, then detached GPU).
# Launches nothing if code tests, native model provenance or GPU admission fail.
set -euo pipefail
[[ $# -eq 3 ]] || {
  echo "usage: bash $0 PINNED_EXECUTOR_HEAD FROZEN_MANIFEST_HEAD FROZEN_PREP_DIR" >&2
  exit 8
}
HEAD="$1"
SOURCE_HEAD="$2"
PREP="$3"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$WT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -f "$PREP/XP_A24_XPI2_PREP_RECEIPT.json"
cd "$WT"
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a24_round15.py' -v
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a24_xpi2_prep.py' -v
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a24_i2_native.py' -v
"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a24_i2_autopilot.py' -v

# Only the first A24 production run can start; no duplicate project ledger.
if compgen -G "$BASE/xp_a24_xpi2_prod_*/XP_A24_XPI2_CONTROLLER_STATE.json" >/dev/null; then
  echo "ACTION NEEDED: I2_PRIOR_RUN_MUST_RECONCILE" >&2
  exit 8
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="$BASE/xp_a24_xpi2_prod_$STAMP"
LOG="$BASE/XP_A24_XPI2_SUPERVISOR_$STAMP.log"
"$PY" scripts/track_r/xp_a24_xpi2_autopilot.py \
  --expected-head "$HEAD" \
  --roots-source-head "$SOURCE_HEAD" \
  --prep "$PREP" --run "$RUN" --preflight-only

# Native 2-bag replay is executed first inside the supervisor, then 100 bags.
# The native family has been tested previously in the DAPS environment.
# No clean-GPU requirement; 80% telemetry and per-GPU worker gate are live.
nohup "$PY" -u scripts/track_r/xp_a24_xpi2_autopilot.py \
  --expected-head "$HEAD" \
  --roots-source-head "$SOURCE_HEAD" \
  --prep "$PREP" --run "$RUN" >"$LOG" 2>&1 </dev/null &
PID=$!
echo "A24_XPI2_DETACHED_PID=$PID"
echo "A24_XPI2_RUN=$RUN"
echo "A24_XPI2_LOG=$LOG"
echo "A24_XPI2_LAUNCHED_NO_TIMING_CLAIMS"
echo "A24_XPI2_NEXT_ESCALATION=A13.4_ONLY"
