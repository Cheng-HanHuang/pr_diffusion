#!/usr/bin/env bash
# A14: one infrastructure-only retry for the native PC-DAPS YAML LR type.
# Never edit upstream, prior failed native attempts, or frozen measurement prep.
set -euo pipefail
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$ROOT/pr_diffusion_track_r_xp_a13"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
PC="$ROOT/external/PC_DAPS_xp"
PREP="$BASE/xp_a12_g3_prepare_20261030T153655Z"
OLD="$BASE/xp_a12_g3_prod_20261030T160238Z"
HEAD="$1"
[[ "$HEAD" =~ ^[a-f0-9]{40}$ ]] || { echo STOP_EXPECTED_SHA40; exit 8; }
test -x "$PY"
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test "$(git -C "$PC" rev-parse HEAD)" = \
  59b2f0461134ca83bb99d61c648f38cd4a9cd4a0
test -z "$(git -C "$PC" status --porcelain --untracked-files=no)"
test "$(cat "$BASE/XP_A14_G3_RECOVERY_LATEST_RUN.txt")" = "$OLD"
test -f "$OLD/XP_A12_CAMPAIGN_STATE.json"
test ! -e "$BASE/XP_A14_G3_NATIVE_LR_LATEST_RUN.txt"
test "$(sha256sum "$PREP/XP_A12_G3_180_BAGS.json" | awk '{print $1}')" = \
  3ad52d84216ccd57c14d8b59c3a643a9b89f59adfce908dac2f43d4d1639ff84
# Exactly one previous native failure directory is allowed; the first
# admission-only failure never created a directory.
if find "$BASE" -maxdepth 1 -type d -name 'xp_a12_g3_prod_*' \
  ! -path "$OLD" -print -quit | grep -q .;then
 echo STOP_UNACCOUNTED_G3_OUTPUT_EXISTS; exit 8
fi
if pgrep -af '[x]p_a12_campaign_autopilot.py|[x]p_a14_g3_after_clone_autopilot.py|[x]p_a12_g6_clean_autopilot.py|[x]p_a12_g7_c2_autopilot.py';then
 echo STOP_CONCURRENT_TRACK_R_CONTROLLER; exit 8
fi
export CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$WT" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
cd "$WT"
PRIOR_RUN=$(cat "$BASE/XP_A13_G2G5_LATEST_RUN.txt")
case "$PRIOR_RUN" in "$BASE"/xp_a12_campaign_prod_*) ;; *) echo STOP_PRIOR_PATH;exit 8;; esac
PRIOR="$PRIOR_RUN/XP_A12_CAMPAIGN_STATE.json"
test -f "$PRIOR"
echo "===== VERIFY PINNED PC-DAPS YAML SCALAR, CPU ONLY ====="
"$PY" - "$PC" <<'PY'
import sys
import yaml
from pathlib import Path
from scripts.track_r.xp_a14_g3_mcmc_config import normalize_mcmc_config
pc=Path(sys.argv[1])
raw=yaml.safe_load((pc/"configs/task/phase_retrieval.yaml").read_text())
raw=raw["pixel"]["mcmc_sampler_config"]
fixed=normalize_mcmc_config(raw)
assert fixed["lr"]==5e-5 and type(fixed["lr"]) is float
assert {k:v for k,v in fixed.items() if k!="lr"} == \
       {k:v for k,v in raw.items() if k!="lr"}
print("G3_NATIVE_YAML_LR_TYPE_FIXED",type(raw["lr"]).__name__,
      "->",type(fixed["lr"]).__name__,"numerical_value",fixed["lr"])
PY
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$BASE/xp_a12_g3_prod_$STAMP"
LOG="$BASE/XP_A14_G3_NATIVE_LR_RETRY_$STAMP.log"
SESSION="prdiff_a14_g3_native_lr_$STAMP"
test ! -e "$RUN" && test ! -e "$LOG"
echo "===== VERIFY OLD FAILED SMOKE COSTS IN SHARED LEDGER ====="
"$PY" scripts/track_r/xp_a14_g3_after_clone_autopilot.py \
 --expected-head "$HEAD" --prior-campaign "$PRIOR" \
 --prep "$PREP" --run "$RUN" --failed-g3-run "$OLD" --preflight-only
test ! -e "$RUN"
echo "===== LAUNCH ONE BOUNDED NATIVE RETRY ====="
printf -v CMD '%q ' "$PY" \
 scripts/track_r/xp_a14_g3_after_clone_autopilot.py \
 --expected-head "$HEAD" --prior-campaign "$PRIOR" \
 --prep "$PREP" --run "$RUN" --failed-g3-run "$OLD"
tmux new-session -d -s "$SESSION" \
 "cd '$WT'; env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$WT' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 $CMD > '$LOG' 2>&1"
printf '%s\n' "$RUN" > "$BASE/XP_A14_G3_NATIVE_LR_LATEST_RUN.txt"
echo "A14_G3_NATIVE_LR_SESSION=$SESSION"
echo "A14_G3_NATIVE_LR_LOG=$LOG"
echo "A14_G3_NATIVE_LR_RUN=$RUN"
echo "A14_G3_NATIVE_LR_RETRY_LAUNCHED"
