#!/usr/bin/env bash
# A14 G3 post-clone: CPU native adapter certification, two-id smoke,
# and conditional PC-DAPS production under the shared project/A12 ledger.
# Separate controller; never repeats successful G2/G4/G5.
set -euo pipefail
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$ROOT/pr_diffusion_track_r_xp_a13"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
DAPS_PY="$ROOT/conda-envs/daps/bin/python"
PC="$ROOT/external/PC_DAPS_xp"
HEAD="$1"
[[ "$HEAD" =~ ^[a-f0-9]{40}$ ]] || { echo STOP_EXPECTED_SHA40;exit 8; }
test -x "$PY" && test -x "$DAPS_PY"
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test "$(git -C "$PC" rev-parse HEAD | cut -c1-8)" = 59b2f046
test -z "$(git -C "$PC" status --porcelain --untracked-files=no)"
if test -e "$BASE/XP_A14_G3_LATEST_RUN.txt" ||
  find "$BASE" -maxdepth 1 -type d -name 'xp_a12_g3_prod_*' -print -quit | grep -q .;then
 echo STOP_EXISTING_G3_RUN; exit 8
fi
if pgrep -af '[x]p_a12_campaign_autopilot.py|[x]p_a14_g3_after_clone_autopilot.py|[x]p_a12_g7_c2_autopilot.py|[x]p_a12_g6_clean_autopilot.py';then
 echo STOP_ALREADY_RUNNING_TRACK_R_CONTROLLER;exit 8
fi
PRIOR_RUN=$(cat "$BASE/XP_A13_G2G5_LATEST_RUN.txt")
case "$PRIOR_RUN" in "$BASE"/xp_a12_campaign_prod_*) ;; *) echo STOP_PRIOR_PATH;exit 8;; esac
PRIOR="$PRIOR_RUN/XP_A12_CAMPAIGN_STATE.json"
test -f "$PRIOR"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
cd "$WT"
echo "===== PC-DAPS NATIVE IMPORT CERTIFICATION, CPU ONLY ====="
"$DAPS_PY" - "$PC" <<'PY'
import os,sys
assert os.environ["CUDA_VISIBLE_DEVICES"]==""
sys.path.insert(0,sys.argv[1])
from sampler import get_sampler
from forward_operator import get_operator
assert callable(get_sampler) and callable(get_operator)
print("A14_G3_PC_NATIVE_IMPORT_CERT_PASS")
PY
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PREP="$BASE/xp_a12_g3_prepare_$STAMP"
RUN="$BASE/xp_a12_g3_prod_$STAMP"
LOG="$BASE/XP_A14_G3_$STAMP.log"
echo "===== G3 SIGNED-Y MANIFEST CPU CERTIFICATION ====="
"$PY" scripts/track_r/xp_a12_g3_prepare_cpu.py \
 --expected-head "$HEAD" --out "$PREP"
"$PY" - "$PREP" "$HEAD" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
r=json.loads((p/"XP_A12_G3_PREP_RECEIPT.json").read_text())
assert r["status"]=="XP_A12_G3_CPU_PREP_PASS"
assert r["head"]==sys.argv[2]
assert r["n_bags"]==180 and r["n_particles"]==720
assert r["two_id_smoke_required"] is True and r["cap_gpu_h"]==45
print("A14_G3_CPU_MANIFEST_CERT_PASS")
PY
echo "===== DETACHED G3 NATIVE SMOKE AND CONDITIONAL PRODUCTION ====="
test ! -e "$RUN" && test ! -e "$LOG"
SESSION=prdiff_a14_g3_$STAMP
printf -v CMD '%q ' \
 "$PY" scripts/track_r/xp_a14_g3_after_clone_autopilot.py \
 --expected-head "$HEAD" --prior-campaign "$PRIOR" \
 --prep "$PREP" --run "$RUN"
tmux new-session -d -s "$SESSION" \
 "cd '$WT'; env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$WT' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 $CMD > '$LOG' 2>&1"
printf '%s\n' "$RUN" > "$BASE/XP_A14_G3_LATEST_RUN.txt"
echo "A14_G3_SESSION=$SESSION"
echo "A14_G3_LOG=$LOG"
echo "A14_G3_RUN=$RUN"
echo "A14_G3_AUTOPILOT_LAUNCHED"
