#!/usr/bin/env bash
# A17 v5 prospective forecasting after resumed native PREFIX4 outputs.
# Never inspect candidates or scoring of the held indices 4..63.
set -euo pipefail
[[ $# -eq 4 ]] || {
 echo "usage: bash $0 RECOVERY_HEAD ORIGINAL_PROD RECOVERY_RUN TASKS" >&2
 exit 8
}
HEAD="$1";GEN="$2";RECOVERY="$3";TASKS="$4"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PHASE=A18_XPN_PILOT_0_3
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test "$GEN" = "$BASE/xp_a18_xpn_prod_20261031T225748Z"
test -d "$RECOVERY" && test -d "$TASKS"
echo "A20_V5_WAIT_UNTIL_ALL400_PREFIX_NATIVE_RECEIPTS_EXIST"
while true; do
 if [ -e "$RECOVERY/STOP_XPN_RECOVERY" ]; then
  echo "A20_V5_OPERATOR_STOP";exit 8
 fi
 count=0
 for start in $(seq 0 20 380); do
  chunk=$(printf '%s/production/%s/chunk_%05d_020/R0_5_CHUNK_COMPLETE.json' "$GEN" "$PHASE" "$start")
  if [ -s "$chunk" ];then count=$((count+1));fi
 done
 if [ "$count" -eq 20 ];then
   echo "A20_V5_ALL400_PREFIX_NATIVE_CHUNKS_EXIST"
   break
 fi
 if [ -f "$RECOVERY/XP_N_RECOVERY_FINAL_STATUS.json" ];then
   echo "A20_V5_RECOVERY_TERMINATED_WITHOUT_FULL_PREFIX_$count"
   exit 8
 fi
 echo "A20_V5_PREFIX_CHUNKS=$count/20_WAITING"
 sleep 60
done
# Now the original A19 helper's pilot-only GT/hash check is legitimate.
# On any native input failure stop without unsealing held candidates.
bash "$WT/scripts/track_r/xp_a19_xpn_v5_autopilot.sh" "$HEAD" "$TASKS" "$GEN"
