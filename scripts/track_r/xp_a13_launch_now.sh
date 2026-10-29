#!/usr/bin/env bash
# A13 one PAC command: pinned G1b now; B16 protocol/F8 and G2-G5
# preparation concurrently, then automatically start G2-G5 after G1b.
# Stages G6/G7/T1/T2 remain separate conditional controllers.
set -euo pipefail
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$ROOT/pr_diffusion_track_r_xp_a13"
SOURCE="$ROOT/pr_diffusion_track_r_xp_a12_g1a"
BRANCH=codex/prdiff-track-r-xp-a12-campaign-autopilot-20261029
HEAD="$1"
[[ "$HEAD" =~ ^[a-f0-9]{40}$ ]] || { echo STOP_EXPECTED_SHA;exit 8; }
test -d "$SOURCE"
test ! -e "$WT"
git -C "$SOURCE" fetch --no-tags origin "$BRANCH"
test "$(git -C "$SOURCE" rev-parse FETCH_HEAD)" = "$HEAD"
git -C "$SOURCE" worktree add --detach "$WT" "$HEAD"
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
git -C "$WT" fetch --no-tags origin \
 claude/prdiff-track-r-planner-20261021 \
 theory/prdiff-track-r-th-20261026
git -C "$WT" cat-file -e "c598176:docs/track_r/protocol/gtfree_protocol.py"
test -z "$(git -C "$WT" status --porcelain)"
echo '===== A13 G1B AND B16 FROZEN PROTOCOL PREPARE + DETACHED LAUNCH ====='
bash "$WT/scripts/track_r/xp_a13_cp1_g1b_start.sh" "$HEAD"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
echo '===== DETACHED F8 CPU ANALYSIS: scoring, CLIP, v1/v3, protocol ====='
F8LOG="$BASE/XP_A13_CP1_F8_$STAMP.log"
test ! -e "$F8LOG"
tmux new-session -d -s "prdiff_a13_f8_$STAMP" \
 "bash '$WT/scripts/track_r/xp_a13_cp1_f8_cpu.sh' '$HEAD' > '$F8LOG' 2>&1"
echo '===== DETACHED G2G5 INPUTS WHILE G1B RUNS; AUTOMATIC NEXT STAGE ====='
QUEUELOG="$BASE/XP_A13_G2G5_PREP_$STAMP.log"
test ! -e "$QUEUELOG"
tmux new-session -d -s "prdiff_a13_queueprep_$STAMP" \
 "bash '$WT/scripts/track_r/xp_a13_after_g1b_prepare.sh' '$HEAD' > '$QUEUELOG' 2>&1"
echo "A13_F8_CPU_LOG=$F8LOG"
echo "A13_G2G5_PREP_LOG=$QUEUELOG"
echo "A13_SOURCE_HEAD=$HEAD"
echo 'A13_G1B_B16_F8_G2G5_CONCURRENT_LAUNCH_COMPLETE'
