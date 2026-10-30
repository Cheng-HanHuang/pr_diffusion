#!/usr/bin/env bash
# CPU-only CP3 continuation after independent CP2 audit and local commit
# have been published to GitHub. Never reruns CP2 and never publishes CP3.
set -euo pipefail
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$ROOT/pr_diffusion_track_r_xp_a13"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
HEAD="$1"
STAMP="$2"
CP2_BRANCH="codex/prdiff-track-r-xp-a15-cp2-20261030T185023Z"
PUB="$ROOT/pr_diffusion_track_r_cp2_publish_20261030T185023Z"
test "$CUDA_VISIBLE_DEVICES" = ""
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -d "$PUB"
test "$(git -C "$PUB" symbolic-ref --short HEAD)" = "$CP2_BRANCH"
LOCAL_CP2=$(git -C "$PUB" rev-parse HEAD)
case "$LOCAL_CP2" in ef1cf10*) ;; *) echo STOP_CP2_WRONG_LOCAL_COMMIT;exit 8;; esac
REMOTE_CP2=$(git -C "$PUB" ls-remote origin "refs/heads/$CP2_BRANCH" | cut -f1)
test "$LOCAL_CP2" = "$REMOTE_CP2" || { echo STOP_CP2_NOT_PUSHED_EXACT;exit 8; }
AUDIT="$PUB/docs/track_r/executors/xp/returns/XP_A15_CP2_20261030T185023Z/XP_A15_CP2_INDEPENDENT_AUDIT.json"
"$PY" - "$AUDIT" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p["status"]=="XP_A15_CP2_INDEPENDENT_AUDIT_PASS"
assert p["CP2_ready_for_sanitized_commit"] is True
assert p["n_candidates"]==6400
print("A15_CP2_REMOTE_CHECKPOINT_PUBLISHED_AND_AUDITED")
PY
cd "$WT"
G202="$BASE/xp_a15_cp3_G2_0p02_$STAMP"
G210="$BASE/xp_a15_cp3_G2_0p10_$STAMP"
G4="$BASE/xp_a15_cp3_G4_$STAMP"
G5="$BASE/xp_a15_cp3_G5_$STAMP"
G3="$BASE/xp_a15_cp3_G3_$STAMP"
echo "===== CP3 ONLY (CP2 PUBLISHED, NEVER RESCORED) ====="
for spec in "G2_0.02:$G202" "G2_0.10:$G210" "G4:$G4" "G5:$G5" "G3:$G3"; do
 IFS=: read -r stage out <<< "$spec"
 echo "===== CP3 FROZEN NATIVE PREFLIGHT $stage ====="
 "$PY" scripts/track_r/xp_a15_cp3_score_cpu.py \
    --expected-head "$HEAD" --stage "$stage" --out "$out" --preflight-only
 echo "===== CP3 FROZEN NATIVE SCORE + CLIP $stage ====="
 "$PY" scripts/track_r/xp_a15_cp3_score_cpu.py \
    --expected-head "$HEAD" --stage "$stage" --out "$out"
done
OUT="$BASE/xp_a15_cp3_protocol_$STAMP"
echo "===== CP3 FROZEN PROTOCOL PREFLIGHT ====="
"$PY" scripts/track_r/xp_a15_cp3_protocol_cpu.py \
  --expected-head "$HEAD" --g2-02 "$G202" --g2-10 "$G210" \
  --g4 "$G4" --g5 "$G5" --g3 "$G3" --out "$OUT" --preflight-only
echo "===== CP3 FROZEN PROTOCOL ====="
"$PY" scripts/track_r/xp_a15_cp3_protocol_cpu.py \
  --expected-head "$HEAD" --g2-02 "$G202" --g2-10 "$G210" \
  --g4 "$G4" --g5 "$G5" --g3 "$G3" --out "$OUT"
echo "CP3_PROVISIONAL_CPU_ANALYSIS_COMPLETE__DO_NOT_RETURN_OR_PUBLISH_CP3"
echo "CP3_LOCAL_DIR=$OUT"
