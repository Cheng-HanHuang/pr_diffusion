#!/usr/bin/env bash
# Runs inside a single detached tmux. Return only at CP2/CP3 checkpoints.
set -euo pipefail
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
WT="$ROOT/pr_diffusion_track_r_xp_a13"
HEAD="$1"
STAMP="$2"
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test "$CUDA_VISIBLE_DEVICES" = ""
cd "$WT"
CP2="$BASE/xp_a15_cp2_d4_$STAMP"
echo "===== CP2: CHECK V4 REGISTRY AND 4800 NATIVE D4 RECEIPTS ====="
"$PY" scripts/track_r/xp_a15_cp2_d4_cpu.py \
 --expected-head "$HEAD" --out "$CP2" --preflight-only
echo "===== CP2: SCORE FULL FAST IMAGENET 100x64, G1 + FROZEN PROTOCOL ====="
"$PY" scripts/track_r/xp_a15_cp2_d4_cpu.py \
 --expected-head "$HEAD" --out "$CP2"
echo "===== CP2: INDEPENDENT U-STATISTIC AND PROTOCOL REPLAY AUDIT ====="
"$PY" scripts/track_r/xp_a15_cp2_audit_cpu.py \
 --expected-head "$HEAD" --score-dir "$CP2"
echo "===== CP2: COMMIT SANITIZED CHECKPOINT ONLY AFTER AUDIT PASS ====="
bash scripts/track_r/xp_a15_cp2_publish.sh "$HEAD" "$CP2" "$STAMP"
echo "CHECKPOINT_CP2_READY__PUBLISHED_NOT_YET_RETURNED_TO_PLANNER"
echo "===== CP3: FIVE INDEPENDENT FROZEN-NATIVE PANELS ====="
G202="$BASE/xp_a15_cp3_G2_0p02_$STAMP"
G210="$BASE/xp_a15_cp3_G2_0p10_$STAMP"
G4="$BASE/xp_a15_cp3_G4_$STAMP"
G5="$BASE/xp_a15_cp3_G5_$STAMP"
G3="$BASE/xp_a15_cp3_G3_$STAMP"
for spec in "G2_0.02:$G202" "G2_0.10:$G210" "G4:$G4" "G5:$G5" "G3:$G3";do
 IFS=: read -r stage out <<< "$spec"
 echo "===== CP3 NATIVE STAGE PREFLIGHT: $stage ====="
 "$PY" scripts/track_r/xp_a15_cp3_score_cpu.py \
  --expected-head "$HEAD" --stage "$stage" --out "$out" --preflight-only
 echo "===== CP3 NATIVE STAGE SCORE+ZERO-SHOT CLIP: $stage ====="
 "$PY" scripts/track_r/xp_a15_cp3_score_cpu.py \
  --expected-head "$HEAD" --stage "$stage" --out "$out"
done
CP3="$BASE/xp_a15_cp3_protocol_$STAMP"
echo "===== CP3 FROZEN PROTOCOL ALL PANELS ====="
"$PY" scripts/track_r/xp_a15_cp3_protocol_cpu.py \
 --expected-head "$HEAD" --g2-02 "$G202" --g2-10 "$G210" \
 --g4 "$G4" --g5 "$G5" --g3 "$G3" --out "$CP3" --preflight-only
"$PY" scripts/track_r/xp_a15_cp3_protocol_cpu.py \
 --expected-head "$HEAD" --g2-02 "$G202" --g2-10 "$G210" \
 --g4 "$G4" --g5 "$G5" --g3 "$G3" --out "$CP3"
echo "CP3_EVALUATED_BUT_NOT_READY_FOR_RELEASE__NFE_MATCH_AND_INDEPENDENT_AUDIT_PENDING"
echo "CP3_LOCAL_DIR=$CP3"
echo "A15_CHECKPOINT_ONLY_CPU_PIPELINE_TERMINAL"
