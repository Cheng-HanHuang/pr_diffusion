#!/usr/bin/env bash
# Only checkpoint-gated sanitized CP2 return; no per-candidate GT metrics.
set -euo pipefail
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r_xp_a13"
BASE="$ROOT/outputs/pr_diffusion/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
HEAD="$1"
SCORE="$2"
STAMP="$3"
[[ "$HEAD" =~ ^[a-f0-9]{40}$ ]] || { echo STOP_CP2_PUBLISH_HEAD;exit 8; }
[[ "$STAMP" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || { echo STOP_CP2_PUBLISH_STAMP;exit 8; }
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -f "$SCORE/XP_A15_CP2_INDEPENDENT_AUDIT.json"
"$PY" - "$SCORE" "$HEAD" <<'PY'
import json,sys,hashlib
from pathlib import Path
p=Path(sys.argv[1])
a=json.loads((p/"XP_A15_CP2_INDEPENDENT_AUDIT.json").read_text())
s=json.loads((p/"XP_A15_CP2_D4_SCORING_RECEIPT.json").read_text())
assert a["status"]=="XP_A15_CP2_INDEPENDENT_AUDIT_PASS"
assert a["CP2_ready_for_sanitized_commit"] is True
assert a["expected_head"]==s["source_head"]==sys.argv[2]
assert s["status"]=="XP_A15_CP2_D4_SCORED_READY_FOR_INDEPENDENT_AUDIT"
assert s["n_candidates"]==6400 and s["n_native_new"]==4800
h=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
assert h(p/"XP_A15_CP2_D4_SCORING_RECEIPT.json")==a["source_receipt_sha256"]
for name,d in s["csv_sha256"].items():assert h(p/name)==d
print("CP2_SANITIZED_PUBLISH_GATE_PASS")
PY
PUBLISH="$ROOT/pr_diffusion_track_r_cp2_publish_$STAMP"
BRANCH="codex/prdiff-track-r-xp-a15-cp2-$STAMP"
test ! -e "$PUBLISH"
test -z "$(git -C "$WT" branch --list "$BRANCH")"
git -C "$WT" worktree add -b "$BRANCH" "$PUBLISH" "$HEAD"
DEST="$PUBLISH/docs/track_r/executors/xp/returns/XP_A15_CP2_$STAMP"
mkdir -p "$DEST"
for file in \
 D4_64RUN_STRINGS.csv \
 D4_64RUN_G1_CLIP.csv \
 D4_PROTOCOL_N32.json \
 D4_PROTOCOL_N64.json \
 CP2_V2_V2E1_V4_FORECAST_COMPARISON.json \
 XP_A15_CP2_D4_SCORING_RECEIPT.json \
 XP_A15_CP2_INDEPENDENT_AUDIT.json; do
 test -f "$SCORE/$file"
 cp "$SCORE/$file" "$DEST/$file"
done
(
 cd "$DEST"
 sha256sum D4_64RUN_STRINGS.csv D4_64RUN_G1_CLIP.csv \
 D4_PROTOCOL_N32.json D4_PROTOCOL_N64.json \
 CP2_V2_V2E1_V4_FORECAST_COMPARISON.json \
 XP_A15_CP2_D4_SCORING_RECEIPT.json \
 XP_A15_CP2_INDEPENDENT_AUDIT.json > SHA256SUMS
 sha256sum -c SHA256SUMS
)
cat > "$DEST/README.md" <<'EOF'
# CP2: XP-D4 native FAST ImageNet 64-run checkpoint
Registered v4 was checked before scoring. This checkpoint contains:
- 100 images x 64-run label/residual strings and the frozen G1 decisions.
- Exact U-statistics vs v2, v2-E1 and registered v4 predictions.
- Frozen c598176 protocol n=32 (primary) and n=64 (secondary).
- Independent numeric and protocol-replay audit.
Candidate-level GT scores, image paths, images and tensors remain local on PAC.
CP3 is a separate checkpoint and is not covered by this return.
EOF
git -C "$PUBLISH" add -- "$DEST"
git -c core.whitespace=trailing-space,space-before-tab,cr-at-eol \
  -C "$PUBLISH" diff --cached --check
git -C "$PUBLISH" commit -m "results(track-r): audited CP2 FAST ImageNet D4 n32/n64 checkpoint"
git -C "$PUBLISH" push -u origin "$BRANCH"
echo "A15_CP2_COMMITTED_AND_PUSHED"
git -C "$PUBLISH" rev-parse HEAD
echo "CP2_BRANCH=$BRANCH"
echo "CP2_DEST=$DEST"
