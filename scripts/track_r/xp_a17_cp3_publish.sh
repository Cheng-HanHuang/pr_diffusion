#!/usr/bin/env bash
# A15 CP3 sanitized checkpoint publication. Run only after independent
# A17 descriptive + clean paired timing + staging have passed.
set -euo pipefail
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r_xp_a13"
BASE="$ROOT/outputs/pr_diffusion/track_r"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
HEAD="$1"
STAMP="$2"
SSH_URL=git@github.com:Cheng-HanHuang/pr_diffusion.git
export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$WT"
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=12"
[[ "$HEAD" =~ ^[0-9a-f]{40}$ ]] || { echo STOP_A17_HEAD;exit 8; }
[[ "$STAMP" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || { echo STOP_A17_STAMP;exit 8; }
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
OUT="$BASE/xp_a17_cp3_return_$STAMP"
"$PY" "$WT/scripts/track_r/xp_a17_cp3_prepare_return_cpu.py" \
 --expected-head "$HEAD" --out "$OUT" --preflight-only
"$PY" "$WT/scripts/track_r/xp_a17_cp3_prepare_return_cpu.py" \
 --expected-head "$HEAD" --out "$OUT"
"$PY" - "$OUT/CP3_STAGING_RECEIPT.json" "$HEAD" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x["status"]=="A15_CP3_SANITIZED_RETURN_STAGED_FOR_COMMIT"
assert x["expected_head"]==sys.argv[2]
assert x["A15_authorized"] is True
assert x["no_matched_NFE_claim"] is True
assert x["XP_I_matched_NFE"]=="WITHDRAWN_UNRESOLVED"
print("A17_CP3_RELEASE_PREFLIGHT_A15_PASS")
PY
(cd "$OUT/returns" && sha256sum -c SHA256SUMS)
(cd "$OUT/theory_inputs" && sha256sum -c SHA256SUMS)
BRANCH="codex/prdiff-track-r-xp-a17-cp3-$STAMP"
PUBLISH="$ROOT/pr_diffusion_track_r_cp3_publish_$STAMP"
test ! -e "$PUBLISH"
test -z "$(git -C "$WT" branch --list "$BRANCH")"
# One detached source worktree, one isolated checkpoint commit.
git -C "$WT" worktree add -b "$BRANCH" "$PUBLISH" "$HEAD"
R="docs/track_r/executors/xp/returns/XP_A17_CP3_$STAMP"
T="docs/track_r/executors/xp/theory_inputs/XP_A17_CP3_$STAMP"
mkdir -p "$PUBLISH/$R" "$PUBLISH/$T"
cp -a "$OUT/returns/." "$PUBLISH/$R/"
cp -a "$OUT/theory_inputs/." "$PUBLISH/$T/"
(cd "$PUBLISH/$R" && sha256sum -c SHA256SUMS)
(cd "$PUBLISH/$T" && sha256sum -c SHA256SUMS)
git -C "$PUBLISH" add -- "$R" "$T"
git -C "$PUBLISH" diff --cached --check
git -C "$PUBLISH" commit -m "results(track-r): A15 CP3 confirmatory G2/G4/G5 and descriptive PC-DAPS, corrected 400x10 NFE"
git -C "$PUBLISH" push "$SSH_URL" "HEAD:refs/heads/$BRANCH"
LOCAL=$(git -C "$PUBLISH" rev-parse HEAD)
REMOTE=$(git -C "$PUBLISH" ls-remote "$SSH_URL" "refs/heads/$BRANCH" | cut -f1)
test "$LOCAL" = "$REMOTE"
echo "A17_CP3_REMOTE_CHECKPOINT_PUBLISHED_AND_VERIFIED=$LOCAL"
echo "CP3_BRANCH=$BRANCH"
echo "CP3_DIGEST=$R/CP3_DIGEST.md"
echo "CP3_THEORY_INPUTS=$T"
echo "CP3_UNRESOLVED_EXACT_MATCHED_NFE_WITHDRAWN_PER_A15"
