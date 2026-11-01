#!/usr/bin/env bash
# A17 prospective XP-N v5 end-to-end CPU lane. Run in detached tmux.
# Generation controller runs independently and may generate sealed 4..63.
# This program reads ONLY the 0..3 pilot complete receipts, GT/y and
# native images. A verified SSH push of v5 (or explicit decline) is
# mandatory before any later-run scores are read.
set -euo pipefail
[[ $# == 3 ]] || { echo "usage: bash $0 EXPECTED_HEAD TASKS_DIR XP_N_GENERATION_RUN" >&2; exit 8; }
HEAD="$1";TASKS="$2";GEN="$3"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
CODE=81003cc5a9167def74a04281501b4eae2837f44c
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$BASE/xp_a19_xpn_v5_$STAMP"
P4="$BASE/xp_a19_xpn_prefix4_$STAMP"
PUB="$ROOT/pr_diffusion_track_r_xpn_prefix_pub_$STAMP"
PUBV="$ROOT/pr_diffusion_track_r_xpn_v5_pub_$STAMP"
CODEWT="$ROOT/pr_diffusion_track_r_xpn_v5_code_$STAMP"
PBR="codex/prdiff-track-r-xp-n-prefix4-$STAMP"
VBR="codex/prdiff-track-r-xp-n-v5-$STAMP"
REL="docs/track_r/executors/xp/theory_inputs/XP_N_RUNS_0_3"
VREL="docs/track_r/executors/xp/returns/XP_N_V5_$STAMP"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$WT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
git -C "$WT" rev-parse HEAD | grep -Fx "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test -d "$TASKS" && test -d "$GEN"
test ! -e "$OUT" && test ! -e "$P4" && test ! -e "$PUB" && test ! -e "$PUBV"
mkdir -m 700 "$OUT"
echo "A19_V5_CPU_AUTOPILOT_BEGIN=$OUT"

# No run>=4 manifest, receipt or reconstruction may be inspected here.
PHASE=A18_XPN_PILOT_0_3
while true; do
  if [ -e "$GEN/STOP_XPN" ] || [ -e "$OUT/STOP_V5" ]; then
    echo "A19_V5_OPERATOR_STOP"; exit 8
  fi
  COMPLETE=0
  for start in $(seq 0 20 380); do
    chunk=$(printf '%s/production/%s/chunk_%05d_020/R0_5_CHUNK_COMPLETE.json' "$GEN" "$PHASE" "$start")
    if [ -s "$chunk" ]; then COMPLETE=$((COMPLETE+1)); fi
  done
  if [ "$COMPLETE" = 20 ]; then break; fi
  # Only check a terminal controller status if the 0..3 prefix is
  # incomplete; never open later-run candidate outcomes.
  if [ -e "$GEN/XP_N_FINAL_STATUS.json" ]; then
    echo "A19_V5_PREFIX_INCOMPLETE_AFTER_TERMINAL=$COMPLETE"; exit 8
  fi
  echo "A19_V5_WAIT_PILOT_CHUNKS=$COMPLETE/20"
  sleep 60
done
echo "A19_V5_PILOT_ALL400_NATIVE_CHUNKS_COMPLETE"

"$PY" "$WT/scripts/track_r/xp_a19_xpn_prefix4_cpu.py" \
   --expected-head "$HEAD" --tasks "$TASKS" --run "$GEN" \
   --out "$P4" --preflight-only
"$PY" "$WT/scripts/track_r/xp_a19_xpn_prefix4_cpu.py" \
   --expected-head "$HEAD" --tasks "$TASKS" --run "$GEN" --out "$P4"

PREFIX="$P4/XP_N_RUNS_0_3_A8_STYLE_PER_IMAGE.csv"
CLIP="$P4/XP_N_RUNS_0_3_G1_DECISIONS.csv"
PS=$(sha256sum "$PREFIX" | cut -d' ' -f1)
CS=$(sha256sum "$CLIP" | cut -d' ' -f1)
echo "A19_V5_FROZEN_INPUT_SHA=$PS/$CS"

# First publish only prospective prefix tables, never GT tensors or
# per-candidate native scoring paths.
git -C "$WT" worktree add -b "$PBR" "$PUB" "$HEAD"
mkdir -p "$PUB/$REL"
cp "$PREFIX" "$PUB/$REL/"
cp "$CLIP" "$PUB/$REL/"
cat > "$PUB/$REL/README.md" <<EOF
# XP-N prospective 100-image FAST prefix-4 inputs for theory v5
These data derive exclusively from run indices 0,1,2,3.
GT-derived class labels are evaluation-only. G1 orientation uses the
frozen CLIP model. The 4–63 candidate outcomes were not opened.
PREFIX4 SHA-256: $PS
CLIP4 SHA-256: $CS
EOF
(cd "$PUB/$REL" && sha256sum XP_N_RUNS_0_3_A8_STYLE_PER_IMAGE.csv XP_N_RUNS_0_3_G1_DECISIONS.csv README.md > SHA256SUMS && sha256sum -c SHA256SUMS)
git -C "$PUB" add "$REL"
git -C "$PUB" -c core.whitespace=cr-at-eol diff --cached --check
git -C "$PUB" commit -m "results(track-r): publish XP-N prospective FAST 100x4 prefix and G1 decisions"
GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=10" \
 git -C "$PUB" push git@github.com:Cheng-HanHuang/pr_diffusion.git "HEAD:refs/heads/$PBR"
PLOCAL=$(git -C "$PUB" rev-parse HEAD)
PREMOTE=$(git -C "$PUB" ls-remote git@github.com:Cheng-HanHuang/pr_diffusion.git "refs/heads/$PBR" | cut -f1)
test -n "$PREMOTE" && test "$PLOCAL" = "$PREMOTE"
echo "A19_XPN_PREFIX4_REMOTE_PUBLISHED_AND_VERIFIED=$PLOCAL"

# Detached theory checkout: never modify/push frozen theory branch.
git -C "$WT" fetch --no-tags origin theory/prdiff-track-r-th-20261026
git -C "$WT" cat-file -e "$CODE^{commit}"
git -C "$WT" worktree add --detach "$CODEWT" "$CODE"
THEORY="$CODEWT/scripts/track_r/theory"
(
cd "$CODEWT"
cat <<'EOF' | sha256sum -c -
5ac8811f0948ee143f0b2cd90406626a01bc9421121472fccb241292e2745292  scripts/track_r/theory/fit_th1_imagenet_v5.py
7313d46ddb5df33e2b3dfb7bae8c301e66584b0d503b6da117560e41bde4d33c  scripts/track_r/theory/check_th1_imagenet_v5_contract.py
6f70c8638eda541ce10786b905403a3ac82c2fe2eefef9bfcf25647819ff3f71  scripts/track_r/theory/seal_th1_imagenet_v5.py
e795ff2233f5d57cac6c4ec12649e5e31271c00fb864a74ff2b586088dcaec8e  scripts/track_r/theory/run_th1_imagenet_v5.sh
8fba9cf6ba1cdfdaea2f93933d6168490eec8dd58495e971eeeebd69095c25b6  scripts/track_r/theory/test_th1_imagenet_v5_synthetic.py
EOF
)
test -z "$(git -C "$CODEWT" status --porcelain)"
"$PY" -c "import numpy, scipy; print('A19_V5_NUMPY_SCIPY_READY',numpy.__version__,scipy.__version__)"

DRAFT="$OUT/XP_N_V5_DRAFT.json"
SEALED="$OUT/TH1_PREDICTIONS_v5_XP_N.json"
FITRC=0
(
 cd "$THEORY"
 "$PY" fit_th1_imagenet_v5.py \
  --prefix4 "$PREFIX" --prefix4-sha256 "$PS" \
  --clip4 "$CLIP" --clip4-sha256 "$CS" \
  --output "$DRAFT" --point-reps 250 --bootstrap 60 --boot-reps 3
 "$PY" check_th1_imagenet_v5_contract.py "$DRAFT"
 "$PY" seal_th1_imagenet_v5.py \
  --draft "$DRAFT" --output "$SEALED" \
  --code-commit "$CODE" \
  --prefix4 "$PREFIX" --prefix4-sha256 "$PS" \
  --clip4 "$CLIP" --clip4-sha256 "$CS"
) > "$OUT/XP_N_V5_NATIVE_FIT_CHECK_SEAL.log" 2>&1 || FITRC=$?
cat "$OUT/XP_N_V5_NATIVE_FIT_CHECK_SEAL.log"

# Any fit/check/seal failure: do not change frozen forecast code/seed/
# parameters, and do not score held runs. Publish an explicit decline
# if this native production attempt failed. Input/environment repair
# requires one deliberate retry before running this path again.
if [ "$FITRC" -ne 0 ]; then
  echo "A19_V5_FIT_CHECK_SEAL_FAILED_RC=$FITRC"
  # A17 permits at most one corrective retry for input formatting or
  # Python environment, never model/seed/parameter changes. A repeated
  # identical computation is not a corrective retry. If the native fit
  # fails in this pinned environment, explicitly decline rather than
  # forging a forecast. Errors are preserved for planner review.
  DECLINE="$OUT/XP_N_V5_DECLINED_PRODUCTION_FAILED.json"
  "$PY" - "$DECLINE" "$OUT/XP_N_V5_NATIVE_FIT_CHECK_SEAL.log" "$FITRC" "$PS" "$CS" "$PLOCAL" <<'PY'
import datetime,json,sys
from pathlib import Path
path,log,rc,ps,cs,prefix=sys.argv[1:]
txt=Path(log).read_text(errors="replace")
status=dict(status="XP_N_V5_DECLINED_PRODUCTION_FAILED",
 decline_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 reason="pinned_native_fit_check_seal_failed_no_model_change",
 returncode=int(rc),stderr_stdout_tail=txt[-6000:],
 code_commit="81003cc5a9167def74a04281501b4eae2837f44c",
 prefix4_sha256=ps,clip4_sha256=cs,
 prefix4_remote_commit=prefix,
 no_held_runs_4_to_63_scored_or_read=True,
 code_seed_and_parameter_modifications=0)
Path(path).write_text(json.dumps(status,sort_keys=True,indent=2)+"\n")
PY
  git -C "$WT" worktree add -b "$VBR" "$PUBV" "$HEAD"
  mkdir -p "$PUBV/$VREL"
  cp "$DECLINE" "$PUBV/$VREL/"
  (cd "$PUBV/$VREL" && sha256sum XP_N_V5_DECLINED_PRODUCTION_FAILED.json > SHA256SUMS && sha256sum -c SHA256SUMS)
  git -C "$PUBV" add "$VREL"
  git -C "$PUBV" commit -m "results(track-r): explicitly decline unmodified prospective XP-N v5 production after native fit failure"
  GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=10" \\
    git -C "$PUBV" push git@github.com:Cheng-HanHuang/pr_diffusion.git "HEAD:refs/heads/$VBR"
  VLOCAL=$(git -C "$PUBV" rev-parse HEAD)
  VREMOTE=$(git -C "$PUBV" ls-remote git@github.com:Cheng-HanHuang/pr_diffusion.git "refs/heads/$VBR" | cut -f1)
  test -n "$VREMOTE" && test "$VLOCAL" = "$VREMOTE"
  echo "A19_XPN_V5_DECLINED_REMOTE_PUBLISHED_AND_VERIFIED=$VLOCAL"
  echo "A19_XPN_V5_DECLINED_NO_FORECAST_TEST"
  exit 0
fi
test -s "$DRAFT" && test -s "$SEALED"
F5=$(sha256sum "$SEALED" | cut -d' ' -f1)
"$PY" - "$OUT" "$PS" "$CS" "$CODE" "$F5" "$PLOCAL" <<'PY'
import datetime,json,platform,sys
from pathlib import Path
import numpy,scipy
out=Path(sys.argv[1]);ps,cs,commit,sha,pp=sys.argv[2:]
rec=dict(status="A17_XPN_V5_NATIVE_FIT_CHECK_SEAL_PASS_PENDING_REMOTE_REGISTRATION",
    freeze_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,
    code_commit=commit,code_sha256={
       "fit":"5ac8811f0948ee143f0b2cd90406626a01bc9421121472fccb241292e2745292",
       "check":"7313d46ddb5df33e2b3dfb7bae8c301e66584b0d503b6da117560e41bde4d33c",
       "seal":"6f70c8638eda541ce10786b905403a3ac82c2fe2eefef9bfcf25647819ff3f71",
       "launcher":"e795ff2233f5d57cac6c4ec12649e5e31271c00fb864a74ff2b586088dcaec8e",
       "synthetic_test":"8fba9cf6ba1cdfdaea2f93933d6168490eec8dd58495e971eeeebd69095c25b6"},
    prefix4_sha256=ps,clip4_sha256=cs,
    prefix4_remote_commit=pp,
    sealed_v5_sha256=sha,
    held_runs_4_to_63_read=False,held_runs_4_to_63_scored=False,
    no_model_seed_or_parameter_changes=True,
    point_reps=250,bootstrap=60,boot_reps=3)
(out/"XP_N_V5_SEAL_RECEIPT.json").write_text(json.dumps(rec,indent=2,sort_keys=True)+"\n")
PY
git -C "$WT" worktree add -b "$VBR" "$PUBV" "$HEAD"
mkdir -p "$PUBV/$VREL"
cp "$DRAFT" "$SEALED" "$OUT/XP_N_V5_SEAL_RECEIPT.json" "$PUBV/$VREL/"
(cd "$PUBV/$VREL" && sha256sum XP_N_V5_DRAFT.json TH1_PREDICTIONS_v5_XP_N.json XP_N_V5_SEAL_RECEIPT.json > SHA256SUMS && sha256sum -c SHA256SUMS)
git -C "$PUBV" add "$VREL"
git -C "$PUBV" -c core.whitespace=cr-at-eol diff --cached --check
git -C "$PUBV" commit -m "results(track-r): register prospective XP-N v5 frozen outputs before any held-run scoring"
GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=10" \
 git -C "$PUBV" push git@github.com:Cheng-HanHuang/pr_diffusion.git "HEAD:refs/heads/$VBR"
VLOCAL=$(git -C "$PUBV" rev-parse HEAD)
VREMOTE=$(git -C "$PUBV" ls-remote git@github.com:Cheng-HanHuang/pr_diffusion.git "refs/heads/$VBR" | cut -f1)
test -n "$VREMOTE" && test "$VLOCAL" = "$VREMOTE"
"$PY" - "$OUT" "$VLOCAL" "$F5" "$VBR" <<'PY'
import datetime,json,sys
from pathlib import Path
p=Path(sys.argv[1])/"XP_N_V5_REMOTE_REGISTERED.json"
r=dict(status="A17_XPN_V5_REMOTE_REGISTERED_BEFORE_RUN4_SCORING",
 registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 remote_commit=sys.argv[2],sealed_sha256=sys.argv[3],
 remote_branch=sys.argv[4],can_score_runs_4_to_63=True)
with p.open("x") as f:json.dump(r,f,sort_keys=True,indent=2)
PY
echo "A19_XPN_V5_REMOTE_REGISTERED_AND_VERIFIED=$VLOCAL"
echo "A19_XPN_V5_REMOTE_BRANCH=$VBR"
echo "A19_XPN_V5_OUTPUT_DIR=$OUT"
echo "A19_XPN_V5_HELD_SCORING_MAY_NOW_PROCEED"
