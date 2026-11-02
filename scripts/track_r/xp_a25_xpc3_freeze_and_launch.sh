#!/usr/bin/env bash
# A20: freeze both exact scorer code AND all 13,600 source rows, then push,
# and ONLY THEN spawn a separate detached GPU controller.
# The controller blocks on the shared XP-I2 project lock, not its GPU workers.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: bash $0 BASE_HEAD BRANCH" >&2;exit 8; }
BASE_HEAD="$1"; BRANCH="$2"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
export PYTHONPATH="$WT"
export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
cd "$WT"
test "$(git rev-parse HEAD)" = "$BASE_HEAD"
test -z "$(git status --porcelain)"
test "$(git branch --show-current)" = ""
test "$(git rev-parse "origin/$BRANCH")" = "$BASE_HEAD"

"$PY" -m unittest discover -s tests -p 'test_track_r_xp_a25_xpc3_a20.py' -v
"$PY" scripts/track_r/xp_a25_xpc3_source_freeze_cpu.py \
  --expected-head "$BASE_HEAD"

"$PY" - <<'PY'
import hashlib,json
from pathlib import Path
root=Path.cwd()
source=root/"docs/track_r/executors/xp/theory_inputs/XP_C3_A20_SOURCE_MANIFEST.json"
code=root/"scripts/track_r/xp_a25_xpc3_mid_gpu.py"
original=root/"prdiffusion/track_r/prior_score.py"
dest=root/"docs/track_r/executors/xp/XP_C3_A20_CODE_FREEZE.json"
assert source.is_file() and code.is_file() and not dest.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads(source.read_text())
assert (manifest["status"]=="A20_XPC3_ALL13600_SOURCE_FROZEN_UNSCORED"
        and manifest["n_candidates"]==13600
        and manifest["panel_order"]==["XP_N","G5","XP_D4"]
        and manifest["checkpoint_sha256"]==
         "a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1")
freeze=dict(
 status="A20_XPC3_OPTION_B_S_MID_GLOBAL2_ONLY",
 planner_commit="a6ea8cc",
 prior_definition="S_mid global2 only; A20.3",
 code_stage="BEFORE_FIRST_XPC3_CANDIDATE_SCORE",
 source_manifest_path=str(source.relative_to(root)),
 source_manifest_sha256=sha(source),
 scorer_script=str(code.relative_to(root)),
 scorer_script_sha256=sha(code),
 original_prior_score_sha256=sha(original),
 uid_rule="XPC3|<panel>|<image_id>|<run>",
 seed_rule="prior_seed(uid,t,j)",
 checkpoint_path="/egr/research-pac/huang248/models/imagenet256.pt",
 checkpoint_sha256=manifest["checkpoint_sha256"],
 timesteps=[300,400,500,600],
 draws=[0,1,2,3],
 orientations=["000","111"],
 image_and_noise_pairing=True,
 alpha_schedule="frozen_alpha_schedule()",
 prior_margin="L000-L111",
 decision_threshold_exclusive=1e-7,
 tie_choice="000",
 n_candidates=13600,
 panel_order=manifest["panel_order"],
 sizes=manifest["sizes"],
 budget_cap_gpu_h=8.,
 project_cap_gpu_h=400.,
 no_2forward_secondary=True,
 science_fixed_before_scoring=True)
dest.write_text(json.dumps(freeze,indent=2,sort_keys=True,allow_nan=False)+"\n")
print("A20_XPC3_CODE_HASH",freeze["scorer_script_sha256"])
print("A20_XPC3_SOURCE_HASH",freeze["source_manifest_sha256"])
print("A20_XPC3_FREEZE_STAGED",str(dest))
PY
git add -- docs/track_r/executors/xp/theory_inputs/XP_C3_A20_SOURCE_MANIFEST.json \
           docs/track_r/executors/xp/XP_C3_A20_CODE_FREEZE.json
test "$(git diff --cached --name-only | wc -l)" -eq 2
git commit -m "freeze(track-r): A20 S_mid global2 exact code and 13600-source UID/SHA manifest"
FROZEN_HEAD="$(git rev-parse HEAD)"
git push origin "HEAD:refs/heads/$BRANCH"
test "$(git rev-parse HEAD)" = "$FROZEN_HEAD"
test -z "$(git status --porcelain)"
git fetch --no-tags origin "$BRANCH"
test "$(git rev-parse FETCH_HEAD)" = "$FROZEN_HEAD"
RUN="$BASE/xp_a25_xpc3_prod_$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$BASE/XP_C3_A20_SUPERVISOR_$(basename "$RUN").log"

# Only now has the code/source SHA freeze been pushed. This controller
# will defer any GPU score until XP-I2 releases the global accounting lock.
nohup env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$WT" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    "$PY" -u scripts/track_r/xp_a25_xpc3_autopilot.py \
    --expected-head "$FROZEN_HEAD" --run "$RUN" --branch "$BRANCH" \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "A20_XPC3_FROZEN_COMMIT=$FROZEN_HEAD"
echo "A20_XPC3_SUPERVISOR_PID=$PID"
echo "A20_XPC3_RUN=$RUN"
echo "A20_XPC3_LOG=$LOG"
echo "A20_XPC3_EXECUTION_ONLY_AFTER_PUSHED_SOURCE_AND_CODE_FREEZE"
echo "A20_XPC3_XPI2_PINNED_WORKTREE_UNTOUCHED"
