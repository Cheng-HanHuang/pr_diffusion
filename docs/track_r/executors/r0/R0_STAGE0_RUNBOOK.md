# R0.0 / R0.0b / R0.1 operator runbook (2026-10-21)

**Status for Cheng:** The R0.0 bounded inventory, optional isolated environment/assets
bootstrap and the initial R0.1 scientific core plus CPU tests are committed. No
PAC command has run, no GPU work is authorized by this runbook, and no protected
cohort has been inspected. Source-specific R0.2 manifest generation must wait for
the inventory's verified PAC paths and native candidate filenames.

Planner base: `8f6302ff9ea1ea16e5849f751a7706193054d9a7`.
Executor branch: `codex/prdiff-track-r-r0-20261021`.
Set `EXPECTED` below to the latest **pushed executor commit**, supplied
in the chat handoff. Do not use the planner base for PAC execution.

## R0.0: one consolidated CPU inventory block

```bash
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
BRANCH=codex/prdiff-track-r-r0-20261021
EXPECTED=REPLACE_WITH_PUSHED_EXECUTOR_SHA
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOGROOT="$ROOT/outputs/pr_diffusion/track_r/logs"
PREP_LOG="$LOGROOT/r0_0_checkout_$STAMP.log"
mkdir -p "$LOGROOT"
r0_prepare_and_inventory() {
  if [ ! -e "$WT" ]; then
    git clone --branch "$BRANCH" https://github.com/Cheng-HanHuang/pr_diffusion "$WT"
    RC=$?
    if [ "$RC" -ne 0 ]; then echo "STOP_CLONE_RC=$RC"; return "$RC"; fi
  elif [ ! -d "$WT/.git" ]; then
    echo "STOP_NOT_A_DEDICATED_CHECKOUT"; return 3
  else
    if [ -n "$(git -C "$WT" status --porcelain)" ]; then
      echo "STOP_DIRTY_CHECKOUT"; return 4
    fi
  fi
  git -C "$WT" fetch origin "$BRANCH"
  RC=$?
  if [ "$RC" -ne 0 ]; then echo "STOP_FETCH_RC=$RC"; return "$RC"; fi
  git -C "$WT" checkout --detach "$EXPECTED"
  RC=$?
  if [ "$RC" -ne 0 ]; then echo "STOP_DETACH_RC=$RC"; return "$RC"; fi
  echo "HEAD=$(git -C "$WT" rev-parse HEAD)"
  echo "TREE_STATUS_BEGIN"
  git -C "$WT" status --porcelain
  echo "TREE_STATUS_END"
  if [ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED" ] || \
     [ -n "$(git -C "$WT" status --porcelain)" ]; then
    echo "STOP_HEAD_OR_TREE_MISMATCH"; return 5
  fi
  bash "$WT/scripts/track_r/r0_0_inventory.sh" "$EXPECTED"
  return $?
}
r0_prepare_and_inventory > "$PREP_LOG" 2>&1
RC=$?
echo "R0_0_PREP_RC=$RC PREP_LOG=$PREP_LOG"
tail -n 32 "$PREP_LOG"
```

Return **only** the `R0_0_INVENTORY.json` and the last 30–40 log lines (or
a metadata-only zip, no model weights). Do not proceed through a mismatch.
The inventory reports model hashes, source roots, package versions and
possible missing assets; it never recursively traverses large datasets.

## R0.0b: environment/assets, opt-in AFTER reviewing R0.0

Only if R0.0 identified missing pieces **and Cheng explicitly confirms**
this step. This may download pinned external dependencies. Keep the checkout
at the same clean pushed commit. This script never modifies legacy envs.

```bash
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
EXPECTED=REPLACE_WITH_PUSHED_EXECUTOR_SHA
bash "$WT/scripts/track_r/r0_0b_create_env_and_assets.sh" "$EXPECTED"
echo "R0_0B_RC=$?"
```

If the pinned UnivFD `fc_weights.pth` is unavailable, R0.0b stops rather
than fetching an unverified weight URL. Return the full logged failure for
planner escalation. No script should download to the user home directory.

## R0.1 CPU-only tests (after the target environment exists)

```bash
ROOT=/egr/research-pac/huang248
WT="$ROOT/pr_diffusion_track_r"
EXPECTED=REPLACE_WITH_PUSHED_EXECUTOR_SHA
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$ROOT/outputs/pr_diffusion/track_r/logs/r0_1_cpu_tests_$STAMP.log"
mkdir -p "$(dirname "$LOG")"
r0_cpu_tests() {
  echo "HEAD=$(git -C "$WT" rev-parse HEAD)"
  echo "TREE_STATUS_BEGIN"
  git -C "$WT" status --porcelain
  echo "TREE_STATUS_END"
  if [ "$(git -C "$WT" rev-parse HEAD)" != "$EXPECTED" ] ||
     [ -n "$(git -C "$WT" status --porcelain)" ]; then
    echo "STOP_HEAD_OR_TREE_MISMATCH"; return 3
  fi
  if [ ! -x "$PY" ]; then echo "STOP_TARGET_PYTHON_MISSING"; return 4; fi
  cd "$WT" || return 5
  export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
  export XDG_CACHE_HOME="$ROOT/outputs/.cache/track_r"
  export TORCH_HOME="$ROOT/outputs/.cache/track_r/torch"
  export PIP_CACHE_DIR="$ROOT/outputs/.cache/track_r/pip"
  export HF_HOME="$ROOT/outputs/.cache/track_r/hf"
  "$PY" -m pytest -q tests/track_r
  return $?
}
r0_cpu_tests > "$LOG" 2>&1
RC=$?
echo "R0_1_CPU_RC=$RC LOG=$LOG"
tail -n 35 "$LOG"
```

## Acceptance boundary

R0.0 is an inventory, not G0. R0.1 CPU tests do not authorize any GPU work.
The native candidate layout mapping, R0.2 G0 replay and later GPU gates are
separate. S5 generation and scored results cannot be launched until G0/G1/smoke
pass and aggregate GPU-hour projection is no greater than 16.
