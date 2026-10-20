#!/usr/bin/env bash
# A1.3.0: run only AFTER separate read-only PAC inspection confirms ownership.
# Invocation: bash scripts/track_a/a13_run_inventory_v1.sh <exact-pushed-A13-HEAD>
# Never use set -e, set -u, pipefail, logout, or a GPU.
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_track_a_a13_candidate_audit"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
A1="$ROOT/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
OUT="$ROOT/outputs/pr_diffusion/track_a_np_paper/a13_candidate_audit_20261020/a13_0_inventory_v1"
EXPECTED="${1:-}"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
if [ -z "$EXPECTED" ] || [ ! -x "$PY" ] || { [ ! -d "$REPO/.git" ] && [ ! -f "$REPO/.git" ]; }; then
  echo "STOP_A13_0: missing exact commit, interpreter or checked-out Git worktree"
elif [ -e "$OUT" ] || [ -L "$OUT" ]; then
  echo "STOP_A13_0: stage0 output already exists; preserve it"
else
  cd "$REPO" || echo "STOP_A13_0: cannot enter dedicated checkout"
  if [ "$(git rev-parse HEAD 2>/dev/null)" != "$EXPECTED" ]; then
    echo "STOP_A13_0: source HEAD does not equal pinned expected commit"
  else
    "$PY" -m unittest scripts.track_a.test_a13_physics -v
    RC=$?
    echo "A13_SYNTHETIC_TEST_RC=$RC"
    if [ "$RC" -ne 0 ]; then
      echo "STOP_A13_0: unit-test gate failed; no inventory run"
    else
      "$PY" -m scripts.track_a.a13_inventory --stage inventory \
        --expected-head "$EXPECTED" \
        --run-root "$A1/a1_2_exposed_b22_k4_v1" \
        --input-root "$A1/a1_2_exposed_b22_inputs_v1" --out "$OUT"
      RC=$?
      echo "A13_INVENTORY_RC=$RC"
      if [ "$RC" -ne 0 ]; then
        echo "STOP_A13_0: inspect error, do not run stage1"
      else
        echo "STAGE0_RECEIPT=$OUT/A13_INVENTORY_COMPLETE.json"
        sha256sum "$OUT/A13_SOURCE_AND_INPUT_AUDIT.json"
      fi
    fi
  fi
fi
