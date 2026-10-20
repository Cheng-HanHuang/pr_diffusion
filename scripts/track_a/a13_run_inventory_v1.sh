#!/usr/bin/env bash
# A1.3.0: requires prior read-only PAC inventory and exact commit argument.
# This script is called by bash; nonzero status returns to caller's terminal.
# No set -e/-u/pipefail, no implicit exit/logout/GPU or historical-output writes.

a13_stage0_main() {
  local ROOT=/egr/research-pac/huang248
  local REPO="$ROOT/pr_diffusion_track_a_a13_candidate_audit"
  local PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
  local A1="$ROOT/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
  local OUT="$ROOT/outputs/pr_diffusion/track_a_np_paper/a13_candidate_audit_20261020/a13_0_inventory_v1"
  local EXPECTED="${1:-}"
  local RC TEST_LOG
  export CUDA_VISIBLE_DEVICES=""
  export PYTHONDONTWRITEBYTECODE=1
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4

  if [ -z "$EXPECTED" ] || [ ! -x "$PY" ] || { [ ! -d "$REPO/.git" ] && [ ! -f "$REPO/.git" ]; }; then
    echo "STOP_A13_0: missing exact commit, interpreter or checkout"
    return 9
  fi
  if [ -e "$OUT" ] || [ -L "$OUT" ]; then
    echo "STOP_A13_0: stage0 output occupied; preserve"
    return 9
  fi
  if ! cd "$REPO"; then
    echo "STOP_A13_0: cannot enter dedicated checkout"
    return 9
  fi
  if [ "$(git rev-parse HEAD 2>/dev/null)" != "$EXPECTED" ] ||
     [ "$(git branch --show-current 2>/dev/null)" != "codex/prdiff-track-a-a13-candidate-audit-20261020" ]; then
    echo "STOP_A13_0: source HEAD/branch does not equal exact pinned identity"
    return 9
  fi

  TEST_LOG=$("$PY" -m unittest scripts.track_a.test_a13_physics -v 2>&1)
  RC=$?
  printf '%s\n' "$TEST_LOG" | tail -24
  echo "A13_SYNTHETIC_TEST_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "STOP_A13_0: synthetic unit-test gate failed"
    return "$RC"
  fi

  "$PY" -m scripts.track_a.a13_inventory --stage inventory \
    --expected-head "$EXPECTED" \
    --run-root "$A1/a1_2_exposed_b22_k4_v1" \
    --input-root "$A1/a1_2_exposed_b22_inputs_v1" --out "$OUT"
  RC=$?
  echo "A13_INVENTORY_RC=$RC"
  if [ "$RC" -ne 0 ]; then
    echo "STOP_A13_0: source/input/operator gate failed; do not run stage1"
    return "$RC"
  fi

  if ! printf '%s\n' "$TEST_LOG" > "$OUT/A13_SYNTHETIC_TESTS.log"; then
    echo "STOP_A13_0: cannot save new stage test receipt"
    return 9
  fi
  echo "STAGE0_RECEIPT=$OUT/A13_INVENTORY_COMPLETE.json"
  sha256sum "$OUT/A13_SOURCE_AND_INPUT_AUDIT.json"
  return $?
}

a13_stage0_main "$@"
