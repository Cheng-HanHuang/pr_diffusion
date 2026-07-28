#!/usr/bin/env bash
# Recover the first B22.1 smoke after the NP reconstruction completed but strict
# JSON serialization rejected an expected NaN diagnostic.
#
# This script:
# - preserves the entire first NP attempt;
# - reruns NP only with the same frozen configuration and seed;
# - requires the rerun reconstruction tensor hash to match attempt 1 exactly;
# - reuses the already successful SITCOM result;
# - runs the independent validator and creates a new return archive.

set -u

GPU_INDEX=${1:-0}
RUN_ROOT=${2:-}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
CONFIG="$REPO_ROOT/configs/b22/b22_1_smoke.json"
NP_PY=/egr/research-pac/huang248/conda-envs/prdiff_ffhq/bin/python
DAPS_PY=/egr/research-pac/huang248/conda-envs/daps/bin/python

if [[ -z "$RUN_ROOT" ]]; then
    echo "usage: bash scripts/b22/resume_b22_1_after_np_json_failure.sh GPU_INDEX RUN_ROOT"
    exit 2
fi

RUN_ROOT=$(readlink -f "$RUN_ROOT")
LOG_DIR="$RUN_ROOT/logs"
STATUS="$RUN_ROOT/status.tsv"
FAILED_DIR="$RUN_ROOT/np1_attempt1_json_failure"
CURRENT_DIR="$RUN_ROOT/np1"
STAMP=$(date +%Y%m%d_%H%M%S)
RECOVERY_STATUS="$RUN_ROOT/RECOVERY_STATUS.txt"

record() {
    local step="$1"
    local status="$2"
    local detail="$3"
    printf '%s\t%s\t%s\n' "$step" "$status" "$detail" >> "$STATUS"
    printf '[%-4s] %s\n' "$status" "$step"
}

archive_and_report() {
    local archive="${RUN_ROOT}_npjson_recovery_${STAMP}.tar.gz"
    if tar -C "$(dirname "$RUN_ROOT")" -czf "$archive" "$(basename "$RUN_ROOT")" \
        >"$LOG_DIR/recovery_archive.log" 2>&1
    then
        record "recovery-archive" "OK" "$archive"
        echo
        echo "Attach this archive:"
        echo "$archive"
    else
        record "recovery-archive" "FAIL" "see recovery_archive.log"
    fi
}

hash_reconstruction() {
    "$NP_PY" - "$1" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
payload = torch.load(path, map_location="cpu")
if isinstance(payload, torch.Tensor):
    value = payload
elif isinstance(payload, dict) and isinstance(payload.get("reconstruction"), torch.Tensor):
    value = payload["reconstruction"]
else:
    raise TypeError(f"No reconstruction tensor found in {path}")

value = value.detach().cpu().contiguous()
if tuple(value.shape) != (1, 3, 256, 256):
    raise ValueError(f"Unexpected reconstruction shape: {tuple(value.shape)}")
if not bool(torch.isfinite(value).all()):
    raise ValueError("Reconstruction is not finite")

header = json.dumps(
    {"dtype": str(value.dtype), "shape": list(value.shape)},
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")
digest = hashlib.sha256()
digest.update(header)
digest.update(b"\0")
digest.update(value.numpy().tobytes(order="C"))
print(digest.hexdigest())
PY
}

mkdir -p "$LOG_DIR"

if [[ ! -f "$RUN_ROOT/input/input_manifest.json" ]]; then
    echo "STOP: missing input manifest: $RUN_ROOT/input/input_manifest.json"
    exit 1
fi
if [[ ! -f "$RUN_ROOT/sitcom1/result.json" ]]; then
    echo "STOP: successful SITCOM result is missing"
    exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "STOP: config missing from updated worktree: $CONFIG"
    exit 1
fi

if [[ -d "$CURRENT_DIR" && ! -d "$FAILED_DIR" ]]; then
    if [[ ! -f "$CURRENT_DIR/reconstruction.pt" ]]; then
        echo "STOP: attempt-1 NP reconstruction is missing"
        exit 1
    fi
    mv "$CURRENT_DIR" "$FAILED_DIR"
elif [[ -d "$FAILED_DIR" && ! -d "$CURRENT_DIR" ]]; then
    : # Recovery was prepared previously but not completed.
else
    echo "STOP: unexpected NP directory state"
    echo "current=$CURRENT_DIR exists=$([[ -d "$CURRENT_DIR" ]] && echo 1 || echo 0)"
    echo "failed=$FAILED_DIR exists=$([[ -d "$FAILED_DIR" ]] && echo 1 || echo 0)"
    exit 1
fi

OLD_HASH=$(hash_reconstruction "$FAILED_DIR/reconstruction.pt") || {
    record "np1-attempt1-hash" "FAIL" "could not hash preserved reconstruction"
    archive_and_report
    exit 1
}
record "np1-attempt1-preserved" "OK" "hash=$OLD_HASH"

BRANCH=$(git -C "$REPO_ROOT" branch --show-current)
HEAD=$(git -C "$REPO_ROOT" rev-parse HEAD)
{
    echo "recovery_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repo_root=$REPO_ROOT"
    echo "repo_branch=$BRANCH"
    echo "repo_head=$HEAD"
    echo "run_root=$RUN_ROOT"
    echo "gpu_index=$GPU_INDEX"
    echo "attempt1_reconstruction_hash=$OLD_HASH"
    echo "failure_class=post_reconstruction_json_serialization"
    echo "method_or_measurement_change=0"
} > "$RECOVERY_STATUS"

if CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$NP_PY" \
    "$REPO_ROOT/scripts/b22/run_b22_1_np_smoke.py" \
    --config "$CONFIG" \
    --repo_root "$REPO_ROOT" \
    --run_root "$RUN_ROOT" \
    >"$LOG_DIR/np1_retry.log" 2>&1
then
    record "np1-retry" "OK" "np1_retry.log"
else
    rc=$?
    record "np1-retry" "FAIL" "np1_retry.log rc=$rc"
    echo "np_retry_ok=0" >> "$RECOVERY_STATUS"
    archive_and_report
    exit 1
fi

NEW_HASH=$(hash_reconstruction "$CURRENT_DIR/reconstruction.pt") || {
    record "np1-retry-hash" "FAIL" "could not hash retry reconstruction"
    archive_and_report
    exit 1
}

echo "retry_reconstruction_hash=$NEW_HASH" >> "$RECOVERY_STATUS"
if [[ "$NEW_HASH" != "$OLD_HASH" ]]; then
    record "np1-replay-identity" "FAIL" "old=$OLD_HASH new=$NEW_HASH"
    echo "exact_replay_match=0" >> "$RECOVERY_STATUS"
    archive_and_report
    exit 1
fi
record "np1-replay-identity" "OK" "exact tensor-content hash match"
echo "exact_replay_match=1" >> "$RECOVERY_STATUS"

if "$DAPS_PY" "$REPO_ROOT/scripts/b22/validate_b22_1_smoke.py" \
    --config "$CONFIG" \
    --run_root "$RUN_ROOT" \
    >"$LOG_DIR/validate_recovery.log" 2>&1
then
    record "validate-recovery" "OK" "validate_recovery.log"
    echo "validate_ok=1" >> "$RECOVERY_STATUS"
    echo "full_panel_authorized=0" >> "$RECOVERY_STATUS"
    echo "gate_state=B22.1_SMOKE_COMPLETE_PENDING_EXECUTION_LEAD_REVIEW" >> "$RECOVERY_STATUS"
else
    rc=$?
    record "validate-recovery" "FAIL" "validate_recovery.log rc=$rc"
    echo "validate_ok=0" >> "$RECOVERY_STATUS"
    echo "full_panel_authorized=0" >> "$RECOVERY_STATUS"
    echo "gate_state=B22.1_SMOKE_FAILED_OR_INCOMPLETE" >> "$RECOVERY_STATUS"
    archive_and_report
    exit 1
fi

archive_and_report

echo
echo "Recovery summary:"
cat "$RECOVERY_STATUS"
