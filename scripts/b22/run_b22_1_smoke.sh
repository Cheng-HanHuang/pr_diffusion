#!/usr/bin/env bash
# Reproducible B22.1 one-image smoke. No full-panel work is performed.
#
# Usage:
#   bash scripts/b22/run_b22_1_smoke.sh [GPU_INDEX] [OUTPUT_BASE]
#
# The script never edits repository files. Verbose output is redirected to logs.

set -u

GPU_INDEX="${1:-0}"
OUTPUT_BASE="${2:-/egr/research-pac/huang248/outputs/pr_diffusion/b22_baselines}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$REPO_ROOT/configs/b22/b22_1_smoke.json"

DAPS_PY=/egr/research-pac/huang248/conda-envs/daps/bin/python
SITCOM_PY=/egr/research-pac/huang248/conda-envs/sitcom_ode_bw/bin/python
NP_PY=/egr/research-pac/huang248/conda-envs/prdiff_ffhq/bin/python

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$OUTPUT_BASE/B22_1_smoke_${STAMP}"
STATUS="$RUN_ROOT/status.tsv"

mkdir -p "$RUN_ROOT/logs"
printf 'step\tstatus\tlog\n' > "$STATUS"

record() {
    local step="$1"
    local status="$2"
    local log="$3"
    printf '%s\t%s\t%s\n' "$step" "$status" "$log" >> "$STATUS"
    printf '[%-4s] %s\n' "$status" "$step"
}

run_step() {
    local step="$1"
    local logfile="$2"
    shift 2
    if "$@" >"$logfile" 2>&1; then
        record "$step" "OK" "$(basename "$logfile")"
        return 0
    else
        local rc=$?
        record "$step" "FAIL" "$(basename "$logfile") rc=$rc"
        return "$rc"
    fi
}

echo "B22.1 smoke output: $RUN_ROOT"
echo "GPU visibility: physical GPU $GPU_INDEX -> process cuda:0"

PREP_OK=0
SITCOM_OK=0
NP_OK=0
VALIDATE_OK=0

if run_step \
    "cpu-preflight" \
    "$RUN_ROOT/logs/prepare.log" \
    "$DAPS_PY" "$REPO_ROOT/scripts/b22/prepare_b22_1_smoke.py" \
        --config "$CONFIG" \
        --repo_root "$REPO_ROOT" \
        --run_root "$RUN_ROOT"
then
    PREP_OK=1
fi

if [[ "$PREP_OK" -eq 1 ]]; then
    if run_step \
        "sitcom1" \
        "$RUN_ROOT/logs/sitcom1.log" \
        env CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
        "$SITCOM_PY" "$REPO_ROOT/scripts/b22/run_b22_1_sitcom_smoke.py" \
            --config "$CONFIG" \
            --repo_root "$REPO_ROOT" \
            --run_root "$RUN_ROOT"
    then
        SITCOM_OK=1
    fi

    if run_step \
        "np1" \
        "$RUN_ROOT/logs/np1.log" \
        env CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
        "$NP_PY" "$REPO_ROOT/scripts/b22/run_b22_1_np_smoke.py" \
            --config "$CONFIG" \
            --repo_root "$REPO_ROOT" \
            --run_root "$RUN_ROOT"
    then
        NP_OK=1
    fi
else
    record "sitcom1" "SKIP" "preflight failed"
    record "np1" "SKIP" "preflight failed"
fi

if [[ "$PREP_OK" -eq 1 && "$SITCOM_OK" -eq 1 && "$NP_OK" -eq 1 ]]; then
    if run_step \
        "validate" \
        "$RUN_ROOT/logs/validate.log" \
        "$DAPS_PY" "$REPO_ROOT/scripts/b22/validate_b22_1_smoke.py" \
            --config "$CONFIG" \
            --run_root "$RUN_ROOT"
    then
        VALIDATE_OK=1
    fi
else
    record "validate" "SKIP" "one or more required method steps failed"
fi

{
    echo "run_root=$RUN_ROOT"
    echo "repo_root=$REPO_ROOT"
    echo "config=$CONFIG"
    echo "gpu_index=$GPU_INDEX"
    echo "preflight_ok=$PREP_OK"
    echo "sitcom_ok=$SITCOM_OK"
    echo "np_ok=$NP_OK"
    echo "validate_ok=$VALIDATE_OK"
    echo "full_panel_authorized=0"
    if [[ "$VALIDATE_OK" -eq 1 ]]; then
        echo "gate_state=B22.1_SMOKE_COMPLETE_PENDING_EXECUTION_LEAD_REVIEW"
    else
        echo "gate_state=B22.1_SMOKE_FAILED_OR_INCOMPLETE"
    fi
} > "$RUN_ROOT/FINAL_STATUS.txt"

ARCHIVE="${RUN_ROOT}.tar.gz"
if tar -C "$(dirname "$RUN_ROOT")" -czf "$ARCHIVE" "$(basename "$RUN_ROOT")" \
    >"$RUN_ROOT/logs/archive.log" 2>&1
then
    printf '[OK  ] archive\n'
else
    rc=$?
    printf '[FAIL] archive (see %s)\n' "$RUN_ROOT/logs/archive.log"
    echo "Archive creation failed with rc=$rc" >> "$RUN_ROOT/FINAL_STATUS.txt"
fi

echo
echo "===== COMPACT STATUS ====="
if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$STATUS"
else
    cat "$STATUS"
fi

echo
cat "$RUN_ROOT/FINAL_STATUS.txt"
echo
echo "Return this archive:"
echo "$ARCHIVE"

if [[ "$VALIDATE_OK" -eq 1 && -f "$ARCHIVE" ]]; then
    exit 0
fi
exit 1
