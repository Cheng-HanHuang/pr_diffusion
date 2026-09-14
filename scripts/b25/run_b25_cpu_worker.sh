#!/usr/bin/env bash

ROOT="/egr/research-pac/huang248"
REPO="$ROOT/pr_diffusion_b25"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
ROLE_DIR="$ROOT/outputs/pr_diffusion/b24/B24_2_7424_extension_20260907T231303Z/case_freeze/method_stage"
DEV80_RUN="$ROOT/outputs/pr_diffusion/b24/B24_3_dev80_overnight_20260913T082146Z"
CLOSEOUT="$ROOT/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_corrected_20260914T060600Z"
DAPS="$ROOT/pr_diffusion_b19_solver/external/daps"
SITCOM="$ROOT/external/SITCOM_ODE"
DIFFFPR="$ROOT/external/DiffFPR"
SPEC="$REPO/configs/b25/b25_cpu_spec.json"
WRAP="$REPO/scripts/b25/run_cpu_stage.py"
SYN="$REPO/scripts/b25/run_b25_synthetic.py"
DEV="$REPO/scripts/b25/run_b25_dev_diagnostics.py"
ANALYZE="$REPO/scripts/b25/analyze_b25_results.py"
MAX_RSS_GIB=16
MAX_TOTAL_WALL=14400

json_status_pass() {
    local file="$1"
    if [ ! -f "$file" ]; then
        return 1
    fi
    "$PY" - "$file" <<'PY' >/dev/null 2>&1
import json,sys
try:
    x=json.load(open(sys.argv[1]))
    raise SystemExit(0 if x.get('status')=='PASS' else 1)
except Exception:
    raise SystemExit(1)
PY
}

next_attempt_dir() {
    local run="$1" stage="$2" n=1 candidate
    while [ "$n" -le 99 ]; do
        candidate="$run/${stage}_attempt$(printf '%02d' "$n")"
        if [ ! -e "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
        n=$((n+1))
    done
    return 1
}

resource_wall() {
    local file="$1"
    if [ ! -f "$file" ]; then
        printf '0\n'
        return 0
    fi
    "$PY" - "$file" <<'PY'
import json,sys
try:
    print(float(json.load(open(sys.argv[1])).get('wall_seconds',0.0)))
except Exception:
    print(0.0)
PY
}

main() {
    local RUN="$1"
    if [ -z "$RUN" ]; then
        echo "STOP|missing_run_root"
        return 2
    fi
    if [ ! -d "$ROOT" ]; then echo "STOP|missing_root:$ROOT"; return 2; fi
    if [ ! -d "$REPO" ]; then echo "STOP|missing_b25_worktree:$REPO"; return 2; fi
    if [ ! -d "$RUN" ]; then echo "STOP|missing_run_root:$RUN"; return 2; fi
    if [ ! -x "$PY" ]; then echo "STOP|missing_python:$PY"; return 2; fi
    for p in "$ROLE_DIR" "$DEV80_RUN" "$CLOSEOUT" "$DAPS" "$SITCOM" "$DIFFFPR"; do
        if [ ! -d "$p" ]; then echo "STOP|missing_required_dir:$p"; return 2; fi
    done
    for p in "$SPEC" "$WRAP" "$SYN" "$DEV" "$ANALYZE" "$RUN/PRE_RUN_IDENTITY.json"; do
        if [ ! -f "$p" ]; then echo "STOP|missing_required_file:$p"; return 2; fi
    done

    local EXPECTED_HEAD HEAD BRANCH DIRTY
    EXPECTED_HEAD=$("$PY" - "$RUN/PRE_RUN_IDENTITY.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['pre_run_commit'])
PY
)
    HEAD=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)
    BRANCH=$(git -C "$REPO" branch --show-current 2>/dev/null)
    DIRTY=$(git -C "$REPO" status --porcelain 2>/dev/null)
    if [ "$BRANCH" != "codex/b25-noise-selection-mechanisms" ]; then echo "STOP|wrong_branch:$BRANCH"; return 3; fi
    if [ "$HEAD" != "$EXPECTED_HEAD" ]; then echo "STOP|head_drift:$HEAD|expected=$EXPECTED_HEAD"; return 3; fi
    if [ -n "$DIRTY" ]; then echo "STOP|b25_worktree_dirty"; git -C "$REPO" status --short; return 3; fi

    export CUDA_VISIBLE_DEVICES=""
    export PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=4
    export NUMEXPR_NUM_THREADS=4

    local SYN_DIR DEV_DIR ANA_DIR SYN_PTR DEV_PTR ANA_PTR
    SYN_PTR="$RUN/LATEST_SYNTHETIC.txt"
    DEV_PTR="$RUN/LATEST_DEV.txt"
    ANA_PTR="$RUN/LATEST_ANALYSIS.txt"

    if [ -f "$SYN_PTR" ]; then SYN_DIR=$(head -n 1 "$SYN_PTR"); else SYN_DIR=""; fi
    if [ -z "$SYN_DIR" ] || ! json_status_pass "$SYN_DIR/SYNTHETIC_SUMMARY.json"; then
        SYN_DIR=$(next_attempt_dir "$RUN" "synthetic") || { echo "STOP|no_synthetic_attempt_slot"; return 4; }
        printf '%s\n' "$SYN_DIR" > "$SYN_PTR"
        echo "B25_STAGE_START|synthetic|$SYN_DIR"
        "$PY" "$WRAP" \
            --resource-json "$SYN_DIR/CPU_RESOURCE.json" \
            --max-rss-gib "$MAX_RSS_GIB" \
            --max-wall-seconds "$MAX_TOTAL_WALL" \
            -- "$PY" "$SYN" --spec "$SPEC" --output "$SYN_DIR"
        local SYN_RC=$?
        echo "B25_STAGE_END|synthetic|rc=$SYN_RC|$SYN_DIR"
        if [ "$SYN_RC" -ne 0 ]; then
            echo "B25_STAGE_FAILED|synthetic|rc=$SYN_RC|preserved=$SYN_DIR"
        fi
    else
        echo "B25_STAGE_REUSE|synthetic|$SYN_DIR"
    fi

    local SYN_WALL REMAIN
    SYN_WALL=$(resource_wall "$SYN_DIR/CPU_RESOURCE.json")
    REMAIN=$("$PY" - "$MAX_TOTAL_WALL" "$SYN_WALL" <<'PY'
import sys
print(max(1.0,float(sys.argv[1])-float(sys.argv[2])))
PY
)

    if [ -f "$DEV_PTR" ]; then DEV_DIR=$(head -n 1 "$DEV_PTR"); else DEV_DIR=""; fi
    if [ -z "$DEV_DIR" ] || ! json_status_pass "$DEV_DIR/EXP3_SYMMETRY_SUMMARY.json"; then
        DEV_DIR=$(next_attempt_dir "$RUN" "dev") || { echo "STOP|no_dev_attempt_slot"; return 5; }
        printf '%s\n' "$DEV_DIR" > "$DEV_PTR"
        echo "B25_STAGE_START|dev|$DEV_DIR"
        "$PY" "$WRAP" \
            --resource-json "$DEV_DIR/CPU_RESOURCE.json" \
            --max-rss-gib "$MAX_RSS_GIB" \
            --max-wall-seconds "$REMAIN" \
            -- "$PY" "$DEV" \
                --spec "$SPEC" \
                --repo "$REPO" \
                --role-dir "$ROLE_DIR" \
                --dev80-run "$DEV80_RUN" \
                --closeout "$CLOSEOUT" \
                --daps-root "$DAPS" \
                --sitcom-root "$SITCOM" \
                --difffpr-root "$DIFFFPR" \
                --output "$DEV_DIR"
        local DEV_RC=$?
        echo "B25_STAGE_END|dev|rc=$DEV_RC|$DEV_DIR"
        if [ "$DEV_RC" -ne 0 ]; then
            echo "B25_STAGE_FAILED|dev|rc=$DEV_RC|preserved=$DEV_DIR"
        fi
    else
        echo "B25_STAGE_REUSE|dev|$DEV_DIR"
    fi

    if json_status_pass "$SYN_DIR/SYNTHETIC_SUMMARY.json" && json_status_pass "$DEV_DIR/EXP3_SYMMETRY_SUMMARY.json"; then
        local DEV_WALL SCI_WALL ANALYSIS_REMAIN
        DEV_WALL=$(resource_wall "$DEV_DIR/CPU_RESOURCE.json")
        SCI_WALL=$("$PY" - "$SYN_WALL" "$DEV_WALL" <<'PY'
import sys
print(float(sys.argv[1])+float(sys.argv[2]))
PY
)
        ANALYSIS_REMAIN=$("$PY" - "$MAX_TOTAL_WALL" "$SCI_WALL" <<'PY'
import sys
print(max(1.0,float(sys.argv[1])-float(sys.argv[2])))
PY
)
        if [ -f "$ANA_PTR" ]; then ANA_DIR=$(head -n 1 "$ANA_PTR"); else ANA_DIR=""; fi
        if [ -z "$ANA_DIR" ] || ! json_status_pass "$ANA_DIR/B25_ANALYSIS.json"; then
            ANA_DIR=$(next_attempt_dir "$RUN" "analysis") || { echo "STOP|no_analysis_attempt_slot"; return 6; }
            printf '%s\n' "$ANA_DIR" > "$ANA_PTR"
            echo "B25_STAGE_START|analysis|$ANA_DIR"
            "$PY" "$WRAP" \
                --resource-json "$ANA_DIR/CPU_RESOURCE.json" \
                --max-rss-gib "$MAX_RSS_GIB" \
                --max-wall-seconds "$ANALYSIS_REMAIN" \
                -- "$PY" "$ANALYZE" \
                    --spec "$SPEC" \
                    --synthetic-dir "$SYN_DIR" \
                    --dev-dir "$DEV_DIR" \
                    --output "$ANA_DIR"
            local ANA_RC=$?
            echo "B25_STAGE_END|analysis|rc=$ANA_RC|$ANA_DIR"
            if [ "$ANA_RC" -ne 0 ]; then echo "B25_STAGE_FAILED|analysis|rc=$ANA_RC|preserved=$ANA_DIR"; fi
        else
            echo "B25_STAGE_REUSE|analysis|$ANA_DIR"
        fi
    else
        echo "B25_ANALYSIS_BLOCKED|upstream_stage_not_pass"
    fi

    local FINAL_STATUS="PARTIAL"
    if [ -f "$SYN_PTR" ] && [ -f "$DEV_PTR" ] && [ -f "$ANA_PTR" ]; then
        SYN_DIR=$(head -n 1 "$SYN_PTR")
        DEV_DIR=$(head -n 1 "$DEV_PTR")
        ANA_DIR=$(head -n 1 "$ANA_PTR")
        if json_status_pass "$SYN_DIR/SYNTHETIC_SUMMARY.json" && \
           json_status_pass "$DEV_DIR/EXP3_SYMMETRY_SUMMARY.json" && \
           json_status_pass "$ANA_DIR/B25_ANALYSIS.json"; then
            FINAL_STATUS="PASS"
        fi
    fi

    "$PY" - "$RUN" "$FINAL_STATUS" "$HEAD" "${SYN_DIR:-}" "${DEV_DIR:-}" "${ANA_DIR:-}" <<'PY'
import json,pathlib,sys,time
run=pathlib.Path(sys.argv[1]); status=sys.argv[2]
value={
  'schema_version':'b25.worker-complete.v1','status':status,'pre_run_commit':sys.argv[3],
  'synthetic_dir':sys.argv[4] or None,'dev_dir':sys.argv[5] or None,'analysis_dir':sys.argv[6] or None,
  'gpu_work_performed':False,'pretrained_model_inference_performed':False,
  'new_ffhq_measurements_generated':False,'new_ffhq_reconstructions_generated':False,
  'confirmation_payloads_accessed':False,'completed_unix_time':time.time(),
}
(run/'WORKER_COMPLETE.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
print(json.dumps(value,sort_keys=True))
PY

    if [ "$FINAL_STATUS" = "PASS" ]; then
        echo "B25_CPU_WORKER_PASS|run=$RUN|head=$HEAD"
        return 0
    fi
    echo "B25_CPU_WORKER_PARTIAL|run=$RUN|inspect_and_resume"
    return 7
}

main "$@"
