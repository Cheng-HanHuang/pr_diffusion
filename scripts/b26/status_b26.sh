#!/usr/bin/env bash
b26_status_main() {
    RUN="$1"; ROOT="/egr/research-pac/huang248"; REPO="$ROOT/pr_diffusion_b26"; PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
    [ -n "$RUN" ] && [ -d "$RUN" ] || { echo "STOP|usage: status_b26.sh RUN"; return 2; }
    echo "B26_STATUS|run=$RUN|git=$(git --version 2>&1)|branch=$(git -C "$REPO" branch --show-current 2>/dev/null)|head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)"
    for pf in "$RUN"/pids/*.pid "$RUN"/gpu/pids/*.pid; do
        [ -f "$pf" ] || continue; pid="$(cat "$pf" 2>/dev/null)"; if kill -0 "$pid" 2>/dev/null; then alive=YES; else alive=NO; fi; echo "PROCESS|$(basename "$pf")|pid=$pid|alive=$alive"
    done
    COMPLETE=$(find "$RUN/gpu/jobs" -name JOB_COMPLETE.json -type f 2>/dev/null | wc -l); FAIL=$(find "$RUN/gpu/jobs" -name FAILURE.json -type f 2>/dev/null | wc -l); echo "GPU_JOBS|complete=$COMPLETE|failures=$FAIL|max=640"
    for g in smoke dev16 dev80; do [ -f "$RUN/gpu/GATE_${g}.json" ] && { echo "----- GATE $g -----"; cat "$RUN/gpu/GATE_${g}.json"; }; done
    [ -f "$RUN/cpu/CPU_RESOURCE.json" ] && { echo "----- CPU RESOURCE -----"; cat "$RUN/cpu/CPU_RESOURCE.json"; }
    [ -f "$RUN/cpu/experiment/CPU_SUMMARY.json" ] && { echo "----- CPU SUMMARY COMPACT -----"; "$PY" - "$RUN/cpu/experiment/CPU_SUMMARY.json" <<'PY'
import json,sys
v=json.load(open(sys.argv[1])); print(json.dumps({k:v.get(k) for k in ['status','observations','total_complete_reverse_trajectories','wall_seconds','confirmation_payloads_accessed']},sort_keys=True))
PY
    }
    for log in $(ls -1t "$RUN"/logs/*.log "$RUN"/gpu/logs/*.log 2>/dev/null | head -n 5); do echo "----- TAIL $log -----"; tail -n 12 "$log"; done
    [ -f "$RUN/B26_COMPLETE.json" ] && { echo "----- B26 COMPLETE -----"; cat "$RUN/B26_COMPLETE.json"; }
    return 0
}
b26_status_main "$@"
