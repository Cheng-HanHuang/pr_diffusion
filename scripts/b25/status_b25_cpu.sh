#!/usr/bin/env bash

ROOT="/egr/research-pac/huang248"
REPO="$ROOT/pr_diffusion_b25"
OUTROOT="$ROOT/outputs/pr_diffusion/b25"

main() {
    local RUN="$1"
    if [ ! -d "$ROOT" ]; then echo "STOP|missing_root:$ROOT"; return 2; fi
    if [ ! -d "$REPO" ]; then echo "STOP|missing_repo:$REPO"; return 2; fi
    if [ -z "$RUN" ]; then
        if [ ! -f "$OUTROOT/B25_LATEST_RUN.txt" ]; then echo "STOP|missing_latest_pointer"; return 2; fi
        RUN=$(head -n 1 "$OUTROOT/B25_LATEST_RUN.txt")
    fi
    if [ ! -d "$RUN" ]; then echo "STOP|missing_run:$RUN"; return 2; fi

    echo "B25_STATUS|run=$RUN|git=$(git --version 2>&1)|branch=$(git -C "$REPO" branch --show-current 2>/dev/null)|head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)"
    if [ -f "$RUN/worker.pid" ]; then
        local PID CMD
        PID=$(head -n 1 "$RUN/worker.pid")
        CMD=$(ps -p "$PID" -o args= 2>/dev/null)
        if [ -n "$CMD" ]; then echo "WORKER|alive=YES|pid=$PID|cmd=$CMD"; else echo "WORKER|alive=NO|pid=$PID"; fi
    else
        echo "WORKER|pid_file=NO"
    fi

    for pair in "synthetic|SYNTHETIC_SUMMARY.json" "dev|EXP3_SYMMETRY_SUMMARY.json" "analysis|B25_ANALYSIS.json"; do
        local STAGE=${pair%%|*} FILE=${pair#*|} PTR="$RUN/LATEST_${STAGE^^}.txt"
        if [ -f "$PTR" ]; then
            local DIR
            DIR=$(head -n 1 "$PTR")
            if [ -f "$DIR/$FILE" ]; then
                echo "STAGE|$STAGE|summary=$DIR/$FILE"
                python3 - "$DIR/$FILE" <<'PY'
import json,sys
try:
 x=json.load(open(sys.argv[1]));
 keep={k:x.get(k) for k in ('status','wall_seconds','dev_images','terminal_rows','shared_failure_count','aggregate_scientific_wall_seconds','max_observed_rss_gib') if k in x}
 print(json.dumps(keep,sort_keys=True))
except Exception as e: print('SUMMARY_READ_ERROR|'+repr(e))
PY
            else
                echo "STAGE|$STAGE|partial=$DIR"
            fi
        else
            echo "STAGE|$STAGE|not_started"
        fi
    done

    if [ -f "$RUN/WORKER_COMPLETE.json" ]; then
        echo "WORKER_COMPLETE|$RUN/WORKER_COMPLETE.json"
        cat "$RUN/WORKER_COMPLETE.json"
    fi
    if [ -f "$RUN/worker.log" ]; then
        echo "----- WORKER LOG TAIL -----"
        tail -n 40 "$RUN/worker.log"
        echo "----- END LOG TAIL -----"
    fi
    return 0
}

main "$@"
