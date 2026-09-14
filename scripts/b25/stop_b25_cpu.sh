#!/usr/bin/env bash

ROOT="/egr/research-pac/huang248"
REPO="$ROOT/pr_diffusion_b25"

main() {
    local RUN="$1"
    if [ ! -d "$ROOT" ]; then echo "STOP|missing_root:$ROOT"; return 2; fi
    if [ ! -d "$REPO" ]; then echo "STOP|missing_repo:$REPO"; return 2; fi
    if [ -z "$RUN" ] || [ ! -d "$RUN" ]; then echo "STOP|missing_or_invalid_run:$RUN"; return 2; fi
    if [ ! -f "$RUN/worker.pid" ]; then echo "B25_STOP|no_pid_file|run=$RUN"; return 0; fi
    local PID CMD
    PID=$(head -n 1 "$RUN/worker.pid")
    case "$PID" in (*[!0-9]*|'') echo "STOP|invalid_pid:$PID"; return 3;; esac
    CMD=$(ps -p "$PID" -o args= 2>/dev/null)
    if [ -z "$CMD" ]; then echo "B25_STOP|already_stopped|pid=$PID"; return 0; fi
    case "$CMD" in
        *"scripts/b25/run_b25_cpu_worker.sh"*"$RUN"*) ;;
        *) echo "STOP|pid_identity_mismatch|pid=$PID|cmd=$CMD"; return 3;;
    esac
    kill -TERM "$PID"
    local RC=$?
    if [ "$RC" -eq 0 ]; then
        echo "B25_STOP_SIGNAL_SENT|pid=$PID|signal=TERM|run=$RUN"
    else
        echo "STOP|kill_failed|pid=$PID|rc=$RC"
    fi
    return "$RC"
}

main "$@"
