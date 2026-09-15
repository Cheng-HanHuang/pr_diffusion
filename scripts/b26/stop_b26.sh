#!/usr/bin/env bash
b26_stop_main() {
    RUN="$1"; [ -n "$RUN" ] && [ -d "$RUN" ] || { echo "STOP|usage: stop_b26.sh RUN"; return 2; }
    RC=0
    for pf in "$RUN"/pids/*.pid "$RUN"/gpu/pids/*.pid; do
        [ -f "$pf" ] || continue; pid="$(cat "$pf" 2>/dev/null)"; [ -n "$pid" ] || continue
        if ! kill -0 "$pid" 2>/dev/null; then echo "PROCESS_ALREADY_DEAD|file=$pf|pid=$pid"; continue; fi
        args="$(ps -p "$pid" -o args= 2>/dev/null)"
        case "$args" in *pr_diffusion_b26/scripts/b26/*|*run_cpu_stage.py*) ;; *) echo "REFUSE_KILL|pid=$pid|args=$args"; RC=3; continue;; esac
        pgid="$(ps -p "$pid" -o pgid= 2>/dev/null | tr -d ' ')"
        if [ "$pgid" = "$pid" ]; then kill -TERM -- "-$pgid" 2>/dev/null; K=$?; else kill -TERM "$pid" 2>/dev/null; K=$?; fi
        echo "TERM_SENT|pid=$pid|pgid=$pgid|rc=$K|args=$args"; [ "$K" -eq 0 ] || RC=4
    done
    echo "NOTE|An interrupted root attempt is preserved and is not automatically retried; return it for a scoped retry decision."
    return "$RC"
}
b26_stop_main "$@"
