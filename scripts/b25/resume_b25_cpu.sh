#!/usr/bin/env bash

ROOT="/egr/research-pac/huang248"
CONTROL="$ROOT/pr_diffusion_b23"
REPO="$ROOT/pr_diffusion_b25"
B24_BRANCH="codex/b24-bestof4-failure-sweep"
B25_BRANCH="codex/b25-noise-selection-mechanisms"
IMMUTABLE="ed162c2f97430804fddb5d9a0bfec7abde201ca0"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"

main() {
    local RUN="$1"
    if [ ! -d "$ROOT" ]; then echo "STOP|missing_root:$ROOT"; return 2; fi
    if [ ! -d "$CONTROL" ]; then echo "STOP|missing_control:$CONTROL"; return 2; fi
    if [ ! -d "$REPO" ]; then echo "STOP|missing_repo:$REPO"; return 2; fi
    if [ -z "$RUN" ] || [ ! -d "$RUN" ]; then echo "STOP|missing_or_invalid_run:$RUN"; return 2; fi
    if [ ! -f "$RUN/PRE_RUN_IDENTITY.json" ]; then echo "STOP|missing_pre_run_identity"; return 2; fi
    if [ ! -x "$PY" ]; then echo "STOP|missing_python:$PY"; return 2; fi

    if [ -f "$RUN/worker.pid" ]; then
        local OLD_PID OLD_CMD
        OLD_PID=$(head -n 1 "$RUN/worker.pid")
        OLD_CMD=$(ps -p "$OLD_PID" -o args= 2>/dev/null)
        if [ -n "$OLD_CMD" ]; then echo "STOP|worker_still_alive|pid=$OLD_PID|cmd=$OLD_CMD"; return 3; fi
    fi

    echo "GIT_VERSION|$(git --version 2>&1)"
    git -C "$CONTROL" fetch origin "+refs/heads/$B24_BRANCH:refs/remotes/origin/$B24_BRANCH"
    local R1=$?
    git -C "$CONTROL" fetch origin "+refs/heads/$B25_BRANCH:refs/remotes/origin/$B25_BRANCH"
    local R2=$?
    if [ "$R1" -ne 0 ] || [ "$R2" -ne 0 ]; then echo "STOP|fetch_failed|b24=$R1|b25=$R2"; return 3; fi
    local RB24 RB25 EXPECTED HEAD BRANCH DIRTY
    RB24=$(git -C "$CONTROL" rev-parse "origin/$B24_BRANCH" 2>/dev/null)
    RB25=$(git -C "$CONTROL" rev-parse "origin/$B25_BRANCH" 2>/dev/null)
    EXPECTED=$("$PY" - "$RUN/PRE_RUN_IDENTITY.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['pre_run_commit'])
PY
)
    HEAD=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)
    BRANCH=$(git -C "$REPO" branch --show-current 2>/dev/null)
    DIRTY=$(git -C "$REPO" status --porcelain 2>/dev/null)
    echo "RESUME_IDENTITY|remote_b24=$RB24|remote_b25=$RB25|expected=$EXPECTED|local=$HEAD|branch=$BRANCH"
    if [ "$RB24" != "$IMMUTABLE" ]; then echo "STOP|B24_remote_advanced"; return 4; fi
    if [ "$RB25" != "$EXPECTED" ]; then echo "STOP|B25_remote_changed_after_pre_run|remote=$RB25|expected=$EXPECTED"; return 4; fi
    if [ "$HEAD" != "$EXPECTED" ] || [ "$BRANCH" != "$B25_BRANCH" ] || [ -n "$DIRTY" ]; then
        echo "STOP|local_pre_run_identity_drift"
        git -C "$REPO" status --short --branch
        return 4
    fi

    nohup bash "$REPO/scripts/b25/run_b25_cpu_worker.sh" "$RUN" >> "$RUN/worker.log" 2>&1 < /dev/null &
    local PID=$!
    printf '%s\n' "$PID" > "$RUN/worker.pid"
    echo "B25_CPU_RESUMED|run=$RUN|pid=$PID|head=$HEAD"
    echo "STATUS|bash $REPO/scripts/b25/status_b25_cpu.sh $RUN"
    return 0
}

main "$@"
