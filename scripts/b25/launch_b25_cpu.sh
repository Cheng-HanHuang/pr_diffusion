#!/usr/bin/env bash

ROOT="/egr/research-pac/huang248"
CONTROL="$ROOT/pr_diffusion_b23"
REPO="$ROOT/pr_diffusion_b25"
OUT_PARENT="$ROOT/outputs/pr_diffusion"
OUTROOT="$OUT_PARENT/b25"
B24_OUT="$OUT_PARENT/b24"
B24_BRANCH="codex/b24-bestof4-failure-sweep"
B25_BRANCH="codex/b25-noise-selection-mechanisms"
IMMUTABLE="ed162c2f97430804fddb5d9a0bfec7abde201ca0"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
ROLE_DIR="$B24_OUT/B24_2_7424_extension_20260907T231303Z/case_freeze/method_stage"
DEV80_RUN="$B24_OUT/B24_3_dev80_overnight_20260913T082146Z"
CLOSEOUT="$B24_OUT/B24_3_zero_gpu_closeout_corrected_20260914T060600Z"

main() {
    if [ ! -d "$ROOT" ]; then echo "STOP|missing_root:$ROOT"; return 2; fi
    if [ ! -d "$CONTROL" ]; then echo "STOP|missing_control_repo:$CONTROL"; return 2; fi
    if [ ! -d "$REPO" ]; then
        echo "STOP|B25_worktree_not_created:$REPO"
        echo "Create it only after verifying the fetched B25 ref; do not reuse another directory."
        return 2
    fi
    if [ ! -d "$OUT_PARENT" ]; then echo "STOP|missing_output_parent:$OUT_PARENT"; return 2; fi
    if [ ! -x "$PY" ]; then echo "STOP|missing_python:$PY"; return 2; fi
    for p in "$ROLE_DIR" "$DEV80_RUN" "$CLOSEOUT"; do
        if [ ! -d "$p" ]; then echo "STOP|missing_frozen_source_dir:$p"; return 2; fi
    done

    echo "GIT_VERSION|$(git --version 2>&1)"
    git -C "$CONTROL" fetch origin "+refs/heads/$B24_BRANCH:refs/remotes/origin/$B24_BRANCH"
    local FB24=$?
    git -C "$CONTROL" fetch origin "+refs/heads/$B25_BRANCH:refs/remotes/origin/$B25_BRANCH"
    local FB25=$?
    echo "FETCH_RESULT|b24=$FB24|b25=$FB25"
    if [ "$FB24" -ne 0 ] || [ "$FB25" -ne 0 ]; then echo "STOP|fetch_failed"; return 3; fi

    local REMOTE_B24 REMOTE_B25
    REMOTE_B24=$(git -C "$CONTROL" rev-parse "origin/$B24_BRANCH" 2>/dev/null)
    REMOTE_B25=$(git -C "$CONTROL" rev-parse "origin/$B25_BRANCH" 2>/dev/null)
    echo "REMOTE_IDENTITY|b24=$REMOTE_B24|b25=$REMOTE_B25"
    if [ "$REMOTE_B24" != "$IMMUTABLE" ]; then echo "STOP|remote_B24_advanced:$REMOTE_B24"; return 3; fi

    local BRANCH HEAD DIRTY
    BRANCH=$(git -C "$REPO" branch --show-current 2>/dev/null)
    HEAD=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)
    DIRTY=$(git -C "$REPO" status --porcelain 2>/dev/null)
    echo "B25_LOCAL_BEFORE|branch=$BRANCH|head=$HEAD|dirty_lines=$(printf '%s' "$DIRTY" | sed '/^$/d' | wc -l)"
    if [ "$BRANCH" != "$B25_BRANCH" ]; then echo "STOP|wrong_B25_branch:$BRANCH"; return 4; fi
    if [ -n "$DIRTY" ]; then echo "STOP|B25_worktree_dirty"; git -C "$REPO" status --short; return 4; fi

    if [ "$HEAD" != "$REMOTE_B25" ]; then
        git -C "$REPO" merge-base --is-ancestor "$HEAD" "$REMOTE_B25" >/dev/null 2>&1
        if [ "$?" -ne 0 ]; then echo "STOP|B25_local_not_ff_ancestor|local=$HEAD|remote=$REMOTE_B25"; return 4; fi
        git -C "$REPO" merge --ff-only "$REMOTE_B25"
        local MRC=$?
        echo "B25_FF_PULL|rc=$MRC|from=$HEAD|to=$REMOTE_B25"
        if [ "$MRC" -ne 0 ]; then return 4; fi
    else
        echo "B25_FF_PULL|rc=0|already_current=YES"
    fi

    HEAD=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)
    DIRTY=$(git -C "$REPO" status --porcelain 2>/dev/null)
    if [ "$HEAD" != "$REMOTE_B25" ] || [ -n "$DIRTY" ]; then echo "STOP|post_sync_identity_failed"; return 4; fi

    export CUDA_VISIBLE_DEVICES=""
    export PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=4
    export NUMEXPR_NUM_THREADS=4

    local FILES=(
        "$REPO/scripts/b25/run_b25_synthetic.py"
        "$REPO/scripts/b25/run_b25_dev_diagnostics.py"
        "$REPO/scripts/b25/analyze_b25_results.py"
        "$REPO/scripts/b25/run_cpu_stage.py"
        "$REPO/scripts/b25/test_b25.py"
    )
    for f in "${FILES[@]}"; do
        if [ ! -f "$f" ]; then echo "STOP|missing_B25_file:$f"; return 5; fi
        "$PY" - "$f" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1])
compile(p.read_text(encoding='utf-8'), str(p), 'exec')
PY
        if [ "$?" -ne 0 ]; then echo "STOP|syntax_failed:$f"; return 5; fi
    done
    CUDA_VISIBLE_DEVICES="" "$PY" "$REPO/scripts/b25/test_b25.py"
    local TRC=$?
    echo "B25_ZERO_GPU_TESTS|rc=$TRC"
    if [ "$TRC" -ne 0 ]; then return 5; fi
    DIRTY=$(git -C "$REPO" status --porcelain 2>/dev/null)
    if [ -n "$DIRTY" ]; then echo "STOP|tests_dirtied_worktree"; git -C "$REPO" status --short; return 5; fi

    if [ ! -d "$OUTROOT" ]; then
        echo "B25_OUTPUT_ROOT_ABSENT|creating=$OUTROOT"
        mkdir -p "$OUTROOT"
        if [ "$?" -ne 0 ] || [ ! -d "$OUTROOT" ]; then echo "STOP|cannot_create_output_root:$OUTROOT"; return 6; fi
    else
        echo "B25_OUTPUT_ROOT_EXISTS|$OUTROOT"
    fi

    local STAMP RUN
    STAMP=$(date -u +%Y%m%dT%H%M%SZ)
    RUN="$OUTROOT/B25_cpu_${STAMP}"
    if [ -e "$RUN" ]; then echo "STOP|run_root_exists:$RUN"; return 6; fi
    mkdir -p "$RUN"
    if [ "$?" -ne 0 ]; then echo "STOP|cannot_create_run:$RUN"; return 6; fi

    CUDA_VISIBLE_DEVICES="" "$PY" "$REPO/scripts/b25/test_b25.py" > "$RUN/PRE_RUN_TESTS.log" 2>&1
    TRC=$?
    if [ "$TRC" -ne 0 ]; then
        echo "STOP|pre_run_tests_failed|log=$RUN/PRE_RUN_TESTS.log|run_preserved=$RUN"
        tail -n 40 "$RUN/PRE_RUN_TESTS.log"
        return 7
    fi

    "$PY" - "$RUN" "$REPO" "$HEAD" "$REMOTE_B24" "$REMOTE_B25" "$ROLE_DIR" "$DEV80_RUN" "$CLOSEOUT" <<'PY'
import hashlib,json,pathlib,sys,time
run,repo=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2])
head,rb24,rb25=sys.argv[3:6]
role,dev,close=map(pathlib.Path,sys.argv[6:9])
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
value={
 'schema_version':'b25.pre-run-identity.v1','status':'PASS','pre_run_commit':head,
 'branch':'codex/b25-noise-selection-mechanisms','remote_b24':rb24,'remote_b25':rb25,
 'dirty_state_at_run':'CLEAN','spec_path':str(repo/'configs/b25/b25_cpu_spec.json'),
 'spec_sha256':sha(repo/'configs/b25/b25_cpu_spec.json'),
 'role_dir':str(role),'role_csv_sha256':sha(role/'B24_METHOD_IMAGE_ROLES.csv'),
 'dev80_run':str(dev),'dev80_manifest_sha256':sha(dev/'DEV80_MANIFEST.json'),
 'corrected_closeout':str(close),'hard_subset_sha256':sha(close/'HARD_SUBSET.csv'),
 'closeout_per_image_sha256':sha(close/'DEV80_CLOSEOUT_PER_IMAGE.csv'),
 'python':sys.executable,'python_version':sys.version.split()[0],
 'cuda_visible_devices':'','gpu_work_authorized':False,'pretrained_model_inference_authorized':False,
 'confirmation_payload_access_authorized':False,'created_unix_time':time.time(),
}
(run/'PRE_RUN_IDENTITY.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
print(json.dumps(value,sort_keys=True))
PY
    if [ "$?" -ne 0 ]; then echo "STOP|pre_run_identity_failed|run=$RUN"; return 7; fi

    nohup bash "$REPO/scripts/b25/run_b25_cpu_worker.sh" "$RUN" > "$RUN/worker.log" 2>&1 < /dev/null &
    local PID=$!
    printf '%s\n' "$PID" > "$RUN/worker.pid"
    printf '%s\n' "$RUN" > "$OUTROOT/B25_LATEST_RUN.txt"
    echo "B25_CPU_LAUNCHED|run=$RUN|pid=$PID|head=$HEAD|cuda_visible_devices=EMPTY|max_threads=4|max_ram_gib=16|max_wall_seconds=14400"
    echo "STATUS|bash $REPO/scripts/b25/status_b25_cpu.sh $RUN"
    echo "STOP_COMMAND|bash $REPO/scripts/b25/stop_b25_cpu.sh $RUN"
    echo "RESUME_IF_NEEDED|bash $REPO/scripts/b25/resume_b25_cpu.sh $RUN"
    return 0
}

main "$@"
