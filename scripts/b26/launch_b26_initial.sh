#!/usr/bin/env bash
b26_initial_main() {
    ROOT="/egr/research-pac/huang248"; REPO="$ROOT/pr_diffusion_b26"; OUTROOT="$ROOT/outputs/pr_diffusion/b26"; PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
    BRANCH="codex/b26-np-conditional-correction"; B25="codex/b25-noise-selection-mechanisms"; BASE="c906d36e0e396a6abbf761e3d65c91433428170f"
    B25CAP="$ROOT/outputs/pr_diffusion/b25/B25_cpu_20260914T224132Z.tar.gz"; B25SHA="5b1e785ba46e6be952ffdbc768c5d9bab365e495cf9efd5b154b075ce0081354"
    GPUCSV="${1:-0,1,2,3}"
    echo "GIT_VERSION|$(git --version 2>&1)"
    for d in "$ROOT" "$REPO"; do [ -d "$d" ] || { echo "STOP|missing_dir=$d"; return 2; }; done
    [ -x "$PY" ] || { echo "STOP|missing_python=$PY"; return 2; }
    [ -f "$B25CAP" ] || { echo "STOP|missing_B25_capsule"; return 2; }
    OBSHA="$(sha256sum "$B25CAP" | awk '{print $1}')"; echo "B25_CAPSULE_SHA=$OBSHA"; [ "$OBSHA" = "$B25SHA" ] || { echo "STOP|B25_capsule_sha_drift"; return 3; }
    BR="$(git -C "$REPO" branch --show-current)"; HEAD="$(git -C "$REPO" rev-parse HEAD)"; DIRTY="$(git -C "$REPO" status --porcelain | wc -l)"
    echo "LOCAL|branch=$BR|head=$HEAD|dirty=$DIRTY"; [ "$BR" = "$BRANCH" ] || { echo "STOP|wrong_branch"; return 3; }; [ "$DIRTY" -eq 0 ] || { echo "STOP|dirty"; return 3; }
    git -C "$REPO" fetch origin "+refs/heads/$B25:refs/remotes/origin/$B25" >/dev/null 2>&1; F1=$?; git -C "$REPO" fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH" >/dev/null 2>&1; F2=$?
    [ "$F1" -eq 0 ] && [ "$F2" -eq 0 ] || { echo "STOP|fetch_failed"; return 4; }
    RB25="$(git -C "$REPO" rev-parse "refs/remotes/origin/$B25")"; RB26="$(git -C "$REPO" rev-parse "refs/remotes/origin/$BRANCH")"; echo "REMOTE|b25=$RB25|b26=$RB26"
    [ "$RB25" = "$BASE" ] || { echo "STOP|B25_remote_advanced"; return 5; }; [ "$HEAD" = "$RB26" ] || { echo "STOP|B26_local_remote_mismatch"; return 5; }
    MANIFEST="$REPO/configs/b26/b26_dev80_manifest.json"; SPEC="$REPO/configs/b26/b26_spec.json"
    [ -f "$MANIFEST" ] && [ -f "$SPEC" ] || { echo "STOP|pre_run_manifest_or_spec_missing"; return 6; }
    CUDA_VISIBLE_DEVICES="" "$PY" "$REPO/scripts/b26/test_b26.py" --require-manifest; T=$?; echo "B26_ZERO_GPU_TESTS|rc=$T"; [ "$T" -eq 0 ] || return 7
    mkdir -p "$OUTROOT" || return 8
    STAMP="$(date -u +%Y%m%dT%H%M%SZ)"; RUN="$OUTROOT/B26_${STAMP}"; [ ! -e "$RUN" ] || { echo "STOP|run_exists=$RUN"; return 8; }; mkdir -p "$RUN/cpu" "$RUN/gpu" "$RUN/logs" "$RUN/pids" || return 8
    "$PY" - "$RUN/PRE_RUN_IDENTITY.json" "$HEAD" "$SPEC" "$MANIFEST" "$B25CAP" "$GPUCSV" <<'PY'
import hashlib,json,os,sys,time
p,head,spec,manifest,cap,gpus=sys.argv[1:]
def sha(x):
 h=hashlib.sha256();
 with open(x,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
v={'schema_version':'b26.pre-run-identity.v1','status':'PASS','pre_run_commit':head,'signed_b25_base':'c906d36e0e396a6abbf761e3d65c91433428170f','spec_path':os.path.realpath(spec),'spec_sha256':sha(spec),'manifest_path':os.path.realpath(manifest),'manifest_sha256':sha(manifest),'b25_capsule_sha256':sha(cap),'gpu_ids':[int(x) for x in gpus.split(',')],'created_unix_time':time.time(),'confirmation_payload_access_authorized':False,'new_ffhq_measurements_authorized':False,'weighted_ffhq_authorized':False}
t=p+'.tmp'; open(t,'w').write(json.dumps(v,indent=2,sort_keys=True)+'\n'); os.replace(t,p); print(json.dumps(v,sort_keys=True))
PY
    [ "$?" -eq 0 ] || return 9
    CPUOUT="$RUN/cpu/experiment"; CPURES="$RUN/cpu/CPU_RESOURCE.json"; CPULOG="$RUN/logs/cpu.log"
    nohup setsid env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
      "$PY" "$REPO/scripts/b26/run_cpu_stage.py" --output "$CPURES" --max-wall-seconds 14400 --max-rss-gib 16 -- \
      "$PY" "$REPO/scripts/b26/run_b26_cpu_estimator.py" --output "$CPUOUT" >"$CPULOG" 2>&1 < /dev/null &
    CPID=$!; echo "$CPID" > "$RUN/pids/cpu.pid"; echo "CPU_LAUNCHED|pid=$CPID|log=$CPULOG"
    bash "$REPO/scripts/b26/launch_b26_gpu_stage.sh" "$RUN" smoke "$GPUCSV"; G=$?
    if [ "$G" -ne 0 ]; then echo "STOP|gpu_smoke_launch_failed=$G|cpu_pid=$CPID"; return 10; fi
    echo "B26_INITIAL_LAUNCHED|run=$RUN|head=$HEAD|cpu_pid=$CPID|gpus=$GPUCSV"
    echo "STATUS|bash $REPO/scripts/b26/status_b26.sh $RUN"
    return 0
}
b26_initial_main "$@"
