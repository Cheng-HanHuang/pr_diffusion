#!/usr/bin/env bash
b26_gpu_stage_main() {
    ROOT="/egr/research-pac/huang248"
    REPO="$ROOT/pr_diffusion_b26"
    PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
    BRANCH="codex/b26-np-conditional-correction"
    B25="codex/b25-noise-selection-mechanisms"
    BASE="c906d36e0e396a6abbf761e3d65c91433428170f"
    RUN="$1"
    STAGE="$2"
    GPUCSV="${3:-0,1,2,3}"
    if [ -z "$RUN" ] || [ -z "$STAGE" ]; then echo "STOP|usage: launch_b26_gpu_stage.sh RUN smoke|dev16|dev80 [0,1,2,3]"; return 2; fi
    case "$STAGE" in smoke|dev16|dev80) ;; *) echo "STOP|bad_stage=$STAGE"; return 2;; esac
    echo "GIT_VERSION|$(git --version 2>&1)"
    for d in "$REPO" "$RUN"; do [ -d "$d" ] || { echo "STOP|missing_dir=$d"; return 2; }; done
    [ -x "$PY" ] || { echo "STOP|missing_python=$PY"; return 2; }
    [ -f "$RUN/PRE_RUN_IDENTITY.json" ] || { echo "STOP|missing_pre_run_identity"; return 2; }
    BR="$(git -C "$REPO" branch --show-current 2>/dev/null)"; HEAD="$(git -C "$REPO" rev-parse HEAD 2>/dev/null)"; DIRTY="$(git -C "$REPO" status --porcelain | wc -l)"
    echo "LOCAL|branch=$BR|head=$HEAD|dirty=$DIRTY"
    [ "$BR" = "$BRANCH" ] || { echo "STOP|wrong_branch"; return 3; }
    [ "$DIRTY" -eq 0 ] || { echo "STOP|dirty_worktree"; return 3; }
    git -C "$REPO" fetch origin "+refs/heads/$B25:refs/remotes/origin/$B25" >/dev/null 2>&1; F1=$?
    git -C "$REPO" fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH" >/dev/null 2>&1; F2=$?
    [ "$F1" -eq 0 ] && [ "$F2" -eq 0 ] || { echo "STOP|fetch_failed|b25=$F1|b26=$F2"; return 4; }
    RB25="$(git -C "$REPO" rev-parse "refs/remotes/origin/$B25")"; RB26="$(git -C "$REPO" rev-parse "refs/remotes/origin/$BRANCH")"
    echo "REMOTE|b25=$RB25|b26=$RB26"
    [ "$RB25" = "$BASE" ] || { echo "STOP|B25_advanced=$RB25"; return 5; }
    [ "$HEAD" = "$RB26" ] || { echo "STOP|B26_local_remote_mismatch"; return 5; }
    "$PY" - "$RUN/PRE_RUN_IDENTITY.json" "$HEAD" <<'PY'
import json,sys
v=json.load(open(sys.argv[1])); h=sys.argv[2]
if v.get('status')!='PASS' or v.get('pre_run_commit')!=h: raise SystemExit('pre-run identity mismatch')
print('PRE_RUN_IDENTITY_PASS|'+h)
PY
    [ "$?" -eq 0 ] || return 6
    if [ "$STAGE" = "dev16" ]; then "$PY" - "$RUN/gpu/GATE_smoke.json" <<'PY'
import json,sys
v=json.load(open(sys.argv[1])); assert v['status']=='PASS' and v['scope']=='smoke'
PY
        [ "$?" -eq 0 ] || { echo "STOP|smoke_gate_not_pass"; return 7; }
    fi
    if [ "$STAGE" = "dev80" ]; then "$PY" - "$RUN/gpu/GATE_dev16.json" <<'PY'
import json,sys
v=json.load(open(sys.argv[1])); assert v['status']=='PASS' and v['scope']=='dev16'
PY
        [ "$?" -eq 0 ] || { echo "STOP|dev16_gate_not_pass"; return 7; }
    fi
    IFS=',' read -r -a GPUS <<< "$GPUCSV"
    N="${#GPUS[@]}"; [ "$N" -ge 1 ] && [ "$N" -le 4 ] || { echo "STOP|bad_gpu_count=$N"; return 8; }
    "$PY" - "$GPUCSV" <<'PY'
import subprocess,sys
ids=[int(x) for x in sys.argv[1].split(',') if x!='']
if len(ids)!=len(set(ids)) or any(x not in (0,1,2,3) for x in ids): raise SystemExit('GPU IDs must be unique explicit subset of 0,1,2,3')
text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
rows={}
for line in text.splitlines():
 p=[x.strip() for x in line.split(',')]; rows[int(p[0])]={'uuid':p[1],'total':int(p[2]),'used':int(p[3]),'free':int(p[4])}
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
active=[]
for line in apps.splitlines() if apps else []:
 p=[x.strip() for x in line.split(',')]; active.append(p)
for i in ids:
 r=rows.get(i)
 if r is None or r['free']<10240: raise SystemExit(f'GPU admission failed {i}: {r}')
 conflicts=[a for a in active if a[0]==r['uuid']]
 if conflicts: raise SystemExit(f'GPU {i} has existing compute process(es): {conflicts}')
 print(f"GPU_READY|index={i}|uuid={r['uuid']}|free_mib={r['free']}|used_mib={r['used']}")
PY
    [ "$?" -eq 0 ] || return 8
    GPUROOT="$RUN/gpu"; mkdir -p "$GPUROOT/logs" "$GPUROOT/pids" "$GPUROOT/receipts"
    MANIFEST="$REPO/configs/b26/b26_dev80_manifest.json"
    [ -f "$MANIFEST" ] || { echo "STOP|missing_manifest"; return 9; }
    for ((i=0;i<N;i++)); do
        g="${GPUS[$i]}"; old="$GPUROOT/WORKER_${STAGE}_$(printf '%02d' "$i").json"
        if [ -f "$old" ]; then mv "$old" "$GPUROOT/receipts/WORKER_${STAGE}_$(printf '%02d' "$i")_$(date -u +%Y%m%dT%H%M%SZ).json" || { echo "STOP|cannot_archive_receipt=$old"; return 10; }; fi
        log="$GPUROOT/logs/${STAGE}_gpu${g}_$(date -u +%Y%m%dT%H%M%SZ).log"
        nohup setsid env CUDA_VISIBLE_DEVICES="$g" PYTHONDONTWRITEBYTECODE=1 \
          "$PY" "$REPO/scripts/b26/run_b26_np_gpu_worker.py" \
          --manifest "$MANIFEST" --output-root "$GPUROOT" --stage "$STAGE" --physical-gpu "$g" --worker-index "$i" --worker-count "$N" --min-free-mib 10240 \
          >"$log" 2>&1 < /dev/null &
        pid=$!; echo "$pid" > "$GPUROOT/pids/${STAGE}_gpu${g}.pid"
        echo "GPU_WORKER_LAUNCHED|stage=$STAGE|gpu=$g|worker=$i/$N|pid=$pid|log=$log"
    done
    "$PY" - "$GPUROOT/STAGE_${STAGE}_LAUNCH.json" "$STAGE" "$HEAD" "$GPUCSV" <<'PY'
import json,os,sys,time
p,stage,head,gpus=sys.argv[1:]
v={'schema_version':'b26.gpu-stage-launch.v1','status':'LAUNCHED','stage':stage,'pre_run_commit':head,'gpu_ids':[int(x) for x in gpus.split(',')],'launched_unix_time':time.time()}
t=p+'.tmp'; open(t,'w').write(json.dumps(v,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
    echo "B26_GPU_STAGE_LAUNCHED|run=$RUN|stage=$STAGE|head=$HEAD|gpus=$GPUCSV"
    return 0
}
b26_gpu_stage_main "$@"
