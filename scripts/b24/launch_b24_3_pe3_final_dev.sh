#!/usr/bin/env bash
set -euo pipefail

ROOT=/egr/research-pac/huang248
CONTROL="$ROOT/pr_diffusion_b23"
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
BRANCH=codex/b24-bestof4-failure-sweep
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
DAPS_PY="$ROOT/conda-envs/daps/bin/python"
SITCOM_PY="$ROOT/conda-envs/sitcom_ode_bw/bin/python"
DEV_PTR="$OUTROOT/B24_3_DEV80_LATEST_RUN.txt"
MIN_FREE_MIB=10240
HARD_CEILING_MIB=52452

[[ -x "$PY" && -x "$DAPS_PY" && -x "$SITCOM_PY" ]] || { echo "STOP|missing_python_env"; exit 2; }
[[ -f "$DEV_PTR" ]] || { echo "STOP|missing_dev80_pointer:$DEV_PTR"; exit 2; }
DEV80=$(cat "$DEV_PTR")
[[ -d "$DEV80" ]] || { echo "STOP|missing_dev80_run:$DEV80"; exit 2; }
for p in "$DEV80/DEV80_MANIFEST.json" "$DEV80/DEV80_SUMMARY.json" "$DEV80/DEV80_PER_IMAGE.csv"; do [[ -f "$p" ]] || { echo "STOP|missing_source:$p"; exit 2; }; done

"$PY" - "$DEV80" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]); m=json.load(open(r/'DEV80_MANIFEST.json')); s=json.load(open(r/'DEV80_SUMMARY.json'))
if m.get('image_count')!=80 or m.get('confirmation_exposed') is not False: raise SystemExit('bad DEV80 manifest')
if s.get('status')!='PASS' or s.get('image_count')!=80 or s.get('confirmation_exposed') is not False: raise SystemExit('bad DEV80 summary')
for g in range(4):
    p=r/f'workers/gpu{g}/WORKER_COMPLETE.json'
    if not p.is_file(): raise SystemExit(f'missing DEV80 worker {p}')
    v=json.load(open(p))
    if v.get('status')!='PASS' or int(v.get('completed',-1))!=20 or int(v.get('failed',-1))!=0: raise SystemExit(f'nonpass DEV80 worker {p}')
print('PE3_SOURCE_DEV80_PASS')
PY

[[ -z "$(git -C "$REPO" status --porcelain)" ]] || { echo "STOP|B24_worktree_dirty"; git -C "$REPO" status --short; exit 3; }
git -C "$CONTROL" fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
LOCAL=$(git -C "$REPO" rev-parse HEAD); REMOTE=$(git -C "$REPO" rev-parse "origin/$BRANCH")
git -C "$REPO" merge-base --is-ancestor "$LOCAL" "$REMOTE" || { echo "STOP|local_not_ancestor|local=$LOCAL|remote=$REMOTE"; exit 4; }
if [[ "$LOCAL" != "$REMOTE" ]]; then git -C "$REPO" merge --ff-only "origin/$BRANCH"; fi
HEAD=$(git -C "$REPO" rev-parse HEAD)

cd "$REPO"
export PYTHONPATH="$REPO:$REPO/scripts/b22${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
CUDA_VISIBLE_DEVICES="" "$PY" -m py_compile \
  scripts/b24/run_b24_3_pe3_refinement.py \
  scripts/b24/run_b24_3_pe3_dev80.py \
  scripts/b24/summarize_b24_3_pe3.py \
  scripts/b24/run_python_cuda_flop_counted.py \
  scripts/b24/run_b24_3_cross_family_flop_audit.py \
  scripts/b24/test_b24_3_pe3.py
CUDA_VISIBLE_DEVICES="" "$PY" scripts/b24/test_b24_3_pe3.py
for P in "$PY" "$DAPS_PY" "$SITCOM_PY"; do
  "$P" -c 'from torch.utils.flop_counter import FlopCounterMode; print("FLOP_COUNTER_READY")'
done
echo "B24_3_PE3_ZERO_GPU_TESTS_PASS|head=$HEAD"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$OUTROOT/B24_3_pe3_final_dev_${STAMP}"
[[ ! -e "$RUN" ]] || { echo "STOP|run_exists:$RUN"; exit 5; }
mkdir -p "$RUN/assignments"

"$PY" - "$DEV80" "$RUN" "$HEAD" <<'PY'
import json,pathlib,sys
src=pathlib.Path(sys.argv[1]).resolve(); run=pathlib.Path(sys.argv[2]).resolve(); head=sys.argv[3]
m=json.load(open(src/'DEV80_MANIFEST.json'))
tasks=[]
for g in range(4): (run/'assignments'/f'gpu{g}').mkdir(parents=True,exist_ok=True)
for idx,t in enumerate(m['tasks']):
    td=src/'workers'/f"gpu{t['assigned_gpu']}"/t['output_subdir']
    comp=json.load(open(td/'IMAGE_COMPLETE.json'))
    if comp.get('status')!='PASS' or comp.get('confirmation_exposed') is not False: raise SystemExit(f'bad source completion {td}')
    inp=pathlib.Path(t['source_input_manifest']).resolve() if t['pilot16'] else (td/'input'/'input_manifest.json').resolve()
    role=(td/'role_row.csv').resolve(); np4=pathlib.Path(comp['np_results']['NP4_INDEPENDENT']).resolve(); metrics=pathlib.Path(comp['baseline_metrics']).resolve()
    for p in (inp,role,np4,metrics):
        if not p.is_file(): raise SystemExit(f'missing source file {p}')
    gpu=idx%4
    task={
      'schema_version':'b24.pe3-task.v1','task_index':idx,'image_id':t['image_id'],'class_label':t['class_label'],
      'fresh_baseline_class':comp['fresh_baseline_class'],'assigned_gpu':gpu,'method_role':'DEVELOPMENT',
      'input_manifest':str(inp),'role_row':str(role),'np4_result':str(np4),'baseline_metrics':str(metrics),
      'source_dev80_task':str(td.resolve()),'output_subdir':f"task{idx:03d}_{t['class_label']}_{t['image_id']}",
    }
    p=run/'assignments'/f'gpu{gpu}'/f"task{idx:03d}_{t['class_label']}_{t['image_id']}.json"; p.write_text(json.dumps(task,indent=2,sort_keys=True)+'\n')
    tasks.append(task)
if len(tasks)!=80 or len({t['image_id'] for t in tasks})!=80: raise SystemExit('PE3 task count drift')
for g in range(4):
    if sum(t['assigned_gpu']==g for t in tasks)!=20: raise SystemExit(f'PE3 gpu imbalance {g}')
out={
 'schema_version':'b24.pe3-launch.v1','status':'FROZEN_BEFORE_EXECUTION','pe3_head':head,'source_dev80_run':str(src),
 'image_count':80,'new_measurement_count':0,'confirmation_exposed':False,'arms':['NP_PE3_SCORE','NP_PE3_RANDOM'],
 'total_unet_evals_per_arm_per_image':8800,'proposal_unet_evals_per_arm_per_image':8796,
 'advancement_gate':{'median_delta_db_min':0.0,'good25_count_at_least_np4':True,'good25_rescues_at_least_harms':True,'large_5db_rescues_at_least_harms':True,'if_neither_passes':'STOP_B24_METHOD_REFINEMENT'},
 'compute_audit_authorized':True,'tasks':tasks,
}
(run/'PE3_MANIFEST.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'PASS','tasks':80,'per_gpu':20,'new_measurements':0,'confirmation_exposed':False},sort_keys=True))
PY

cat > "$RUN/LAUNCH_IDENTITY.txt" <<EOF
head=$HEAD
source_dev80_run=$DEV80
images=80
new_measurements=0
arms=NP_PE3_SCORE,NP_PE3_RANDOM
total_unet_evals_per_arm=8800
min_free_mib=$MIN_FREE_MIB
hard_ceiling_mib=$HARD_CEILING_MIB
confirmation_exposed=0
EOF

for GPU in 0 1 2 3; do
  ASSIGN="$RUN/assignments/gpu${GPU}"; COUNT=$(find "$ASSIGN" -maxdepth 1 -type f -name 'task*.json' | wc -l)
  [[ "$COUNT" -eq 20 ]] || { echo "STOP|bad_assignment_gpu=$GPU|count=$COUNT"; exit 6; }
  nohup bash "$REPO/scripts/b24/run_b24_3_pe3_worker.sh" "$GPU" "$ASSIGN" "$RUN" "$MIN_FREE_MIB" > "$RUN/gpu${GPU}.log" 2>&1 &
  PID=$!; printf '%s\n' "$PID" > "$RUN/gpu${GPU}.pid"
  echo "PE3_GPU_LAUNCHED|gpu=$GPU|pid=$PID|assigned=$COUNT"
done

# The FLOP audit waits for GPU0's PE3 worker, then reuses GPU0.  It does not
# expose new data and never preempts another job.
nohup bash "$REPO/scripts/b24/run_b24_3_cross_family_flop_audit.sh" "$RUN" "$DEV80" 0 > "$RUN/flop_audit.log" 2>&1 &
AUDIT_PID=$!; printf '%s\n' "$AUDIT_PID" > "$RUN/flop_audit.pid"

printf '%s\n' "$RUN" > "$OUTROOT/B24_3_PE3_LATEST_RUN.txt"
echo "B24_3_PE3_LAUNCHED|run=$RUN|images=80|new_measurements=0|audit_pid=$AUDIT_PID|head=$HEAD"
echo "STATUS|bash $REPO/scripts/b24/status_b24_3_pe3.sh"
