#!/usr/bin/env bash
set -euo pipefail

ROOT=/egr/research-pac/huang248
CONTROL="$ROOT/pr_diffusion_b23"
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
BRANCH=codex/b24-bestof4-failure-sweep
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
PILOT_PTR="$OUTROOT/B24_3_PILOT16_LATEST_RUN.txt"
PILOT_SHA=124d3759e4fd540d2e870618dde59ff73d02cbb798d773a785296da5b140e98a
MIN_FREE_MIB=10240
HARD_CEILING_MIB=52452

[[ -x "$PY" ]] || { echo "STOP|missing_python:$PY"; exit 2; }
[[ -f "$PILOT_PTR" ]] || { echo "STOP|missing_pilot_pointer:$PILOT_PTR"; exit 2; }
PILOTRUN=$(cat "$PILOT_PTR")
[[ -d "$PILOTRUN" ]] || { echo "STOP|missing_pilot_run:$PILOTRUN"; exit 2; }
PILOT="$PILOTRUN/B24_METHOD_PILOT16.csv"
MANIFEST="$PILOTRUN/PILOT16_MANIFEST.json"
[[ -f "$PILOT" ]] || { echo "STOP|missing_pilot_csv:$PILOT"; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "STOP|missing_pilot_manifest:$MANIFEST"; exit 2; }
[[ "$(sha256sum "$PILOT" | awk '{print $1}')" == "$PILOT_SHA" ]] || { echo "STOP|pilot_sha_mismatch"; exit 2; }
(( MIN_FREE_MIB <= HARD_CEILING_MIB )) || { echo "STOP|min_free_exceeds_hard_ceiling"; exit 2; }

# The source Pilot16 must be fully complete before reusing any locked measurement.
for g in 0 1 2 3; do
  WC="$PILOTRUN/workers/gpu${g}/WORKER_COMPLETE.json"
  [[ -f "$WC" ]] || { echo "STOP|source_worker_incomplete:$WC"; exit 3; }
  STATUS=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$WC")
  [[ "$STATUS" == "PASS" ]] || { echo "STOP|source_worker_nonpass:$WC:$STATUS"; exit 3; }
done

[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  echo "STOP|B24_worktree_dirty"; git -C "$REPO" status --short; exit 4;
}
git -C "$CONTROL" fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
LOCAL=$(git -C "$REPO" rev-parse HEAD)
REMOTE=$(git -C "$REPO" rev-parse "origin/$BRANCH")
git -C "$REPO" merge-base --is-ancestor "$LOCAL" "$REMOTE" || {
  echo "STOP|local_not_ancestor|local=$LOCAL|remote=$REMOTE"; exit 4;
}
if [[ "$LOCAL" != "$REMOTE" ]]; then git -C "$REPO" merge --ff-only "origin/$BRANCH"; fi
HEAD=$(git -C "$REPO" rev-parse HEAD)

cd "$REPO"
export PYTHONPATH="$REPO:$REPO/scripts/b22${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
CUDA_VISIBLE_DEVICES="" "$PY" -m py_compile \
  scripts/b24/run_b24_3_np_branching.py \
  scripts/b24/run_b24_3_epp321_refinement.py \
  scripts/b24/test_b24_3_epp321_refinement.py
CUDA_VISIBLE_DEVICES="" "$PY" scripts/b24/test_b24_3_epp321_refinement.py

echo "B24_3_EPP321_ZERO_GPU_TESTS_PASS|head=$HEAD"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$OUTROOT/B24_3_epp321_refinement_${STAMP}"
[[ ! -e "$RUN" ]] || { echo "STOP|run_exists:$RUN"; exit 5; }
mkdir -p "$RUN/assignments"
cp "$PILOT" "$RUN/B24_METHOD_PILOT16.csv"

"$PY" - "$PILOTRUN" "$MANIFEST" "$RUN" "$PILOT_SHA" "$HEAD" <<'PY'
import csv,json,pathlib,sys
pilotrun=pathlib.Path(sys.argv[1]).resolve()
manifest_path=pathlib.Path(sys.argv[2]).resolve()
run=pathlib.Path(sys.argv[3]).resolve()
pilot_sha=sys.argv[4]
head=sys.argv[5]
manifest=json.load(open(manifest_path))
rows=manifest.get('rows',[])
if len(rows)!=16: raise SystemExit(f'expected 16 frozen pilot rows, got {len(rows)}')

# Locate the 15 worker sources by image id.
sources={}
for rowdir in sorted(pilotrun.glob('workers/gpu*/row*')):
    role=rowdir/'role_row.csv'; inp=rowdir/'input'/'input_manifest.json'; summ=rowdir/'methods'/'SMOKE_COMPLETE.json'
    if not (role.is_file() and inp.is_file() and summ.is_file()): continue
    with role.open(newline='',encoding='utf-8') as f: rr=list(csv.DictReader(f))
    if len(rr)!=1: raise SystemExit(f'bad role row: {role}')
    image=rr[0]['image_id']
    if json.load(open(summ)).get('status')!='PASS': raise SystemExit(f'source not PASS: {summ}')
    sources[image]={'role_row':str(role.resolve()),'input_manifest':str(inp.resolve()),'source_smoke_summary':str(summ.resolve()),'source_methods_dir':str((rowdir/'methods').resolve())}

# Add the reused calibration source.
cal=manifest['calibration_reuse']
cal_summary=pathlib.Path(cal['smoke_summary']).resolve()
cal_root=cal_summary.parent.parent
cal_image=cal['image_id']
if json.load(open(cal_summary)).get('status')!='PASS': raise SystemExit('calibration source not PASS')
sources[cal_image]={
    'role_row':str((cal_root/'role_row.csv').resolve()),
    'input_manifest':str((cal_root/'input'/'input_manifest.json').resolve()),
    'source_smoke_summary':str(cal_summary),
    'source_methods_dir':str((cal_root/'methods').resolve()),
}

ids=[r['image_id'] for r in rows]
if set(ids)!=set(sources):
    raise SystemExit(f'source/pilot identity mismatch missing={sorted(set(ids)-set(sources))} extra={sorted(set(sources)-set(ids))}')
if len(set(ids))!=16: raise SystemExit('pilot image ids not unique')

for gpu in range(4): (run/'assignments'/f'gpu{gpu}').mkdir(parents=True,exist_ok=True)
tasks=[]
for idx,row in enumerate(rows):
    image=row['image_id']; gpu=idx%4
    task={
        'task_index':idx,
        'image_id':image,
        'class_label':row['class_label'],
        'assigned_gpu':gpu,
        'output_subdir':f'task{idx:02d}_{image}',
        **sources[image],
    }
    p=run/'assignments'/f'gpu{gpu}'/f'task{idx:02d}_{image}.json'
    p.write_text(json.dumps(task,indent=2,sort_keys=True)+'\n')
    tasks.append(task)

source_head=''
identity=pilotrun/'LAUNCH_IDENTITY.txt'
if identity.is_file():
    for line in identity.read_text().splitlines():
        if line.startswith('head='): source_head=line.split('=',1)[1]

out={
    'schema_version':'b24.epp321-launch.v1',
    'status':'FROZEN_BEFORE_EXECUTION',
    'refinement_head':head,
    'source_pilot_run':str(pilotrun),
    'source_pilot_head':source_head,
    'pilot_csv_sha256':pilot_sha,
    'image_count':16,
    'measurement_generation_authorized':False,
    'measurements_reused_from_completed_pilot':True,
    'arms':['NP_EPP_321','NP_EPP_321_RANDOM_PRUNE','NP_EPP_321_NO_REALLOCATION'],
    'compute_note':'Main and random arms: 8800 total NP UNet evaluations = NP4. No-reallocation: 6900 total and must be reported at its lower Work-FRE.',
    'tasks':tasks,
}
(run/'REFINEMENT_MANIFEST.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
PY

cat > "$RUN/LAUNCH_IDENTITY.txt" <<EOF
head=$HEAD
source_pilot_run=$PILOTRUN
pilot_csv_sha256=$PILOT_SHA
min_free_mib=$MIN_FREE_MIB
hard_ceiling_mib=$HARD_CEILING_MIB
measurement_generation=0
main_total_unet_evals=8800
random_total_unet_evals=8800
no_reallocation_total_unet_evals=6900
EOF

for GPU in 0 1 2 3; do
  ASSIGN="$RUN/assignments/gpu${GPU}"
  COUNT=$(find "$ASSIGN" -maxdepth 1 -type f -name 'task*.json' | wc -l)
  [[ "$COUNT" -gt 0 ]] || { echo "STOP|empty_assignment_gpu=$GPU"; exit 6; }
  nohup bash "$REPO/scripts/b24/run_b24_3_epp321_worker.sh" "$GPU" "$ASSIGN" "$RUN" "$MIN_FREE_MIB" \
    > "$RUN/gpu${GPU}.log" 2>&1 &
  PID=$!
  printf '%s\n' "$PID" > "$RUN/gpu${GPU}.pid"
  echo "EPP321_GPU_LAUNCHED|gpu=$GPU|pid=$PID|assigned=$COUNT"
done

printf '%s\n' "$RUN" > "$OUTROOT/B24_3_EPP321_LATEST_RUN.txt"
echo "B24_3_EPP321_LAUNCHED|run=$RUN|images=16|new_measurements=0|min_free_mib=$MIN_FREE_MIB|head=$HEAD"
echo "STATUS|bash $REPO/scripts/b24/status_b24_3_epp321_refinement.sh"
