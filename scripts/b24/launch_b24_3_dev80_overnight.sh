#!/usr/bin/env bash
set -euo pipefail

ROOT=/egr/research-pac/huang248
CONTROL="$ROOT/pr_diffusion_b23"
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
BRANCH=codex/b24-bestof4-failure-sweep
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
DAPS="$ROOT/pr_diffusion_b19_solver/external/daps"
SITCOM="$ROOT/external/SITCOM_ODE"
MODEL="$ROOT/models/ffhq_10m.pt"
MODEL_SHA=81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa
ROLE_PTR="$OUTROOT/B24_METHOD_STAGE_LATEST_FREEZE.txt"
PILOT_PTR="$OUTROOT/B24_3_PILOT16_LATEST_RUN.txt"
EPP_PTR="$OUTROOT/B24_3_EPP321_LATEST_RUN.txt"
PILOT_SHA=124d3759e4fd540d2e870618dde59ff73d02cbb798d773a785296da5b140e98a
MIN_FREE_MIB=10240
HARD_CEILING_MIB=52452
export TORCH_HOME="$ROOT/models/torch_cache"
mkdir -p "$TORCH_HOME"

[[ -x "$PY" ]] || { echo "STOP|missing_python:$PY"; exit 2; }
[[ -f "$MODEL" ]] || { echo "STOP|missing_model:$MODEL"; exit 2; }
[[ "$(sha256sum "$MODEL" | awk '{print $1}')" == "$MODEL_SHA" ]] || { echo "STOP|model_sha_mismatch"; exit 2; }

for p in "$ROLE_PTR" "$PILOT_PTR" "$EPP_PTR"; do
  [[ -f "$p" ]] || { echo "STOP|missing_pointer:$p"; exit 2; }
done
ROLE_DIR=$(cat "$ROLE_PTR")
PILOTRUN=$(cat "$PILOT_PTR")
EPPRUN=$(cat "$EPP_PTR")
for d in "$ROLE_DIR" "$PILOTRUN" "$EPPRUN"; do
  [[ -d "$d" ]] || { echo "STOP|missing_source_dir:$d"; exit 2; }
done
PANEL="$ROLE_DIR/B24_METHOD_IMAGE_ROLES.csv"
ROLE_SUMMARY="$ROLE_DIR/B24_METHOD_ROLE_FREEZE_SUMMARY.json"
PILOT="$PILOTRUN/B24_METHOD_PILOT16.csv"
EPP_SUMMARY="$EPPRUN/EPP321_REFINEMENT_SUMMARY.json"
for p in "$PANEL" "$ROLE_SUMMARY" "$PILOT" "$EPP_SUMMARY"; do
  [[ -f "$p" ]] || { echo "STOP|missing_source_file:$p"; exit 2; }
done
[[ "$(sha256sum "$PILOT" | awk '{print $1}')" == "$PILOT_SHA" ]] || { echo "STOP|pilot_sha_mismatch"; exit 2; }
(( MIN_FREE_MIB <= HARD_CEILING_MIB )) || { echo "STOP|min_free_exceeds_hard_ceiling"; exit 2; }

verify_source() {
  local name="$1" path="$2" exp_head="$3" exp_tree="$4" exp_index="$5" exp_diff="$6"
  local head tree index_digest diff_digest
  head=$(git -C "$path" rev-parse HEAD)
  tree=$(git -C "$path" rev-parse 'HEAD^{tree}')
  index_digest=$(git -C "$path" ls-files -s | sha256sum | awk '{print $1}')
  diff_digest=$(git -C "$path" diff --binary HEAD -- . | sha256sum | awk '{print $1}')
  [[ "$head" == "$exp_head" ]] || { echo "STOP|source_head|name=$name|observed=$head|expected=$exp_head"; return 21; }
  [[ "$tree" == "$exp_tree" ]] || { echo "STOP|source_tree|name=$name|observed=$tree|expected=$exp_tree"; return 22; }
  [[ "$index_digest" == "$exp_index" ]] || { echo "STOP|source_index|name=$name|observed=$index_digest|expected=$exp_index"; return 23; }
  [[ "$diff_digest" == "$exp_diff" ]] || { echo "STOP|source_diff|name=$name|observed=$diff_digest|expected=$exp_diff"; return 24; }
  echo "SOURCE_READY|name=$name|head=$head|tree=$tree"
}
verify_source DAPS "$DAPS" \
  e7a77d094167084faed19b599b96673b7bb11447 \
  e63f9715e4704d9cd7a43a166559496d9d94e781 \
  d5487cdba570dbaac0c1909e549da361a0a0fc3fed81e5c13f59fa12925876b6 \
  fbb5b42369ecf0d3b9b67f8fc162053bc40ec32aed41dbd92a67e8d81dcfad69
verify_source SITCOM "$SITCOM" \
  275ab67efbd8146bffca20155171ba6be1169c09 \
  80263442e3606824a06dc003504c28da5c59c2c5 \
  3ef63a8a29d0ba65cc642027a57ec102257fd9b387b0e9a5b4aae7f46d6a949f \
  a9f0076d6f852b6898000142c19a09131ffc49ceba0e3d935cd465e85df26e6e

"$PY" - "$PANEL" "$ROLE_SUMMARY" "$PILOTRUN" "$EPPRUN" "$EPP_SUMMARY" <<'PY'
import hashlib,json,pathlib,sys
panel,role_summary,pilotrun,epprun,epp_summary=map(pathlib.Path,sys.argv[1:])
rs=json.load(open(role_summary)); es=json.load(open(epp_summary))
if rs.get('status')!='PASS' or rs.get('development_count')!=80 or rs.get('confirmation_count')!=305:
    raise SystemExit('method-role freeze is not the accepted 80/305 PASS split')
h=hashlib.sha256(panel.read_bytes()).hexdigest()
if h!=rs.get('panel_csv_sha256'): raise SystemExit(f'panel SHA mismatch {h} != {rs.get("panel_csv_sha256")}')
if es.get('status')!='PASS' or es.get('image_count')!=16:
    raise SystemExit('EPP321 source refinement summary is not PASS/16')
for root,label in ((pilotrun,'pilot'),(epprun,'epp321')):
    for gpu in range(4):
        p=root/f'workers/gpu{gpu}/WORKER_COMPLETE.json'
        if not p.is_file(): raise SystemExit(f'{label} source worker missing: {p}')
        v=json.load(open(p))
        if v.get('status')!='PASS': raise SystemExit(f'{label} source worker non-PASS: {p}')
print('DEV80_SOURCE_GATES_PASS')
PY

[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  echo "STOP|B24_worktree_dirty"; git -C "$REPO" status --short; exit 3;
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
  scripts/b24/run_b24_3_dev80_np.py \
  scripts/b24/run_b24_3_dev80_image.py \
  scripts/b24/summarize_b24_3_dev80.py \
  scripts/b24/test_b24_3_dev80.py
CUDA_VISIBLE_DEVICES="" "$PY" scripts/b24/test_b24_3_dev80.py

echo "B24_3_DEV80_ZERO_GPU_TESTS_PASS|head=$HEAD"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$OUTROOT/B24_3_dev80_overnight_${STAMP}"
[[ ! -e "$RUN" ]] || { echo "STOP|run_exists:$RUN"; exit 5; }
mkdir -p "$RUN/assignments" "$RUN/workers"

"$PY" - "$PANEL" "$ROLE_SUMMARY" "$PILOTRUN" "$EPPRUN" "$RUN" "$HEAD" "$MIN_FREE_MIB" <<'PY'
import csv,hashlib,json,pathlib,sys
panel_path=pathlib.Path(sys.argv[1]).resolve()
role_summary_path=pathlib.Path(sys.argv[2]).resolve()
pilotrun=pathlib.Path(sys.argv[3]).resolve()
epprun=pathlib.Path(sys.argv[4]).resolve()
run=pathlib.Path(sys.argv[5]).resolve()
head=sys.argv[6]; gate=int(sys.argv[7])
SCREEN_SHA='b516c8154cbbb790d8a3592b86736bb0d4bd47d0833d85ecf3d6a9d710e950ba'
NP_ARMS=('NP4_INDEPENDENT','NP_EPP_321','NP_EPP_321_RANDOM_PRUNE','NP_EPP_321_NO_REALLOCATION')

def readj(p): return json.load(open(p))
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def seed63(domain,label,image,rep):
    material='|'.join([domain,SCREEN_SHA,label,image,str(rep)])
    return int(hashlib.sha256(material.encode()).hexdigest()[:16],16)&((1<<63)-1)
def check_result(path,arm):
    p=pathlib.Path(path).resolve()
    if not p.is_file(): raise SystemExit(f'missing source result: {p}')
    v=readj(p)
    expected={'NP4_INDEPENDENT':8800,'NP_EPP_321':8800,'NP_EPP_321_RANDOM_PRUNE':8800,'NP_EPP_321_NO_REALLOCATION':6900}[arm]
    if v.get('status')!='PASS' or v.get('arm')!=arm or int(v.get('total_unet_evals',-1))!=expected:
        raise SystemExit(f'bad source NP result: {p}')
    if bool(v.get('runtime_decisions_use_ground_truth',True)) or bool(v.get('terminal_selection_uses_ground_truth',True)):
        raise SystemExit(f'bad clean-free flags in source result: {p}')
    return str(p)

pilot_manifest=readj(pilotrun/'PILOT16_MANIFEST.json')
pilot_sources={}
for rowdir in sorted(pilotrun.glob('workers/gpu*/row*')):
    role=rowdir/'role_row.csv'; inp=rowdir/'input'/'input_manifest.json'; summ=rowdir/'methods'/'SMOKE_COMPLETE.json'
    if not (role.is_file() and inp.is_file() and summ.is_file()): continue
    with role.open(newline='',encoding='utf-8') as f: rr=list(csv.DictReader(f))
    if len(rr)!=1: raise SystemExit(f'bad pilot role row: {role}')
    image=rr[0]['image_id']
    if readj(summ).get('status')!='PASS': raise SystemExit(f'pilot source non-PASS: {summ}')
    pilot_sources[image]={
        'input_manifest':str(inp.resolve()),
        'NP4_INDEPENDENT':check_result(rowdir/'methods'/'NP4_INDEPENDENT'/'result.json','NP4_INDEPENDENT'),
    }
cal=pilot_manifest['calibration_reuse']
cal_summary=pathlib.Path(cal['smoke_summary']).resolve(); cal_root=cal_summary.parent.parent; image=cal['image_id']
if readj(cal_summary).get('status')!='PASS': raise SystemExit('calibration source non-PASS')
pilot_sources[image]={
    'input_manifest':str((cal_root/'input'/'input_manifest.json').resolve()),
    'NP4_INDEPENDENT':check_result(cal_root/'methods'/'NP4_INDEPENDENT'/'result.json','NP4_INDEPENDENT'),
}
if len(pilot_sources)!=16: raise SystemExit(f'expected 16 pilot sources, got {len(pilot_sources)}')

epp_sources={}
for taskdir in sorted(epprun.glob('workers/gpu*/task*')):
    taskp=taskdir/'task.json'; summ=taskdir/'methods'/'SMOKE_COMPLETE.json'
    if not (taskp.is_file() and summ.is_file()): continue
    task=readj(taskp); image=task['image_id']
    if readj(summ).get('status')!='PASS': raise SystemExit(f'EPP source non-PASS: {summ}')
    epp_sources[image]={
        arm:check_result(taskdir/'methods'/arm/'result.json',arm)
        for arm in NP_ARMS if arm!='NP4_INDEPENDENT'
    }
if len(epp_sources)!=16: raise SystemExit(f'expected 16 EPP321 sources, got {len(epp_sources)}')
if set(epp_sources)!=set(pilot_sources): raise SystemExit('Pilot/EPP source image mismatch')

with panel_path.open(newline='',encoding='utf-8') as f: rows=[dict(r) for r in csv.DictReader(f)]
dev=[r for r in rows if r['method_role']=='DEVELOPMENT']
if len(dev)!=80 or len({r['image_id'] for r in dev})!=80: raise SystemExit('DEV80 identity/count drift')
counts={c:sum(r['class_label']==c for r in dev) for c in 'ABCD'}
if counts!={'A':20,'B':20,'C':20,'D':20}: raise SystemExit(f'DEV80 stratum drift: {counts}')
pilot_ids={r['image_id'] for r in dev if r['pilot16']=='TRUE'}
if len(pilot_ids)!=16 or pilot_ids!=set(pilot_sources): raise SystemExit('Pilot16 subset drift')
dev.sort(key=lambda r:(r['class_label'],int(r['method_role_rank']),r['image_id']))

dev_csv=run/'B24_METHOD_DEV80.csv'
with dev_csv.open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(dev[0].keys()),lineterminator='\n'); w.writeheader(); w.writerows(dev)

for gpu in range(4): (run/'assignments'/f'gpu{gpu}').mkdir(parents=True,exist_ok=True)
tasks=[]
for idx,row in enumerate(dev):
    image=row['image_id']; label=row['class_label']; pilot=row['pilot16']=='TRUE'; gpu=idx%4
    daps=[seed63('B24_DEV80_DAPS_SOLVER_V1',label,image,r) for r in range(4)]
    sitcom=[seed63('B24_DEV80_SITCOM_SOLVER_V1',label,image,r) for r in range(4)]
    if len({x%(2**32) for x in daps})!=4 or len({x%(2**32) for x in sitcom})!=4:
        raise SystemExit(f'native seed collision: {label}/{image}')
    source_np={}; source_input=None
    if pilot:
        source_input=pilot_sources[image]['input_manifest']
        source_np['NP4_INDEPENDENT']=pilot_sources[image]['NP4_INDEPENDENT']
        source_np.update(epp_sources[image])
    task={
        'schema_version':'b24.dev80-task.v1','task_index':idx,'image_id':image,
        'class_label':label,'assigned_gpu':gpu,'pilot16':pilot,
        'method_role':'DEVELOPMENT','measurement_seed':int(row['dev_measurement_seed']),
        'input_mode':'REUSE_PILOT16' if pilot else 'GENERATE_FROZEN_DEV',
        'source_input_manifest':source_input,'run_np':not pilot,'source_np_results':source_np,
        'daps_solver_seeds':daps,'sitcom_solver_seeds':sitcom,
        'role_row':row,'output_subdir':f'task{idx:03d}_{label}_{image}',
    }
    p=run/'assignments'/f'gpu{gpu}'/f'task{idx:03d}_{label}_{image}.json'
    p.write_text(json.dumps(task,indent=2,sort_keys=True)+'\n')
    tasks.append(task)

if any(sum(t['assigned_gpu']==g for t in tasks)!=20 for g in range(4)):
    raise SystemExit('GPU assignment imbalance')
manifest={
    'schema_version':'b24.dev80-launch.v1','status':'FROZEN_BEFORE_EXECUTION',
    'dev80_head':head,'source_role_summary':str(role_summary_path),
    'source_panel_csv':str(panel_path),'source_panel_csv_sha256':sha(panel_path),
    'dev80_csv':str(dev_csv.resolve()),'dev80_csv_sha256':sha(dev_csv),
    'source_pilot_run':str(pilotrun),'source_epp321_run':str(epprun),
    'image_count':80,'screening_stratum_counts':counts,'pilot_reuse_count':16,
    'new_measurement_count':64,'fresh_daps4_count':80,'fresh_sitcom4_count':80,
    'new_np_count':64,'confirmation_exposed':False,'min_free_mib':gate,
    'baseline_seed_domains':{'DAPS':'B24_DEV80_DAPS_SOLVER_V1','SITCOM':'B24_DEV80_SITCOM_SOLVER_V1'},
    'np_arms':list(NP_ARMS),'tasks':tasks,
}
(run/'DEV80_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':'PASS','dev80_csv_sha256':manifest['dev80_csv_sha256'],'tasks':80,'pilot_reuse':16,'new_measurements':64,'per_gpu':20},sort_keys=True))
PY

cat > "$RUN/LAUNCH_IDENTITY.txt" <<EOF
head=$HEAD
source_role_dir=$ROLE_DIR
source_pilot_run=$PILOTRUN
source_epp321_run=$EPPRUN
min_free_mib=$MIN_FREE_MIB
hard_ceiling_mib=$HARD_CEILING_MIB
images=80
pilot_measurements_reused=16
new_measurements=64
fresh_daps4=80
fresh_sitcom4=80
new_np_images=64
confirmation_exposed=0
EOF

for GPU in 0 1 2 3; do
  ASSIGN="$RUN/assignments/gpu${GPU}"
  COUNT=$(find "$ASSIGN" -maxdepth 1 -type f -name 'task*.json' | wc -l)
  [[ "$COUNT" -eq 20 ]] || { echo "STOP|bad_assignment_gpu=$GPU|count=$COUNT"; exit 6; }
  nohup bash "$REPO/scripts/b24/run_b24_3_dev80_worker.sh" "$GPU" "$ASSIGN" "$RUN" "$MIN_FREE_MIB" \
    > "$RUN/gpu${GPU}.log" 2>&1 &
  PID=$!
  printf '%s\n' "$PID" > "$RUN/gpu${GPU}.pid"
  echo "DEV80_GPU_LAUNCHED|gpu=$GPU|pid=$PID|assigned=$COUNT"
done

printf '%s\n' "$RUN" > "$OUTROOT/B24_3_DEV80_LATEST_RUN.txt"
echo "B24_3_DEV80_LAUNCHED|run=$RUN|images=80|new_measurements=64|fresh_baselines=80|new_np=64|min_free_mib=$MIN_FREE_MIB|head=$HEAD"
echo "STATUS|bash $REPO/scripts/b24/status_b24_3_dev80.sh"
