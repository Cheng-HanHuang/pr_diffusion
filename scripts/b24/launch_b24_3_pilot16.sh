#!/usr/bin/env bash
set -euo pipefail

ROOT=/egr/research-pac/huang248
CONTROL="$ROOT/pr_diffusion_b23"
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
BRANCH=codex/b24-bestof4-failure-sweep
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
ROLE_PTR="$OUTROOT/B24_METHOD_STAGE_LATEST_FREEZE.txt"
SMOKE_PTR="$OUTROOT/B24_3_ONE_IMAGE_LATEST_RUN.txt"
PILOT_SHA=124d3759e4fd540d2e870618dde59ff73d02cbb798d773a785296da5b140e98a
HARD_CEILING_MIB=52452

[[ -f "$ROLE_PTR" ]] || { echo "STOP|missing_role_pointer:$ROLE_PTR"; exit 2; }
[[ -f "$SMOKE_PTR" ]] || { echo "STOP|missing_one_image_pointer:$SMOKE_PTR"; exit 2; }
ROLEDIR=$(cat "$ROLE_PTR")
PILOT="$ROLEDIR/B24_METHOD_PILOT16.csv"
SMOKERUN=$(cat "$SMOKE_PTR")
SMOKESUM="$SMOKERUN/methods/SMOKE_COMPLETE.json"
[[ -f "$PILOT" ]] || { echo "STOP|missing_pilot:$PILOT"; exit 2; }
[[ -f "$SMOKESUM" ]] || { echo "STOP|one_image_smoke_not_complete:$SMOKESUM"; exit 2; }
[[ "$(sha256sum "$PILOT" | awk '{print $1}')" == "$PILOT_SHA" ]] || { echo "STOP|pilot_sha_mismatch"; exit 2; }

read -r SMOKE_STATUS CAL_IMAGE MIN_FREE < <("$PY" - "$SMOKESUM" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
print(p.get('status',''), p.get('image_id',''), p.get('recommended_pilot_min_free_mib',''))
PY
)
[[ "$SMOKE_STATUS" == "PASS" ]] || { echo "STOP|one_image_smoke_not_PASS:$SMOKE_STATUS"; exit 3; }
[[ "$MIN_FREE" =~ ^[0-9]+$ ]] || { echo "STOP|bad_calibrated_gate:$MIN_FREE"; exit 3; }
(( MIN_FREE <= HARD_CEILING_MIB )) || { echo "STOP|calibrated_gate_exceeds_hard_ceiling:$MIN_FREE"; exit 3; }

[[ -z "$(git -C "$REPO" status --porcelain)" ]] || { echo "STOP|B24_worktree_dirty"; git -C "$REPO" status --short; exit 4; }
git -C "$CONTROL" fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
LOCAL=$(git -C "$REPO" rev-parse HEAD)
REMOTE=$(git -C "$REPO" rev-parse "origin/$BRANCH")
git -C "$REPO" merge-base --is-ancestor "$LOCAL" "$REMOTE" || { echo "STOP|local_not_ancestor"; exit 4; }
if [[ "$LOCAL" != "$REMOTE" ]]; then git -C "$REPO" merge --ff-only "origin/$BRANCH"; fi
HEAD=$(git -C "$REPO" rev-parse HEAD)

SMOKE_HEAD=$(awk -F= '$1=="head"{print $2}' "$SMOKERUN/LAUNCH_IDENTITY.txt")
[[ -n "$SMOKE_HEAD" ]] || { echo "STOP|missing_smoke_head"; exit 4; }
[[ "$SMOKE_HEAD" == "$HEAD" ]] || {
  echo "STOP|head_changed_since_calibration|smoke=$SMOKE_HEAD|current=$HEAD"
  echo "Rerun one-image calibration after any runner-affecting head change."
  exit 4
}

cd "$REPO"
export PYTHONPATH="$REPO:$REPO/scripts/b22${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
CUDA_VISIBLE_DEVICES="" "$PY" -m py_compile scripts/b24/run_b24_3_np_branching.py scripts/b24/test_b24_3_np_branching.py
CUDA_VISIBLE_DEVICES="" "$PY" scripts/b24/test_b24_3_np_branching.py

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$OUTROOT/B24_3_pilot16_${STAMP}"
[[ ! -e "$RUN" ]] || { echo "STOP|run_exists:$RUN"; exit 5; }
mkdir -p "$RUN/assignments"
cp "$PILOT" "$RUN/B24_METHOD_PILOT16.csv"

"$PY" - "$PILOT" "$SMOKERUN" "$SMOKESUM" "$RUN" "$CAL_IMAGE" <<'PY'
import csv,json,pathlib,sys
pilot,smokerun,smokesum,run,cal=sys.argv[1:]
run=pathlib.Path(run)
with open(pilot,newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
if len(rows)!=16: raise SystemExit(f'expected 16 pilot rows, got {len(rows)}')
ids=[r['image_id'] for r in rows]
if cal not in ids: raise SystemExit(f'calibration image {cal} not in Pilot16')
for gpu in range(4): (run/'assignments'/f'gpu{gpu}').mkdir(parents=True,exist_ok=True)
for idx,row in enumerate(rows):
    if row['image_id']==cal: continue
    gpu=idx%4
    p=run/'assignments'/f'gpu{gpu}'/f"row{idx:02d}_{row['image_id']}.csv"
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(row)); w.writeheader(); w.writerow(row)
manifest={
 'schema_version':'b24.pilot16-launch.v1','pilot_count':16,
 'calibration_reuse':{'image_id':cal,'smoke_run':smokerun,'smoke_summary':smokesum},
 'new_worker_images':15,
 'assignments':{str(g):len(list((run/'assignments'/f'gpu{g}').glob('row*.csv'))) for g in range(4)},
 'rows':rows,
}
(run/'PILOT16_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
PY

cat > "$RUN/LAUNCH_IDENTITY.txt" <<EOF
head=$HEAD
pilot_csv_sha256=$PILOT_SHA
calibration_smoke_run=$SMOKERUN
calibration_image_id=$CAL_IMAGE
calibrated_min_free_mib=$MIN_FREE
hard_ceiling_mib=$HARD_CEILING_MIB
EOF

for GPU in 0 1 2 3; do
  ASSIGN="$RUN/assignments/gpu${GPU}"
  COUNT=$(find "$ASSIGN" -maxdepth 1 -type f -name 'row*.csv' | wc -l)
  if (( COUNT == 0 )); then
    echo "PILOT16_GPU_SKIP|gpu=$GPU|reason=no_assignment"
    continue
  fi
  nohup bash "$REPO/scripts/b24/run_b24_3_pilot_worker.sh" "$GPU" "$ASSIGN" "$RUN" "$MIN_FREE" \
    > "$RUN/gpu${GPU}.log" 2>&1 &
  PID=$!
  printf '%s\n' "$PID" > "$RUN/gpu${GPU}.pid"
  echo "PILOT16_GPU_LAUNCHED|gpu=$GPU|pid=$PID|assigned=$COUNT"
done

printf '%s\n' "$RUN" > "$OUTROOT/B24_3_PILOT16_LATEST_RUN.txt"
echo "B24_3_PILOT16_LAUNCHED|run=$RUN|calibration_reused=$CAL_IMAGE|new_images=15|min_free_mib=$MIN_FREE|head=$HEAD"
echo "STATUS|bash $REPO/scripts/b24/status_b24_3_pilot16.sh"
