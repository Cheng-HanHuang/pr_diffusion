#!/usr/bin/env bash
set -euo pipefail

ROOT=/egr/research-pac/huang248
CONTROL="$ROOT/pr_diffusion_b23"
REPO="$ROOT/pr_diffusion_b24"
OUTROOT="$ROOT/outputs/pr_diffusion/b24"
BRANCH=codex/b24-bestof4-failure-sweep
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
LATEST7424="$OUTROOT/B24_2_7424_LATEST_RUN.txt"
EXPECTED_PRE_B24_SHA=d475c9c29b4f6ab2839ae21f4b19e33a52fa46f2fd7f0a6a7c5fff491e4b3068

[[ -x "$PY" ]] || { echo "STOP|missing_python:$PY"; exit 2; }
[[ -f "$LATEST7424" ]] || { echo "STOP|missing_7424_pointer:$LATEST7424"; exit 2; }
RUN7424=$(cat "$LATEST7424")
[[ -d "$RUN7424" ]] || { echo "STOP|missing_7424_runroot:$RUN7424"; exit 2; }
[[ -f "$RUN7424/B24_2_baseline_7424.json" ]] || {
  echo "STOP|missing_7424_manifest:$RUN7424/B24_2_baseline_7424.json"; exit 2;
}

# This closeout is intentionally zero-GPU.  No model/reconstruction command is invoked.
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1

[[ -z "$(git -C "$REPO" status --porcelain)" ]] || {
  echo "STOP|B24_worktree_dirty"; git -C "$REPO" status --short; exit 3;
}

git -C "$CONTROL" fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
LOCAL=$(git -C "$REPO" rev-parse HEAD)
REMOTE=$(git -C "$REPO" rev-parse "origin/$BRANCH")
git -C "$REPO" merge-base --is-ancestor "$LOCAL" "$REMOTE" || {
  echo "STOP|local_not_ancestor|local=$LOCAL|remote=$REMOTE"; exit 4;
}
if [[ "$LOCAL" != "$REMOTE" ]]; then
  git -C "$REPO" merge --ff-only "origin/$BRANCH"
fi
HEAD=$(git -C "$REPO" rev-parse HEAD)

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

OBS_PRE_B24_SHA=$(sha256sum manifests/b24/PRE_B24_EXPOSURE.csv | awk '{print $1}')
[[ "$OBS_PRE_B24_SHA" == "$EXPECTED_PRE_B24_SHA" ]] || {
  echo "STOP|PRE_B24_sha|observed=$OBS_PRE_B24_SHA|expected=$EXPECTED_PRE_B24_SHA"; exit 4;
}
echo "PRE_B24_READY|sha256=$OBS_PRE_B24_SHA"

"$PY" -m py_compile \
  scripts/b24/freeze_b24_7424_case_universe.py \
  scripts/b24/freeze_b24_abc300.py
"$PY" -m unittest discover -s tests/b24 -p 'test_*.py'
echo "B24_FREEZE_TESTS_PASS|head=$HEAD|gpu_work=0"

FREEZE="$RUN7424/case_freeze"
if [[ -e "$FREEZE/B24_ABC300_FREEZE_SUMMARY.json" ]]; then
  echo "STOP|ABC300_freeze_already_exists:$FREEZE/B24_ABC300_FREEZE_SUMMARY.json"
  exit 5
fi
mkdir -p "$FREEZE"

"$PY" scripts/b24/freeze_b24_7424_case_universe.py \
  --run7424 "$RUN7424" \
  --out-dir "$FREEZE" \
  | tee "$FREEZE/freeze_7424_case_universe.log"

"$PY" scripts/b24/freeze_b24_abc300.py \
  --universe-csv "$FREEZE/B24_7424_CASE_UNIVERSE.csv" \
  --universe-summary "$FREEZE/B24_7424_CASE_FREEZE_SUMMARY.json" \
  --out-dir "$FREEZE" \
  | tee "$FREEZE/freeze_abc300.log"

FILES=(
  B24_7424_CASE_UNIVERSE.csv
  B24_7424_CLASS_RANKED.json
  B24_7424_CASE_FREEZE_SUMMARY.json
  B24_ABC300_FROZEN.csv
  B24_ABC300_FROZEN.json
  B24_ABC300_FREEZE_SUMMARY.json
)
for f in "${FILES[@]}"; do
  [[ -f "$FREEZE/$f" ]] || { echo "STOP|missing_freeze_output:$FREEZE/$f"; exit 6; }
done

(
  cd "$FREEZE"
  sha256sum "${FILES[@]}" > SHA256SUMS.txt
)

MAN_SHA=$(sha256sum "$RUN7424/B24_2_baseline_7424.json" | awk '{print $1}')
"$PY" - "$FREEZE" "$RUN7424" "$HEAD" "$MAN_SHA" "$OBS_PRE_B24_SHA" <<'PY'
import hashlib,json,sys
from pathlib import Path
freeze=Path(sys.argv[1]); run=Path(sys.argv[2]); head=sys.argv[3]; manifest_sha=sys.argv[4]; exposure_sha=sys.argv[5]
files=[
    'B24_7424_CASE_UNIVERSE.csv',
    'B24_7424_CLASS_RANKED.json',
    'B24_7424_CASE_FREEZE_SUMMARY.json',
    'B24_ABC300_FROZEN.csv',
    'B24_ABC300_FROZEN.json',
    'B24_ABC300_FREEZE_SUMMARY.json',
    'SHA256SUMS.txt',
]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
summary=json.loads((freeze/'B24_7424_CASE_FREEZE_SUMMARY.json').read_text())
cohort=json.loads((freeze/'B24_ABC300_FREEZE_SUMMARY.json').read_text())
assert summary['status']=='PASS', summary
assert cohort['status']=='PASS', cohort
assert cohort['frozen_class_counts']=={'A':100,'B':100,'C':100}, cohort
value={
    'schema_version':'b24.7424-abc300-freeze-provenance.v1',
    'status':'PASS',
    'b24_head':head,
    'run7424':str(run),
    'pre_b24_exposure_sha256':exposure_sha,
    'manifest_file_sha256':manifest_sha,
    'zero_gpu':True,
    'method_execution_performed':False,
    'class_counts':summary['class_counts'],
    'frozen_class_counts':cohort['frozen_class_counts'],
    'cohort_role':'FROZEN_ABC300_ROLE_PENDING_PLANNER',
    'files':{name:sha(freeze/name) for name in files},
    'next':'RETURN_TO_PLANNER_FOR_METHOD_PORTFOLIO_AND_ROLE_POLICY',
}
tmp=freeze/'FREEZE_PROVENANCE.json.tmp'
tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
tmp.replace(freeze/'FREEZE_PROVENANCE.json')
print(json.dumps(value,sort_keys=True))
PY

printf '%s\n' "$FREEZE" > "$OUTROOT/B24_ABC300_LATEST_FREEZE.txt"

echo "B24_7424_ABC300_FREEZE_PASS|runroot=$RUN7424|freeze=$FREEZE|head=$HEAD|manifest_sha256=$MAN_SHA"
echo "RETURN_FILES|$FREEZE/B24_7424_CASE_FREEZE_SUMMARY.json|$FREEZE/B24_ABC300_FREEZE_SUMMARY.json|$FREEZE/B24_ABC300_FROZEN.csv|$FREEZE/FREEZE_PROVENANCE.json|$FREEZE/SHA256SUMS.txt"
