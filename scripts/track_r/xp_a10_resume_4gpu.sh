#!/usr/bin/env bash
# A10 in-place GT-free B16 continuation, with 4-device batch admissions.
# Creates no new B16 manifest; keeps original native runner and all receipts.
set -euo pipefail

test "$#" -eq 1 || { echo "USAGE: xp_a10_resume_4gpu.sh PINNED_CONTROLLER_SHA"; exit 8; }
HEAD="$1"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
RUN="$BASE/xp_a11_b16_prod_20261028T113034Z"
RUNNER="$ROOT/pr_diffusion_track_r_xp_handoff_20261028"
A10="$ROOT/pr_diffusion_track_r_xp_a10_20261028"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"

OLD=079228b3ebe55dba2ac7bc7ae2c8187113290409
PLANNER=4144e0bc2cc073b05260b18781bd620f36e37763
MANIFEST=778ce4fa7d5ef9299e2530bbb32e8a28b3512e5abe97f0569bb3ec74e00247a9

export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONPATH="$A10:$BASE/a7_cpu_dependencies"

test "$(git -C "$A10" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$A10" status --porcelain)"
test "$(git -C "$RUNNER" rev-parse HEAD)" = "$OLD"
test -z "$(git -C "$RUNNER" status --porcelain)"
test ! -e "$RUN/STOP_AFTER_ACTIVE" || {
    echo "ACTION NEEDED: existing STOP_AFTER_ACTIVE marker; do not override user stop"
    exit 8
}

echo "===== A10 CODE AND GPU ADMISSION TESTS ====="
cd "$A10"
"$PY" -m unittest discover -s tests \
    -p test_track_r_xp_a11_b16_handoff.py -v
"$PY" -m unittest discover -s tests \
    -p test_track_r_xp_a10_d1p2_cpu.py -v
"$PY" - scripts/track_r/xp_a11_b16_handoff_autopilot.py <<'PY'
import ast,sys
from pathlib import Path
p=Path(sys.argv[1]);ast.parse(p.read_text(),filename=str(p))
print("A10_CONTROLLER_SYNTAX_PASS")
PY

echo "===== READ-ONLY CAMPAIGN INVENTORY ====="
"$PY" - "$RUN" "$OLD" "$MANIFEST" <<'PY'
import json,hashlib,sys
from pathlib import Path
run=Path(sys.argv[1]);head=sys.argv[2];expected=sys.argv[3]
p=run/"XP_A11_B16_PROD_TASKS.json"
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
s=json.loads((run/"B16_PRODUCTION_STATE.json").read_text())
ad=json.loads((run/"B16_PRODUCTION_ADMISSION.json").read_text())
assert s["head"]==head and s["manifest_sha256"]==expected
assert ad["head"]==head and ad["gate_n_pass"]==400
assert ad["no_new_outcomes_opened"] is True
print("STATE:",s["status"],"DONE_TASKS:",len(s["done"]),
      "SAVED_ACTIVE:",len(s["active"]),
      "RECORDED_GPU_H:",round((s["spent_gpu_seconds"]+
                             s["gate_spent_gpu_seconds"])/3600,3))
print("PREVIOUS_DEADLINE_EPOCH:",s["deadline_epoch"])
if s["status"] in (
    "B16_AND_B32_GENERATION_PASS_EMBARGOED",
    "A10_B16_GENERATION_COMPLETE_WAIT_F8",
):
    print("NO_B16_CONTINUATION_NEEDED: full B16 is already complete")
    raise SystemExit(12)
if s["status"] not in (
    "B16_PRODUCTION_RUNNING",
    "B16_PRODUCTION_DRAINED_RESUMABLE",
    "B16_PRODUCTION_DRAINED_BUDGET",
    "STOP_B16_PRODUCTION_WORKER_OR_RECEIPT",
):
    raise RuntimeError("STOP_UNKNOWN_OLD_STATUS: "+s["status"])
if s.get("A10_planner_sha"):
    raise RuntimeError("STOP_ALREADY_A10_EXTENDED_CHECK_EXISTING_RUN")
bad=[(i,count) for i,count in s["attempts"].items()
     if int(count)>=2 and int(i) not in set(s["done"])]
if bad:raise RuntimeError("STOP_RETRY_EXHAUSTED_NEEDS_AUDIT "+str(bad[:12]))
PY
RC=$?
if [ "$RC" = 12 ]; then exit 0; fi
test "$RC" = 0

# No external PID is stopped. If the old *Track R controller* is still
# alive, request its own drain via the existing file protocol.
echo "===== CHECK OLD CONTROLLER AND OWN WORKERS ====="
old_alive=$("$PY" - "$RUN" <<'PY'
import sys
from pathlib import Path
needle=sys.argv[1].encode()
ctl=[]
for p in Path("/proc").iterdir():
    if not p.name.isdigit():continue
    try:cmd=(p/"cmdline").read_bytes()
    except (OSError,PermissionError):continue
    if needle in cmd and b"xp_a11_b16_handoff_autopilot.py" in cmd:
        ctl.append(int(p.name))
print(len(ctl))
PY
)
if [ "$old_alive" -gt 0 ]; then
    echo "OLD_CONTROLLER_PRESENT: requesting graceful drain (no kills)"
    touch "$RUN/STOP_AFTER_ACTIVE"
    drained=0
    for _ in $(seq 1 180); do
        "$PY" - "$RUN" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]+"/B16_PRODUCTION_STATE.json"))
print("DRAIN_PROGRESS",s["status"],"active",len(s["active"]))
PY
        old_alive=$("$PY" - "$RUN" <<'PY'
import sys
from pathlib import Path
needle=sys.argv[1].encode()
a=[]
for p in Path("/proc").iterdir():
    if not p.name.isdigit():continue
    try:cmd=(p/"cmdline").read_bytes()
    except (OSError,PermissionError):continue
    if needle in cmd and b"xp_a11_b16_handoff_autopilot.py" in cmd:
        a.append(p.name)
print(len(a))
PY
)
        if [ "$old_alive" -eq 0 ]; then drained=1; break; fi
        sleep 5
    done
    if [ "$drained" -ne 1 ]; then
        echo "ACTION NEEDED: old controller still draining. Leave stop marker."
        exit 8
    fi
fi

echo "===== REQUIRE NO ORPHANED TRACK R PROCESSES ====="
"$PY" - "$RUN" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1]);needle=str(run).encode()
state=json.loads((run/"B16_PRODUCTION_STATE.json").read_text())
live=[]
for p in Path("/proc").iterdir():
    if not p.name.isdigit():continue
    try:cmd=(p/"cmdline").read_bytes()
    except (OSError,PermissionError):continue
    if needle in cmd and (
        b"xp_a11_b16_prod_worker_gpu.py" in cmd or
        b"xp_a11_b16_handoff_autopilot.py" in cmd):
        live.append(p.name)
assert not live, "STOP_ORPHANED_TRACK_R_PROCESSES "+str(live)
assert state["active"]==[], "STOP_STALE_ACTIVE_REQUIRE_MANUAL_RECEIPT_AUDIT"
print("SAFE_TO_RESUME",state["status"],"completed_tasks",len(state["done"]))
PY

test ! -e "$RUN/B16_A10_PRE_EXTENSION_STATE_SNAPSHOT.json" || {
    echo "ACTION NEEDED: existing A10 snapshot; audit before repeating"
    exit 8
}
cp -p "$RUN/B16_PRODUCTION_STATE.json" \
      "$RUN/B16_A10_PRE_EXTENSION_STATE_SNAPSHOT.json"

if [ -e "$RUN/STOP_AFTER_ACTIVE" ]; then
    rm "$RUN/STOP_AFTER_ACTIVE"
fi

echo "===== START A10 4-GPU CONTROLLER ====="
SESSION="prdiff_a10_batch_$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$RUN/A10_BATCH_$(date -u +%Y%m%dT%H%M%SZ).log"
test ! -e "$LOG"
tmux new-session -d -s "$SESSION" -c "$A10" \
    "env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$PYTHONPATH' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 '$PY' -u '$A10/scripts/track_r/xp_a11_b16_handoff_autopilot.py' --run '$RUN' --expected-head '$OLD' --controller-head '$HEAD' --a10 --planner-sha '$PLANNER' --reconcile-completed --max-parallel 4 --max-gpu-hours 70 --poll 3 > '$LOG' 2>&1"

sleep 12
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ACTION NEEDED: A10_CONTROLLER_EARLY_EXIT"
    tail -n 70 "$LOG"
    exit 8
fi
echo "A10_4GPU_BATCH_CONTROLLER_ALIVE"
echo "SESSION=$SESSION"
echo "LOG=$LOG"
echo "RUN=$RUN"
echo "CONTROLLER_HEAD=$HEAD"
echo "SCIENTIFIC_WORKER_HEAD=$OLD"
echo "ORIGINAL_MANIFEST_SHA=$MANIFEST"
echo "POLL_SECONDS=3"
echo "MAX_ONE_TRACK_R_WORKER_PER_PHYSICAL_GPU=YES"
echo "A10_CO_TENANCY_ALLOWED=YES"
echo "CAP_TOTAL_GPU_H=70"
echo "DEADLINE_OCT30_12_EDT=true"
"$PY" -m scripts.track_r.xp_a11_b16_status_cpu --run "$RUN"
tail -n 25 "$LOG"
