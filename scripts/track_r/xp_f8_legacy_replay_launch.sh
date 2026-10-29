#!/usr/bin/env bash
# A10 DAPS 400x5 historical eight-root replay; NEVER starts F8 new roots.
# launch: CPU tests + frozen preflight bytes, starts detached tmux.
# execute: wave1 -> offline check -> wave2 -> offline check.
set -euo pipefail
test "$#" -ge 2 || { echo "USAGE: launch|execute PINNED_GIT_SHA [RUN]"; exit 8; }
MODE="$1"
HEAD="$2"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
PREP="$BASE/xp_f8_legacy_preflight_20261029T052005Z"
INPUT_SHA=f077dd896759d2e3eeaa879394680a0fcef5b816559ee84db46a06a199af2be5
PRECHECK_SHA=c19c632a831c47926301814f2abfa05675c4f46cf104449e34ba98bc78287237

export CUDA_VISIBLE_DEVICES=""
export PYTHONPYCACHEPREFIX="$BASE/.cache/f8_legacy/pycache"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
mkdir -p "$(dirname "$PYTHONPYCACHEPREFIX")"

test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test "$(sha256sum "$PREP/XP_F8_REPLAY_INPUTS_GT_FREE.json" | cut -d' ' -f1)" = "$INPUT_SHA"
test "$(sha256sum "$PREP/XP_F8_PRECHECK_RECEIPT.json" | cut -d' ' -f1)" = "$PRECHECK_SHA"
test -f "$PY"
cd "$WT"

if [ "$MODE" = "launch" ]; then
    echo "===== DAPS LEGACY 400x5 PYTHON TESTS ====="
    "$PY" -m unittest discover -s tests \
        -p 'test_track_r_xp_f8_legacy_preflight.py' -v
    "$PY" -m unittest discover -s tests \
        -p 'test_track_r_xp_f8_legacy_replay.py' -v
    "$PY" - <<'PY'
import ast
from pathlib import Path
for name in (
 "scripts/track_r/xp_f8_legacy_preflight_cpu.py",
 "scripts/track_r/xp_f8_legacy_replay_gpu.py",
 "scripts/track_r/xp_f8_legacy_replay_supervisor.py",
 "scripts/track_r/xp_f8_legacy_replay_offline_cpu.py"):
 ast.parse(Path(name).read_text(),filename=name)
 print("F8_STATIC_PASS",name)
PY
    echo "===== CPU SMOKE: ORIGINAL DAPS IMAGEDATASET WITHOUT GT ====="
    DAPS_PY="$ROOT/conda-envs/daps/bin/python"
    DAPS_NATIVE="$ROOT/pr_diffusion_b19_solver/external/daps"
    test -f "$DAPS_PY"
    CUDA_VISIBLE_DEVICES="" "$DAPS_PY" - "$DAPS_NATIVE" "$BASE" <<'PY'
import sys,tempfile
from pathlib import Path
from PIL import Image
root=Path(sys.argv[1])
base=Path(sys.argv[2])
sys.path.insert(0,str(root))
from data import get_dataset
cache=base/".cache/f8_legacy"
cache.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix="dummy_dataset_",dir=str(cache)) as name:
    dest=Path(name)
    Image.new("RGB",(256,256),(0,0,0)).save(dest/"00000.png")
    dataset=get_dataset("image",root=str(dest),
        resolution=256,start_id=0,end_id=1,device="cpu")
    image=dataset.get_data(1,0)
    assert tuple(image.shape)==(1,3,256,256),image.shape
    assert float(image.max())==-1.0 and float(image.min())==-1.0
    print("F8_NATIVE_DUMMY_DATASET_CPU_SMOKE_PASS",
          "one_original_image_record","no_GT",flush=True)
PY
    echo "===== PRE-LAUNCH FOUR-GPU INVENTORY ====="
    nvidia-smi \
       --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu \
       --format=csv,noheader,nounits
    STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    RUN="$BASE/xp_f8_legacy_replay_$STAMP"
    LOG="$BASE/XP_F8_LEGACY_FULL_GATE_$STAMP.log"
    SESSION="prdiff_f8_8root_gate_$STAMP"
    test ! -e "$RUN" && test ! -e "$LOG"
    tmux new-session -d -s "$SESSION" -c "$WT" \
      "env CUDA_VISIBLE_DEVICES='' PYTHONPATH='$WT' PYTHONPYCACHEPREFIX='$PYTHONPYCACHEPREFIX' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 bash '$WT/scripts/track_r/xp_f8_legacy_replay_launch.sh' execute '$HEAD' '$RUN' > '$LOG' 2>&1"
    echo "F8_REPLAY_SESSION=$SESSION"
    echo "F8_REPLAY_RUN=$RUN"
    echo "F8_REPLAY_LOG=$LOG"
    echo "F8_REPLAY_CODE_HEAD=$HEAD"
    echo "F8_REPLAY_PREP_MANIFEST_SHA=$INPUT_SHA"
    echo "DIRTY_GPU_GENERATION_ALLOWED=true"
    echo "F8_NEW_ROOTS_LAUNCHED=0"
    echo "EIGHT_ROOT_OFFLINE_GATE_AUTO_CHAINED=true"
    exit 0
fi

if [ "$MODE" = "execute" ]; then
    test "$#" -eq 3 || { echo "STOP_MISSING_EXECUTION_RUN"; exit 8; }
    RUN="$3"
    echo "===== START HISTORICAL DAPS REPLAY GATE ====="
    date -u +"UTC_START=%Y-%m-%dT%H:%M:%SZ"
    echo "RUN=$RUN"
    "$PY" -u -m scripts.track_r.xp_f8_legacy_replay_supervisor \
        --expected-head "$HEAD" --wave 1 --run "$RUN"
    echo "===== OFFLINE WAVE1: R0.7 HISTORICAL CANDIDATES ONLY ====="
    "$PY" -u -m scripts.track_r.xp_f8_legacy_replay_offline_cpu \
        --expected-head "$HEAD" --run "$RUN" --wave 1
    echo "===== WAVE1 PASSED; START DEV80 NATIVE REPLAY ====="
    "$PY" -u -m scripts.track_r.xp_f8_legacy_replay_supervisor \
        --expected-head "$HEAD" --wave 2 --run "$RUN"
    echo "===== OFFLINE WAVE2: EIGHT HISTORICAL ROOTS ====="
    "$PY" -u -m scripts.track_r.xp_f8_legacy_replay_offline_cpu \
        --expected-head "$HEAD" --run "$RUN" --wave 2
    echo "===== ALL HISTORICAL ROOTS VALIDATED ====="
    "$PY" - "$RUN" <<'PY'
import sys,json,hashlib
from pathlib import Path
p=Path(sys.argv[1]);receipts=[]
for wave in (1,2):
    gpu=json.loads((p/f"XP_F8_LEGACY_REPLAY_WAVE{wave}.json").read_text())
    cpu=json.loads((p/f"XP_F8_LEGACY_REPLAY_WAVE{wave}_OFFLINE.json").read_text())
    assert gpu["status"]==f"XP_F8_LEGACY_WAVE{wave}_GPU_PASS"
    assert cpu["status"]==f"XP_F8_LEGACY_WAVE{wave}_OFFLINE_PASS"
    receipts.append((gpu,cpu))
print("WAVE1_GPU_H",round(receipts[0][0]["incremental_gpu_h"],4))
print("WAVE2_GPU_H",round(receipts[1][0]["incremental_gpu_h"],4))
print("DAPS_REPLAY_ADDITIONAL_GPU_H",round(
 sum(x[0]["incremental_gpu_h"] for x in receipts),4))
print("EIGHT_ROOTS_OFFLINE_PASS",receipts[1][1]["n_pass"],"/8")
for filename in ("XP_F8_LEGACY_REPLAY_WAVE1_OFFLINE.json",
                 "XP_F8_LEGACY_REPLAY_WAVE2_OFFLINE.json"):
    path=p/filename
    print("SHA256",hashlib.sha256(path.read_bytes()).hexdigest(),filename)
PY
    echo "F8_LEGACY_EIGHT_ROOT_REPLAY_ALL_GATES_PASS"
    echo "F8_920_NEW_ROOTS_NOT_YET_STARTED"
    date -u +"UTC_END=%Y-%m-%dT%H:%M:%SZ"
    exit 0
fi
echo "STOP_F8_UNSUPPORTED_MODE=$MODE"
exit 8
