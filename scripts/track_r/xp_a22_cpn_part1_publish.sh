#!/usr/bin/env bash
# CP-N part 1 A18 CPU-only on frozen all100/n16; SSH-publish sanitized
# strings, G1 choices, scalar reports. Never publish GT/local path scores.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Usage: bash $0 PINNED_HEAD" >&2;exit 8; }
HEAD="$1"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$BASE/xp_a22_cpn_part1_$STAMP"
PUB="$ROOT/pr_diffusion_track_r_cpn1_pub_$STAMP"
BRANCH="codex/prdiff-track-r-cpn-part1-$STAMP"
REL="docs/track_r/executors/xp/returns/XP_N_CPN_PART1_$STAMP"
THEORY="docs/track_r/executors/xp/theory_inputs/XP_N_RUNS_0_15"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$WT" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
cd "$WT"

"$PY" scripts/track_r/xp_a22_cpn_part1_cpu.py \
    --expected-head "$HEAD" --out "$OUT" --preflight-only
"$PY" scripts/track_r/xp_a22_cpn_part1_cpu.py \
    --expected-head "$HEAD" --out "$OUT"
"$PY" - "$OUT" <<'PY'
import csv,json,sys
from pathlib import Path
p=Path(sys.argv[1])
rec=json.loads((p/"XP_N_CPN_PART1_RECEIPT.json").read_text())
assert rec["status"]=="A18_CPN_PART1_100_IMAGE_16RUN_PASS"
assert rec["v5_not_scored_at_K16"] and rec["no_held_16_to_63_outcomes_read"]
assert rec["n_images"]==100 and rec["n_runs"]==1600
with (p/"XP_N_CPN16_STRINGS.csv").open(newline="") as f:
 rows=list(csv.DictReader(f))
assert len(rows)==100
assert all(len(r["labels_by_run"].split(";"))==16 for r in rows)
with (p/"XP_N_CPN16_G1_DECISIONS.csv").open(newline="") as f:
 rows=list(csv.DictReader(f))
assert len(rows)==1600
assert all(int(r["run"])<16 for r in rows)
lines=(p/"CPN_PART1_PLANNER_RETURN.txt").read_text().splitlines()
assert len(lines)<=20
print("A22_CPN_PART1_SANITIZED_RETURN_PREFLIGHT_PASS",len(lines))
PY

test ! -e "$PUB"
git -C "$WT" worktree add -b "$BRANCH" "$PUB" "$HEAD"
mkdir -p "$PUB/$REL" "$PUB/$THEORY"
cp "$OUT/XP_N_CPN_PART1_RECEIPT.json" "$PUB/$REL/"
cp "$OUT/CPN_PART1_PLANNER_RETURN.txt" "$PUB/$REL/"
cp "$OUT/XP_N_CPN16_STRINGS.csv" "$PUB/$THEORY/"
cp "$OUT/XP_N_CPN16_G1_DECISIONS.csv" "$PUB/$THEORY/"
cat > "$PUB/$REL/README.md" <<'EOF'
# CP-N Part 1, prereg A18, complete 100 ImageNet images × 16 runs
This is a GT-derived offline evaluation of an unchanged native solver,
frozen residual selection and zero-shot G1 CLIP orientation.
v1 protocol c598176, N_MAX=16, primary tau 0.0500829,
secondary tau 0.050434426327793. Only runs 0-15 used.
The separately registered v5 n=64 forecast is NOT scored here.
No GT tensors, native reconstruction paths, local scores or matched-NFE
claims are released in this return.
EOF
(cd "$PUB/$REL" && sha256sum XP_N_CPN_PART1_RECEIPT.json CPN_PART1_PLANNER_RETURN.txt README.md > SHA256SUMS && sha256sum -c SHA256SUMS)
(cd "$PUB/$THEORY" && sha256sum XP_N_CPN16_STRINGS.csv XP_N_CPN16_G1_DECISIONS.csv > SHA256SUMS && sha256sum -c SHA256SUMS)
git -C "$PUB" add "$REL" "$THEORY"
git -C "$PUB" -c core.whitespace=cr-at-eol diff --cached --check
git -C "$PUB" commit -m "results(track-r): A18 CP-N part1 all100 n16 pre-registered protocol and GT-free inputs"
GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=10" \
 git -C "$PUB" push git@github.com:Cheng-HanHuang/pr_diffusion.git "HEAD:refs/heads/$BRANCH"
LOCAL="$(git -C "$PUB" rev-parse HEAD)"
REMOTE="$(git -C "$PUB" ls-remote git@github.com:Cheng-HanHuang/pr_diffusion.git "refs/heads/$BRANCH" | cut -f1)"
test -n "$REMOTE" && test "$LOCAL" = "$REMOTE"
echo "A22_CPN_PART1_REMOTE_RETURN_PUBLISHED=$LOCAL"
echo "A22_CPN_PART1_REMOTE_BRANCH=$BRANCH"
echo "A22_CPN_PART1_REPORT=$OUT/CPN_PART1_PLANNER_RETURN.txt"
