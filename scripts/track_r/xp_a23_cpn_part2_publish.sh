#!/usr/bin/env bash
# A18 CP-N Part 2 full100x64, CPU offline only, no native GPU generation.
# Validates SHA-pinned v5 forecast prior to outcome scoring and publishes
# only sanitized data, quantitative prereg comparisons, and receipts.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: bash $0 PINNED_A23_HEAD" >&2;exit 8; }
HEAD="$1"
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$BASE/xp_a23_cpn_part2_$STAMP"
PUB="$ROOT/pr_diffusion_track_r_cpn2_pub_$STAMP"
BRANCH="codex/prdiff-track-r-cpn-part2-$STAMP"
REL="docs/track_r/executors/xp/returns/XP_N_CPN_PART2_$STAMP"
THEORY="docs/track_r/executors/xp/theory_inputs/XP_N_RUNS_0_63"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$WT" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
cd "$WT"
echo "A23_CPN_PART2_CPU_PREFLIGHT"
"$PY" scripts/track_r/xp_a23_cpn_part2_cpu.py \
  --expected-head "$HEAD" --out "$OUT" --preflight-only
echo "A23_CPN_PART2_GT_EVALUATION_AND_FROZEN_G1_CLIP_START"
"$PY" scripts/track_r/xp_a23_cpn_part2_cpu.py \
  --expected-head "$HEAD" --out "$OUT"
"$PY" - "$OUT" <<'PY'
import csv,json,sys
from pathlib import Path
p=Path(sys.argv[1])
r=json.loads((p/"XP_N_CPN_PART2_RECEIPT.json").read_text())
assert r["status"]=="A18_XPN_CPN_PART2_ALL100_N64_SCORED"
assert r["full100_K64"] and r["n_images"]==100
assert r["v5_all_K_prediction_rows"]==100
assert r["protocol_v1_unmodified"]
assert r["no_matched_compute_claim"] is True
v=json.loads((p/"XP_N_V5_PREDICTION_COMPARISON.json").read_text())
assert v["n_predictions"]==100
assert v["original_registration_commit"]=="55ba86b71cafbd8eb066c777ac141823d9cd9a1d"
with (p/"XP_N_CPN64_STRINGS.csv").open(newline="") as f:
 s=list(csv.DictReader(f))
assert len(s)==100
assert all(len(x["labels_by_run"].split(";"))==64 for x in s)
with (p/"XP_N_CPN64_G1_DECISIONS.csv").open(newline="") as f:
 c=list(csv.DictReader(f))
assert len(c)==6400
assert all(0<=int(x["run"])<64 for x in c)
assert len((p/"CPN_PART2_PLANNER_RETURN.txt").read_text().splitlines())<=20
print("A23_CPN_PART2_SANITIZED_RESULT_GATE_PASS")
PY
test ! -e "$PUB"
git -C "$WT" worktree add -b "$BRANCH" "$PUB" "$HEAD"
mkdir -p "$PUB/$REL" "$PUB/$THEORY"
for f in XP_N_CPN_PART2_RECEIPT.json XP_N_V5_PREDICTION_COMPARISON.json \
  XP_N_CPN64_PROTOCOL_PRIMARY.json XP_N_CPN64_PROTOCOL_SECONDARY.json \
  CPN_PART2_PLANNER_RETURN.txt;do
  cp "$OUT/$f" "$PUB/$REL/"
done
cp "$OUT/XP_N_CPN64_STRINGS.csv" "$PUB/$THEORY/"
cp "$OUT/XP_N_CPN64_G1_DECISIONS.csv" "$PUB/$THEORY/"
cat > "$PUB/$REL/README.md" <<'EOF'
# A18 CP-N Part 2: 100 fresh ImageNet images / 64 native FAST runs
- All 6,400 native roots accepted and checkpointed under <28 GPU-hours.
- Frozen v1 protocol c598176, M=2, primary tau 0.0500829,
  secondary tau 0.050434426327793.
- v5 at K=4/8/16/32/64 evaluated only on the 64-run full panel,
  compared against predictions registered BEFORE held outcomes.
- Frozen CLIP G1 orientation choices; GT offline quality only.
- No matched-compute claim, no native image/GT file paths published.
- P-N5 partial-only images are a diagnostic of an unrecoverable-orbit
  component; not a causal attribution of the full E1-oracle gap.
EOF
(
 cd "$PUB/$REL"
 sha256sum XP_N_CPN_PART2_RECEIPT.json XP_N_V5_PREDICTION_COMPARISON.json \
  XP_N_CPN64_PROTOCOL_PRIMARY.json XP_N_CPN64_PROTOCOL_SECONDARY.json \
  CPN_PART2_PLANNER_RETURN.txt README.md > SHA256SUMS
 sha256sum -c SHA256SUMS
)
(
 cd "$PUB/$THEORY"
 sha256sum XP_N_CPN64_STRINGS.csv XP_N_CPN64_G1_DECISIONS.csv > SHA256SUMS
 sha256sum -c SHA256SUMS
)
git -C "$PUB" add "$REL" "$THEORY"
git -C "$PUB" -c core.whitespace=cr-at-eol diff --cached --check
git -C "$PUB" commit -m "results(track-r): A18 CP-N part2 all100 n64 v5 prospective scoring and exact v1 frontier"
GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=10" \
 git -C "$PUB" push git@github.com:Cheng-HanHuang/pr_diffusion.git "HEAD:refs/heads/$BRANCH"
LOCAL="$(git -C "$PUB" rev-parse HEAD)"
REMOTE="$(git -C "$PUB" ls-remote git@github.com:Cheng-HanHuang/pr_diffusion.git "refs/heads/$BRANCH" | cut -f1)"
test -n "$REMOTE" && test "$LOCAL" = "$REMOTE"
echo "A23_CPN_PART2_REMOTE_RETURN_PUBLISHED=$LOCAL"
echo "A23_CPN_PART2_REMOTE_BRANCH=$BRANCH"
echo "A23_CPN_PART2_REPORT=$OUT/CPN_PART2_PLANNER_RETURN.txt"
