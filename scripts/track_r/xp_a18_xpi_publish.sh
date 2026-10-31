#!/usr/bin/env bash
# A16 XP-I metric table CPU return, 180x4. No GPU, no GT file paths.
# Create a separate clean checkpoint branch and verify SSH remote commit.
set -euo pipefail
[[ $# == 2 && "$1" == "publish" ]] || {
  echo "Usage: bash $0 publish PINNED_HEAD" >&2
  exit 8
}
HEAD="$2"
ROOT=/egr/research-pac/huang248
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/conda-envs/prdiff_track_r/bin/python"
SRC="$ROOT/outputs/pr_diffusion/track_r/xp_a18_xpi_particles_20261031T224601Z"
CSV="$SRC/A16_XPI_PER_PARTICLE_METRICS.csv"
RECEIPT="$SRC/A16_XPI_PER_PARTICLE_METRICS_RECEIPT.json"
EXPECTED_CSV_SHA=1e16f70ebd36abf81c9662edef2eef813d061ce98296d1c3bfbaba2a4cc17f02
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WT"
test "$(git -C "$WT" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$WT" status --porcelain)"
test "$(sha256sum "$CSV" | cut -d' ' -f1)" = "$EXPECTED_CSV_SHA"

"$PY" - "$CSV" "$RECEIPT" "$EXPECTED_CSV_SHA" <<'PY'
import csv,hashlib,json,math,sys
from collections import Counter
from pathlib import Path
csvfile,receipt,digest=(
    Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3])
r=json.loads(receipt.read_text())
assert r["status"]=="A16_XPI_PER_PARTICLE_METRIC_TABLE_PASS"
assert r["sha256"]==digest
assert r["n_images"]==180 and r["n_particles"]==720
assert r["no_native_tensor_or_path"] and r["no_gpu_calls"]
assert r["score_columns_are_gt_evaluation_only"]
assert r["face_columns_are_gt_free_decisions"]
assert r["no_matched_compute_claim"]
with csvfile.open(newline="",encoding="utf-8") as f:
    dr=csv.DictReader(f)
    expected=["image_id","particle","label","PSNR","SSIM","VGG_LPIPS",
             "rho","face_choice","face_margin"]
    assert dr.fieldnames==expected
    rows=list(dr)
assert len(rows)==720
cnt=Counter(x["image_id"] for x in rows)
assert len(cnt)==180 and set(cnt.values())=={4}
assert all(sorted(int(x["particle"]) for x in rows
           if x["image_id"]==iid)==list(range(4)) for iid in cnt)
for x in rows:
    assert x["face_choice"] in ("000","111")
    assert all(math.isfinite(float(x[k])) for k in (
        "PSNR","SSIM","VGG_LPIPS","rho","face_margin"))
    assert float(x["rho"])>=0
    assert not any(y in json.dumps(x).lower() for y in (
        "/egr/","gt_path","measurement_path","root_","confirmation"))
print("A18_XPI_PUBLISH_720X180_SOURCE_GATE_PASS")
PY

STAMP=20261031T224601Z
BRANCH="codex/prdiff-track-r-xp-a18-xpi-$STAMP"
PUBLISH="$ROOT/pr_diffusion_track_r_a18_xpi_publish_$STAMP"
REL="docs/track_r/executors/xp/returns/XP_A18_XPI_PARTICLES_$STAMP"
test ! -e "$PUBLISH"
git -C "$WT" worktree add -b "$BRANCH" "$PUBLISH" "$HEAD"
mkdir -p "$PUBLISH/$REL"
cp "$CSV" "$PUBLISH/$REL/"
cp "$RECEIPT" "$PUBLISH/$REL/"
cat > "$PUBLISH/$REL/README.md" <<EOF
# XP-I 720-row particle metric table (A16)
CP3 accepted; this is an independent, evaluation-only CPU readout.
All 180 images have four particles. PSNR is RGB8 raw; SSIM and
VGG-LPIPS are from frozen original-float PiQ evaluation.
Face orientation choice/margin is a frozen GT-free decision.
No GT tensors, file paths, solver reruns, or matched-NFE claim.
Source CSV SHA-256: $EXPECTED_CSV_SHA
EOF
(cd "$PUBLISH/$REL" && sha256sum A16_XPI_PER_PARTICLE_METRICS.csv \
   A16_XPI_PER_PARTICLE_METRICS_RECEIPT.json README.md > SHA256SUMS)
(cd "$PUBLISH/$REL" && sha256sum -c SHA256SUMS)
git -C "$PUBLISH" add "$REL"
git -C "$PUBLISH" -c core.whitespace=cr-at-eol diff --cached --check
git -C "$PUBLISH" commit -m "results(track-r): A16 XP-I 180x4 independent per-particle metric table"
GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=10" \
  git -C "$PUBLISH" push git@github.com:Cheng-HanHuang/pr_diffusion.git \
  "HEAD:refs/heads/$BRANCH"
LOCAL="$(git -C "$PUBLISH" rev-parse HEAD)"
REMOTE="$(git -C "$PUBLISH" ls-remote \
  git@github.com:Cheng-HanHuang/pr_diffusion.git \
  "refs/heads/$BRANCH" | cut -f1)"
test -n "$REMOTE" && test "$LOCAL" = "$REMOTE"
echo "A18_XPI_REMOTE_CHECKPOINT_PUBLISHED_AND_VERIFIED=$LOCAL"
echo "A18_XPI_REMOTE_BRANCH=$BRANCH"
