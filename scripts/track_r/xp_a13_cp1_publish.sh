#!/usr/bin/env bash
# A13 CP1 curated publication. No GT paths or per-candidate metric tables.
set -euo pipefail
ROOT=/egr/research-pac/huang248
BASE="$ROOT/outputs/pr_diffusion/track_r"
SOURCE="$ROOT/pr_diffusion_track_r_xp_a13"
WT="$ROOT/pr_diffusion_track_r_xp_a13_cp1_publish"
HEAD="$1"
[[ "$HEAD" =~ ^[a-f0-9]{40}$ ]] || { echo STOP_SOURCE_SHA;exit 8; }
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$HEAD"
test -z "$(git -C "$SOURCE" status --porcelain)"
test ! -e "$WT"
B16=$(cat "$BASE/XP_A13_B16_PROTOCOL_LATEST_RUN.txt")
F8S=$(cat "$BASE/XP_A13_F8_SCORE_LATEST.txt")
F8C=$(cat "$BASE/XP_A13_F8_CLIP_LATEST.txt")
F8F=$(cat "$BASE/XP_A13_F8_COMPARE_LATEST.txt")
F8P=$(cat "$BASE/XP_A13_F8_PROTOCOL_LATEST.txt")
for p in "$B16" "$F8S" "$F8C" "$F8F" "$F8P"; do
 case "$p" in "$BASE"/xp_a13_*) test -d "$p";;
  *) echo STOP_INPUT_DIR_NOT_CANONICAL;exit 8;; esac
done
python - "$B16" "$F8S" "$F8C" "$F8F" "$F8P" <<'PY'
import json,sys
from pathlib import Path
b,s,c,f,p=map(Path,sys.argv[1:])
expected=[
 (b,"XP_A13_B16_PROTOCOL_COMPLETE.json","XP_A13_B16_PROTOCOL_V1_FROZEN_COMPLETE"),
 (s,"XP_A13_F8_SCORING_RECEIPT.json","XP_A13_F8_1440_CPU_SCORED_PASS"),
 (c,"XP_A13_F8_FACECLIP_RECEIPT.json","XP_A13_F8_1440_N2_FACE_CLIP_PASS"),
 (f,"XP_A13_F8_COMPARISON_RECEIPT.json","XP_A13_F8_V1_PRIMARY_V3_SECONDARY_COMPARISON_PASS"),
 (p,"XP_A13_F8_PROTOCOL_COMPLETE.json","XP_A13_F8_FROZEN_PROTOCOL_V1_COMPLETE")]
for parent,name,status in expected:
    assert json.loads((parent/name).read_text())["status"]==status,(name,status)
PY
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BRANCH=codex/prdiff-track-r-xp-a13-cp1-$STAMP
git -C "$SOURCE" worktree add -b "$BRANCH" "$WT" "$HEAD"
OUT="$WT/docs/track_r/executors/xp/returns/XP_A13_CP1_$STAMP"
THEORY="$WT/docs/track_r/executors/xp/theory_inputs"
mkdir -p "$OUT" "$THEORY"
cp "$F8S/XP_A13_F8_8RUN_STRINGS.csv" "$OUT/"
cp "$F8C/XP_A13_F8_8RUN_FACE_CLIP_DECISIONS.csv" "$OUT/"
cp "$F8F/XP_A13_F8_PROSPECTIVE_V1_V3.json" "$OUT/"
for VARIANT in primary E0 self_mad3 frontier; do
 cp "$B16/B16_"$VARIANT"_POOLED.json" "$OUT/"
 cp "$F8P/F8_"$VARIANT"_POOLED.json" "$OUT/"
done
cp "$F8P/XP_A13_F8_PROTOCOL_COMPLETE.json" "$OUT/"
cp "$B16/XP_A13_B16_PROTOCOL_COMPLETE.json" "$OUT/"
cp "$F8F/XP_A13_F8_COMPARISON_RECEIPT.json" "$OUT/"
cp "$F8S/XP_A13_F8_8RUN_STRINGS.csv" "$THEORY/XP_A13_F8_8RUN_STRINGS.csv"
cp "$F8C/XP_A13_F8_8RUN_FACE_CLIP_DECISIONS.csv" "$THEORY/XP_A13_F8_8RUN_FACE_CLIP_DECISIONS.csv"
(cd "$OUT"; sha256sum * > SHA256SUMS)
python - "$OUT" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1])
b=json.loads((out/"B16_primary_POOLED.json").read_text())
f=json.loads((out/"F8_primary_POOLED.json").read_text())
lines=[
"# A13 checkpoint CP1 — B16 and F8, protocol v1",
"",
"Full native F8 scoring/CLIP on PAC, frozen v1/v3; unchanged protocol c598176.",
"No GT paths, reconstructions, or per-candidate metric tables committed.",
"",
"| Panel | Success | Mean runs | Gain vs equal-compute E1 |",
"| --- | ---: | ---: | ---: |",
f"| B16 | {b['weighted']['success_all']:.6f} | {b['weighted']['runs_mean']:.6f} | {b['weighted']['gain_vs_fixed_equal_runs']:+.6f} |",
f"| F8 | {f['weighted']['success_all']:.6f} | {f['weighted']['runs_mean']:.6f} | {f['weighted']['gain_vs_fixed_equal_runs']:+.6f} |",
"",
f"H1 B16: {b['H1_PASS']}. H1 F8: {f['H1_F8']}. H2: {f['H2']}. H3 F8: {f['H3_F8']}.",
"SHA256SUMS authenticates published artifacts; full raw frozen results remain on PAC."]
(out/"README.md").write_text("\n".join(lines)+"\n")
PY
git -C "$WT" add docs/track_r/executors/xp
git -C "$WT" diff --cached --check
git -C "$WT" commit -m 'results(track-r): A13 CP1 B16/F8 frozen protocol and F8 v1/v3 theory inputs'
git -C "$WT" push -u origin "$BRANCH"
echo 'A13_CP1_COMMITTED_AND_PUSHED'
git -C "$WT" rev-parse HEAD
echo "A13_CP1_BRANCH=$BRANCH"
echo "A13_CP1_OUTPUT=$OUT"
