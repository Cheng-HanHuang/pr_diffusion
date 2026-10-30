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
echo '===== A13 CP1 CONSERVATIVE PROJECT LEDGER RECEIPT ====='
LEDGER="$BASE/XP_A12_PROJECT_LEDGER_VERIFIED.json"
G1A="$BASE/xp_a12_g1a_prod_20261029T180624Z/XP_A12_G1A_FINAL_STATUS.json"
G1BRUN=$(cat "$BASE/XP_A13_G1B_LATEST_RUN.txt")
case "$G1BRUN" in "$BASE"/xp_a13_g1b_prod_*) ;; *) echo STOP_G1B_LEDGER_PATH;exit 8;; esac
G1B="$G1BRUN/XP_A13_G1B_FINAL_STATUS.json"
python - "$LEDGER" "$G1A" "$G1B" "$OUT" <<'PYLEDGER'
import hashlib,json,sys
from pathlib import Path
ledger,g1a,g1b,out=map(Path,sys.argv[1:])
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(ledger)=="3eabf7b978e1341549f72690061d65d2aee1c29046c0749df0e2b821d74a2f8c"
L=json.loads(ledger.read_text())
a=json.loads(g1a.read_text())
b=json.loads(g1b.read_text())
assert L["status"]=="XP_A12_CUMULATIVE_PROJECT_LEDGER_VERIFIED"
assert L["has_every_stage_since_XP0"] is True
assert L["project_reported_total_is_conservative_upper_bound"] is True
assert a["status"]=="XP_A12_G1A_1600_GENERATION_COMPLETE_SEALED"
assert a["completed"]==1600 and a["failed_attempts"]==0 and a["replay_pass"] is True
assert a["ledger_sha256"]==digest(ledger)
assert b["status"]=="XP_A13_G1B_3200_GENERATION_COMPLETE"
assert b["completed"]==3200 and b["failed_attempts"]==0
assert b["source_head"]=="bce6fc3ae3afc8c821f0eb8f6a8e47547b1c0264"
assert b["G1a_pixel_exact_replay"] is True and b["no_GT_scoring"] is True
prior=float(L["project_total_before_A12_gpu_h"])
ga=float(a["stage_gpu_h"]); gb=float(b["stage_gpu_h"])
assert abs(prior+ga-float(a["project_total_gpu_h"]))<1e-5
assert abs(prior+ga+gb-float(b["project_total_gpu_h"]))<1e-5
assert abs(ga+gb-float(b["A12_total_gpu_h"]))<1e-5
assert abs(prior-153.581581288)<1e-3
assert abs(gb-6.951031519319707)<1e-4
doc={
 "schema":"prdiff.track_r.a13.cp1.project-ledger.v1",
 "status":"XP_A13_CP1_G1A_G1B_PROJECT_LEDGER_VERIFIED",
 "source_references":{
   "historical_ledger_sha256":digest(ledger),
   "g1a_receipt_sha256":digest(g1a),
   "g1b_receipt_sha256":digest(g1b),
   "g1b_execution_head":b["source_head"],
   "planner_A13_head":"988aaed5fb61099ca43cbe950a181a6c38ab74c7"},
 "project_cap_gpu_h":400.,
 "project_conservative_prior_to_A12_gpu_h":prior,
 "g1a_gpu_h":ga,
 "g1b_gpu_h":gb,
 "project_conservative_after_g1b_gpu_h":float(b["project_total_gpu_h"]),
 "project_remaining_after_g1b_gpu_h":400.-float(b["project_total_gpu_h"]),
 "A12_cap_gpu_h":150.,
 "A12_after_g1b_gpu_h":float(b["A12_total_gpu_h"]),
 "A12_remaining_after_g1b_gpu_h":150.-float(b["A12_total_gpu_h"]),
 "A10_included_in_project_prior_but_excluded_from_A12":True,
 "A10_historical_gpu_h":55.54578541457511,
 "historical_uncertain_gate_conservative_upper_bill_gpu_h":31.314365,
 "further_G2_G7_charge_not_yet_in_this_CP1_snapshot":True,
}
assert 0<doc["A12_remaining_after_g1b_gpu_h"]<150
assert 0<doc["project_remaining_after_g1b_gpu_h"]<400
(out/"XP_A13_CP1_PROJECT_GPU_LEDGER.json").write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
print("CP1_LEDGER_VERIFIED",doc["project_conservative_after_g1b_gpu_h"],doc["A12_after_g1b_gpu_h"])
PYLEDGER
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
"Project ledger: `XP_A13_CP1_PROJECT_GPU_LEDGER.json` (conservative after G1b; later stages separately charged).",
"SHA256SUMS authenticates published artifacts; full raw frozen results remain on PAC."]
(out/"README.md").write_text("\n".join(lines)+"\n")
PY
(cd "$OUT"; sha256sum * > SHA256SUMS)
git -C "$WT" add docs/track_r/executors/xp
git -C "$WT" diff --cached --check
git -C "$WT" commit -m 'results(track-r): A13 CP1 B16/F8 protocol, F8 v1/v3 and project ledger'
git -C "$WT" push -u origin "$BRANCH"
echo 'A13_CP1_COMMITTED_AND_PUSHED'
git -C "$WT" rev-parse HEAD
echo "A13_CP1_BRANCH=$BRANCH"
echo "A13_CP1_OUTPUT=$OUT"
