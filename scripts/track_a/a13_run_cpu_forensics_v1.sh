#!/usr/bin/env bash
# A1.3.1: CPU only, never run until stage0 receipt user-return independently verified.
# Invocation: bash scripts/track_a/a13_run_cpu_forensics_v1.sh <exact-pushed-A13-HEAD>
ROOT=/egr/research-pac/huang248
REPO="$ROOT/pr_diffusion_track_a_a13_candidate_audit"
PY="$ROOT/conda-envs/prdiff_ffhq/bin/python"
A1="$ROOT/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
BASE="$ROOT/outputs/pr_diffusion/track_a_np_paper/a13_candidate_audit_20261020"
ST0="$BASE/a13_0_inventory_v1/A13_INVENTORY_COMPLETE.json"
OUT="$BASE/a13_1_cpu_forensics_v1"
ZIP="$BASE/A13_CPU_METADATA_RETURN_V1.zip"
EXPECTED="${1:-}"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
if [ -z "$EXPECTED" ] || [ ! -x "$PY" ] || [ ! -f "$ST0" ]; then
  echo "STOP_A13_1: missing exact HEAD, interpreter or verified stage0 receipt"
elif [ -e "$OUT" ] || [ -e "$ZIP" ] || [ -L "$OUT" ] || [ -L "$ZIP" ]; then
  echo "STOP_A13_1: stage1 output or ZIP occupied; no overwrite"
else
  cd "$REPO" || echo "STOP_A13_1: cannot enter dedicated source"
  if [ "$(git rev-parse HEAD 2>/dev/null)" != "$EXPECTED" ]; then
    echo "STOP_A13_1: HEAD mismatch"
  else
    TEST_LOG=$("$PY" -m unittest scripts.track_a.test_a13_physics -v 2>&1)
    TEST_RC=$?
    printf "%s\\n" "$TEST_LOG" | tail -24
    echo "A13_TEST_RC=$TEST_RC"
    if [ "$TEST_RC" -ne 0 ]; then
      echo "STOP_A13_1: synthetic tests failed"
    else
      ANALYSIS_LOG=$("$PY" -m scripts.track_a.a13_analysis --stage analyze \
        --expected-head "$EXPECTED" \
        --run-root "$A1/a1_2_exposed_b22_k4_v1" \
        --input-root "$A1/a1_2_exposed_b22_inputs_v1" --out "$OUT" 2>&1)
      ANALYSIS_RC=$?
      printf "%s\\n" "$ANALYSIS_LOG" | tail -25
      echo "A13_ANALYSIS_RC=$ANALYSIS_RC"
      if [ "$ANALYSIS_RC" -ne 0 ]; then
        echo "STOP_A13_1: no packaging; preserve any partial A1.3 output"
      else
        printf "%s\\n" "$TEST_LOG" > "$OUT/A13_SYNTHETIC_TESTS.log"
        printf "%s\\n" "$ANALYSIS_LOG" > "$OUT/A13_ANALYSIS_STDOUT_STDERR.log"
        # Compact metadata-only archive, exclusive creation; never include GT/candidates.
        "$PY" - "$OUT" "$ZIP" <<'PYTHON'
import pathlib,sys,zipfile,hashlib
src=pathlib.Path(sys.argv[1]); dst=pathlib.Path(sys.argv[2])
allowed={p.name for p in src.iterdir() if p.suffix in {".csv",".json",".md",".log"} and p.is_file() and not p.is_symlink()}
needed={"A13_CANDIDATE_METRICS.csv","A13_K_POLICY_ROWS.csv","A13_SUMMARY.csv",
 "A13_SOURCE_AND_INPUT_AUDIT.json","A13_CPU_RESOURCE.json","A13_ORACLE_REPLAY_CHECKS.json"}
if not needed.issubset(allowed) or len(allowed)>30 or dst.exists():
    raise RuntimeError("STOP_A13_ZIP unexpected metadata roster/overwrite")
with zipfile.ZipFile(dst,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=8) as z:
    for name in sorted(allowed):z.write(src/name,arcname=name)
with zipfile.ZipFile(dst) as z:
    assert z.testzip() is None
h=hashlib.sha256(dst.read_bytes()).hexdigest()
print("A13_METADATA_ZIP_SHA256",h,"MEMBERS",len(allowed),"ZIP",dst)
PYTHON
        PACKAGE_RC=$?
        echo "A13_PACKAGE_RC=$PACKAGE_RC"
        if [ "$PACKAGE_RC" -ne 0 ]; then echo "STOP_A13_1: packaging failed; preserve stage1 results"; fi
      fi
    fi
  fi
fi
