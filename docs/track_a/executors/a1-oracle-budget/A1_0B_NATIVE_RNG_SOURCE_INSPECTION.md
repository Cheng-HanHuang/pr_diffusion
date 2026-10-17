# A1.0b — native restart and GT-input semantic audit (source-only)

**2026-10-17**, stage `A1.0b`. Proposed source-only check after A1.0 V2 PASS. No GPU/no model import/no measurement or real-image data. Does NOT certify an independent prospective cohort.

## Why

The archived B22 SITCOM one-trajectory execution passes `gt` to `sampler.get_start(gt)` (see `scripts/b22/run_b22_1_sitcom_smoke.py`). This argument may only provide shape to a random initialization; **do not assume the source is harmless until the pinned native implementation is inspected**. The B24 prior source `scripts/b24/run_b24_1_method_smoke.py` (READ-ONLY) demonstrates true process isolation for native DAPS and SITCOM, using four separately seeded solver processes per image. A1 should implement its own wrappers and never run historical B24 data or mutate source.

## Minimal bounded PAC source inspection

Read only the indicated native code and A1 predecessor invocation files:

```bash
ROOT=/egr/research-pac/huang248
SITCOM="$ROOT/external/SITCOM_ODE"
DAPS="$ROOT/pr_diffusion_b19_solver/external/daps"
A1="$ROOT/pr_diffusion_track_a_a1_oracle_budget"
BASE="$ROOT/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
LOG="$BASE/A1_0B_NATIVE_RNG_SOURCE_V1.txt"
if [ -e "$LOG" ]; then
  echo "A1_STOP: source audit log already exists"
elif [ ! -f "$SITCOM/sampler.py" ] || [ ! -f "$DAPS/posterior_sample.py" ]; then
  echo "A1_STOP: expected native source file missing"
else
  {
    echo "A1_0B_NATIVE_SOURCE_INSPECTION"
    echo "SITCOM=$(git -C "$SITCOM" rev-parse HEAD)"
    echo "DAPS=$(git -C "$DAPS" rev-parse HEAD)"
    echo "--- get_start implementation ---"
    grep -n -B 6 -A 60 'def get_start' "$SITCOM/sampler.py"
    echo "--- other get_start definitions, when present ---"
    grep -R -n --include='*.py' 'def get_start' "$SITCOM" 2>/dev/null
    echo "--- SITCOM seed and GT-entrypoint usage ---"
    grep -n -E 'np.random.seed|torch.manual_seed|manual_seed_all|get_start\(gt\)|sampler.sample\(' \
      "$A1/scripts/b22/run_b22_1_sitcom_smoke.py"
    echo "--- DAPS native seed and measurement path symbols ---"
    grep -n -E 'B21_START_NOISE_SEED|B21_SOURCE_SEED|B21_MEASUREMENT_PATH|manual_seed|np.random.seed|torch.Generator' \
      "$DAPS/posterior_sample.py" "$DAPS/sampler.py"
  } > "$LOG" 2>&1
  RC=$?
  echo "A1_0B_SOURCE_INSPECTION_RC=$RC"
  wc -c "$LOG"
  sha256sum "$LOG"
  tail -80 "$LOG"
fi
```

**Provenance requirements**: no `B24` output or confirmation305 metadata/payload accessed; only source code is read. In a return review, inspect whether `get_start` uses numerical GT data, where native measurements are loaded, how solver RNG is set per process, and whether the DAPS `B21_START_NOISE_SEED` override is genuinely consumed. A static grep alone cannot rule out all indirect GT paths. If source semantics are ambiguous, keep GPU and candidate evaluation STOP-gated until an isolated synthetic proof with no real GT exposure can be constructed.

**Further A1.1 blockers:** approved separate cohort provenance and seed namespace; A1-specific process-isolated launcher with exact work accounting; independent diffusion restart replay; metric representation; issue #41 DATA/RUN reservations and user command. A1.0b is source review only, not a new performance experiment.
