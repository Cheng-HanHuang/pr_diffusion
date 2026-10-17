# A1.0c — bounded native DAPS start-noise semantics: source inspection

**Date:** 2026-10-17. **Scope:** source-only and no GPU/data usage. Starts from confirmed A1.0 V2 `PASS_SOURCE_AND_EXPOSED_METADATA_PREFLIGHT` and A1.0b SITCOM GT-independent random initialization audit.

## Why this additional source read is necessary

The A1.0b PAC grep proved that native DAPS invokes `np.random.seed(args.seed)`, `torch.manual_seed(args.seed)` and `torch.cuda.manual_seed_all(args.seed)` and **reads** `B21_START_NOISE_SEED`, but did **not** show the actual sampler body that generates `x_start` / source noise. A1 must ensure different K4 solver seeds correspond to genuinely independent full runs, not only different logged IDs.

## User-run bounded source command

**No Git checkout update necessary.** Inspect the pinned existing native source only and preserve the return in a fresh A1-owned file; no `B24` data/payload/locked confirmation file or full-tree scans.

```bash
ROOT=/egr/research-pac/huang248
DAPS="$ROOT/pr_diffusion_b19_solver/external/daps"
BASE="$ROOT/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016"
LOG="$BASE/A1_0C_DAPS_SEED_SOURCE_V1.txt"

if [ -e "$LOG" ]; then
  echo "A1_STOP: source audit log already exists; preserve it"
elif [ ! -f "$DAPS/sampler.py" ] || [ ! -f "$DAPS/posterior_sample.py" ]; then
  echo "A1_STOP: expected pinned DAPS sources unavailable"
else
  {
    echo "A1_0C_DAPS_SEED_SOURCE"
    echo "DAPS_HEAD=$(git -C "$DAPS" rev-parse HEAD)"
    echo "=== DAPS sampler RNG initialization ==="
    sed -n '155,195p' "$DAPS/sampler.py"
    echo "=== DAPS measurement/trace metadata ==="
    sed -n '210,255p' "$DAPS/sampler.py"
    echo "=== DAPS fresh-step initial noise override ==="
    sed -n '415,480p' "$DAPS/sampler.py"
    echo "=== DAPS high-level seed initialization ==="
    sed -n '220,248p' "$DAPS/posterior_sample.py"
    echo "=== exact start-noise references in sampler (bounded match lines) ==="
    grep -n -E 'B21_START_NOISE_SEED|B21_SOURCE_SEED|B21_MEASUREMENT_PATH|randn_like|randn\(|randn\[|manual_seed|seed_noise' "$DAPS/sampler.py"
  } > "$LOG" 2>&1
  RC=$?
  echo "A1_0C_SOURCE_RC=$RC"
  wc -c "$LOG"
  sha256sum "$LOG"
  tail -155 "$LOG"
fi
```

Please provide the terminal text, or the small `A1_0C_DAPS_SEED_SOURCE_V1.txt` log. The output contains source text and no user/image data. On return, verify source semantics against B21/B24 independent restart specifications and record a dated `A1_0C_SOURCE_RETURN.md`.

**Remaining launch firewalls:** no novel GPU experiment before A1 owns independent DAPS/SITCOM/NP four-restart wrappers with the same frozen measurement/PSNR representation, prospective cohort is certified eligible **without** any B24 confirmation305 access and [DATA RESERVATION] plus [RUN RESERVATION] are registered in issue #41 with Track B's GPU priority.
