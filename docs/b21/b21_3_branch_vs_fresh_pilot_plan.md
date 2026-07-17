# B21.3 equal-cost branch-vs-fresh mini-pilot plan

Status: runner and analyzer implemented; GPU pilot not yet executed.

## Question

At equal annealing-transition cost, is it better to spend the remaining budget on one independent full DAPS trajectory or on two late continuations from a shared step-200 state?

## Paired policies

Each case is one `(image, parent_seed)` pair. Both policies share the exact same `source_full` ann400 candidate.

```text
Fresh2:
  source_full (400) + fresh_extra (400) = 800 transitions

Branch3@200:
  source_full (400)
  + branch_a from step 200 (200)
  + branch_b from step 200 (200)
  = 800 transitions
```

This design avoids charging either policy for an unused prefix-only run. The paired difference is one additional independent ann400 run versus two independent ann200 continuations.

## Mini-pilot panel

```text
images:       00046 00171 00746
parent seeds: 7000 7001 7002 7003 7004 7005 7006 7007
cases:        24
candidates:   96 total
```

Candidate seed mapping is deterministic:

```text
source_full = parent_seed
fresh_extra = parent_seed + 10000
branch_a    = parent_seed + 20000
branch_b    = parent_seed + 30000
```

All runs use the locked `meas5001` payload, base DAPS, ann400/diff5, and no LF guidance.

## Primary endpoint

Oracle any-good at PSNR >= 25 for each paired case:

```text
Fresh2 any-good  = max(source_full, fresh_extra)
Branch3 any-good = max(source_full, branch_a, branch_b)
```

Secondary endpoints:

- best PSNR;
- number of good candidates;
- clean-free exact-loss-selected PSNR and good25;
- candidate SHA256 diversity;
- measured wall time;
- exact paired McNemar test.

## Mini-pilot promotion gate

Promote to a larger fresh validation only when all hold:

1. Branch3 gains at least two net successful cases out of 24.
2. Fresh2 has no more than one policy-only win on any individual image.
3. Aggregate Branch3/Fresh2 wall-time ratio is at most 1.25.

The registry-scale criterion remains at least +5 percentage points in paired any-good rate or exact McNemar `p < 0.05`; the 24-case mini-pilot is primarily a directional and implementation gate.

## Scripts

- `scripts/b21/run_b21_3_branch_vs_fresh_pilot_multigpu.sh`
- `scripts/b21/analyze_b21_3_branch_vs_fresh_pilot.py`

The runner is resumable with `B21_FORCE=0` and uses four independent GPU workers by default.
