# B21.3 split-200 branch-vs-fresh decision

Status: **do not scale the current split-200 policy**.

## Experiment

Three-image paired equal-cost mini-pilot:

- images: `00046`, `00171`, `00746`
- parent seeds: `7000`–`7007`
- cases: `24`
- candidates: `96`
- Fresh2 cost: shared `source_full` ann400 + one independent ann400 = 800 annealing transitions
- Branch3@200 cost: shared `source_full` ann400 + two 200-step continuations from step 200 = 800 annealing transitions

## Result

| group | Fresh2 any-good | Branch3 any-good | net | branch-only | fresh-only |
|---|---:|---:|---:|---:|---:|
| `00046` | 0/8 | 1/8 | +1 | 1 | 0 |
| `00171` | 3/8 | 0/8 | -3 | 0 | 3 |
| `00746` | 5/8 | 5/8 | 0 | 0 | 0 |
| **ALL** | **8/24** | **6/24** | **-2** | **1** | **3** |

Additional observations:

- mean best PSNR: Fresh2 `20.6771`, Branch3 `18.1669` (`-2.5103 dB`)
- exact-loss-selected good cases: Fresh2 `8/24`, Branch3 `6/24`
- McNemar exact two-sided `p = 0.625`
- Branch3/Fresh2 wall-time ratio: `1.0228`
- every Branch3 policy produced three unique candidate hashes on average, so the negative result is not caused by identical continuations

## Decision

The registered mini-pilot promotion gate failed:

- required net successful-case gain at least `+2`; observed `-2`
- required at most one Fresh2-only win per image; observed `3` on `00171`
- wall-time gate passed

Therefore:

1. The continuation implementation itself remains valid and reproducible.
2. The general equal-cost `Branch3@200` reallocation policy is **rejected** and must not be expanded to a larger validation panel.
3. The failure is category-specific: the shared late prefix is especially harmful for the diffuse-collapse image `00171`; independent restarts are materially better there.
4. The isolated `00046` rescue is insufficient to justify a general policy. Any future continuation work must be adaptive or category-specific and must start with a new small development experiment.
5. The next main candidate-generation track should move to B21.5 warm-start/HIO rather than scaling B21.3.

Runtime artifacts:

```text
/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_3_branch_vs_fresh_pilot_3img_8seed_split200
```
