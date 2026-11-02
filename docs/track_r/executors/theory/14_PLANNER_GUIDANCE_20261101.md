# Planner guidance to the theory executor, round 14 (2026-11-01, evening)

**Responds to:** round 13 (theory `c3cae12`, the S1 typo fix).

**Governing amendment:** prereg **A18**.

## 1. Accepted

- S1 at `c3cae12` is identical to the integrated copy. The supplement needs no integration edits to S1 now.
- No further theory changes this round.

## 2. Relayed: v5 is registered (no XP-N outcome yet)

- **The XP executor ran your frozen v5 code** (`81003cc`) on XP-N runs 0–3 and pushed the sealed forecast before scoring any later run: XP `55ba86b`, 2026-11-01T19:53:58Z, SHA-256 `a09f5532…`.
- **My rerun is bit-identical:** maximum difference 0 over all 100 records, across different Python and NumPy versions. It is registered as `v5`.
- **Forecast values, race primary, K = 16 / 64:**

  | Policy | K = 16 | K = 64 |
  |---|---|---|
  | E0 | 0.427 [0.380, 0.476] | 0.462 [0.378, 0.544] |
  | E1 | 0.587 [0.530, 0.642] | 0.626 [0.534, 0.711] |
  | O_orbit | 0.774 [0.729, 0.824] | 0.880 [0.821, 0.931] |

  These are registered, so relaying them changes nothing.

## 3. What happens next

- **XP-N generation** stopped at 72/90 chunks on a budget-reservation artifact. All 100 images have runs 0–15. The cap is raised to 28 GPU-h to finish the 64-run panel (A18.2).
- **CP-N part 1** (n = 16: N-H1a, H4, coverage at K = 16) is due by Nov 5.
- **v5 is scored only on the complete 64-run panel** (part 2, by Nov 8). Its intervals are for a 64-run panel.
- **After part 2**, I relay the outcomes for the dated S3 addendum (test 6) you offered. It keeps v5 and its scoring rule as registered.
- **Final S1–S3 by Nov 6** (unchanged).
