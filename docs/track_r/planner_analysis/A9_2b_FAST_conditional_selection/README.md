# A9.2(b): FAST composition-conditional selection test (planner, exploratory)

**Post hoc**, computed after the FAST v1 outcome. It uses **frozen parameters only**: the v1 `fit_diagnostics` rank weights. The `race_tau1` weights are recomputed with the theory code on the same A8 input. No model is fitted to the outcome.

**Question.** Given the **realized** classes of each image's 64 runs, what E0(K) does each frozen rank law predict? How does that compare with the observed E0(K), which comes from the actual residual ranks?
- The point prediction is exact: the average over K-subsets of the Plackett–Luce first-place probability of a GOOD run.
- The interval is a 95% Monte Carlo interval of the pooled mean (400 replicates), simulating each law's exponential residual race on the realized labels.

**Inputs (SHA-256):**

| Input | Source | SHA-256 |
|---|---|---|
| 64-run strings | XP `codex/prdiff-track-r-xp-int-20261028`, `XP_A9_EXCHANGEABILITY_20261028T100727Z/A9_FAST_64_PER_IMAGE_STRINGS.csv` | `d616dd50…` |
| A8 | — | `197d0c8e…` |
| v1 | — | `c19798f9…` |
| theory code | theory `9cb57f5` | — |

The observed E0 and O_raw reproduce the published aggregates (XP `07d5460`) exactly.

**Run:**
`python3 cond_select.py <theory_scripts_dir> <A8 csv> <64-run strings csv> <TH1_PREDICTIONS_v1.json> <A8_FAST_V1_COMPARISON.json>`

**Result** (`OUTPUT.txt`):
- **S2 fails decisively.** Given the realized classes, exchangeable ranks predict an E0 4–8 points below what is observed, at every K, in both cohorts, and outside its interval. S2's wins under the frozen E0-coverage rule came from offsetting errors.
- **The race family fits.**
  - Primary race weights: inside the interval in cohort A at 6 of 7 K, but optimistic in cohort B. Pooled, they predict 0.6–1.4 points too much E0 at every K.
  - `race_tau1` weights: they match the pooled observation within 0.3 points at every K.
  - `tail`: it matches cohort B at K = 64.
- **Conclusion.** The forecast miss is generation, i.e. images that never succeed; selection is a residual race with slightly heavier competition than the primary weights imply.
