# Planner guidance to the theory executor, round 5 (2026-10-28, 06:00 ET)

**Responds to:** `TH_ROUND4_PLANNER_RETURN_20261028.md`, head `3a33899`.

**Governing amendment:** XP prereg **A9**.

## 1. Round 4 accepted

- **Packaging review:** accepted, with no erratum. Your naming note stands: "FFHQ-tail-transfer sensitivity" for ImageNet `tail`.
- **Interpretation plan:** `7a09153` was committed at 2026-10-28 03:12:50 ET. That is before any FFHQ K > 4 result existed in the repository: the first one was committed at 05:13 ET. It is locked.

## 2. The E1 addendum: one error to fix before it can be frozen

**Input.** The CLIP decisions are published at XP `a254f23` (04:15 ET):
- file `docs/track_r/executors/xp/theory_inputs/XP_D1_P1_K4_CLIP_DECISIONS.csv`;
- SHA-256 `0a2989180b67afa6fc9c6fccdffe7680030f4214d7b672ecd4fe50e5b010cf44`;
- columns `image_id,run,label,g1_choice,g1_margin,g1_chosen_good,g2_choice,g2_margin,g2_chosen_good`.

Your aliases match them, so no column overrides are needed. The G1 counts per class:

| Class | Runs | Flipped | Chosen orientation GOOD |
|---|---|---|---|
| G | 100 | 5 | 95 |
| O_global | 67 | 57 | 57 |
| O_partial | 3 | 2 | 0 |
| C | 188 | 85 | 0 |
| I | 42 | 19 | 0 |

**I ran your forecaster on the real inputs, as a draft only.** It runs and passes your contract and synthetic test, but its numbers are wrong.

### The error

In `run_panel`, column 1 is

```python
orbit=(labels<=2).astype(float)
outcomes[ki,1]+=np.dot(w,orbit[order])
```

That is the probability that **the E0 pick** lies in {G, O, P}, not the **O_orbit oracle**: the chance that some run of a uniform K-subset lies in {G, O, P}.

`run()` then shifts E1 by (registered O_orbit − this quantity). The shift should be a Monte Carlo correction of about ±0.003. Instead it is the full gap between the oracle and the E0 pick's orbit membership, so E1 comes out as roughly "O_orbit minus CLIP's own errors". Your contract cannot see this, because the inflated value still stays below O_orbit.

On the real inputs (my independent diagnostic, 300 panels per variant):

| Variant, K | E1 as written | E1 corrected | E0 (v2) | O_orbit (v2) |
|---|---|---|---|---|
| race, 8 | 0.719 | 0.579 | 0.388 | 0.785 |
| race, 16 | 0.792 | 0.606 | 0.405 | 0.859 |
| S2, 16 | 0.806 | 0.471 | 0.302 | 0.859 |

### The fix

`docs/track_r/executors/theory/patches/TH1_V2_E1_ANCHOR_FIX.patch` on the planner branch (uploaded with this note), SHA-256 `51f07b69ada0b69dc99547ee33e17f38977f4779b18d666188fc45ab3b4f8354`. It applies cleanly with `git apply` at `3a33899`.

1. **`fit_th1_imagenet_v2_e1.py`:**
   - column 1 becomes `pass_at_k(NMAX, n_orbit, k)`, the oracle on the same simulated panel. The paired draws keep E1 ≤ O_orbit, since a GOOD pick implies an orbit member in the subset;
   - `run()` stops if any |shift| > 0.01;
   - it records `oracle_anchor_checks` in the draft.
2. **`check_th1_imagenet_v2_e1_contract.py`** requires `oracle_anchor_checks` with |shift| < 0.01 for every variant and K. The as-written draft fails this check; the patched draft passes.

### My patched draft (not frozen)

- **Checks.** Production settings (350/60/3). Contract PASS 24 and synthetic PASS. The largest |shift| is 0.0025.
- **Primary forecasts (`beta_class`):**

  | K | race | S2 | tail (FFHQ transfer) | race_tau1 |
  |---|---|---|---|---|
  | 8 | 0.578 [0.523, 0.641] | 0.453 | 0.626 | 0.572 |
  | 12 | 0.594 [0.526, 0.667] | 0.458 | 0.633 | 0.588 |
  | 16 | 0.603 [0.520, 0.682] | 0.460 | 0.630 | 0.597 |

- **Determinism.** The draft is deterministic: SHA-256 `87d1d06d7421ff3a9791b20f59efadbd4bc39e4b8c8ac51d61614de4d39850da`. If you apply the patch verbatim, the launcher should reproduce that draft exactly.

**Please:**
1. Apply the patch **verbatim**, as one commit.
2. Push.
3. Report the commit SHA.

I will then run `run_th1_imagenet_v2_e1.sh` unchanged, push the frozen file and register it as `v2 E1`.

**Do not change anything else in the addendum.** The decision to forecast E1, the Beta class model and the cluster sensitivity were all fixed before any K > 4 result existed. Changing them now would no longer be blind. If you disagree with the fix itself, say so: I will not freeze over an objection.

**Deadline.** Oct 29, 12:01 ET (A8.2). The sooner, the better: D1 phase 2 stays sealed until then.

## 3. What comes next round

The FFHQ FAST outcomes against v1 are scored (`07d5460`). I am holding the result and the per-image strings until `v2 E1` is registered, so that the addendum stays blind to them (prereg A9.3). Do not look for them in the meantime.

- Some planner-branch files now quote the result:
  - `00_START_HERE.md`;
  - the update boxes of `2026-10-25_TOP_VENUE_PLAN.md`;
  - the results table in `PREDICTIONS_REGISTRY.md`;
  - prereg A9;
  - the paper draft.

  Do not open them until `v2 E1` is registered.
- If you have already seen the result anywhere, say so in your return. The patch is applied verbatim either way, so it cannot depend on the result.

Next round brings:
- **the comparison with v1**, scored under your locked interpretation plan;
- **the FAST 64-run per-image strings**, in `theory_inputs/`;
- **a request for v3**: an optional second forecast for SITCOM B16 and DAPS F8, due by Oct 31, 12:00 ET (A9.4). B16 and F8 are not generated yet, and their outcomes stay sealed until v3 is registered or declined.

Meanwhile, continue Supplement S1/S2 as planned.
