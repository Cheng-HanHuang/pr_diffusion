# Current experiment plan: LF/S2 selector validation and failure recovery

Updated: 2026-05-23

## Current state

The project has moved beyond the original May 12 near-term plan.  We tested scheduled S2, adaptive S2, memory-bank variants, diagnostic selector traces, a lightweight LF/S2 trajectory selector, a seed tie-break, and validation on a different seed pair.

The central result is:

```text
The LF/S2 config selector is strong when good candidates exist.
The 2-seed validation failure is caused by missing good candidates for some images, not mainly by a wrong LF-vs-S2 config decision.
```

Therefore, the next experiments should answer whether the remaining failures can be fixed by parameter tuning, better scoring/selection metrics, or a fallback/recovery mechanism.  A 4-seed run is useful, but it should not be the only next step.

## Evaluation standard

For every experiment, report raw alignment first:

```text
mean PSNR
median PSNR
minimum PSNR
number of images below 20 dB
number of images below 25 dB
SSIM mean
LPIPS mean when available
worst-image identities
```

For selector experiments, also report:

```text
oracle over available candidates
selected result
selector regret vs oracle
selected config counts
selected seed rule counts
failure images where oracle itself fails
```

Promotion target remains:

```text
raw mean PSNR > 28.8
raw min PSNR > 25
raw below20 = 0
raw below25 = 0
```

## Current candidate method

Current candidate method:

```text
LF/S2 trajectory selector with seed tie-break
```

Algorithm:

```text
Run LF and pre-projection S2 with the chosen seed budget.
For each config, compute mean post-projection winner LF-MSE vs noisy observation.
Choose the config with the lower mean statistic.
Inside the chosen config, choose the seed by post-projection winner LF-MSE unless the seed-stat gap is <= 5e-5.
If tied, choose the seed with lower final noisy low-frequency magnitude residual.
```

This is not a ground-truth PSNR oracle.  It chooses using computed trajectory/measurement statistics only.

Known results:

```text
Seeds 100,101:
  tie-break selector mean ≈ 29.220
  min ≈ 25.077
  below25 = 0

Seeds 102,103:
  tie-break selector mean ≈ 27.921
  min ≈ 9.117
  below25 = 2
  unavoidable LF/S2 candidate failures: 00005, 00014
```

The selector is promising but not final.

## Priority 1: seed-budget/candidate-availability control

### Experiment 1A: LF/S2 selector with four seeds

Run the same LF/S2 tie-break selector with:

```text
seeds = 100,101,102,103
```

Motivation:

- The config selector is nearly oracle-level over the available LF/S2 candidates.
- The `102,103` validation failed because `00005` and `00014` had no good LF/S2 candidate.
- The immediate control is to ask whether a larger seed budget supplies good candidates.

Expected interpretations:

```text
If four seeds pass:
  The current selector is viable under a larger seed budget, but per-run robustness remains unsolved.

If four seeds still fail:
  LF/S2 candidate generation has a structural blind spot and needs better scoring/tuning/fallback.
```

Do not stop at this experiment even if it passes; it answers candidate availability, not algorithmic robustness.

## Priority 2: failure-focused tuning for 00005 and 00014

The validation failures identify two concrete hard cases under seeds `102,103`:

```text
00005
00014
```

These should be used for cheap focused tuning before spending full FFHQ-25 budgets.

### Experiment 2A: projection-start and S2-lambda microgrid on failure images

Run LF/S2-style configs on only `00005,00014` with seeds `102,103`:

```text
proj_start in {200, 300, 400, 500}
s2_lambda in {0.005, 0.01, 0.02, 0.05}
score_radius in {0.4, 0.6}
proj_radius fixed at 0.2 initially
soft/hard fixed at 5/1 initially
```

Motivation:

- S2 can rescue some LF failures but also moves failures.
- The failed images may need a different timing or regularization strength.
- This tests whether failures can be fixed by local tuning within the existing algorithm family.

Promotion from focused tuning:

```text
At least one setting gives >25 dB on both 00005 and 00014 for seeds 102,103
without obviously collapsing on known LF-good/S2-bad guard images 00007 and 00009.
```

Guard-image check:

```text
Always include 00007 and 00009 in the focused tuning subset, because S2-like changes can break them.
```

Recommended focused subset:

```text
00005,00014,00007,00009,00018,00028,00034
```

### Experiment 2B: projection radius sanity check on failure images

Only if Experiment 2A suggests timing/lambda helps, test:

```text
proj_radius in {0.15, 0.2, 0.25}
score_radius in {0.4, 0.6}
```

Do not revisit broad projection (`0.4+`) unless there is a new soft/robust projection mechanism, because broad hard projection was already harmful.

## Priority 3: better complementary metrics for executable selection

The current config selector uses:

```text
mean post-projection winner LF-MSE vs noisy observation
```

The seed tie-break uses:

```text
final noisy low-frequency magnitude residual
```

These worked for `100,101` but could not fix `102,103` because no good candidate existed.  Still, better metrics may identify no-good-candidate cases and trigger a fallback.

### Experiment 3A: selector feature audit on full-25 traces

Use existing diagnostic outputs and/or lightweight trace summaries to study features for images where all LF/S2 candidates fail.

Candidate features:

```text
post_winner_lf_mse_mean
post_winner_lf_mse_max
post_winner_full_mse_mean
final noisy_lowfreq_mag_l2
final noisy_mag_l2
seed disagreement in final residuals
seed disagreement in selector statistic
raw-vs-resolve PSNR gap when available for analysis only
winner_is_lf_best_frac_post
post_lf_mse_margin_mean
```

Goal:

```text
Find a non-ground-truth signal that says: LF/S2 candidate pool likely has no reliable reconstruction.
```

If such a signal exists, use it to trigger a fallback rather than pretending the selector has solved the image.

### Experiment 3B: add a confidence flag to LF/S2 selector

Implement a selector confidence score based on:

```text
config-stat margin
seed-stat margin
final residual tie-break consistency
seed disagreement
```

The output should not only choose a reconstruction.  It should report:

```text
selected reconstruction
confidence/high-risk flag
reason for high-risk flag
```

This is useful even before a fallback is implemented.

## Priority 4: fallback and recovery strategies

### Experiment 4A: memory hard2 as fallback only

Memory hard2 was not good as the main solver, but it was stable:

```text
memory_k=1, hard=2:
  min around 24.9 in earlier test
  no below20 failures
  lower median quality
```

Test it only as a fallback when LF/S2 selector is low-confidence or when the LF/S2 candidate pool looks bad.

Motivation:

- Memory hard2 is too conservative for all images.
- It may still rescue catastrophic LF/S2 failures or provide a non-catastrophic fallback.

Focused test:

```text
Run LF, S2, and memory hard2 on 00005,00014 plus guard images.
Check whether memory hard2 beats the catastrophic LF/S2 candidates.
```

Full test only if focused test helps.

### Experiment 4B: SITCOM-ODE fallback on high-risk cases

If LF/S2 confidence flags can identify likely failures, test running SITCOM-ODE only on high-risk images.

Motivation:

- SITCOM-ODE is stronger on many low-noise FFHQ images.
- NP/LF-S2 has complementary failures.
- A fallback strategy may be cheaper and cleaner than forcing NP to solve every case.

This should be treated as a hybrid solver/fallback pipeline, not as an oracle.

## Priority 5: algorithmic scoring improvements

If focused tuning and fallback do not explain `00005/00014`, return to scoring/selection within the reconstruction loop.

Promising directions:

1. **Soft projection instead of hard projection**

   ```text
   x_new = (1 - eta) * x_prior + eta * projected_x
   eta in {0.1, 0.25, 0.5}
   radius = 0.2
   ```

   Motivation: hard projection is useful at small radius but broad or overly rigid projection is harmful.

2. **Frequency-weighted robust score**

   Replace simple LF score with a weighted residual:

   ```text
   weighted residual = low-frequency emphasis + robust loss
   ```

   Test only after defining clear logging and failure criteria.

3. **Two-stage score schedule**

   Use LF for early global basin selection and S2 only in a controlled window, not throughout all pre-projection steps.  Previous adaptive margins were too blunt; use diagnostics to define a time window rather than a scalar LF-margin gate.

## Recommended immediate launch order

Given current evidence, the next four practical experiments should be:

1. **Four-seed LF/S2 tie-break selector**

   ```text
   seeds = 100,101,102,103
   full FFHQ-25
   goal: candidate-availability control
   ```

2. **Focused failure microgrid: proj_start × S2 lambda**

   ```text
   images = 00005,00014,00007,00009,00018,00028,00034
   seeds = 102,103
   goal: see whether failures are fixable by timing/lambda tuning
   ```

3. **Focused memory fallback check**

   ```text
   images = same focused subset
   configs = LF, S2, memory_k=1 hard=2
   seeds = 102,103
   goal: test whether memory helps failure images without being main method
   ```

4. **Selector confidence-feature audit**

   ```text
   use existing run_level/diagnostic summaries
   goal: identify no-good-candidate flags before adding more solver complexity
   ```

Only after these should we spend on a larger full-25 grid.

## What not to prioritize immediately

Do not prioritize these unless the above experiments fail to clarify the issue:

- More fixed global S2 lambda sweeps on all 25 images.
- More adaptive LF-margin thresholds without new diagnostics.
- Broad hard projection radii (`0.4+`).
- Memory as the main solver with no fresh candidate.

These have already shown failure modes or low payoff.
