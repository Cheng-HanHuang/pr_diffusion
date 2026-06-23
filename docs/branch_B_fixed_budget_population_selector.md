# Branch B fixed-budget SITCOM population selector

Updated: 2026-06-23

This note records the current Branch B conclusion after the B3--B18D phase-retrieval experiments.  It supersedes the earlier interpretation of Branch B as primarily an NP-to-SITCOM sigma-handoff branch.

## 1. Current Branch B answer

The cleanest Branch B result is a fixed-budget SITCOM population selector:

```text
Given one phase-retrieval measurement:
  run 4 independent SITCOM-ODE trajectories;
  at tau = 0.8, read correction_norm for each run;
  select the run with lowest correction_norm;
  return its final reconstruction.
```

This is a fair fixed-budget method: it costs four SITCOM reconstructions per image and does not rely on an offline oracle, a ground-truth image, or unbounded fallback retries.

The pooled fixed-budget audit over B11, B12, and B16-stage1 gives:

| source | n images | mean selected PSNR | min selected PSNR | bad25 | bad20 |
|---|---:|---:|---:|---:|---:|
| B11 | 25 | 30.741 | 23.774 | 1 | 0 |
| B12 | 25 | 29.826 | 5.087 | 1 | 1 |
| B16-stage1 | 25 | 31.127 | 29.548 | 0 | 0 |
| pooled | 75 | 30.565 | 5.087 | 2 | 1 |

The two pooled failures are different failure modes:

| case | selected PSNR | oracle4 PSNR | interpretation |
|---|---:|---:|---|
| B11 / image `00027` | 23.774 | 29.936 | selector failure: a good SITCOM candidate existed |
| B12 / image `00017` | 5.087 | 5.828 | generation failure: all four SITCOM candidates failed |

This is the current Branch B baseline to compare against.

## 2. Why the original NP-to-SITCOM handoff is no longer the main path

B3--B8 tested the intended Branch B handoff idea: export NP reconstructions as sigma states and continue with the patched SITCOM-ODE sampler.  That pipeline is technically useful, but it did not become a competitive solver.

The main empirical lesson was:

```text
NP clean states are not automatically good SITCOM continuation states.
Even tiny/no-noise handoff and local refinements can damage a good NP estimate.
```

Therefore NP should not currently be forced through SITCOM continuation as the default Branch B mechanism.  If NP is used again, it should be treated as an independent candidate or diagnostic second opinion, not as a state to be denoised by SITCOM.

## 3. Health trigger: useful warning, not an action policy

B14 introduced a population-health diagnostic for the 4S selector.  The useful warning feature was:

```text
tau = 0.8
feature = correction_norm
health statistic = max(feature across 4 runs) - min(feature across 4 runs)
trigger if pop_spread_correction >= 0.003186
```

Across B11+B12 this trigger caught both observed selected failures, including the SITCOM-generation failure at image `00017`.  However, it also fired on many good selected outputs.  Across the pooled B11/B12/B16-stage1 audit:

```text
health-triggered cases: 24 / 75 = 32%
triggered selected bad25: 2
triggered oracle4 bad25: 1
```

So the trigger should be read as:

```text
This population looks unstable enough to inspect.
```

It should not be read as:

```text
The current selected reconstruction is wrong and must be replaced.
```

## 4. Why 4-to-8 escalation is diagnostic, not final

B15 showed that if a triggered 4-run population is allowed to pool with another 4 SITCOM runs, an oracle or residual-based 8-candidate rule can often remove failures.  That is useful diagnostically, because it separates candidate-generation failures from selector failures.

However, B16 and B16A showed why this should not be the final algorithm:

| policy | mean PSNR | min PSNR | bad25 | bad20 | compute |
|---|---:|---:|---:|---:|---:|
| B16 stage1 4S selector | 31.127 | 29.548 | 0 | 0 | 4x SITCOM |
| B16 replace-if-triggered 4-to-8 | 30.659 | 29.100 | 0 | 0 | 5.44x SITCOM |

Every triggered replacement in B16A hurt PSNR by roughly 0.8--1.7 dB.  The fallback candidate had lower measurement-side residual, but worse reconstruction quality.

The lesson is important for phase retrieval:

```text
A lower Fourier-magnitude residual is not a clean certificate of a better reconstruction.
```

Therefore 4-to-8 escalation should remain an oracle/diagnostic curve, not the headline method.

## 5. Same-budget NP hybrid: oracle-complementary but not executable yet

B18 tested whether NP should replace one of the four SITCOM candidates under a fixed candidate budget.

The same-budget candidate-set oracle looked promising:

| policy | pooled mean PSNR | pooled min PSNR | bad25 | bad20 |
|---|---:|---:|---:|---:|
| 4S SITCOM baseline | 30.565 | 5.087 | 2 | 1 |
| 3S + 1NP candidate-set oracle | 31.067 | 25.209 | 0 | 0 |
| NP only | 26.398 | 10.434 | 15 | 15 |

But the executable rule `3S_health_to_NP` was worse than the 4S baseline:

```text
3S_health_to_NP pooled: mean 29.179, min 5.827, bad25 6, bad20 5
```

The unique-image complementarity audit B18D explains why:

```text
25 unique images:
  NP bad25: 5 / 25
  S4 selected any bad25: 2 / 25
  S4 oracle any bad25: 1 / 25
  NP rescues an S4 selected failure: 1 image
  NP hurts all-good S4 selected populations: 4 images
  NP hurts at least one good S4 source: 5 images
```

The only clear NP rescue was:

| image | worst S4 selected | worst S4 oracle | NP PSNR |
|---|---:|---:|---:|
| `00017` | 5.087 | 5.828 | 31.286 |

But NP catastrophically failed on several images where SITCOM was good:

| image | NP PSNR | S4 selected behavior |
|---|---:|---|
| `00013` | 10.434 | SITCOM about 29--30+ |
| `00028` | 11.804 | SITCOM about 31+ |
| `00034` | 11.917 | SITCOM about 32+ |
| `00018` | 17.528 | SITCOM about 30+ |
| `00027` | 19.617 | mixed SITCOM selector failure but good SITCOM candidate exists |

Thus NP and SITCOM show possible oracle-level complementarity, but the current one-shot NP candidate is not safe enough for an executable same-budget hybrid.

## 6. Current recommendation

For now, Branch B should report the fixed-budget 4S SITCOM selector as the main result:

```text
4 independent SITCOM runs
+ tau=0.8 lowest correction_norm selection
+ optional population-spread warning flag for diagnosis only
```

Do not present the following as final algorithms yet:

```text
4S + extra fallback candidates
4-to-8 replace-if-triggered policies
3S+1NP or 2S+2NP hybrids
```

Those are useful diagnostics only unless a clean-free certificate is found that can distinguish:

```text
SITCOM failure / NP rescue regime: image 00017
NP failure / SITCOM success regime: images 00013, 00028, 00034, 00018, 00027
```

## 7. Next research questions

The next Branch B work should be diagnostic rather than another GPU-heavy policy loop:

1. Why do some SITCOM populations collapse stochastically, as in B12 image `00017`?
2. Why does the current NP seed/config rescue `00017` but fail badly on `00013`, `00028`, `00034`, `00018`, and `00027`?
3. Is there a clean-free prior-consistency or cross-candidate certificate that distinguishes those regimes?
4. Can selector failures like B11 image `00027` be fixed without increasing the 4S compute budget?

Until those questions are answered, the fair solver story is fixed-budget 4S SITCOM population selection, with failure anatomy clearly reported.
