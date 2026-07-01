# B20.10: Matched-seed schedule sensitivity for hard image `00046`

This note records the B20.10 follow-up to the B20.7--B20.9 schedule-diversity diagnostics.  B20.7--B20.9 showed that changing the annealing length could produce large changes in final reconstruction quality for hard image `00046`, but the first schedule-response maps were not clean enough to distinguish a true schedule effect from candidate-generation / RNG-stream effects.

The goal of B20.10 was therefore to separate two questions:

```text
Q1. Does annealing length causally affect basin entry for the same explicit candidate seed?
Q2. If yes, is there a universal or monotone schedule choice that solves the hard case?
```

## Experimental setting

- Image: FFHQ `00046`
- Task: phase retrieval, `sigma = 0.05`
- Measurement seed: `5001`
- Success threshold: `PSNR >= 25 dB`
- DAPS schedule family: `diff_steps = 5`, varying annealing steps
- Main schedules: `ann300_diff5`, `ann350_diff5`, `ann375_diff5`, `ann400_diff5`, `ann450_diff5`
- Matched-seed panel: `NUM_RUNS = 1`, explicit run seeds `6100`--`6107`

The key methodological change from the earlier `NUM_RUNS=16` experiments is that B20.10C uses one explicit candidate seed per run.  This avoids the hidden batch-RNG confound discovered in B20.10-0: for later run indices in a `NUM_RUNS=16` batch, the initial `x_t` values were not matched across schedules because earlier candidates consume schedule-dependent random numbers.  Under `NUM_RUNS=1`, every schedule comparison is instead a clean `run_index=0` matched-seed panel.

## B20.10-0: batch run-index matching was mostly invalid

A direct initialization audit compared the first saved `x_t` vector for nominally matched `(run_seed, run_index)` pairs across schedules.  The result was:

```text
run_index = 0:
  relative initial x_t differences were tiny, around 1e-3.

later run indices such as 4, 6, and 10:
  relative initial x_t differences were about 1.41,
  consistent with unrelated high-dimensional random vectors.
```

Conclusion:

```text
The previous NUM_RUNS=16 matched-run-index comparisons were only clean for run0.
For later run indices, schedule length also changes the candidate RNG stream.
```

This does not invalidate B20.8/B20.9 as a schedule-and-seed-bank response map, but it means those runs cannot be read as same-initialization schedule comparisons except for `run_index=0`.

## B20.10C: clean matched-seed schedule panel

B20.10C reran `00046`, `meas_seed=5001`, with `NUM_RUNS=1` for explicit candidate seeds `6100`--`6107` and schedules `ann300/350/375/400/450_diff5`.

The resulting final PSNR table was:

```text
 candidate_seed  ann300_diff5  ann350_diff5  ann375_diff5  ann400_diff5  ann450_diff5  best_psnr best_schedule  worst_psnr  schedule_spread  num_good_schedules  any_good  all_good
           6100     15.170509     11.104074     16.419008     31.360647     18.155453  31.360647  ann400_diff5   11.104074        20.256573                   1         1         0
           6101      9.745761      9.726417     16.532373     15.936679     16.587799  16.587799  ann450_diff5    9.726417         6.861382                   0         0         0
           6102     14.400032     11.524717     15.963045     11.303112      9.742174  15.963045  ann375_diff5    9.742174         6.220871                   0         0         0
           6103     31.138315     16.511124     16.494108     16.841272     16.566187  31.138315  ann300_diff5   16.494108        14.644207                   1         1         0
           6104     11.354647     16.175703     16.628736     31.309807     29.141502  31.309807  ann400_diff5   11.354647        19.955160                   2         1         0
           6105     13.016407     19.214869     19.154869     16.622364     14.484200  19.214869  ann350_diff5   13.016407         6.198462                   0         0         0
           6106     15.382439     16.472967     16.461952     17.415272     16.398018  17.415272  ann400_diff5   15.382439         2.032833                   0         0         0
           6107      9.762808     31.006844     31.358423     20.260447     16.437012  31.358423  ann375_diff5    9.762808        21.595615                   2         1         0
```

Schedule summary:

```text
         tag  cases  good25  bad25  mean_psnr  min_psnr  max_psnr
ann400_diff5      8       2      6  20.131200 11.303112 31.360647
ann375_diff5      8       1      7  18.626565 15.963045 31.358423
ann450_diff5      8       1      7  17.189043  9.742174 29.141502
ann350_diff5      8       1      7  16.467089  9.726417 31.006844
ann300_diff5      8       1      7  14.996365  9.745761 31.138315
```

Best-schedule counts:

```text
best_schedule  count
 ann400_diff5      3
 ann375_diff5      2
 ann300_diff5      1
 ann350_diff5      1
 ann450_diff5      1
```

Aggregate interpretation:

```text
Total candidates: 40
Good candidates: 6 / 40
Best fixed schedule: ann400_diff5 with 2 / 8 good seeds
Oracle over five schedules: 4 / 8 seeds have at least one good schedule
All schedules good: 0 / 8 seeds
```

Therefore B20.10C gives clean evidence that annealing length is a real causal basin-entry variable under matched explicit seeds.  However, it is also clearly not a universal solution: no fixed schedule dominates, no seed succeeds under all schedules, and half of the explicit seeds are not rescued by any of the tested schedules.

## B20.10D: early trajectory divergence

B20.10D compared early trajectory features for matched-seed schedule panels.  The diagnostic seeds were chosen to show different schedule-success patterns:

```text
seed 6107:
  good: ann350, ann375
  bad:  ann300, ann400, ann450

seed 6100:
  good: ann400
  bad:  ann300, ann350, ann375, ann450

seed 6103:
  good: ann300
  bad:  ann350, ann375, ann400, ann450

seed 6104:
  good: ann400, ann450
  bad:  ann300, ann350, ann375
```

The feature contrasts show that good and bad schedules can separate in trajectory statistics, but the direction is seed-dependent.

For example, seed `6103` behaves like a conventional healthy-prefix case: the good `ann300` trajectory has lower early `x0hat` loss and smaller `xt_jump_rms` than the bad longer schedules, with separation already visible around steps 20--50.

In contrast, seed `6100` is almost the opposite: the good `ann400` trajectory often has higher early `x0hat` loss and larger early jumps than the bad schedules.  Thus a simple rule such as "lower early residual is better" would select the wrong direction on this seed.

Seed `6107` shows weaker early separation at the first checkpoints, then clearer differences around steps 50--125.  This makes it a useful microscope case, but not yet a reliable early schedule-selector case.

Conclusion from B20.10D:

```text
The trajectory features describe schedule tempo and basin path, not a universal scalar health score.
Good and bad schedules can separate, but the sign and timing of separation are seed-dependent.
```

## Main B20.10 conclusion

B20.10 resolves the main ambiguity left by B20.8/B20.9:

```text
Annealing length causally affects basin entry under matched explicit candidate seeds.
```

But it also rules out the simple schedule-tuning story:

```text
Schedule choice is seed-specific and nonmonotone.
No fixed schedule is reliable.
An oracle over five schedules rescues only half of the tested explicit seeds.
Early trajectory features do not provide a universal health direction.
```

Therefore schedule diversity should be treated as a microscope for basin geometry, not as the final reliability mechanism.

The next algorithmic target should be candidate initialization / early trajectory guidance.  In particular, B20.11 should test low-frequency or measurement-guided initialization, with the goal of increasing the probability that an individual candidate enters the correct structural basin before the final measurement-consistency corrections become too late.

## Recommended next experiment: B20.11

A good first B20.11 question is:

```text
Can weak low-frequency measurement information improve basin-entry probability under a fixed or small set of schedules?
```

The matched-seed B20.10C panel gives the baseline:

```text
fixed schedule best: 2 / 8 good seeds
five-schedule oracle: 4 / 8 good seeds
total schedule candidates: 6 / 40 good
```

B20.11 should try to improve the fixed-schedule or small-schedule success rate without relying on a broad schedule oracle.  The most useful microscope seeds are:

```text
6107: good at ann350/375, bad at ann300/400/450
6100: good only at ann400
6103: good only at ann300
6104: good at ann400/450
```

However, the longer-term benchmark should not overfit to these four seeds.  The next overnight run should include additional explicit seeds and compare a baseline schedule against one or more low-frequency / measurement-guided initialization variants under the same fixed budget.
