# Branch A future controller directions

Updated: 2026-06-21

This note records the two next algorithmic directions that should guide future Branch A work after the A14/A16 prospective validations.

The current empirical state is:

```text
A frozen aggressive residual+consensus controller reduces most catastrophic SITCOM failures on fresh runs, but it still leaves a persistent image-specific floor case around image 00017.
```

A17 added a broad anytime-visibility diagnostic result, but A17.5 showed that the strongest candidate anytime rules do not survive strict cross-fit freeze budgets in a useful way. The current takeaway is therefore:

```text
anytime signal exists;
stable budgeted anytime control is not solved;
do not freeze a new anytime policy yet.
```

Therefore the next Branch A work should not be another ad hoc threshold patch on A14/A16. It should move from fixed-window posthoc triage toward clean-free trajectory control.

## 1. Direction A: anytime risk detection

### Goal

Replace fixed checkpoints such as `first50pct` and `first80pct` with a per-step cumulative risk process.

The current detector summarizes a fixed prefix of the trajectory:

```text
first50pct:
  closer to early warning, but misses late-developing failures

first80pct:
  stronger late-triage signal, but not really early intervention
```

A more solver-like controller should instead ask at every step:

```text
At step t, does this run show enough persistent clean-free evidence that it is entering a bad basin?
```

### Basic structure

For each image `i`, run `r`, and step `t`, compute a risk score such as:

```text
risk(i, r, t) =
  evidence from current residual rank
  + evidence from local residual-rank slope
  + evidence from persistent high-rank behavior
  + evidence from x0hat/x0y disagreement
  + evidence from correction-norm or update instability
  + evidence from step-to-step jumps
  + evidence from consensus isolation when available
```

Then replace the fixed rule

```text
check first50 or first80 summary once
```

by an anytime rule such as:

```text
flag if risk(i, r, t) stays high for m consecutive steps
```

or

```text
flag if cumulative risk evidence exceeds a threshold before the trajectory ends
```

### Why this matters

A12 showed that many A11 first50 misses were late-developing rather than permanently invisible. A fixed first50 checkpoint can miss runs whose failure signal appears at step 60, 70, or 80 percent of the trajectory. An anytime detector could fire as soon as the evidence becomes persistent instead of waiting for a predeclared window summary.

This does not solve truly certificate-invisible failures. If a trajectory never looks risky under the available certificates, no timing rule can rescue it. Direction A is about removing the artificial `50` / `80` boundary and measuring when risk becomes visible.

### Offline development experiment

A good next offline experiment is:

```text
A17_offline_anytime_detector_design
```

Use existing A8, A11, A14, and A16 trajectories only. Do not run new SITCOM jobs.

Suggested outputs:

```text
anytime_feature_table.csv
anytime_detection_event_table.csv
anytime_policy_summary.csv
unflaggable_bad_runs.csv
SUMMARY.md
```

Questions to answer:

```text
How early can each bad run be flagged?
Which bad runs are visible before step 50, 60, 70, or 80 percent?
Which bad runs only become visible near the end?
Which bad runs remain invisible even under cumulative evidence?
Does image 00017 ever become visible under any existing trajectory-side certificate?
```

A useful diagnostic is an event-time table:

```text
image_id, run_index, final_psnr, first_step_rank_high, first_step_slope_high,
first_step_persistent_high_risk, final_flag, visible_by_50, visible_by_80
```

### Prospective follow-up

Only after an anytime rule is selected from development data should it be frozen and tested on a fresh SITCOM trajectory run. The prospective claim should be explicit:

```text
This is an anytime clean-free risk detector, not a posthoc threshold sweep.
```

## 2. Direction B: population / beam controller

### Goal

Move from independent late replacement toward maintaining a population of plausible SITCOM trajectories.

The current Branch A controller does this:

```text
run 4 full SITCOM attempts;
score them late or after completion;
replace flagged outputs by NP-selected fallback.
```

A population controller would instead do this:

```text
maintain several active trajectories;
score them repeatedly with clean-free certificates;
select, prune, respawn, or fallback based on population health.
```

### Motivation

The geometry viewpoint of diffusion inverse problems often asks whether the trajectory stays in an in-distribution corridor while also satisfying the measurement. Branch A certificates can be read as clean-free proxies for this:

```text
measurement consistency:
  does the run fit the observed Fourier magnitude?

trajectory stability:
  is the residual rank getting worse over time?

cross-run consensus:
  is the run isolated from other plausible trajectories?

prior compatibility:
  are correction norms, jumps, or denoiser disagreements abnormal?

fallback agreement:
  is the run wildly inconsistent with a safer NP anchor?
```

The population idea is to use these certificates not only to reject outputs at the end, but also to decide which trajectories are worth continuing.

### Simple multi-start version

The first version can be simple:

```text
1. run K independent SITCOM initializations;
2. monitor anytime risk for each run;
3. keep low-risk runs;
4. mark high-risk runs unsafe;
5. if no safe run remains, use NP fallback or spawn more initializations;
6. at the end, return the best safe candidate or a small certified candidate set.
```

This helps when at least one initialization reaches a good basin and the certificates can identify it.

It can fail when all runs are bad or when bad runs form a mutually consistent cluster. Therefore the controller needs a population-health check, not just per-run outlier detection.

Examples of unhealthy population evidence:

```text
all runs have high absolute residual;
all runs show unstable correction norms;
all runs disagree strongly with NP fallback;
all runs form a consensus cluster that is measurement-plausible but trajectory-unstable;
no candidate passes a minimum clean-free certificate set.
```

### Beam version

A more ambitious version is a beam controller:

```text
Start with one or more root states.

For each step t:
  branch each active state into B candidate continuations;
  score candidates using clean-free certificates;
  keep the top M candidates or all candidates below a risk threshold;
  optionally respawn candidates if the beam becomes unhealthy.

At the end:
  return the best certified candidate,
  or return a small candidate set,
  or fallback to NP if the beam has no healthy trajectory.
```

This is closer to a solver than the current Branch A triage rule. It tries to keep the trajectory inside the intersection of:

```text
measurement-consistent set
learned-prior / in-distribution corridor
stable trajectory basin
cross-run consensus region
```

### Main risk

The main risk is greedy pruning. A trajectory that looks slightly worse at one step may recover later, while a wrong trajectory may look temporarily clean. Therefore the beam should keep multiple candidates when possible, and the scoring rule should include persistence rather than single-step alarms.

### Offline development experiment

A first development experiment can still be offline:

```text
A18_offline_population_policy_design
```

Use existing independent trajectories as if they were a population. Do not branch new trajectories yet.

Questions to answer:

```text
If we had K=4 independent SITCOM runs, could we select a safe run without PSNR?
When the aggressive policy flags some runs, is there usually at least one unflagged high-quality run?
How often is the entire population unhealthy?
Can a small candidate set preserve high best-of-4 PSNR while excluding most catastrophes?
What would the controller do on image 00017?
```

Suggested outputs:

```text
population_health_table.csv
candidate_set_policy_summary.csv
image_level_population_decisions.csv
unsafe_population_cases.csv
SUMMARY.md
```

## 3. Relationship between the two directions

The two directions are complementary:

```text
Direction A: anytime risk detection
  tells us when a trajectory appears to be leaving a good basin.

Direction B: population / beam control
  tells us what to do when some trajectories look unsafe.
```

A good long-term Branch A solver would combine them:

```text
monitor each trajectory continuously;
prune or downweight unsafe trajectories;
keep several plausible alternatives;
spawn or fallback when the population becomes unhealthy;
return a certified candidate or candidate set.
```

This reframes Branch A from:

```text
posthoc NP/SITCOM replacement
```

to:

```text
clean-free population control for diffusion-prior inverse problems.
```

## 4. What should be avoided

Future work should avoid:

```text
retuning thresholds on A14/A16 misses and calling the result prospective;
adding a pixel-space feature solely to catch image 00017 without a new development split;
claiming first80 is an early intervention method;
using ground-truth PSNR in any executable decision;
selecting the best run by oracle over final reconstructions.
```

Image `00017` should be treated as a motivating failure case for new certificate families, not as a target to patch after the fact.

## 5. Suggested immediate next Codex task

The next useful Codex task is a population / beam-controller design pass using existing trajectories first. The anytime work should remain in diagnostic mode until a budget-feasible frozen rule is genuinely available.
