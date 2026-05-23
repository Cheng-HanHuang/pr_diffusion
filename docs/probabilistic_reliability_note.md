# Probabilistic reliability note for diffusion phase retrieval

Updated: 2026-05-23

## Purpose

This note records a first mathematical framework for the current project goal:

```text
Build a diffusion-model phase-retrieval method that reliably returns a good
reconstruction on every execution, up to a small prescribed failure probability.
```

The target is not an oracle `best-of-many` report.  The target is a pre-specified executable algorithm which may use internal randomized branches, adaptive compute, and a non-ground-truth selector, but which returns one reconstruction without using the true image.

The current empirical evidence is summarized in:

- `docs/progress_report.md`
- `docs/empirical_success_probability_multilambda_ffhq25.md`
- `docs/current_experiment_plan.md`

The main empirical lesson is:

```text
The current multi-lambda selector is usually near-oracle over the candidates
that were actually generated.

The remaining failures are usually oracle failures over the available candidate
pool.

Therefore, the present bottleneck is candidate generation / seed diversity,
not primarily selector error.
```

The purpose of this note is to formalize that observation and turn it into a probabilistic reliability framework.

## Executive summary

For one image, one measurement, and one adaptive candidate pool, write the final algorithmic failure event as

```text
algorithm fails = no good candidate is generated
                  OR selector chooses a bad candidate although a good one exists.
```

Mathematically, for a PSNR threshold tau, this becomes

\[
  \mathbb P(\mathrm{Fail}_\tau)
  \le
  \mathbb P(\mathrm{NoGood}_\tau)
  +
  \mathbb P(\mathrm{SelFail}_\tau \mid \mathrm{GoodExists}_\tau).
\]

This decomposition is the correct first theory target because it separates two different problems:

1. **Candidate generation:** randomized diffusion trajectories must produce at least one good reconstruction.
2. **Selector consistency:** the non-oracle statistic must choose a good reconstruction when one exists.

The completed experiments suggest that term 1 is currently the larger bottleneck.  Term 2 is not solved, but it appears substantially smaller under the current multi-lambda LF/S2 selector.

The most useful first theorem is therefore not a deterministic recovery theorem.  It is a reliability theorem of the form:

\[
  \mathbb P(\mathrm{Fail}_\tau(x))
  \le
  (1-p_x)^K + \eta_x,
\]

where:

- \(p_x\) is the per-branch or per-seed probability that the candidate generator creates a tau-good reconstruction for image \(x\),
- \(K\) is the number of randomized seeds / branches / adaptive attempts,
- \(\eta_x\) is the selector failure probability conditional on at least one tau-good candidate existing.

This statement is modest, but it directly guides algorithm design.  Improving the algorithm means increasing \(p_x\), reducing \(\eta_x\), or reducing the compute needed to make \((1-p_x)^K\) small.

## 1. What is being guaranteed?

### 1.1 The phase retrieval problem

Let \(x_\star \in \mathbb R^n\) denote the unknown image.  The phaseless measurement is modeled abstractly as

\[
  y = |\mathcal F_{\mathrm{os}} x_\star| + \xi,
\]

where \(\mathcal F_{\mathrm{os}}\) denotes the oversampled Fourier transform used by the current benchmark and \(\xi\) is measurement noise.  In the current FFHQ-25 experiments, the main setting is noisy magnitude data with \(\sigma_y = 0.05\), oversampling factor 2, and raw-alignment evaluation.

The loss used by an algorithm is not necessarily the true PSNR.  Typical algorithmic residuals include full Fourier magnitude residuals, low-frequency magnitude residuals, and trajectory-level statistics.  The true PSNR is only used for evaluation after the experiment.

### 1.2 Success event

Fix an evaluation threshold \(\tau\), for example \(25\), \(28\), or \(30\) dB.  For an output \(\widehat x\), define the success event

\[
  G_\tau(\widehat x; x_\star)
  :=
  \{ \operatorname{PSNR}_{\mathrm{raw}}(\widehat x,x_\star) \ge \tau \}.
\]

Raw alignment is written explicitly because the current project treats raw PSNR as the primary benchmark.  Other alignments can be evaluated as diagnostics, but the reliability theorem should be stated for the primary evaluation convention.

### 1.3 What “always successful” should mean

The phrase “always successful” should be interpreted probabilistically:

```text
For a target threshold tau and failure tolerance delta, the algorithm should
return a tau-good reconstruction with probability at least 1 - delta.
```

For one fixed image \(x\), this means

\[
  \mathbb P(\operatorname{PSNR}(\widehat x,x) \ge \tau) \ge 1-\delta.
\]

For a benchmark set \(\mathcal X = \{x_1,\ldots,x_m\}\), a stricter benchmark-level statement is

\[
  \mathbb P\left(\min_{i=1,\ldots,m}
    \operatorname{PSNR}(\widehat x_i,x_i) \ge \tau\right)
  \ge 1-\delta.
\]

A simple sufficient condition is the union bound:

\[
  \sum_{i=1}^m \mathbb P(\mathrm{Fail}_\tau(x_i)) \le \delta.
\]

For the FFHQ-25 benchmark, if the desired benchmark-level failure probability is \(\delta=0.05\), then one conservative target is roughly

\[
  \mathbb P(\mathrm{Fail}_\tau(x_i)) \le 0.05/25 = 0.002
\]

for each image.

## 2. Current algorithmic object

The current best empirical method is a multi-lambda LF/S2 selector.  Abstractly, it defines a finite branch family

\[
  \mathcal B
  =
  \{\text{seed}\}\times\{\text{score/config branch}\}.
\]

The current branch family is

```text
configs = LF, S2 lambda=0.005, S2 lambda=0.02, S2 lambda=0.05
score_radius = 0.6
proj_radius  = 0.2
proj_start   = 300
```

Each branch \(b\in\mathcal B\) runs a randomized diffusion reconstruction trajectory and returns:

\[
  (\widehat x_b, Z_b),
\]

where \(\widehat x_b\) is the final reconstruction and \(Z_b\) denotes trajectory diagnostics.  The current selector statistic is based on the post-projection winner low-frequency MSE mean.  Abstractly, write it as

\[
  T_b = T(\widehat x_b, Z_b, y).
\]

The executable selector chooses

\[
  \widehat b = \arg\min_{b\in\mathcal B} T_b,
  \qquad
  \widehat x = \widehat x_{\widehat b}.
\]

This is important: the selector does not use PSNR and does not use \(x_\star\).  Oracle quantities are only used afterward to diagnose whether the candidate pool contained a good reconstruction.

## 3. Candidate-generation success probability

### 3.1 Per-branch success

For image \(x\), measurement \(y\), threshold \(\tau\), and branch \(b\), define

\[
  p_{x,b}^{\tau}
  :=
  \mathbb P\bigl(
    \operatorname{PSNR}(\widehat x_b,x) \ge \tau
  \bigr).
\]

The probability is over all branch randomness: initial noise, diffusion stochasticity, candidate sampling, and any randomized choices inside the solver.

For a candidate pool \(\mathcal B_K\), define

\[
  \mathrm{GoodExists}_\tau
  :=
  \bigcup_{b\in\mathcal B_K}
  \{\operatorname{PSNR}(\widehat x_b,x)\ge\tau\}.
\]

The complementary event is

\[
  \mathrm{NoGood}_\tau
  :=
  \bigcap_{b\in\mathcal B_K}
  \{\operatorname{PSNR}(\widehat x_b,x)<\tau\}.
\]

### 3.2 Independent branch model

If the branch success events are independent, then

\[
  \mathbb P(\mathrm{NoGood}_\tau)
  =
  \prod_{b\in\mathcal B_K}(1-p_{x,b}^{\tau}).
\]

Using \(1-u\le e^{-u}\),

\[
  \mathbb P(\mathrm{NoGood}_\tau)
  \le
  \exp\left(-\sum_{b\in\mathcal B_K} p_{x,b}^{\tau}\right).
\]

If all branches have the same success probability \(p_x^\tau\), then

\[
  \mathbb P(\mathrm{NoGood}_\tau)
  =
  (1-p_x^\tau)^K.
\]

To make this at most \(\delta\), it suffices to choose

\[
  K
  \ge
  \frac{\log \delta}{\log(1-p_x^\tau)}.
\]

This is the simplest adaptive-compute rule.

### 3.3 Empirical interpretation

The current empirical table estimates image-specific candidate-generation difficulty.  For example, the aggregate table over seeds 100--109 and four configs per seed reports:

```text
00028: 6 / 40 successful candidates, 4 / 10 seeds with any success
00005: 12 / 40 successful candidates, 6 / 10 seeds with any success
00013: 18 / 40 successful candidates, 7 / 10 seeds with any success
```

The seed-level success rate is more relevant for seed-adaptive compute than the candidate-level rate, because configs within one seed are correlated.  A rough point estimate is therefore

\[
  \widehat p_x^{\mathrm{seed}}
  =
  \frac{\#\{\text{seeds with at least one successful config}\}}
       {\#\{\text{seeds}\}}.
\]

For \(00028\), this gives \(\widehat p_x^{\mathrm{seed}}=0.4\).  Under the optimistic independent-seed model,

\[
  (1-0.4)^K \le 0.05
  \quad\Rightarrow\quad
  K\ge 6,
\]

and

\[
  (1-0.4)^K \le 0.002
  \quad\Rightarrow\quad
  K\ge 13.
\]

The second number is the more conservative per-image target if one wants a 25-image benchmark-level failure probability around 5% by union bound.

This calculation should not be over-interpreted.  It is a design heuristic, not yet a certified guarantee, because ten empirical seeds are not enough to estimate small failure probabilities tightly.

### 3.4 Lower-confidence version

A more honest empirical reliability bound replaces \(p_x\) by a lower confidence bound \(\underline p_x\).  Suppose we run \(N\) independent seeds and observe \(S\) seeds that produce at least one tau-good candidate.  Then \(S\sim\mathrm{Binomial}(N,p_x)\) under the seed-level model.  A lower confidence bound \(\underline p_x\) satisfies

\[
  \mathbb P(p_x\ge \underline p_x) \ge 1-\alpha.
\]

Then the candidate-generation failure bound becomes

\[
  \mathbb P(\mathrm{NoGood}_\tau)
  \lesssim
  (1-\underline p_x)^K
\]

with statistical confidence \(1-\alpha\).

This reveals a key limitation of the current data: observing 4 successes out of 10 seeds for a hard image suggests nontrivial success probability, but the conservative lower confidence bound is much smaller than 0.4.  Therefore, if the paper eventually wants a serious reliability claim, the hard images need more repeated seeds.

## 4. Selector consistency

Candidate generation alone is not enough.  If a good candidate exists, the algorithm still has to choose it without seeing the ground truth.

### 4.1 Good and bad candidate sets

For a fixed threshold \(\tau\), define

\[
  \mathcal G_\tau
  :=
  \{b\in\mathcal B_K:
    \operatorname{PSNR}(\widehat x_b,x)\ge\tau\},
\]

and

\[
  \mathcal H_\tau
  :=
  \mathcal B_K\setminus\mathcal G_\tau.
\]

The selector succeeds conditional on \(\mathcal G_\tau\ne\emptyset\) if

\[
  \widehat b \in \mathcal G_\tau.
\]

It fails if

\[
  \exists g\in\mathcal G_\tau,\ h\in\mathcal H_\tau
  \quad\text{such that}\quad
  T_h \le T_g
\]

and the minimum selected branch is bad.

### 4.2 Separation assumption

A useful selector theorem requires a separation condition.  Assume there exist values \(\mu_G\), \(\mu_H\), and margin \(\Gamma>0\) such that

\[
  \mu_H - \mu_G \ge \Gamma,
\]

and for good candidates,

\[
  T_g \approx \mu_G,
\]

while for bad candidates,

\[
  T_h \approx \mu_H.
\]

More concretely, suppose

\[
  T_g \le \mu_G + \varepsilon_g
  \quad\text{for all }g\in\mathcal G_\tau,
\]

and

\[
  T_h \ge \mu_H - \varepsilon_h
  \quad\text{for all }h\in\mathcal H_\tau.
\]

If

\[
  \varepsilon_g + \varepsilon_h < \Gamma,
\]

then every good candidate has better selector statistic than every bad candidate, and the selector succeeds.

### 4.3 Concentration version

The current statistic is an average over post-projection trajectory steps.  A stylized model is

\[
  T_b = \frac{1}{M}\sum_{t\in\mathcal I_{\mathrm{post}}} R_{b,t},
\]

where \(R_{b,t}\) is the post-projection winner low-frequency residual at step \(t\).

If the trajectory residuals have concentration around branch-type means, then for some effective sample size \(M_{\mathrm{eff}}\),

\[
  \mathbb P(|T_b-\mu_b|\ge u)
  \le
  2\exp\left(
    -c M_{\mathrm{eff}} u^2/\sigma_T^2
  \right).
\]

This gives the selector error bound

\[
  \mathbb P(\mathrm{SelFail}_\tau\mid\mathrm{GoodExists}_\tau)
  \le
  2|\mathcal G_\tau||\mathcal H_\tau|
  \exp\left(
    -c M_{\mathrm{eff}} \Gamma^2/\sigma_T^2
  \right),
\]

under the simplified assumption that good and bad candidates have a positive statistic gap \(\Gamma\).

This theorem is useful even if the exact concentration assumptions are not yet provable.  It says what needs to be measured:

```text
For every generated candidate, compare PSNR success/failure against
post_winner_lf_mse_mean and related trajectory statistics.

Estimate whether good and bad candidates are separated by the selector statistic.

Measure the margin and its stability over seeds, configs, images, and noise.
```

### 4.4 Why this matches the experiments

The current experiments already report four views:

```text
selected_config_seed_by_selector
selected_config_bestofk
global_run_by_selector
oracle_all_candidates
```

When the selected result fails and the oracle also fails, the failure is not a selector failure.  It is a candidate availability failure.

When the oracle succeeds but the selected result fails, the failure is a selector consistency failure.

The current evidence suggests that the second case is less common than the first.  This supports prioritizing candidate generation and adaptive compute while still measuring selector regret.

## 5. Full reliability theorem

### Theorem 1: reliability decomposition

Fix an image \(x\), measurement \(y\), candidate pool \(\mathcal B_K\), selector statistic \(T_b\), and threshold \(\tau\).  Let

\[
  \mathrm{Fail}_\tau
  :=
  \{\operatorname{PSNR}(\widehat x_{\widehat b},x)<\tau\}.
\]

Let

\[
  \mathrm{NoGood}_\tau
  :=
  \{\mathcal G_\tau=\emptyset\},
\]

and

\[
  \mathrm{SelFail}_\tau
  :=
  \{\widehat b\notin\mathcal G_\tau\}
  \quad\text{on the event }\mathcal G_\tau\ne\emptyset.
\]

Then

\[
  \mathbb P(\mathrm{Fail}_\tau)
  \le
  \mathbb P(\mathrm{NoGood}_\tau)
  +
  \mathbb P(\mathrm{SelFail}_\tau\mid\mathcal G_\tau\ne\emptyset).
\]

This is simply the union bound, but it is the central conceptual decomposition.

### Corollary 1: independent homogeneous branches

If each independent branch produces a tau-good reconstruction with probability at least \(p_x\), and if the selector failure probability conditional on at least one good candidate is at most \(\eta_x\), then

\[
  \mathbb P(\mathrm{Fail}_\tau(x))
  \le
  (1-p_x)^K + \eta_x.
\]

To ensure

\[
  \mathbb P(\mathrm{Fail}_\tau(x))\le\delta_x,
\]

it is enough to choose \(K\) so that

\[
  (1-p_x)^K \le \delta_x-\eta_x.
\]

Equivalently,

\[
  K
  \ge
  \frac{\log(\delta_x-\eta_x)}{\log(1-p_x)}.
\]

This requires \(\delta_x>\eta_x\).

### Corollary 2: benchmark-level reliability

For a benchmark \(\mathcal X=\{x_1,\ldots,x_m\}\), if

\[
  \mathbb P(\mathrm{Fail}_\tau(x_i))\le\delta_i
\]

for each image and

\[
  \sum_{i=1}^m \delta_i \le \delta,
\]

then

\[
  \mathbb P\left(
    \min_i \operatorname{PSNR}(\widehat x_i,x_i)\ge\tau
  \right)
  \ge
  1-\delta.
\]

This gives an adaptive compute target: allocate more seeds to hard images until their estimated failure probability is comparable to the easy images.

## 6. Adaptive compute as a theorem-guided algorithm

The current fixed two-seed and four-seed protocols can be replaced by adaptive compute.

### 6.1 Ideal adaptive policy

For each image:

```text
Initialize candidate pool with a small number of seeds/configs.
Compute selector statistic and risk features.
Estimate whether candidate availability and selector confidence are sufficient.
If risk is low, stop and return selected candidate.
If risk is high, add more seeds/configs.
Repeat until confidence target is reached or max budget is reached.
```

### 6.2 Risk features

Useful risk features include:

```text
best selector statistic
selector margin between best and second-best candidates
config-level statistic margin
seed-level statistic margin
agreement/disagreement among configs
post-projection LF residual trajectory shape
full magnitude residual
whether selected branch belongs to known fragile config patterns
whether the image resembles known hard-image behavior
```

The key distinction is:

```text
A low selector statistic is not itself a proof of PSNR success.
It is a statistical certificate whose reliability must be calibrated.
```

### 6.3 Adaptive reliability bound

Let \(K(x)\) be the number of branches actually run for image \(x\).  If the stopping rule ensures an estimated lower success probability \(\underline p_x\) and selector risk estimate \(\widehat\eta_x\), then the algorithm can require

\[
  (1-\underline p_x)^{K(x)} + \widehat\eta_x \le \delta_x.
\]

This is the reliability stopping criterion.

In practice, the first version should not claim formal certification.  It should be presented as a calibrated risk rule:

```text
Continue generating candidates until the empirical risk score is below a
predefined threshold.
```

Then validate the rule on held-out seeds and images.

## 7. Role of the diffusion prior

### 7.1 Why pure phase retrieval is not enough

The Fourier magnitude map is not globally injective over all natural images.  Even ignoring trivial ambiguities, phaseless measurements can admit many plausible or implausible solutions.  Therefore, a deterministic theorem of the form

```text
low measurement residual implies high PSNR
```

is false without additional assumptions.

This is why the pretrained diffusion model matters.

### 7.2 Prior-restricted feasible set

Let \(\mathcal C\) denote the set of images that are reachable or high-probability under the pretrained diffusion prior and the reconstruction dynamics.  This is not a simple known manifold, but it is the effective feasible set searched by the algorithm.

A useful assumption is **restricted identifiability**:

For every \(x\in\mathcal C\), if \(x\) is far from \(x_\star\), then its phaseless measurement is separated from the true measurement:

\[
  d(x,x_\star)\ge\varepsilon
  \quad\Longrightarrow\quad
  \mathcal L_y(x)
  \ge
  \mathcal L_y(x_\star) + \kappa(\varepsilon) - \mathrm{noise\ term}.
\]

Here \(d\) can be an image metric related to PSNR, and \(\mathcal L_y\) can be a magnitude residual.

This assumption says:

```text
The phase retrieval problem may be ill-posed in ambient space, but it may become
better posed after restricting to the diffusion-prior-reachable set.
```

### 7.3 What this assumption buys

If restricted identifiability holds, then a sufficiently good candidate under the prior and measurement residual should be close to the ground truth.  This would connect selector statistics to reconstruction quality.

However, for a modern pretrained diffusion model, \(\mathcal C\) is not explicitly characterized.  Therefore, this assumption is not currently provable from first principles for FFHQ and the guided diffusion checkpoint.

The theory note should therefore treat restricted identifiability as an assumption to be empirically tested, not a proven fact.

## 8. What can and cannot be proven now

### 8.1 What can be proven cleanly

The following statements can be proven under explicit assumptions:

1. **Failure decomposition:** algorithm failure is bounded by candidate-generation failure plus conditional selector failure.
2. **Adaptive compute budget:** if branch success probability is lower-bounded by \(p_x\), then \(K\) branches reduce candidate-generation failure like \((1-p_x)^K\).
3. **Benchmark union bound:** per-image failure guarantees imply whole-benchmark reliability.
4. **Selector consistency under separation:** if the selector statistic separates good and bad candidates with concentration, then selector failure is small.
5. **Empirical confidence bounds:** repeated seeds can give lower confidence bounds on \(p_x\), which can be converted into conservative compute budgets.

These are real mathematical statements and are already useful.

### 8.2 What cannot be proven without stronger modeling

The following cannot honestly be proven yet:

1. **Universal deterministic recovery** for arbitrary images from Fourier magnitude using the current diffusion heuristic.
2. **PSNR guarantee from measurement residual alone**, because wrong images may have similar magnitude residuals.
3. **Exact posterior correctness**, because the implemented trajectory is not an exact posterior sampler under a fully specified likelihood-prior pair.
4. **Architecture-only lower bound on \(p_x\)** for a pretrained guided diffusion model.
5. **Guaranteed reliability on unseen images** without either prior modeling assumptions or empirical validation over a representative image distribution.

These limitations are not failures of the project.  They clarify the correct scope of the theory.

## 9. Failed or incomplete proof directions

### 9.1 Deterministic phase retrieval recovery

Attempt:

```text
Prove that the phaseless Fourier measurement uniquely determines x.
```

Why it fails:

```text
The phase retrieval map is not globally injective in the ambient image space.
Even if oversampling reduces ambiguity, deterministic uniqueness over all images
is not the mechanism used by diffusion methods.
```

What would be needed:

```text
A restricted uniqueness theorem over the diffusion-prior feasible set.
```

### 9.2 Residual-only certificate

Attempt:

```text
Prove that small low-frequency or full magnitude residual implies high PSNR.
```

Why it fails:

```text
Magnitude residual is not equivalent to perceptual/image-space correctness.
Wrong candidates can satisfy low-frequency magnitude constraints.
```

What would be needed:

```text
A prior-restricted stability condition showing that, within the reachable set,
small residual implies closeness to the target.
```

### 9.3 Selector theorem without separation

Attempt:

```text
Prove that the selector chooses a good candidate whenever one exists.
```

Why it fails:

```text
A non-oracle statistic can only be guaranteed if it separates good candidates
from bad candidates. Without a gap assumption, a bad candidate can have an equal
or better statistic.
```

What would be needed:

```text
Empirical or theoretical evidence that post_winner_lf_mse_mean is stochastically
smaller for good candidates than for bad candidates, with a measurable margin.
```

### 9.4 Exact Bayesian posterior theorem

Attempt:

```text
Model the diffusion solver as exact posterior sampling and use posterior
concentration.
```

Why it fails:

```text
The current algorithm uses heuristic projection, candidate scoring, lambda
branches, and trajectory selection. It is not an exact sampler from a specified
posterior distribution.
```

What would be needed:

```text
A carefully defined approximate posterior sampler and a bound on the
approximation error. This is likely much harder than the current project needs.
```

## 10. Empirical quantities to estimate next

The theory suggests the following measurements.

### 10.1 Candidate-generation quantities

For each image and threshold \(\tau\):

```text
seed_success_rate_tau
config_success_rate_tau
candidate_success_rate_tau
best_config_family
hard-image class
lower confidence bound for seed_success_rate_tau
```

The existing empirical table already starts this analysis for \(\tau=25\) dB.

### 10.2 Selector quantities

For each candidate:

```text
PSNR success/failure label
post_winner_lf_mse_mean
post_winner_full_mse_mean
pre_winner_lf_mse_mean
selector margins
config margins
seed margins
```

Then evaluate:

```text
Does the statistic rank successful candidates above failures?
How often does oracle success exist but selector fails?
How large is selector regret when it fails?
Which images/configs break the selector?
```

### 10.3 Adaptive-compute quantities

For each image:

```text
number of seeds needed until first success
number of seeds needed until selector chooses success
estimated stopping time under proposed risk rules
average compute budget
worst-case compute budget
failure rate after stopping
```

This directly connects theory to an executable algorithm.

## 11. Recommended next work

### Recommendation

The best next step is **not** to spend a long time trying to prove a deeper deterministic theorem.  The best next step is:

```text
Build an empirical reliability calibration layer and simulate adaptive compute
from existing traces.
```

Reason:

```text
The current bottleneck is candidate availability, and the existing data already
contains enough structure to design and test adaptive compute policies.
```

A deeper mathematical study is still valuable, but it should be focused on the reliability framework above rather than on full recovery theory.

### Immediate task A: adaptive-compute simulation from existing traces

Using the existing seed/config traces, simulate policies such as:

```text
Policy 1:
  start with 1 seed;
  if risk score is high, add 1 seed;
  if still high, add 2 more seeds.

Policy 2:
  start with 2 seeds;
  if risk score is high, add 2 more seeds.

Policy 3:
  choose configs adaptively;
  start with LF + S2 lambda=0.02;
  add lambda=0.05 or lambda=0.005 depending on risk features.
```

Report:

```text
raw mean PSNR
raw min PSNR
below25 count
average seeds used per image
max seeds used per image
which images trigger extra compute
selector failures vs oracle failures
```

This is the most direct path toward an “always successful with adaptive compute” method.

### Immediate task B: selector calibration plot/table

Build a table or plot of

```text
post_winner_lf_mse_mean vs raw PSNR
```

across all candidates.  Do this globally and per hard image.  This tests whether the selector statistic has a meaningful separation margin.

If the plot shows good separation, the selector-consistency theorem becomes plausible.

If the plot shows overlap, the selector needs better features before a selector theorem can be useful.

### Immediate task C: targeted hard-image candidate-generation study

Focus on:

```text
00028
00005
00013
00034
00027
00007
00000
```

For these images, test whether new branches increase seed-level success probability:

```text
lambda = 0.01, 0.02, 0.05, 0.1
proj_start = 300, 350, 400
soft_candidates = 5, 8
hard_candidates = 1, 2
```

The key metric is not average PSNR.  The key metric is:

```text
Does this branch increase p_x for the hard images without destroying easy images?
```

### Medium-term task: reliability-calibrated algorithm

Implement an algorithm with the following structure:

```text
Run initial seed/config pool.
Compute selected candidate and risk features.
If calibrated risk is low, stop.
If calibrated risk is high, add targeted branches.
Repeat until risk threshold or max budget.
Return selected candidate.
```

This would be a true algorithmic improvement over fixed best-of-many reporting, because the stopping and selection rules are pre-specified and do not use ground truth.

## 12. Possible paper-level framing

A future paper contribution could be framed as:

```text
We study reliability, not only average reconstruction quality, for diffusion
phase retrieval.

We decompose failure into candidate-generation failure and selector failure.

We show empirically that recent diffusion phase-retrieval failures are often
candidate-availability failures.

We introduce an adaptive reliability strategy that allocates compute until a
non-ground-truth risk certificate is satisfied.

The resulting method reduces below-threshold failures without using PSNR oracle
selection.
```

This is a more honest and potentially stronger story than claiming deterministic recovery.

## 13. Current conclusion

The correct mathematical direction is a probabilistic reliability framework:

\[
  \boxed{
  \mathbb P(\mathrm{Fail}_\tau(x))
  \le
  \mathbb P(\mathrm{NoGood}_\tau(x))
  +
  \mathbb P(\mathrm{SelFail}_\tau(x)\mid\mathrm{GoodExists}_\tau(x))
  }
\]

and under an idealized independent candidate-generation model,

\[
  \boxed{
  \mathbb P(\mathrm{Fail}_\tau(x))
  \le
  (1-p_x)^K+\eta_x.
  }
\]

This framework does not prove full deterministic phase retrieval.  Instead, it gives a practical and mathematically transparent way to design an always-successful-up-to-small-probability algorithm.

The next project step should be adaptive-compute simulation and selector calibration, followed by targeted hard-image candidate-generation ablations.