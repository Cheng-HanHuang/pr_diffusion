# B24 — start here

B24 is a separate study in `Cheng-HanHuang/pr_diffusion`, isolated from B23. PR #37 must remain untouched.

## Immutable base

Signed B23.1 final head:

`27505e6328157ac9296c95dc5e611cbeef80de98`

B24 branch:

`codex/b24-bestof4-failure-sweep`

Draft PR: #38.

## Completed baseline/cohort evidence

- B24.0 exposure freeze PASS.
- B24.1 serial/concurrent equivalence + memory/throughput smoke PASS.
- B24.2 baseline screen complete at 7424 rows.
- Final screen census: A=6925, B=107, C=307, D=85.
- Primary ABC300 and secondary C1 cohorts frozen.
- Method roles frozen prospectively: DEV80=80 and confirmation=305; Pilot16=4 per screening stratum.

## Completed B24.3 development evidence

The following development stages completed operationally:

- one-image NP branching calibration;
- Pilot16;
- bounded EPP321 refinement;
- full DEV80 with fresh DAPS-4 + pinned SITCOM-4 and the NP4/EPP321 portfolio;
- final protected-explorer refinement (`NP_PE3_SCORE`, `NP_PE3_RANDOM`);
- one development-only cross-family dispatch-supported FLOP audit.

### Frozen scientific verdict

`NP_EPP_321` failed to improve robustly over compute-matched NP4 on DEV80.

The final PE3 arms were then evaluated under a prospectively frozen advancement gate requiring all of:

- median paired PSNR delta vs NP4 >= 0 dB;
- Good25 count >= NP4;
- Good25 rescues >= harms;
- >=5 dB rescues >= harms.

Neither PE3 arm passed. The binding decision is:

`STOP_B24_METHOD_REFINEMENT`

No confirmation image has been exposed.

## Current authorized stage — zero-GPU development closeout

The user/planner authorized one combined analysis/packaging stage only:

1. derive historical Fresh2 from the stored DEV80 DAPS trajectories using its existing clean-free exact-operator-loss selector;
2. build DEV80 complementarity/failure-overlap tables;
3. finish the compute audit with explicit unsupported Fourier/operator-work accounting and conservative interpretation guards;
4. package B24 as a negative development result.

Exact closeout authorization/spec:

- `docs/b24/B24_3_ZERO_GPU_CLOSEOUT_AUTHORIZATION.md`
- `configs/b24/b24_3_zero_gpu_closeout.json`

### Historical Fresh2 rule

Fresh2 uses DAPS reps 0 and 1 only. Compute exact pinned DAPS phase-retrieval `operator.loss(x,y)` for each terminal candidate and select rep 1 iff

`loss1 < loss0 - 0.7`.

Otherwise retain rep 0. Ground truth is not used for selection.

### Run closeout

Use the fail-closed synchronous launcher:

```bash
cd /egr/research-pac/huang248/pr_diffusion_b23

git fetch origin \
  '+refs/heads/codex/b24-bestof4-failure-sweep:refs/remotes/origin/codex/b24-bestof4-failure-sweep'

git show \
  origin/codex/b24-bestof4-failure-sweep:scripts/b24/launch_b24_3_zero_gpu_closeout.sh \
  | bash
```

The launcher hides CUDA, verifies that the DEV80 and PE3 sources are complete and confirmation-free, runs zero-GPU tests, derives Fresh2, creates complementarity and compute-closeout artifacts, and emits a `.tar.gz` plus `.sha256`.

Status helper after completion:

```bash
bash /egr/research-pac/huang248/pr_diffusion_b24/scripts/b24/status_b24_3_zero_gpu_closeout.sh
```

## Current hard boundary

Authorized now: zero-GPU closeout/analysis only on already-exposed development artifacts.

Not authorized:

- any GPU work;
- any new measurement generation;
- any of the 305 confirmation images/measurements;
- C1-only extra exposure;
- any further NP/EPP/PE3 method, selector, score, checkpoint, schedule, or threshold tuning;
- changing the frozen PE3 stop verdict;
- merge/rebase/squash/retarget/force-push/history rewrite;
- modification of PR #37.

Confirmation remains locked after the closeout unless the planner separately changes project direction later; the closeout itself does not authorize that.
