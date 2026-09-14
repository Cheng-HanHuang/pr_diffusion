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
- one development-only cross-family dispatch-supported FLOP audit;
- final zero-GPU DEV closeout/packaging.

### Frozen scientific verdict

`NP_EPP_321` failed to improve robustly over compute-matched NP4 on DEV80.

The final PE3 arms were then evaluated under a prospectively frozen advancement gate requiring all of:

- median paired PSNR delta vs NP4 >= 0 dB;
- Good25 count >= NP4;
- Good25 rescues >= harms;
- >=5 dB rescues >= harms.

Neither PE3 arm passed. The binding decision is:

`STOP_B24_METHOD_REFINEMENT`

The zero-GPU closeout preserved this verdict. No confirmation image has been exposed.

## Completed zero-GPU development closeout

The authorized closeout completed from scientific-run head

`c3f13963cd267094231302bf9bc8d3a6e8c754c9`

at PAC run root

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_20260914T044033Z`.

It completed all required source, test, CUDA-hidden, and artifact gates and reported:

- `gpu_work_performed=false`;
- `measurement_generation_performed=false`;
- `confirmation_exposed=false`;
- `decision=STOP_B24_METHOD_REFINEMENT`.

Exact authorization/spec:

- `docs/b24/B24_3_ZERO_GPU_CLOSEOUT_AUTHORIZATION.md`
- `configs/b24/b24_3_zero_gpu_closeout.json`

Published result summary:

- `docs/b24/B24_3_ZERO_GPU_CLOSEOUT_RESULT.md`

### Historical Fresh2 closeout result

Fresh2 used DAPS reps 0 and 1 only with its historical clean-free rule:

`loss1 < loss0 - 0.7` => choose rep1, otherwise rep0.

Ground truth was not used for selection.

On DEV80:

- Fresh2 Good25 = `57/80`, mean PSNR `26.1596 dB`, median `29.9255 dB`;
- DAPS2 oracle Good25 = `59/80`;
- Fresh2 vs DAPS1: `10/0` Good25 rescues/harms;
- Fresh2 vs executable NP4: mean `+0.4249 dB`, median `+1.0939 dB`, Good25 rescues/harms `14/12`.

Fresh2 remains historical methodology and was not promoted into a new post-hoc B24 advancement candidate.

### Compute interpretation

Dispatch-supported dynamic FLOPs remain useful for relative accounting but are not exact total FLOPs because FFT/custom work is nonzero and implementation dependent. The closeout explicitly inventories unsupported Fourier/operator work and forbids an exact total-FLOP-equivalence claim.

## Current hard boundary

B24 method refinement is stopped under the frozen gate.

Not authorized:

- any GPU work;
- any new measurement generation;
- any of the 305 confirmation images/measurements;
- C1-only extra exposure;
- any further NP/EPP/PE3 method, selector, score, checkpoint, schedule, or threshold tuning;
- changing the frozen PE3 stop verdict;
- merge/rebase/squash/retarget/force-push/history rewrite;
- modification of PR #37.

Confirmation remains locked unless the planner separately authorizes a new project direction.
