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
- one zero-GPU DEV80 closeout capsule.

### Frozen scientific verdict

`NP_EPP_321` failed to improve robustly over compute-matched NP4 on DEV80.

The final PE3 arms were evaluated under a prospectively frozen advancement gate requiring all of:

- median paired PSNR delta vs NP4 >= 0 dB;
- Good25 count >= NP4;
- Good25 rescues >= harms;
- >=5 dB rescues >= harms.

Neither PE3 arm passed. The binding decision is accepted:

`STOP_B24_METHOD_REFINEMENT`

No confirmation image has been exposed.

## Original zero-GPU closeout capsule

Original run root:

`/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_20260914T044033Z`

Verified archive SHA-256:

`fe4ac6e0ac5c6554973cc171067f52fe0b729e6a33ded7503594fd424da3b45d`

The original capsule is preserved unchanged. Its metrics remain valid, but one reporting-scope error was identified: `unique_good25_image_ids` was computed over all DEV80 and then described in repository prose as if those counts belonged to the 10-case shared-failure subset.

The correction does not affect any reconstruction, PSNR, Good25 label, Fresh2 comparison, PE3 gate, or stop decision.

## Correct shared-failure interpretation

The shared-failure subset is defined by:

`DAPS4_ORACLE < 25 dB` and `SITCOM4_ORACLE < 25 dB`.

It contains 10 DEV images.

Good25 successes / exclusive successes among the compared executable methods inside that subset are:

- `DAPS1`: `0 / 0`;
- `FRESH2_SELECTED`: `0 / 0`;
- `SITCOM1`: `0 / 0`;
- `NP4_SELECTED`: `2 / 1`;
- `EPP321_SELECTED`: `2 / 2`;
- `PE3_SCORE_SELECTED`: `1 / 0`;
- `PE3_RANDOM_SELECTED`: `0 / 0`.

PE3 score's subset success is image `34587`, shared with NP4. Fresh2's globally unique Good25 success is image `17146`, which lies outside the shared-failure subset.

Because Fresh2 selects from DAPS reps 0/1, the corrected reporting stage enforces per image:

`FRESH2_SELECTED <= DAPS2_ORACLE <= DAPS4_ORACLE`.

Therefore Fresh2 cannot rescue a failure of that same DAPS4 candidate set.

## Current authorized stage — reporting-only successor capsule

The planner accepted the stop decision and authorized only a zero-GPU reporting correction:

1. separate all-DEV80 uniqueness from shared-failure-subset success and exclusivity;
2. correct the result document and PR description;
3. add nested-oracle and subset-consistency checks;
4. preserve the verified original capsule and publish a corrected successor.

Implementation:

- `configs/b24/b24_3_zero_gpu_reporting_correction.json`
- `scripts/b24/correct_b24_3_zero_gpu_closeout.py`
- `scripts/b24/test_b24_3_zero_gpu_reporting_correction.py`
- `scripts/b24/launch_b24_3_zero_gpu_reporting_correction.sh`

The successor performs no reconstruction, no model/operator re-evaluation, no metric recomputation, no GPU work, no measurement generation, and no confirmation exposure. It reads the already-published per-image closeout table, checks the reporting invariants, copies valid artifacts, and regenerates only the ambiguous summaries/report.

### Run corrected successor

```bash
cd /egr/research-pac/huang248/pr_diffusion_b23

git fetch origin \
  '+refs/heads/codex/b24-bestof4-failure-sweep:refs/remotes/origin/codex/b24-bestof4-failure-sweep'

git show \
  origin/codex/b24-bestof4-failure-sweep:scripts/b24/launch_b24_3_zero_gpu_reporting_correction.sh \
  | bash
```

The launcher verifies the original archive SHA-256 and sidecar, verifies all original internal checksums, hides CUDA, runs correction tests, creates a new successor directory/archive, and leaves the original capsule untouched.

## Current hard boundary

Authorized now: reporting-only zero-GPU correction of already-exposed DEV80 closeout artifacts.

Not authorized:

- any GPU work;
- any reconstruction or model/operator re-evaluation;
- any new measurement generation;
- any of the 305 confirmation images/measurements;
- C1-only extra exposure;
- any further NP/EPP/PE3/DPS method, selector, score, checkpoint, schedule, threshold, or root-budget tuning;
- changing the frozen PE3 stop verdict;
- merge/rebase/squash/retarget/force-push/history rewrite;
- modification of PR #37.

B24 closes as a negative development result after the corrected successor capsule is published. Any Fresh2 follow-up requires a separate prospective scientific question.
