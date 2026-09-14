# B24.3 — method specification and bounded development pilot

B24.2 baseline screening is complete at cumulative 7424 rows. The baseline class universe is frozen as A=6925, B=107, C=307, D=85. ABC300 is frozen as 100 A + 100 B + 100 C by `B24_CLASS_RANK_V1`. C1 is a secondary severity-enriched diagnostic containing the 100 class-C rows with the lowest pinned-SITCOM best-of-four PSNR; it does not redefine C.

## Scientific question

The next question is whether changing how native NP retains and allocates proposals improves recovery beyond independent NP populations at comparable compute.

The frozen parent is NP-1 from `configs/b23/np1_frozen.yaml`: 1000 scheduler steps, projection start 300, five soft proposals before projection, one hard proposal after projection, LF score radius 0.6, projection radius 0.2, and post-projection mean winner LF MSE as the executable terminal selector statistic.

## Frozen role policy

Before project-method execution, the primary images are split by a new fixed hash:

- A100: 20 development / 80 confirmation
- B100: 20 development / 80 confirmation
- C100: 20 development / 80 confirmation
- D85: 20 development / 65 confirmation

Development total: 80. Confirmation total: 305. Pilot16: the first four hash-ranked development rows in each original screen stratum A/B/C/D.

Each development image gets one new deterministic locked measurement. Confirmation images remain unmaterialized until a main method is frozen; then they are intended to receive two new locked measurements per image.

Original A/B/C/D labels remain screening strata. New-measurement baseline outcomes may transition class and are reported rather than filtered away.

## Frozen candidate algorithms

### NP_EPP — early population pruning with reallocation

Start four independent native NP roots using the role-manifest root seeds.

- transitions 0..71: 4 live roots, k=5 each;
- before transition 72: rank by trailing-32 mean selected-state LF measurement MSE, keep 2;
- transitions 72..147: 2 live branches, k=10 each;
- before transition 148: rank by the same score, keep 1;
- transitions 148..299: 1 live branch, k=20;
- projection start 300: fork the selected complete native state into four deterministic hard-phase RNG descendants;
- transitions 300..998: four descendants, k=1.

This is exactly proposal-compute matched to four independent NP-1 trajectories: 8796 proposal UNet evaluations plus four root-initial UNet evaluations = 8800 total.

The random-pruning ablation uses the same roots/checkpoints/live counts/candidate counts but chooses survivors by the domain-separated `B24_METHOD_RANDOM_PRUNE_V1` hash, with no measurement information.

The no-reallocation ablation uses the measurement-based survivors but leaves k=5 after pruning. Its lower work is reported honestly rather than pretending compute equality.

### NP_DPS — delayed proposal selection

Preferred main-method hypothesis.

Start one native NP root and run ordinary greedy NP through transition 71. At designated transitions, retain all five ordinary NP proposals instead of immediately collapsing to one:

1. expand at transition 72; retain five; advance each branch greedily through 73..147; prune to one before 148;
2. expand at 148; retain five; advance through 149..223; prune to one before 224;
3. expand at 224; retain five; advance through 225..299; prune to one at projection start 300.

Each inter-window branch uses the frozen k=5 native proposal rule. Pruning uses trailing-32 mean LF measurement MSE. Child RNG streams are deterministically domain-separated by `B24_METHOD_DPS_BRANCH_V1`.

At projection start, the selected complete state is forked to four hard-phase descendants and run through transitions 300..998. Proposal work is exactly 8796, matching NP4. Total model evaluations are 8797 rather than 8800 because NP_DPS starts from one initial root instead of four.

## Controls and later development comparison

Pilot16 arms:

- NP1
- NP4_INDEPENDENT
- NP_EPP
- NP_EPP_RANDOM_PRUNE
- NP_EPP_NO_REALLOCATION
- NP_DPS

The later 80-image development comparison also includes the historical NP-8-RS identity (two scoring configs × four seeds), and rerun DAPS-4 / pinned SITCOM-4 on the same new measurement. Fresh2 may be derived from its preregistered DAPS pair using its unchanged historical selector. LF-v1 and Branch-A/B are outside the initial portfolio.

## Information and accounting rules

No ground truth in runtime proposal retention, pruning, allocation, routing, stopping, or executable terminal selection. Checkpoint decisions use only information available at that checkpoint.

Count every evaluated proposal, including discarded proposals. Preserve branch lineage, complete native state, selected epsilon/noise, timestep, and named RNG identity.

Executable final selection: minimum `selector_post_winner_lf_mse_mean`, stable tie by terminal index. Ground-truth best terminal is offline oracle reporting only. Report the selector PSNR gap.

Early checkpoint score is explicitly `preprojection_trailing_lf_mse_mean32`: mean selected-state LF measurement MSE over the immediately preceding 32 native transitions, radius 0.6. It is a hypothesis, not a certificate.

## Resource policy

Global B24 hard process/group ceiling remains 52,452 MiB. The baseline 10,240-MiB admission gate is not assumed valid for NP branching.

The first one-image Pilot16 run is a full-method engineering/calibration smoke. It launches conservatively, records PyTorch peak allocated/reserved memory plus sampled B24-process and whole-device GPU memory, and derives a pilot-only admission gate from observed B24 process peak + 4096 MiB reserve, rounded upward to 1024 MiB with a 10,240-MiB floor.

Report exact proposal count, total UNet count, NP1-equivalent work, wall time, GPU memory, branch count/events, terminal count, clean-free selected metrics, and oracle terminal metrics.

## Current authorization

The user/planner has explicitly authorized:

- implementation of the frozen branching runner;
- one full-method engineering/calibration image from the already-frozen Pilot16 on its prospectively frozen new measurement;
- if and only if all one-image integrity/resource gates pass, the full frozen Pilot16.

The calibration image is part of Pilot16 and must be reused. The remaining 64 development images and all confirmation execution remain unauthorized.

See `docs/b24/B24_3_GPU_AUTHORIZATION.md`.
