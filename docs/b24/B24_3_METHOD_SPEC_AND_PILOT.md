# B24.3 method specification and bounded pilot

## Status

This document is the human-readable companion to `configs/b24/b24_3_method_dev_spec.json`.

The baseline screen is complete and baseline image collection is closed. The next scientific question is whether changing **when native NP alternatives are discarded** improves recovery beyond independent NP populations at comparable compute.

No confirmation measurement is authorized before development selects and freezes one main method.

## Image roles

Before method execution, run `scripts/b24/freeze_b24_method_roles.sh`. The role freeze is outcome-blind with respect to all project methods and uses new domain-separated hashes.

- A100: 20 development / 80 confirmation.
- B100: 20 development / 80 confirmation.
- C100: 20 development / 80 confirmation.
- D85: 20 development / 65 confirmation.
- Pilot: four fixed development images per A/B/C/D = 16 total.

Development receives one new locked measurement per image. Confirmation receives two new locked measurements per image only after the main method is frozen.

C1 remains secondary. C1 images inside primary C100 inherit the primary role; the 69 C1 images outside primary C100 are development-locked until the main method is frozen.

## Frozen NP parent

Parent: NP-1 from `configs/b23/np1_frozen.yaml`.

- 1000 diffusion steps;
- projection start 300;
- 5 soft candidates before projection;
- 1 hard candidate after projection;
- LF measurement score, radius 0.6;
- LF projection radius 0.2;
- no runtime ground-truth use.

A pre-projection pruning score is frozen as the **trailing-32 mean selected-state LF measurement MSE** over the same diffusion-time window. Pointwise and trailing-16 variants may be logged diagnostically, but they do not control the primary pilot methods unless a later development amendment explicitly freezes such a change.

## NP_EPP: early population pruning

Purpose: test whether measurement-only early allocation improves on completing every independent NP root.

1. Start four independent native NP roots.
2. Transitions 0..71: keep four roots; each uses k=5 proposals.
3. At transition 72: rank roots by trailing-32 LF-MSE; retain two.
4. Transitions 72..147: each survivor uses k=10 proposals.
5. At transition 148: rank the two survivors by trailing-32 LF-MSE; retain one.
6. Transitions 148..299: the survivor uses k=20 proposals.
7. At projection start 300: fork the selected full native state into four deterministic independent hard-phase RNG descendants.
8. Transitions 300..998: four descendants, k=1.
9. Clean-free terminal output: minimum `selector_post_winner_lf_mse_mean`; terminal-oracle PSNR is diagnostic only.

This preserves 6000 pre-projection proposal evaluations and 2796 post-projection proposal evaluations: 8796 proposal evaluations total, matching four independent NP-1 trajectories. Including initial root model evaluations, both NP_EPP and NP4 use approximately 8800 model evaluations.

Primary ablations:

- random pruning at the same checkpoints and branch counts, using a fixed hash independent of the measurement;
- no reallocation: measurement-based pruning remains, but survivor k stays 5 and the lower Work-FRE is reported honestly.

## NP_DPS: delayed proposal selection

Purpose: test whether NP's immediate greedy proposal discard removes alternatives that would improve after further native evolution.

1. Start one native NP root.
2. Transitions 0..71: standard greedy NP-1.
3. At transition 72: generate the ordinary five NP proposals but retain all five complete proposal states.
4. Advance those five branches through transitions 73..147 using ordinary per-branch NP greedy semantics; at transition 148 prune to one by trailing-32 LF-MSE.
5. At transition 148: expand the survivor into its five ordinary proposals; retain all five and advance through 149..223; prune to one at 224.
6. At transition 224: repeat; advance five branches through 225..299; prune to one at projection start 300.
7. Fork the selected full native state into four deterministic independent hard-phase RNG descendants.
8. Transitions 300..998: four descendants, k=1.
9. Clean-free terminal output: minimum `selector_post_winner_lf_mse_mean`; terminal-oracle PSNR is diagnostic only.

The delayed windows are 76 transitions each. This gives exactly 6000 pre-projection proposal evaluations and 2796 post-projection proposal evaluations = 8796 proposals total. Because DPS starts from one root rather than four independent roots, it uses three fewer initial model evaluations than NP4: approximately 8797 versus 8800 total model evaluations.

## Comparison portfolio

Initial pilot engineering/scientific arms:

- NP-1;
- four independent NP-1;
- NP_EPP;
- NP_EPP random-pruning ablation;
- NP_EPP no-reallocation ablation;
- NP_DPS.

Subsequent 80-image development comparison adds:

- DAPS-4 on the new locked measurement;
- pinned SITCOM-4 on the same new locked measurement;
- historical NP-8-RS with its actual identity: two scoring configurations × four seeds;
- Fresh2 may be reported from a preregistered DAPS pair using its unchanged historical selector.

LF-v1 and Branch-A/B controllers are excluded from the initial comparison.

## Pilot gate

The 16-image pilot is reused in development and must verify:

- exact branch-state continuation;
- branch lineage and RNG identity;
- proposal-count and Work-FRE accounting;
- measurement-only checkpoint-score logging;
- whether measurement-based pruning discards branches that later would have been strong terminals, assessed offline with GT only after execution;
- GPU memory and throughput.

The first GPU action should be a one-image NP branching smoke. The baseline 10,240-MiB admission gate must not be inherited blindly. The global B24 hard process/group ceiling remains 52,452 MiB.

## Stop rules

If the new methods do not improve on compute-matched independent NP populations on the bounded 80-image development panel, record the negative result rather than expanding an open-ended method search.

No confirmation measurement or confirmation method run may begin before one main method is selected and frozen.
