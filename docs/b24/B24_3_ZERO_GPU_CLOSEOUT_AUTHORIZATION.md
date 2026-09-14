# B24.3 zero-GPU development closeout authorization

## Planner authorization

After the frozen PE3 DEV80 gate returned `STOP_B24_METHOD_REFINEMENT`, the user/planner authorized one combined closeout stage with no new scientific exposure:

1. derive the already-defined historical Fresh2 clean-free DAPS selector from the stored DEV80 DAPS trajectories;
2. produce complementarity/failure-overlap tables across the already-completed DEV80 methods;
3. close the cross-family compute audit with an explicit analytical inventory of unsupported Fourier/operator work and interpretation guards;
4. package B24 as a negative development result for planner/manuscript review.

This is analysis/packaging only. It is not a new method-development stage.

## Hard scope

Authorized:

- read the completed DEV80 and PE3 development artifacts;
- reuse the exact stored DEV80 measurements and terminal reconstructions;
- evaluate DAPS terminal candidates on CPU with the pinned DAPS phase-retrieval operator;
- derive historical Fresh2 using its already-frozen clean-free rule;
- compute zero-GPU summary/complementarity/compute-accounting artifacts;
- package the closeout artifacts with SHA-256 manifests and a tarball.

Not authorized:

- any GPU execution;
- any new measurement generation;
- any of the 305 confirmation images or measurements;
- any C1-only extra exposure;
- any new NP/EPP/PE3 method, score, checkpoint, schedule, threshold, or selector tuning;
- changing the PE3 frozen advancement verdict;
- merge/rebase/squash/retarget/force-push/history rewrite;
- modification of PR #37.

## Frozen Fresh2 definition

Fresh2 is historical project methodology, not newly fit on B24:

- candidate 0 = DAPS rep 0 (`base_full` historically);
- candidate 1 = DAPS rep 1 (`base_extra` historically);
- both are complete independent DAPS trajectories;
- compute the exact DAPS phase-retrieval operator loss on the terminal PNG against the locked raw measurement;
- frozen margin `theta = 0.7`;
- choose rep 1 iff

  `loss(rep1) < loss(rep0) - 0.7`;

  otherwise keep rep 0.

The exact loss is `operator.loss(x, y)` from the pinned DAPS `PhaseRetrieval` operator. The selector never uses ground-truth PSNR. B24 DEV80 DAPS uses the same full DAPS protocol family required for this derivation (`annealing num_steps=400`, `diffusion num_steps=5`).

Fresh2 reporting must distinguish:

- `FRESH2_SELECTED`: executable clean-free selected output;
- `DAPS2_ORACLE`: GT-best of reps 0 and 1, diagnostic ceiling only;
- `DAPS4_ORACLE`: existing GT-best of all four DEV80 DAPS trajectories, diagnostic ceiling only.

## Compute-closeout policy

The existing FLOP audit remains authoritative for dispatch-supported dynamic FLOPs. The closeout must not silently count unsupported FFT/custom kernels as zero.

The analytical inventory records:

- the exact DAPS phase-retrieval operator structure and loss semantics;
- the fact that the pinned DAPS phase-retrieval forward uses one centered orthonormal 2-D complex FFT on the 384x384 padded grid per operator evaluation;
- NP/PE3 measurement scoring/projection Fourier call counts that are statically determined from the frozen runner;
- Fresh2's two extra terminal operator-loss evaluations in addition to two full DAPS trajectories;
- any resolved DAPS scheduler/MCMC counts recoverable from the stored audit configuration;
- conventional FFT work-unit estimates only as estimates, never exact total FLOPs.

No exact total-FLOP equivalence claim is allowed unless unsupported work is fully incorporated.

## Scientific closeout interpretation

The PE3 frozen gate has already stopped B24 method refinement. Closeout may report that NP-native branching/reallocation can generate qualitatively different basins, but did not robustly outperform independent NP populations under the frozen DEV80 tests. It may compare already-completed methods and derive historical Fresh2, but it must not promote any post-hoc portfolio/selector as a new method.

Confirmation remains locked after this closeout.
