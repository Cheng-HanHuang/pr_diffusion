# Current experiment plan: reliable phase retrieval solver after FFHQ NP/SITCOM study

## Guiding principle

Optimize one solver for reliability; avoid post-hoc oracle selection.

## Priority directions

1. Timestep-dependent/adaptive score regularization.
2. Candidate-bank/noise-memory NP.
3. NP-style early branching inside SITCOM-ODE.
4. Robust measurement weighting over hard broad projection.

## Promotion criteria

- mean PSNR > 28.8 (screen)
- min PSNR > 25
- no catastrophic failures (<20 dB)
- report best-of-1/2/4 plus all-run metrics and failures.
