# Current experiment plan: reliable phase retrieval solver after FFHQ NP/SITCOM study

## Guiding principle

Optimize for a single solver that reliably produces good reconstructions; avoid post-hoc oracle selection.

## Direction A: score improvement as one principled algorithm

- Timestep-dependent `prev_l2` regularization.
- Adaptive regularization based on LF score uncertainty.
- Candidate-bank/noise-memory strategy: retain winning noise from prior k steps and sample only remaining candidates.

## Direction B: NP inside SITCOM-ODE

Use NP-style branch selection in early timesteps and SITCOM-style refinement late, as one integrated solver.

## Direction C: robust measurement weighting

Prefer soft and robust, frequency-dependent measurement weighting over broader hard projection.

## Direction D: evaluation standard

For every method, report all-run mean/median/min, best-of-k curves, failure counts, per-image failures, SSIM/LPIPS, and runtime.
