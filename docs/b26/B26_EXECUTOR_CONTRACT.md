# B26 executor contract

## Scientific purpose

B26 answers two bounded questions without reopening B24 method search:

1. Does the historical NP observation clamp itself matter when only the scoring/selector target is corrected to the signed locked observation and projection remains nonnegative?
2. In an exact finite-support model, how much intermediate-likelihood accuracy survives a finite Monte Carlo inner estimator, and how does posterior fidelity relate to reconstruction risk when truths/observations vary?

Neither question presupposes improvement. B26.1 continuation gates are technical/resource gates only.

## Repository invariants

- signed base must remain `c906d36e0e396a6abbf761e3d65c91433428170f`;
- branch must be `codex/b26-np-conditional-correction`;
- every write explicitly targets that branch;
- scientific execution occurs only from a clean pushed pre-run SHA containing the exact PAC-derived DEV80 manifest;
- historical files are imported/read, never edited;
- `main`, PRs #37–39, historical branches/worktrees/outputs are untouched.

## B26.1 invariants

For a root seed and image, H and R share initialization, scheduler, proposal RNG semantics, K schedule, incumbent-noise reuse, UNet, projection start/radius, and hard winner rule. Define `y_raw` as the verified locked signed measurement and `y_plus=max(y_raw,0)`.

- H: score/selector against `y_plus`; project against `y_plus`.
- R: score/selector against `y_raw`; project against `y_plus`.

The historical low-frequency score remains the L2 norm of the masked predicted-magnitude residual. No mask/reduction/normalization change is permitted. At every realized candidate set, record both raw- and plus-target scores/winners; only the arm-defined winner changes the trajectory. Post-projection selector histories are accumulated against both targets. Terminal reporting includes H+/Hraw and Rraw/R+ clean-free selectors plus a GT-only four-root oracle.

Each root must use exactly the frozen root seed. The expected historical count is 2,200 UNet calls/root (1 initialization + 2,199 proposal calls), but the implementation counts actual calls and fails if the frozen loop no longer yields that contract.

Smoke is root0 on the rank-0 image of each original A/B/C/D stratum, H and R = 8 trajectories. Historical H smoke must match the accepted B24 NP4 terminal tensor SHA-256 and available selector/noise/RNG provenance for that same root. After smoke PASS, DEV16 completes all roots/arms; after DEV16 integrity/resource/budget PASS, the same frozen configuration continues to DEV80. No PSNR-based stopping or method edits are allowed.

## B26.2 invariants

The four B25 prior families/templates, sigma=0.08, and alpha schedule `[1,.72,.36,.12,0]` are frozen. For every family, 32 observation seeds independently draw truth from the family prior and Gaussian measurement noise. Outer K=5 proposals come from the exact unconditional reverse kernel. Six methods run 512 complete reverse trajectories/observation.

`L_hat_M(z)=(1/M) sum_m p(y_raw|X_m)` with `X_m~p(x0|z)` exactly sampled from analytical finite-support conditional probabilities. Compute this arithmetic mean using log-sum-exp. Inner samples, outer proposals, categorical resampling, truth/noise, and hard-tie RNG have separate namespaces. Inner samples never become terminal candidates.

For the exact reversal-equivalent pair, first verify numerical Fourier-magnitude equivalence, then impose exactly equal template log-likelihood values for the pair. Hard exact-intermediate ties are randomized uniformly using the dedicated tie RNG; do not use deterministic argmax index bias.

## Resource/safety gates

B26.1: explicit physical IDs only; >=10,240 MiB free before worker; <=52,452 MiB B26 process/group memory; one worker/GPU; 24 aggregate reservation GPU-hours maximum. Stop before overrun. Preserve failures; no silent retry.

B26.2: `CUDA_VISIBLE_DEVICES=""`, <=4 threads, <=16 GiB RSS, <=14,400 s.

All completion files are written atomically. Resume skips PASS canonical jobs, never overwrites them, and writes new numbered failure-attempt records. Confirmation payload access is a hard failure.

## Return boundary

B26 may return exactly one recommendation: `stop`, `isolate another identified defect`, or `request a precisely specified weighted-NP pilot`. B26 itself does not authorize weighted FFHQ reconstruction.
