# B25 checkpoint report

## Status

`B25.0 IMPLEMENTED — PRE-RUN FREEZE REVIEW`

No B25 scientific experiment has been executed at the time of this checkpoint. The numerical choices, RNG streams, experiment matrix, protected-data policy, and interpretation/falsification criteria are frozen in `configs/b25/b25_cpu_spec.json` before PAC scientific execution.

## Repository identity

- immutable B24 start: `ed162c2f97430804fddb5d9a0bfec7abde201ca0`
- signed B23.1 ancestor: `27505e6328157ac9296c95dc5e611cbeef80de98`
- branch: `codex/b25-noise-selection-mechanisms`
- draft PR: `#39`
- PR base: `codex/b24-bestof4-failure-sweep`
- B24 decision remains `STOP_B24_METHOD_REFINEMENT`

The user-supplied PAC inventory at UTC `20260914T083919Z` had SHA-256

`e780d6bc8412bc8b8d20dbebe8428e4d13ea50254e9431efecfc48eab4128be1`.

It verified the remote B24 tip at the immutable start, fetched the B25 branch, found no pre-existing B25 worktree or B25 output root, and identified the frozen 80-development/305-confirmation registry and accepted DEV80 run. Historical DAPS and SITCOM source trees remain intentionally dirty and read-only; B25 verifies their exact B24-accepted HEAD/tree/index/diff identities rather than cleaning or changing them.

## Frozen source inputs

- role registry: `/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_2_7424_extension_20260907T231303Z/case_freeze/method_stage`
- accepted DEV80 run: `/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_dev80_overnight_20260913T082146Z`
- corrected closeout: `/egr/research-pac/huang248/outputs/pr_diffusion/b24/B24_3_zero_gpu_closeout_corrected_20260914T060600Z`
- DAPS source: `/egr/research-pac/huang248/pr_diffusion_b19_solver/external/daps`
- SITCOM source: `/egr/research-pac/huang248/external/SITCOM_ODE`
- DiffFPR source: `/egr/research-pac/huang248/external/DiffFPR`
- CPU interpreter: `/egr/research-pac/huang248/conda-envs/prdiff_ffhq/bin/python`

## Implemented B25 experiments

### Experiment 1

Independent Gaussian proposal selection with `K={1,4,8}` over a prospectively fixed logarithmic `h` grid. It includes an analytic linear order-statistic reference, an analytic smooth quadratic exponential-tilt reference, unselected proposals, hard minimum selection, and finite-candidate likelihood-weighted selection. Distributional output includes directional histograms/quantiles in addition to mean/covariance.

The prospective mechanism prediction is that fixed-`K>1` hard minimum selection gives a leading `O(sqrt(h))` state displacement along the score direction whereas a smooth likelihood tilt gives `O(h)` displacement. This is not asserted as a theorem about native NP.

### Experiment 2

Exact finite-support nonnegative `6x6` prior with distinguishable, exact-reversal-ambiguous, and near-ambiguous families and both equal/unequal prior weights. The coherent VP forward Markov chain and exact Gaussian reverse bridges are explicitly derived. Exact intermediate likelihood is computed by summing over template support. Hard selection, exact-intermediate finite-candidate weighted resampling, and denoised-point-likelihood weighted resampling use the same unconditional reverse-kernel proposal construction.

The analysis reports terminal posterior TV/mode errors, per-mode Monte Carlo standard errors and standardized deviations, intermediate mean/covariance errors, reconstruction MSE to the frozen truth as a separate metric, operation counts, and CPU time.

### Experiment 3

Manifest-driven existing-DEV80 diagnostic. Before opening any payload, it proves the 80 DEV IDs and 305 confirmation IDs are disjoint and that the accepted DEV80 manifest equals the DEV allowlist. It then evaluates at most four existing DAPS, four SITCOM, and four NP4 terminals per image. No terminal is regenerated. Fresh2 is not counted as a new candidate.

The exact tested symmetry group is the eight per-channel identity/180-degree-reversal combinations under the channelwise padded centered orthonormal FFT magnitude operator. Synthetic invariance is a mandatory gate before real DEV evaluation. Raw canonical RGB8 PSNR is recomputed and cross-checked against historical B24 records before the GT-assisted symmetry oracle is reported.

### Experiment 4

Source and existing-measurement audit only. It records the accepted dirty-source identities and bounded line excerpts for the relevant DAPS/SITCOM/NP forward/preprocessing paths. The numerical DEV audit counts negative stored amplitudes and quantifies the difference between raw and `clamp_min(0)` targets. The synthetic clipping demonstration is descriptive and cannot by itself establish which solver uses which preprocessing path.

## Safety/resource implementation

- every scientific subprocess requires `CUDA_VISIBLE_DEVICES=""`;
- no B25 runner imports the project model loader or names the FFHQ checkpoint;
- BLAS/OpenMP thread counts are capped at four;
- each stage is wrapped with wall/RSS resource accounting;
- aggregate scientific wall budget is 14,400 seconds and RAM limit 16 GiB;
- failures are preserved in numbered attempt directories;
- resume refuses a changed remote B25 head after `PRE_RUN_IDENTITY.json` is frozen;
- launch/status/resume/stop scripts do not use `set -e`, `set -u`, `pipefail`, shell replacement, or broad process-killing;
- the stop helper sends `SIGTERM` only after exact PID/command/run-root identity checking.

## Required pre-run gate on PAC

Before launch the executor must confirm:

1. Git version and fetched remote refs;
2. remote B24 still equals the immutable start;
3. the B25 worktree is newly created or already the intended clean worktree;
4. branch/head equal the current pushed B25 pre-run commit;
5. zero-GPU syntax/integrity tests pass;
6. the corrected B24 closeout and frozen DEV sources exist;
7. `PRE_RUN_IDENTITY.json` is written before the first scientific subprocess.

## Post-run work still required

After the CPU worker returns PASS, the executor must inspect the machine-readable outputs, especially `SOURCE_AUDIT.json`, before drawing the B25.4 preprocessing conclusion. Small evidence and the final checkpoint/planner return must then be committed to GitHub. The PAC run must be packaged with `scripts/b25/package_b25_run.py`; archive member safety and internal checksums must pass and the actual `.tar.gz` SHA-256 must be recorded.
