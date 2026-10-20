# A1.4 conditional smallest prospective benchmark protocol (DESIGN ONLY — NOT AUTHORIZED)

**Decision:** This is an experiment **proposal if** the novelty review and source/data prerequisites pass. No PAC commands, GPU reservation, source cloning or new measurements are requested or authorized by A1.4. Stage name for a hypothetical new executor: `A1.5`, **not** a continuation of completed A1.2/A1.3. Planner/user decide separately.

## 0. Gate conditions BEFORE any PAC/GPU action

1. The paper claim must be more substantial than NP's intermediate position between DAPS and SITCOM: e.g. *algorithm-specific* tail coverage at equal budget, or a carefully motivated physics/reliability failure analysis not in FAST-DIPS or Xue. Read complete Xue manuscript before claiming first failure detection.
2. A true **independently identified/licensed** evaluation source/custodian is frozen before running, not B21/B22 final100, not B24 DEV80, and **never** B24 confirmation305 (including its registry/membership). Avoid choosing FFHQ IDs by retrospective oracle/PSNR screening. Check ledger #41 for Track B dataset reservations and avoid common final-test overlap. Prefer a separate dataset with independent provenance and train/test license permissions.
3. Pin Git commits and exact source/sha for official FAST-DIPS, native DAPS, native SITCOM(-ODE flavor separately) and NP. Confirm **same** model checkpoint *bytes* or prove that models differ and account for that confound; recorded A1 checkpoint SHA256 `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`, FAST public repo checkpoint download not yet verified identical. An operator source/formula match is not a model-byte match.
4. Prototype an **external frozen measurement adapter** for FAST and others that loads **the same signed measurement tensor** instead of invoking `operator.measure` separately in each method. Numerically test `A(x)` on deterministic 256 RGB impulse/gradient probes, noise variance and 384 output shape, reference per-channel alignment and round-trip 180° physics. Don't change a method's algorithm to hide a mismatch; NP's historical `y.clamp_min(0)` should be explicitly disclosed or normalized across all arms in separately preregistered ablation.
5. Agree an exact common PSNR metric before opening any test outputs. Suggested **primary**: uniformly saved **PNG8 RGB raw oriented PSNR** (because existing A1 historical choice); DAPS, SITCOM, FAST, NP all must export identically quantized candidate PNGs. Also record full-precision RGB float PSNR for *all* methods **only if** uniformly archived, strictly secondary until a new protocol preregisters it. 180° GT-orientation-aware PSNR is diagnostic, not deployable.
6. Track A planner registers `[DATA RESERVATION]` and `[RUN RESERVATION]` on #41 with source SHA, exact permissible sample manifest SHA, signed observation/noise seed SHA, model SHA, runtime envelope, exact new PAC output path and Track B competing GPU calendar. User executes an explicit bounded PAC command *only after* these reservations and source/CPU/smoke gates. Four GPUs if justified for multi-hour independent batch throughput, with source/seed/checkpoint isolation and per-run batch1 timings; **not** arbitrary 4× speedup to reported per-candidate latency.

## 1. Falsifiable primary candidate-set question

Primary: *At the same exact oversampled 256 RGB Fourier measurement, prior weights, noise and image cohort, does NP's independent noise proposal/picking produce a reproducible conditional **raw-PSNR upper envelope and better bad-tail reliability** relative to FAST-DIPS, DAPS, SITCOM at the same physically charged computational budget?*

Candidate portfolio by method `m` and image `i`: `Q_star(i,m,K)=max_{j<=K} PSNR_RGB8_raw(x[i,m,j],x_gt[i])`. This is GT-selected **offline**, not available at deployment. Separate *executable selector* frozen pre-test (if an auditable native one exists), report its actual chosen candidate and regret to oracle. Do not combine a GT-free scorer chosen after seeing exposed eight GT values and call it frozen.

**Arms (common prior/data/measurement):**
- FAST-DIPS pixel, official default PR `T=150,K_ADMM=2,S=5,rho=200,epsilon=.05` (code verified) with independent trajectory seeds 0–3. Later budget-matched FAST step variants must be frozen **before** test.
- DAPS native frozen baseline, independent 4 roots; SITCOM-ODE native independent 4 roots; NP-LF independent 4 roots and NP-S2 independent 4 roots. NP mixed LF/S2 K4 if it has a prespecified allocation, separately labeled hybrid, **no implicit K8**.
- Report first 1/2/4 independent-root prefixes, with *same K* comparisons and separate *equal measured sampling-time* comparisons. A method's K4 timing is the **sum** of all four runs plus explicit external selector/measurement/preprocessing and I/O when the paper metric includes them. FAST-DIPS published **10.354 s per run** is NOT a 4-root total. If experimental time differs, report differences; never match budget by omitting extra roots.

**Technical/nuisance controls:** deterministic seed namespace per method/image/run; one **shared** observation tensor for all arms; signed y frozen before selection; model initialization standard; no GT or prior output from future test used for steering. Operator `oversample=2`, pad=64, `sigma_y=.05`; negative-entry frequency measured and reported, NP clipping disclosed. Native DiffFPR vs DAPS/SITCOM solver differences remain fully native rather than pretending one regularization is equivalent to another.

## 2. Cost accounting and source statistics

Report alongside `Q_star`:
- `K`, n independent trajectory roots (vs within-trajectory proposals/inner loops separately), per-sample single-worker **batch1 latency**, full K subprocess wall with I/O, sampling-only duration, **GPU-active-event** duration (counter, not utilization), actual GPU occupancy/memory and total device-seconds, NFE including denoiser forwards/backwards, operator FFT/VJP/JVP and projection attempts.
- For parallel runs, aggregate makespan, full device-hours, and *single-worker isolated latency* are three different metrics. No change of x-axis between methods. No 100-image cross-cohort comparisons from different hardware as if measured jointly.
- Every GT PSNR max over K must be presented with **its K roots fully charged** even when native project/paper plots report `mean(t_j)` per run. Show `sum_{j=1}^K t_{i,m,j}` and a table of `mean per-run latency` separately.
- Lower-tail summary: raw mean and paired mean differences, median, q05/q10, Good25/Good30, Bad20, raw-to-180° disambiguation tags; q05 from pilot n<12 is meaningless, label not estimable.

## 3. Staged sizing and kill conditions (not a GPU request)

**Desk/CPU source preflight (now, A1.4):** pin repo paths and config, inspect FAST operator and its evaluator in GitHub only, compare model identities *without* fetching/using PAC, enumerate unresolved paper methods and prospective data licensing. This is A1.4's **only authorized stage**.

**Engineering pilot (hypothetical later, **separately authorized**):** at most **8–12 legitimately independent licensed images**, initially **K1** per method to check source/data measurement identity, finite outputs, GPU memory, time measurement, PNG8 PSNR replay. A tiny K4 follow-up only if fixed upfront and all K1 gates pass; no stage auto-advance. This is a code-compatibility/variability check, **not** statistical paper evidence. Pre-register stop for any source drift, operator mismatch, checkpoint mismatch, OOM, unlogged retries, cross-method unshared measurements, inability to produce uniform RGB8 outputs, GT used to select test method, or false timing accounting. No retuning based on GT of the pilot while calling the same cohort confirmatory.

**Prospective scientific test (hypothetical later):** at least ~100 independent images or a precision-based sample-size calculation tied to paired tail differences, with full K1/K2/K4 and matched device-second comparisons. Predeclare effect direction, finite paired-bootstrap confidence interval and McNemar/paired binary tests when appropriate, and a stopping rule that excludes using early observed GT to decide the final score or data IDs. For exploratory OOD, include in-domain, moderate/severe independently specified domain shifts and a prior/difficulty control; a training-prior mismatch must not be confounded with drastically altered resolution or imaging operator.

**Scientific falsification:** if at matched budget FAST gives NP no credible advantage in raw PSNR tail/coverage, and ablation of noise picking is negligible, stop NP standalone method-paper development. If only NP's intermediate Pareto cost is confirmed, publication motivation remains weak without new mechanism. If an advantage appears only on the 8 exposed cases or via GT-fitted selector/rotation selection, kill confirmatory claim.

## 4. Optional H2 selector-specific pilot (NOT currently endorsed)

First obtain full ISCAS2025 Xue detector methodology. Any candidate selection should be predeclared using actually available at-inference inputs and prior information, with orientation impossibility honestly recognized. Report not just PSNR regret but whether score predicts Good25 and calibrated confidence on previously unseen images, using cross-root aligned candidate portfolios and charging all additional FFT/denoiser evaluations. Do not include this expensive nested test in a hardware benchmark automatically.

## 5. Output/document boundaries

If planner later approves an A1.5 executor, all new source, tests and output roots must be **fresh Track A namespaces**, never editing B21–B26/A1.2/A1.3/Track B worktrees. Use exact host paths only after bounded inventory on PAC; don't fabricate a new run's timestamp, physical GPU ID, model hash or candidate manifest. No PAC shell commands exist at A1.4. Documents are not executable authorization.

**Current decision:** PROPOSED_ONLY / NO_NEW_GPU_OR_DATA / FAST-DIPS_FIRST / NOVELTY_GATE_UNRESOLVED.
