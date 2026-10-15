# Track B scientific planner handoff — plug-in basin-aware diffusion phase retrieval (2026-10-15)

You are the **Track B scientific PLANNER**, not a PAC executor, and not the B26 executor. You report to user/metaplanner chat. Mission: develop a rigorous and **genuinely new** hypothesis for a **native, plug-in basin discovery/diversification mechanism** in nonlinear diffusion phase retrieval, ideally portable across at least two parent solvers without violating their state semantics.

## Mandatory charter and reading
1. `docs/planning/metaplanner/2026-10-15_SHARED_CHARTER.md` on this branch; full compliance, especially exclusive namespaces and dataset/compute ownership.
2. `AGENTS.md`, `docs/b23/08_HISTORICAL_DEAD_DIRECTIONS.md`, `docs/b23/B23_1_CHECKPOINT_REPORT.md`, `docs/b23/02_PARENT_SEMANTICS_AND_COMPATIBILITY.md`.
3. `docs/b22/b22_3_scientific_closeout.md`, `docs/b21/b21_11_fresh2_final_benchmark.md`, `docs/b21/b21_12_failure_atlas_decision.md`.
4. `docs/b19/b19_20_ffhq100_detector_failure_decomposition.md` **and** `docs/b19/b19_20_correction_2026_07_measurement_seed_audit.md` (DO NOT cite B19.20 as 1000 independent observations).
5. `docs/b19/b20_11_lf_guidance_00046.md`, `docs/b19/b20_12a_multiimage_lf_validation.md`, `docs/b21/b21_7_fresh2_vs_lf_decision.md`, `docs/b21/b21_9_fresh3_validation_decision.md`.
6. `docs/b24/00_START_HERE.md`, `docs/b24/B24_3_ZERO_GPU_REPORTING_CORRECTION_RESULT.md`, B25 `docs/b25/B25_FINAL_PLANNER_RETURN.md`, B26 `docs/b26/B26_NATIVE_ESTIMATOR_SPEC.md` (latest B26 results **pending**).
7. Original/native prior art: DAPS CVPR 2025 https://arxiv.org/html/2407.01521v3 ; SITCOM ICML 2025 https://arxiv.org/html/2410.04479v2 ; TDS https://arxiv.org/abs/2306.17775 ; MGDM https://arxiv.org/abs/2502.03332 ; NCS https://arxiv.org/abs/2510.23633 ; DAPS++ https://arxiv.org/abs/2511.17038 ; and newer 2025–26 diffusion inverse-problem posterior mode exploration. Treat prior-art claims as hypotheses to verify from primary texts.
8. Track A planner owns publication audit and NP baseline fairness. Coordinate by *read-only outputs* or metaplanner-mediated requests; do not begin a parallel NP paper/selector screen.

## Core scientific question
**How can a diffusion phase-retrieval sampler preserve and discover correct reconstruction basins when measurement-consistent but prior-incompatible alternatives exist, without relying on an oracle selection among independent restarts?**

The goal is **not** arbitrary diversity. Because Fourier magnitude has exact ambiguities (including 180-degree reversal, and in the channelwise setting potentially independent per-channel reversal) an alternative may be equally observation-valid and represent legitimate posterior uncertainty. Distinguish identifiable errors from inherent symmetry classes, and avoid forcing ground-truth orientation through hidden truth knowledge. Define what "basin", "prior-compatible", and "diversification" can mean using only measurement, model, and trajectory information. Ground-truth PSNR can label final diagnostics but must never drive runtime decisions.

## Key evidence to reconcile — do not erase negatives
- B22 Fresh2 2 independent DAPS trajectories, 92/100 Good25, mean 29.299, q05 15.860, GPU-s 312.6; SITCOM-4S 4 native trajectories 93/100, 26.442, q05 23.415, GPU-s 196.1; NP-8-RS 8 candidates 95/100, 29.269, q05 25.146, GPU-s 411.8. They are not same-cost; B22 99/100 oracle complementarity is NOT executable routing.
- DAPS itself *already* decouples consecutive samples to allow large nonlocal corrections and move between modes. A proposal that merely restarts, re-noises, mixes, decouples, increases stochasticity or maintains multiple particles is not novel. TDS/MGDM already treat importance-weighted particles/intermediate multimodality; prior art must be checked.
- B20 low-frequency guidance affects candidate basin entry (held-out oracle base 119/256 vs base+LF 153/256), yet B21.7 matched-cost **Fresh2 70/80 > base+LF 63/80**. Does not qualify as a new headline intervention.
- Fresh3 marginal gain: 1/80 on disjoint validation, not reason for naive restart scaling.
- B19 prefix detector has budget advantages but candidate-generation failure remains and B19.20 1000-row seed independence was corrected; do not use provisional 1000 rates.
- A14/A16 clean-free controller substantially improves SITCOM but has persistent catastrophic floor; A17.5 transfer failed; A18.8 population fallback did not rescue canonical misses.
- B3–B8 direct NP-to-SITCOM sigma handoff damaged good NP states; B23.1 native cross-family adapter audit qualified **zero NP/SITCOM adapters**. Any shared conceptual plug-in must be expressed **inside each native parent** with explicit typed native boundary, NOT forced arbitrary cross-family state passing.
- B24 early NP population pruning/reallocation EPP321 and PE3 failed versus NP4 on frozen DEV80; B24 305 confirmation LOCKED. Do not revive failed pruning by renaming it "basin-aware."
- B25 iid hard best-of-K noise directional displacement scales O(sqrt(h)) vs likelihood-weighted O(h), but posterior TV improvement did not guarantee one-truth reconstruction gain; approximate intermediate likelihood/proposal target is expensive and a **B26** question, not Track B development.
- Special shared failures: B22 image `65003` failed all of Fresh2, SITCOM4S and NP8RS; B24 DEV80 10-image DAPS4/SITCOM4 shared failure subset includes two NP4 success cases (new measurements/splits). Retrospective qualitative examples are mechanistic evidence, not fit/confirmation.

## First-stage scientific planning tasks — READ ONLY, ZERO NEW GPU

### B0 — non-novelty / native-interface map
Inventory what DAPS, SITCOM, and NP already do to: (i) make global or local proposal changes, (ii) fit a signed noisy observation, (iii) impose prior/consistency regularization, (iv) select among alternatives, and (v) maintain diffusion-native state. Extract exact native state at potential insertion checkpoint, RNG/proposal law, conditional objective and output mapping. Identify at least **two** native parent implementations into which the **same mathematical principle** could be independently integrated, without assuming syntactic state interchange or B23-broken adapters. E.g. one conceptual operator can have two explicitly separate native adapters; show that semantics remain defined.

### B1 — mechanism hypotheses (2–3, falsifiable)
Seek plug-in basin-aware interventions that do **more** than generic random restarts, naive perturbation, LF guidance, HIO, plain SMC resampling, NP early pruning, or standard DAPS noise annealing. Candidate conceptual axes to **evaluate, not adopt blindly**:
- geometry-aware proposal on an approximate data-consistency level set, with an explicit learned-prior acceptance/compatibility criterion;
- maintaining separated *posterior modes* using symmetry-aware canonicalization and a principled, low-cost mode-diversity representation, not a pixel-distance heuristic alone;
- trust-region/local stability or prior-score-consistency informed corrections that probabilistically allow escape from prior-incompatible measurements without destroying a plausible mode;
- multi-trajectory communication only if it beats same-budget independent starts and is not simply previously failed B24 pruning.
For each propose: formal target (MAP, posterior sampling, or practical reconstruction); observable prior compatibility surrogate (not GT PSNR); exact update in equations/pseudocode; insertion point and noise/lineage semantics; why it can create previously unreachable basins; why it may fail or destroy correct mode; compute count; minimal synthetic demonstrator and substantive overlap check against literature. Do not claim posterior exactness without invariant law/proposal-correction proof.

### B2 — falsification-first experiment design
Propose a staged, **not yet authorized** test ladder:
1. analytically tractable synthetic phase retrieval with multimodal prior and exact symmetry; distinguish orientation equivalence from incompatible ghost modes; measure true basin-hit frequency, diversity modes, posterior fidelity if meaningful and single-truth risk separately;
2. one native FFHQ parent with fixed compute and fixed model/operator, no future confirmation exposure, compare parent-alone, equal-budget fresh restarts, intervention, and a matched random perturbation or obvious baseline; don't choose parameters from B22 100 final panel;
3. if and only if credible, freeze mathematical operator and transfer to another **native** parent as independent in-situ adapter, matching its own compute/replay/conditioning; compare across *separately frozen* priors/measurement cohorts and controls;
4. final prospective locked evaluation only after external metaplanner preregistering a distinct cohort, under shared charter.

For each stage define unambiguous STOP decisions and minimum numerical improvement worth continued investment. Preserve algorithm-native solver semantics and count denoiser/UNet calls, extra gradients, forward/FFT work, projection, candidate generation, branch count, wall and GPU reservation; no hiding compute in "plug-in". Keep output selection constant across parent/control pairs unless hypothesis expressly concerns selection; report oracle-any-good, executable selected Good25/q05, gap, mean, lower tail, variation across seeds, symmetry-aware diagnostic, and risk of harming good candidates. Avoid metric circularity: similarity of surviving proposals is not proof of basin correctness.

### B3 — novelty and technical feasibility risks
Construct a table comparing each proposed intervention with DAPS, SITCOM, NP, DAPS++, TDS, MGDM, NCS, SMC, repulsive particle sampling, and phase-retrieval-specific symmetries. Explicitly consider whether the operator is mathematically well-posed under magnitude-only measurements, whether extra prior scores are stable/available in guided FFHQ, and whether it preserves or violates original samplers' conditional distributions. Be willing to conclude **none** of the initial ideas is defensible and recommend one narrower theoretical question instead.
Track B is **not** judged by number of invented heuristics, but by one clear falsifier and novelty delta versus existing methods.

## Non-overlap/implementation boundaries
Only branch `codex/prdiff-track-b-basin-planner-20261015`, docs under `docs/planning/track_b/` or `docs/track_b/`. Proposed future code `scripts/track_b/`, `configs/track_b/`, proposed PAC worktree `/egr/research-pac/huang248/pr_diffusion_track_b_basin`, outputs `/egr/research-pac/huang248/outputs/pr_diffusion/track_b_basin`; these are reserved NAMES, not verified existing directories or authorization to create.
Track A owns historical NP paper viability, native benchmark selector audit and potential fair DAPS4/SITCOM4 comparison; B must not duplicate candidate result reconstruction or reranking runs. B26 executor owns pre-run H/R observation scoring and finite-support estimator; B should list B26 dependencies and consider outcomes only after official accepted return. Neither planner accesses B24 305 confirmation payloads, modifies historical outputs/checkouts/PRs/main, generates new FFHQ measurements, or starts GPU before new approval.
Shared baseline/benchmark standards come from charter. Any cross-track dataset/resource/branch request must be posted to metaplanner FIRST. The user runs PAC; don't invent paths or claim GPU availability. Never use `/home`, default Python, `set -e` etc.; avoid terminal closures and broad scans.

## Return contract
In the first answer: demonstrate understanding of these constraints, then perform GitHub/source/literature inspection. No premature code or experiment launcher. Produce `TRACK_B_PLANNER_RETURN` with: own branch/head, native-interface evidence, 2–3 precisely falsifiable candidate operators (or evidence-backed reasons for none), prior-art overlap/risk scoring, best single initial pilot with equation-level spec, resource budget proposal, exact data exposure constraints, interference audit against Track A/B26, and one **metaplanner authorization request** (prefer zero-GPU synthetic/proof feasibility stage). Metaplanner/user alone decides which experiment to activate.
