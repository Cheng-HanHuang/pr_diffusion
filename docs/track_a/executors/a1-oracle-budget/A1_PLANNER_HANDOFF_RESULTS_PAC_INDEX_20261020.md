# Track A A1 executor return to scientific planner — verified native K4 results, PAC inventory, and proposed next scientific gates

**Report date:** 2026-10-20 (America/Detroit).  
**Origin:** Track A A1 oracle-budget executor, branch `codex/prdiff-track-a-a1-oracle-budget-20261016`.  
**Destination:** Track A scientific planner, branch `codex/prdiff-track-a-np-paper-planner-20261015`, via shared coordination issue [#41](https://github.com/Cheng-HanHuang/pr_diffusion/issues/41).  
**Status:** `A1.2_NATIVE_K4_ENGINEERING_PASS / SCIENTIFIC_INTERPRETATION_NEEDS_PLANNER_DECISION`. No future GPU/data reservation requested or created by this handoff.  
**Scope warning:** eight images in A1.2 are from **previously exposed B22 final100**, not an eligible new held-out or B24-confirmation-disjoint cohort. The protected B24 confirmation305 remains completely inaccessible, even for hashed membership checks.

## Executive return and what is genuinely new

The A1 executor has now *actually executed* four independent native starts for each of DAPS, SITCOM-ODE, NP-LF and NP-S2 on each of eight fixed, identical-observation FFHQ images (128 full native trajectories total), after validating a procedural native smoke and provenance/CPU input freeze. This repairs two critical limitations of the B22 retrospective metadata: the previously missing **DAPS K4** and truly **independently seeded SITCOM K4**. Source/checkpoint/measurement identities, seeds, output digests and timings are recorded per candidate, with no native candidate failures. This is important engineering evidence but not a prospective scientific test.

**Main quantitative outcome:** K4 *offline ground-truth PSNR oracle* (identical saved RGB8 representation, raw orientation):
- DAPS: **31.518 dB**, Good25=8/8, Good30=8/8; **260.11 s/image** charged subprocess wall.
- NP-LF: **30.388 dB**, Good25=8/8, Good30=4/8; **147.40 s/image**.
- NP-S2: **30.222 dB**, Good25=8/8, Good30=4/8; **147.17 s/image**.
- SITCOM: **27.439 dB**, Good25=8/8, Good30=0/8; **122.48 s/image**.
- DAPS thus leads NP-LF by **+1.130 dB on average and on all eight images**, at **1.765x NP-LF's wall time**. NP-LF leads SITCOM by **+2.949 dB**, at **1.203x SITCOM's wall time**.
- **NP-LF K4 vs DAPS K2:** NP-LF 30.388 dB at 147.40 s/image versus DAPS 28.634 dB at 133.58 s/image; *not* a clean robustness win because DAPS's K2 raw mean is disproportionately affected by an exact-orientation ambiguity (details below).
- NP-MIXED8 (= LF0,S20,LF1,S21,LF2,S22,LF3,S23): **30.392 dB**, only +0.004 dB vs NP-LF K4 at **twice the charged standalone candidate wall** (~294.58 vs 147.40 s/image); **no additional experimental GPU** because existing candidates were recombined offline. No evidence for a useful LF/S2 mixture advantage on these eight images.
- SITCOM K2 reaches Good25=8/8 at ~60.98 s/image, while K4 produces only +0.014 dB in mean PSNR; higher-quality Good30 ceiling remains 0/8.
- All four families saturate Good25=8/8 by K4, so Good25 alone cannot distinguish these policies on this exposed pilot. Importantly, the A1 preregistered *primary reliability contrast* NP-S2 K4 versus SITCOM-independent4 yields **0 percentage-point difference** in Good25 here; it does **not** meet the planned +5 pp gate. This is only exposed-development evidence, not a confirmatory failure test.

**User's interpretation and requested research priorities:** NP may be a practical middle-ground on a quality–efficiency frontier: ~43% less time/−1.13 dB versus DAPS K4, ~20% more time/+2.95 dB versus SITCOM K4. User wants to **retain raw PSNR as the primary practical outcome**, because orientation mistakes are a real application failure. The secondary 180° ambiguity-aware metric is **diagnostic**, not a substitute. User also wants the planner to evaluate **prior mismatch/OOD robustness** (e.g., pretrained face prior applied to unknown images from other domains), possibly more compelling than a generic GT-free selector. These are planner hypotheses, **not findings of OOD robustness**.

## 1. Relation to planner's existing oracle-first mandate (do not inadvertently overturn it)

The planner's [2026-10-16 oracle-first reframe](https://github.com/Cheng-HanHuang/pr_diffusion/blob/codex/prdiff-track-a-np-paper-planner-20261015/docs/planning/track_a/2026-10-16_ORACLE_FIRST_PUBLICATION_REFRAME.md) correctly notes that published DAPS/SITCOM phase-retrieval protocols report GT best-of-four quality and often one-run runtime; a legitimate **candidate-generation** contribution can be expressed via offline best-of-K *if K is fully charged*, even without a deployable GT-free selector. Please **retain this scientific distinction**. The new proposed GT-free and OOD investigations **augment**, rather than automatically replace, the oracle-first candidate-set reliability analysis. Where a claim is about what a practical user obtains from one observation, GT-free cross-run selection matters. The strongest publication narrative has not been decided.

Comparison to archived B22 final100 (retrospective only): DAPS Fresh2 K2 Good25 92/100; SITCOM sequential K4 native 93/100 / oracle 95/100; NP mixed8 native 95/100 / oracle 96/100. B22 was not an independent DAPS4/native independently seeded SITCOM4 fair comparison. A1.2 remedies native independence on **only eight exposed B22 images**, not the final-cohort or robustness questions. Do not combine 8-image A1.2 estimates with exposed B22 100 as if independent cohorts.

## 2. Complete experiment progression and independently verified returns

All listed PAC file paths were extracted from user-submitted PAC receipts / hard-coded frozen source paths, or from source-verified relative layout; **the executor does not have direct PAC filesystem access at the time of this report**. The planner must send read-only commands to the user to verify current presence; do not equate a historical receipt with current file availability.

| Stage | Validation / action | Result, immutable receipt |
|---|---|---|
| A0 B22 replay (planner precursor) | Archived native/policy metadata replay, no new native GPU work | 92/100 DAPS2, 93/100 SITCOM4 native, 95/100 NP8 native; A0 planner return and A0 GitHub branch `codex/prdiff-track-a-a0-native-selector-audit` |
| A1.0d | deterministic CPU restart/test simulation | PASS 14/14; PAC archive `A1_0D_RESTART_SYNTHETIC_RETURN.zip` |
| A1.1a | synthetic 256² RGB, DAPS native Fourier measurement generation on CPU | PASS; `A1_1A_PROBE_METADATA_RETURN_V1.zip` SHA256 `a2f61731638602f38563502c0c86e3d0896fa9a95f8fbf6fbb7937c507bc290f` |
| A1.1b V1 | native GPU launch intended; **time gate expired** | Controlled STOP before GPU inference; V1 archive SHA `ea631b9b581adc07b86511197574f247262ccf6dc62931c84f4574fa3b681672`; preserve as STOP |
| A1.1b V2 | one full native DAPS, SITCOM, NP LF and NP S2 trajectory each on same procedural measurement | **4/4 PASS**, 173.03 s total wall, peak reported full-device 1667 MiB; V2 archive SHA `875fae9cc3a546c82d72c15cf31964dc8c5728c52528b4e2f5005b5f7b63df3a` |
| A1.2 inventory | CPU metadata-only deterministic SHA ranking among **exposed B22 100** | PASS 2/2 tests, 8 IDs; ZIP SHA `720f30adfa04a7629b766e597afbdc4ae1d853cca7e98ac20757020ebfd2b90b` |
| A1.2 input freeze | CPU original image/observation byte checks, exact B22 GT transform check, write 8 GT tensors | PASS 3/3 tests, 8/8 historical image/measurement byte and tensor-content checks; ZIP SHA `219412e50d9dc40d2e028d5f41ad4cc8480b4166a7bc9e7bc04593daa7a55d04` |
| A1.2 K4 CPU plan | source/model/measurement integrity, 128 deterministic independent seed/candidate IDs | PASS 9/9 tests, **0 reconstructions**; ZIP SHA `7777d36f78464a0782d3cde446f88f53fd7b98d0a2b1fde70afff767e0d4dd20` |
| **A1.2 K4 actual GPU run** | 8 exposed images × four native methods × four isolated starts | **128/128 PASS** in **5432.002 s** (90.533 min). ZIP SHA `e82ad0c0314b4cff0fed82db656500307cd82f6927f42c7bc667f94878d7e9fc`. [Full executor audit](A1_2_GPU128_VERIFIED_SCIENTIFIC_ANALYSIS_20261020.md) |

A1.2 source execution A1 HEAD `f4cd28dcd3ec4860cc9f3990fb24964bb97e97bd`. Parent executor documentation HEAD at the time of this return may be newer because the analysis and handoff commits followed the run; **do not reinterpret a newer doc commit as the execution source**.

## 3. PAC PATH INDEX — exact study locations / where to find each result

**Root:** `/egr/research-pac/huang248` (no home-directory storage).

**Dedicated A1 executor source worktree** (run occurred here):
`/egr/research-pac/huang248/pr_diffusion_track_a_a1_oracle_budget`
On executor branch `codex/prdiff-track-a-a1-oracle-budget-20261016`. The planner's **different** reserved worktree is `/egr/research-pac/huang248/pr_diffusion_track_a_np_paper`; do not assume it is cloned or current, and do not mutate this A1 checkout without coordination.

**A1 output BASE:** `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016`  
Abbreviated `BASE` below denotes exactly that path.

### A1.1 procedural smoke locations
- A1.1a source synthetic inputs directory: `BASE/a1_1a_synthetic_probe_v1/`, including `A1_1A_PROBE_MANIFEST.json`, `synthetic_probe.png`, `ground_truth_tensor.pt` and `measurement.pt`.
- A1.1a verified metadata return: `BASE/A1_1A_PROBE_METADATA_RETURN_V1.zip`.
- A1.1b CPU source-plan return: `BASE/A1_1B_CPU_PLAN_RETURN_V1.zip`.
- A1.1b V1 **expired** GPU STOP return: `BASE/A1_1B_GPU_NATIVE_RETURN_V1.zip` (not a native-method failure).
- A1.1b V2 complete native procedural outputs: `BASE/a1_1b_one_each_v2/`; metadata and native log return `BASE/A1_1B_GPU_NATIVE_RETURN_V2.zip`.

### A1.2 exposed B22 development cohort and input freeze
- **Allowed historical B22 exposed manifest (read-only):** `/egr/research-pac/huang248/outputs/pr_diffusion/b22_baselines/B22_2_overnight_20260727_155135/full/manifest.json`, file SHA256 `94184e36b27aac0defb93b31ae7b47ea332909bcb0c0a80445be2949f422a219`.
- **Eight frozen IDs in original SHA ranking:** `69073,68539,67948,60140,67293,67174,61101,66631`, selected-list SHA256 `faa2e292e915690dc8dd7d477591bb47a12a5f5977fd1c9648849e25c2d09f32`.
- **A1.2 cohort inventory:** `BASE/a1_2_exposed_b22_dev8_inventory_v1/A1_2_EXPOSED_DEV8_INVENTORY.json` (per-image paths, source SHA, measurement SHA, original source row ID); metadata return `BASE/A1_2_EXPOSED_B22_INVENTORY_RETURN_V1.zip`.
- **Exact archived noisy B21 measurement for image ID:** `/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_11_fresh2_final_val100_meas5401/measurements/ffhq<ID>_phase_noise005_meas5401.pt`. Do NOT regenerate; the per-ID recorded byte/tensor SHA is in the A1.2 inventory and input-freeze receipts.
- **Original allowed FFHQ face source:** `/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024/<THOUSAND_BLOCK>/<ID>.png`; for `69073` specifically `.../images1024x1024/69000/69073.png`. All eight already exposed B22 images; no new browsing of other source IDs needed for archived A1.2 audit.
- **Input-freeze manifest:** `BASE/a1_2_exposed_b22_inputs_v1/A1_2_INPUT_FREEZE_COMPLETE.json`, full file SHA256 `fda3b5c3a26d3ed761b81cee026eea8dc2eb134b490d6369c2feda29965c2490`. Each image `BASE/a1_2_exposed_b22_inputs_v1/<ID>/ground_truth_tensor.pt` and `.../<ID>/item.json`. Input-freeze return `BASE/A1_2_INPUT_FREEZE_METADATA_RETURN_V1.zip`.
- Historical GT preprocessing audited: `ToTensor → Resize(256) → CenterCrop(256) → 2*x−1`. Observation tensor `torch.float32 [1,3,384,384]`, sigma_y `0.05`, oversampling `2.0`. Raw observation is signed and method-specific effective preprocessing differs.

### A1.2 K4 planning and full native GPU outputs — **highest-value artifacts**
- CPU plan return (no solver execution): `BASE/A1_2_K4_CPU_PLAN_RETURN_V1.zip`; its `A1_2_K4_CPU_PLAN_V1.json` and test log are at BASE.
- **Full GPU output root:** `BASE/a1_2_exposed_b22_k4_v1/`. Contains complete reconstructions and logs **on PAC**; this directory was not zipped in full.
- **Run source/seed/observation registry:** `BASE/a1_2_exposed_b22_k4_v1/RUN_IDENTITY.json`; 128 planned candidate entries `candidate_id`, seed63/seed32, source HEAD and recorded observation content SHA.
- **Overall result:** `BASE/a1_2_exposed_b22_k4_v1/A1_2_COMPLETE.json` (128/128, method counts, wall and GPU event totals, max memory, provenance).
- **Per-image 16-candidate receipts:** `BASE/a1_2_exposed_b22_k4_v1/<ID>/IMAGE_CANDIDATES_COMPLETE.json`.
- **Per-candidate independent receipt:** `BASE/a1_2_exposed_b22_k4_v1/<ID>/<METHOD>_rep<R>_COMPLETE.json` where `METHOD` ∈ `DAPS,SITCOM,NP_LF,NP_S2` and `R` ∈ `0,1,2,3`. Each includes output path/hash, seed, measurement hash, timer/log paths, subprocess wall and CUDA event elapsed.
- **Raw PSNR oracle K1/K2/K4 and NP mixed4/mixed8, full per-image 128-candidate metric list:** `BASE/a1_2_exposed_b22_k4_v1/A1_2_OFFLINE_ORACLE_SUMMARY.json`. This is the primary **post-run GT oracle** result; never pass its chosen winners into a clean-free selector.
- **Main GPU stdout record:** `BASE/A1_2_K4_GPU_MAIN_V1.log`.
- **Metadata-only ZIP for sharing:** `BASE/A1_2_K4_GPU_METADATA_RETURN_V1.zip`, verified SHA256 `e82ad0c0314b4cff0fed82db656500307cd82f6927f42c7bc667f94878d7e9fc`. ZIP contains 368 members: 128 candidate receipts, 128 CUDA timers, 96 native SITCOM/NP result JSON, 8 image summaries, 3 root identity/summary/oracle JSON, main log and last four native log tails. **Does NOT contain reconstructed tensors or PNGs.**

**Full physical reconstruction locations (PAC only):** for each `<ID>`, replicate `<R>`:
- DAPS: `BASE/a1_2_exposed_b22_k4_v1/<ID>/daps/rep<R>/daps_results/rep<R>/samples/00000_run0000.png`.
- SITCOM: `BASE/a1_2_exposed_b22_k4_v1/<ID>/sitcom/rep<R>/sitcom1/reconstruction.pt` and `reconstruction.png` in same `sitcom1` folder; `result.json` in same folder.
- NP-LF: `BASE/a1_2_exposed_b22_k4_v1/<ID>/np_lf/rep<R>/np1/reconstruction.pt`, `reconstruction.png` and `result.json` in `np1`.
- NP-S2: `BASE/a1_2_exposed_b22_k4_v1/<ID>/np_s2/rep<R>/np1/reconstruction.pt`, `reconstruction.png` and `result.json` in `np1`.
- Candidate native log/timer: `BASE/a1_2_exposed_b22_k4_v1/<ID>/<lower_method>/rep<R>/run.log` and `cuda_timing.json`; `<lower_method>` = `daps,sitcom,np_lf,np_s2`.
- SITCOM/NP frozen runtime config and A1 per-candidate input manifest under each candidate `rep<R>/frozen_runtime_config.json` and `rep<R>/input/input_manifest.json`.
- Historical DAPS ephemeral symlink/Hydra registration was exclusively A1-namespaced, removed conservatively after each attempt; use the A1-owned resulting PNG/receipt rather than looking for temporary DAPS dataset registrations.

**Important:** all PAC paths are case-sensitive; `RUN_IDENTITY.json` is the source of truth if any guessed pattern conflicts with stored `terminal`/`log_file`/`timing_json` in per-candidate receipts. Use per-candidate recorded paths for robust scripts.

**Safe next investigator workflow:** a new CPU-only analyzer should open `RUN_IDENTITY.json` and `A1_2_INPUT_FREEZE_COMPLETE.json` and check their committed SHA/source/ID list before reading only the 128 A1.2 candidate outputs. Do not scan other historical runs or protected B24 paths. Do not edit original output files or issue an implicit GPU rerun; place analyses under new A1.3 own output paths after issue #41 coordination.

### External solver/source and compute environment locations (read-only)
- Model checkpoint: `/egr/research-pac/huang248/models/ffhq_10m.pt`; SHA256 `81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa`.
- DAPS frozen native: `/egr/research-pac/huang248/pr_diffusion_b19_solver/external/daps` at HEAD `e7a77d094167084faed19b599b96673b7bb11447`, additional previously verified tracked diff SHA256 `fbb5b42369ecf0d3b9b67f8fc162053bc40ec32aed41dbd92a67e8d81dcfad69`; do NOT clean/reset/patch this native repo.
- SITCOM frozen native: `/egr/research-pac/huang248/external/SITCOM_ODE` at HEAD `275ab67efbd8146bffca20155171ba6be1169c09`.
- NP/DiffFPR frozen external: `/egr/research-pac/huang248/external/DiffFPR` at HEAD `a45ffe58f18fed8a63d3446600424e2b08733524`.
- Conda/Python: `/egr/research-pac/huang248/conda-envs/prdiff_ffhq/bin/python` (controller/NP), `.../conda-envs/daps/bin/python`, `.../conda-envs/sitcom_ode_bw/bin/python`.
- PAC hardware 4 RTX PRO 6000 Blackwell-class GPUs; A1.2 ran on physical GPU3 UUID `GPU-7d65c050-d7e8-5a6b-ee38-1d72d7a5696a`; **historical reservation complete/released**, not reserved for future A1.3/OOD work.
- A1 execution source `scripts/track_a/a1_2_k4_exposed_b22.py` and wrapper `scripts/track_a/a1_2_run_k4_gpu_v1.sh` in pinned A1 branch HEAD `f4cd28dcd3ec4860cc9f3990fb24964bb97e97bd`; common timing wrapper `scripts/b23/run_python_cuda_timed.py`; historical launch helpers `scripts/b24/run_b24_1_method_smoke.py` and `scripts/b22/run_b22_1_{sitcom,np}_smoke.py`. Do not use the already-used one-shot wrapper to restart A1.2.

## 4. Audited numerical results and compute

All quality below is **GT-oracle max over complete candidate prefixes**; raw orientation of uniformly saved 8-bit RGB images; NOT an executable selector. Candidate count K is charged in full. Due to n=8 exposed images, avoid inferential/population generalization.

| Candidate family/prefix | Mean oracle raw PSNR dB | Good25/8 | Good30/8 | Total charged subprocess wall s across 8 |
|---|---:|---:|---:|---:|
| DAPS K1 | 21.805 | 4 | 4 | 555.39 |
| DAPS K2 | 28.634 | 7 | 7 | 1068.65 |
| **DAPS K4** | **31.518** | **8** | **8** | **2080.85** |
| SITCOM K1 | 22.072 | 5 | 0 | 245.93 |
| SITCOM K2 | 27.425 | 8 | 0 | 487.83 |
| SITCOM K4 | 27.439 | 8 | 0 | 979.85 |
| NP-LF K1 | 21.145 | 4 | 2 | 295.82 |
| NP-LF K2 | 23.963 | 5 | 2 | 612.38 |
| **NP-LF K4** | **30.388** | **8** | **4** | **1179.24** |
| NP-S2 K1 | 21.273 | 4 | 3 | 279.37 |
| NP-S2 K2 | 26.409 | 6 | 3 | 583.59 |
| NP-S2 K4 | 30.222 | 8 | 4 | 1177.40 |
| NP mixed4, fixed interleaving | 26.467 | 6 | 3 | 1195.97 |
| NP mixed8, same 8 available trajectories | 30.392 | 8 | 4 | 2356.63 |

Per-image K4 comparison (raw oracle dB):

| Image | DAPS | SITCOM | NP LF | NP S2 | NP mixed8 |
|---|---:|---:|---:|---:|---:|
| 69073 | 32.18 | 27.82 | 31.17 | 31.19 | 31.19 |
| 68539 | 32.23 | 28.04 | 31.31 | 31.24 | 31.31 |
| 67948 | 30.82 | 27.08 | 29.91 | 29.82 | 29.91 |
| 60140 | 30.78 | 27.06 | 29.63 | 29.51 | 29.63 |
| 67293 | 30.54 | 26.93 | 29.48 | 29.27 | 29.48 |
| 67174 | 30.40 | 27.11 | 29.11 | 28.23 | 29.11 |
| 61101 | 33.31 | 28.07 | 31.86 | 31.86 | 31.86 |
| 66631 | 31.89 | 27.41 | 30.64 | 30.64 | 30.64 |

Run engineering: **128/128 complete**, 32 per method, 128 unique recorded terminal SHA, all 128 CUDA timers status0; for all 96 SITCOM/NP native result.json, raw measurement/GT/source/seed/model/finite flags agree with archived receipt. DAPS terminal original PNG bytes are still on PAC, not in uploaded metadata ZIP; recorded SHA cannot be independently rehashed without a PAC read. Whole wall **5432.002s** (90.533m), subprocess wall sum **5417.332s**, CUDA event elapsed **4833.754s** (not direct SM-busy time). Method subprocess walls: DAPS 2080.85s; SITCOM 979.85s; NP LF 1179.24s; NP S2 1177.40s. Peak physical device used 1667 MiB; peak PyTorch reserved 966 MiB. No GPU work needed merely to reanalyze these candidates.

Nuance: **~1.13 dB difference is a mean of PSNRs, not a multiplier on image quality**; runtimes measured in these specific native configurations on one GPU and include method-specific overhead. Same frozen **raw signed** measurement is shared across families, but NP performs frozen `clamp_min_zero_in_memory`; SITCOM uses raw; these numerical effective inputs are intentionally not identical. Existing SITCOM/NP native float PSNR deviates from uniform saved PNG8 score by less than ~0.018dB, but DAPS has no uniformly archived unquantized float output in A1.2. Do not replace original future **float** preregistration with observed PNG8 scores post hoc.

## 5. Physical ambiguity, basin structure and GT-free selection: what is established

**Raw PSNR must remain PRIMARY** (explicit user decision). Fourier phase-retrieval magnitude is invariant to certain symmetries, including conjugate reflection under standard real-signal assumptions; raw-oriented reconstruction can be practically wrong even when data fit. Report 180°-corrected/ambiguity-aware scores only as **diagnosis**, never silently substitute them as primary or call an oracle GT rotation physically executable.

Concrete observations from saved per-candidate oracle metrics:
- Image `68539`, DAPS rep0: raw PSNR **6.36 dB**, rotated180 **31.83 dB**. A dramatic *orientation error*, not automatically a poor numerical fit. DAPS K2 raw best across the image very low (~9.39dB) whereas 180°-aware GT oracle K2 is ~31.83dB.
- DAPS K2 mean across eight images changes from raw **28.634dB / Good25 7/8** to offline 180°-aware **~31.44dB / Good25 8/8**; DAPS 180°-aware K4 ~31.52dB. Therefore its apparent raw K2→K4 +2.88dB gain is **mostly correct orientation selection**, not uniform creation of new physical solution quality.
- Image `67293` DAPS rep1–3: raw ~11.4–11.6dB, after rotation only ~14–14.5dB, i.e. **other errors or wrong basins**, not simple 180° ambiguity.
- Image `67174` NP LF first three reps raw around 12.7–13.1dB; fourth reaches 29.11dB, not accounted for by 180° alone.
- Single-candidate Good25 raw counts over 32 starts/family: DAPS19, SITCOM24, NP LF21, NP S2 22. With offline rotation-aware GT metric: DAPS21, SITCOM24, NP LF21, NP S2 23. Neither all failures orientation nor all failures optimization.
- The phase retrieval ambiguities may include **more than exactly a pixel-array 180° rotation** (and FFT centering conventions matter). Planner should verify native operator's mathematical equivalence before assigning exact symmetry categories.

**GT-free selection remains UNTESTED across independent restarts.** NP's *within-trajectory* candidate scoring (`LF` or `prev_l2` regularized) is not equivalent to ranking four **completed** independent native trajectories. DAPS/SITCOM have their own posterior/regularization structures; a universal raw residual cross-method ranking may be unfair and also cannot solve exact symmetry. No native cross-run selector may be fabricated from the GT oracle. If a method lacks a defined GT-free K4 selector, record `N/A`; new candidate policies require preregistration on **development only** before out-of-domain testing.

## 6. New scientific motivation and planner decisions — hypotheses, NOT observed A1.2 results

User and executor discussed three potential paper narratives:

**(A) Candidate-quality versus charged compute Pareto frontier** (directly evidenced in exposed dev): NP-LF at K4 has ~43% lower wall time than DAPS K4 with ~1.13dB lower mean raw PSNR, while NP-LF is ~20% slower than SITCOM K4 but +2.95dB higher. Strongest near-matched-cost point is NP-LF K4 ~147s/image vs DAPS K2 ~134s/image, but raw orientation confounds it; include symmetry diagnostics and **actual cost**. Counterpoint: DAPS K4 oracle dominates raw PSNR on all eight. SITCOM K2 reaches Good25=8/8 at far lower cost, and Good25 K4 saturates. Better metrics may include Good30, q05, Bad20, mean/min and paired failures. These are *oracle* frontiers; do not present them as deployable reconstructions.

**(B) Prior-mismatch/generalization stress testing** (motivating but no data yet): Real-world inverse problems often apply public pretrained diffusion priors to images from different, unknown domains. Test whether NP low-frequency guidance/projection is more resilient to mismatch than DAPS/SITCOM, **not assume it**. Freeze same FFHQ checkpoint and same oversampled Fourier magnitude operator/noise across arms for a particular input. A small staged test may include FFHQ in-domain control, moderate shift (e.g. stylized faces), natural images/animals, and strong shift (e.g. landscapes/scientific images). These domains are **examples, not selected datasets or permissions**. Require source/license/provenance review, deterministic per-image measurement and seed namespaces, preprocessing compatibility, no overlap with protected B24 confirmation305, and separate independent-custodian/fresh test gating. Baseline matched-prior checkpoint or other ablation can distinguish model misspecification from inherent image difficulty; do not treat OOD failures as uniquely NP-related without controls. Prior mismatch of a pretrained model is established conceptual concern, not a new novelty claim by itself. The planner must perform literature/novelty assessment before new GPU budgets.

**(C) Post-hoc GT-free restart policy and basin reliability** (first stage can be CPU-only): Work on the **existing 128** A1.2 outputs. Define method-specific or physically grounded GT-free confidence metrics, independently log clean measurement residual and prior/stability/trajectory scores, then rank *within* method K (where well defined) without GT. Offline evaluate chosen raw PSNR and regret to GT best-of-K, cost, and rotation failure taxonomy. Compare raw/180° diagnostic and measurement residual. Separately diagnose whether bad candidates are wrong-orientation-equivalent, measurement-consistent alternate basins, or truly high-residual/nonconverged. **Selection cannot solve an exact non-identifiability without extra prior/side information**, and cannot create a good candidate absent from a generated set. If no meaningful GT-free selector exists, publish oracle candidate-generation evaluation honestly rather than treating clean-free selection as necessary for best-of-K benchmark publication.

**Planner-level integration A/B:** Track B owns *new basin-transition methods* and its own independent code/outputs. Track A A1.3 may conduct **read-only offline evaluation/diagnosis** of its 128 candidates and compare posterior/selection hypotheses, but must not implement or overwrite Track B's basin-transition solver in this A1 branch. Coordinate scientific overlap and GPU use via issue #41 rather than transferring or merging source worktrees.

### Suggested falsifiable OOD experiment design (requires planner revision/approval)

Separate **method choice** (choose DAPS/NP/SITCOM once using in-domain *training/development* benchmark) from **restart choice** (choose one output among K for each measurement). Fix the in-domain selection rule **before** seeing OOD images/results; compare quality and regret on OOD versus the post-hoc method oracle. For each method/target, evaluate:
1. `Q_oracle(K) = max_{r<=K} rawPSNR(x_r,GT)` — generation coverage ceiling; GT only for offline evaluation.
2. `Q_select(K) = rawPSNR(x_{s(y,{x_r})},GT)` — candidate selected by a *frozen GT-free* rule; if missing, N/A.
3. `Q_oracle-Q_select` — GT-selection regret; can shift even when coverage stays high.
4. Observation residual, symmetry diagnostic, quantiles/Good25/Good30/bad20, full charged compute and candidate diversity.
5. For canonical in-domain 180°-rotated inputs, check learned prior's orientation convention under **unchanged observation magnitude**; this is a symmetry stress test, **not genuine OOD evidence**.

Interpretation: if OOD oracle collapses, candidate generation is the problem; if oracle remains high but GT-free selection drops, the selector is the problem; if both collapse, prior/algorithm changes may be needed. No current A1.2 evidence implies NP is OOD-robust. Compare method/compute budget rigorously; do not attribute cross-domain change solely to the prior when low-level image complexity differs.

## 7. Planner action requested (self-directed autonomy)

Please return a new **committed planner scientific decision** under `docs/planning/track_a/` on planner branch `codex/prdiff-track-a-np-paper-planner-20261015` and link its commit in issue #41. The executor requests prioritization, not a preselected direction:

1. Reassess whether **oracle-first candidate-generator quality/compute** remains the primary paper framing given the new independent DAPS4 and SITCOM4 controls. Reconcile prereg primary **NP-S2-4 vs SITCOM4 Good25 reliability** (tie at 8/8 exposed) and NP-LF's stronger mean versus DAPS dominance. Decide minimum publishable differentiator and explicit **kill criteria**; do not rebrand this small exposed study as a win.
2. Decide whether next A1.3 should be **CPU-only selection/trajectory/symmetry audit** (recommended as low-cost first step), and which selector metrics and baseline-native selection comparators are scientifically justified. State missing scores/no selector as N/A. Determine permissible selection-policy training vs test exposure; raw PSNR primary.
3. Conduct independent OOD literature/novelty audit, then design a **small preregistered pilot** with distinct moderate and strong shift controls (if justified), appropriate candidate family/counts, checkpoint and operator identity, matched charged compute, genuinely allowable data provenance/splits. No OOD GPU run is authorized by this report; obtain [DATA RESERVATION] and [RUN RESERVATION] separately.
4. Distinguish Track A's offline basin diagnostic from Track B's new basin-aware transition method. Share ideas but do not overlap paths or modify Track B sources.
5. Decide whether homogeneous saved PNG8 raw PSNR can remain the **engineering** primary, while original future publication float-PSNR prereg should retain a true common float-tensor exporter for DAPS or be formally amended *before any fresh test GT*. Preserve user decision: **raw PSNR primary; ambiguity-aware auxiliary.**
6. If useful, delegate an **A1.3 CPU-only executor** with exact read-only PAC paths above and a new A1.3 output namespace. Planner owns the study protocol; executors implement source/tests, user performs PAC steps. Do not schedule a large GPU run until these scientific gates have been reviewed.

**No additional PAC computation, reservations or mutation authorized by this return.** A1.2 GPU3 lease ended and was released. Preserve all previous ZIPs, terminal logs, raw tensors, native checkouts, model, measurements and B24 protected confirmation305 unchanged.

## 8. Durable source documents / GitHub entrypoints

- [Track A planner's oracle-first scientific mandate](https://github.com/Cheng-HanHuang/pr_diffusion/blob/codex/prdiff-track-a-np-paper-planner-20261015/docs/planning/track_a/2026-10-16_ORACLE_FIRST_PUBLICATION_REFRAME.md)
- [A1 frozen original preregistration](A1_STUDY_PREREGISTRATION.md)
- [A1.1b native integration/GPU provenance](A1_1B_V2_VERIFIED_GPU_RETURN.md)
- [A1.2 exposed cohort inventory audit](A1_2_EXPOSED_B22_INVENTORY_VERIFIED_RETURN.md)
- [A1.2 eight-input tensor verification audit](A1_2_INPUT_FREEZE_VERIFIED_RETURN.md)
- [A1.2 K4 CPU plan audit](A1_2_K4_CPU_PLAN_VERIFIED_RETURN.md)
- [A1.2 full 128 native reconstructions scientific analysis](A1_2_GPU128_VERIFIED_SCIENTIFIC_ANALYSIS_20261020.md)
- [A1.2 execution implementation handoff](A1_2_GPU128_EXECUTION_HANDOFF_20261019.md)
- [A1.2 exact execution controller and source](../../../../scripts/track_a/a1_2_k4_exposed_b22.py)
- [A1.2 GPU wrapper](../../../../scripts/track_a/a1_2_run_k4_gpu_v1.sh)
- [Cross-track scientific and machine ledger issue #41](https://github.com/Cheng-HanHuang/pr_diffusion/issues/41)

### Return bottom line

**Scientific evidence today:** NP-LF occupies a middle quality/cost point on eight exposed FFHQ development images *under full GT-oracle selection*. Independent DAPS4 has a higher raw PSNR ceiling on every image; native independent SITCOM4 is faster and saturates Good25; NP mixed8 adds negligible benefit over LF4. Large K-effects include genuine basin escapes and practical orientation ambiguity. The work **does not yet establish** superior tail reliability, an operational GT-free NP result, robust prior-mismatch generalization or novelty over existing OOD diffusion inverse-problem studies.

**The next research allocation should be decided by the Track A planner** using the preserved 128 native PAC candidates, the existing oracle-first mandate, and the user's OOD motivation—rather than reflexively repeating more in-domain K4 GPU trajectories.
