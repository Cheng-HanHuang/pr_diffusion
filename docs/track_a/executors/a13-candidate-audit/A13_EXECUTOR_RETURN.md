# A1.3 executor final scientific return — saved-candidate phase-retrieval forensics

**2026-10-20, America/Detroit.** Executor: Track A A1.3 candidate audit. Owner branch: `codex/prdiff-track-a-a13-candidate-audit-20261020`. This report is retrospective analysis of **8 previously exposed B22 development images** and **128 immutable A1.2 outputs**, **not** a new study, a new sampling method, or publication-level prospective evidence. The researcher ran commands on PAC; the executor had no direct PAC access. This return is based on the user-returned terminal receipts and complete uploaded metadata-only archive, which was read and independently summarized.

## Executive scientific determination

**PASS A1.3 CPU forensics; GO to planner review; NO automatic A1.4/GPU escalation.** Measurement residual can select good independent restarts **within** several fixed solver families on this panel, but its reliability depends strongly on signed-versus-clipped noisy measurements. More importantly, **a physically valid residual is not a universally calibrated quality/routing score across solver families**: retrospectively minimizing either residual across all 16 candidates/image chooses SITCOM on **all 8/8 images**, producing about **27.42/27.40 dB**, even though the best raw-PSNR candidate in every image is DAPS and the pooled oracle mean is **31.518 dB**. This cross-family calculation is a **post-hoc diagnostic**, not a preregistered policy and not a proposal for an executable router.

Do **not** interpret the severe raw-PSNR failure of DAPS/signed on image `68539` as a clean non-symmetry basin failure: it is the exact physical **180-degree Fourier-magnitude ambiguity**. Conversely, NP clipped-residual failures on `66631` and `67174` often remain bad after the 180-degree check, illustrating genuine practical measurement-consistent alternative reconstructions without claiming mathematically distinct posterior basins. The positive family-specific sign/clipping pattern is descriptive only and **cannot** be used to choose a method-specific score based on these GT outcomes and then claim GT-free generalization.

## 1. Actual PAC receipts and immutable provenance

- A1.3 analysis **execution source HEAD:** `b55c5a47f010fad019270b77c4c9366a9c8b99cc`; the final reporting commit(s) come later and do **not** invalidate already executed receipts.
- A1.2 **GPU candidate generation source:** `f4cd28dcd3ec4860cc9f3990fb24964bb97e97bd`, distinct from A1.3.
- Stage0 source/input audit: `A13_SOURCE_AND_INPUT_AUDIT_PASS`; **128/128** terminal and saved PNG8 SHA checks, 8 frozen measurements and GT input file digests, archived per-candidate receipt/seed/timing linkage. Native DAPS and SITCOM `forward_operator.PhaseRetrieval` compared with independent CPU FFT at relative error **0.0 for both**. SITCOM imported motionblur source and one-line mkdir patch were pinned and vetted, with unrelated tracked Python bytecode change separated. Seven synthetic unit tests PASS. Stage0 audit SHA256 **`ca67a9b94da73464b6eee3795b6f21370c444e096592473bf0dc0739e70b4f9f`**, matching terminal receipt and printed checksum.
- Stage1 user PAC output: **`A13_CPU_FORENSICS_COMPLETE`, exit 0**; **128** candidate scores/PSNR replays, **112** exact per-image historical A1.2 oracle policy replays, **224** within-family selector rows, **7/7** synthetic tests PASS again, **96** additional SITCOM/NP unquantized-float terminal diagnostics, **128** common PNG8 external FFT scores.
- Stage1 total Python CPU wall time **6.3815 seconds**, 144 visible CPUs (does NOT imply 144 CPUs utilized); **zero GPU inference**, zero new observations/candidates. Separate recorded candidate-generation wall times come from completed A1.2 GPU experiments and MUST NOT be interpreted as the 6.38-second CPU execution.
- Stage1 metadata-only ZIP SHA256 **`6b889c5fdca0db20b58d1ffa4c44b339021af1ae4a387da44cdd5d8ccd7bb7bb`**. The uploaded ZIP's SHA256 was independently recomputed and **matched** the PAC terminal value; it contains **16 metadata files**, no raw GT/reconstruction assets. A1.3 preexisting files remain read-only. Protected B24 confirmation305 was not accessed.

## 2. Exact comparison protocol

Eight frozen exposed IDs: `69073,68539,67948,60140,67293,67174,61101,66631`. Four families `DAPS, SITCOM, NP_LF, NP_S2` × independent replicates `0,1,2,3`: 8 × 4 × 4 = 128 terminal candidates. Main quality number: **raw, not rotation-corrected, GT RGB8 PSNR** on uniformly saved PNG8; all 128 PSNR entries and 112 per-image oracle winners replayed exactly within the predeclared tolerance. GT is used **only after** all predeclared candidate indices are selected; selection itself never reads GT. Three K policies per family: first 1,2,4 independent roots; fixed interleaved `LF0,S20,LF1,S21,...` NP mixed K4/K8. **Oracle best-of-K is an unrealizable GT upper reference**, not a deployable selector.

Predeclared GT-free common scores, computed on each **saved PNG8 candidate** with identical `A` across families:

- `r_signed = ||A(x_png8) - y_raw||_2 / (||y_raw||_2 + 1e-12)`.
- `r_clipped = ||A(x_png8) - max(y_raw,0)||_2 / (||max(y_raw,0)||_2 + 1e-12)`.

`A` converts model `[-1,1]` to physical `[0,1]`, symmetrically zero-pads 256→384 (`64` per edge), computes channelwise orthonormal centered 2D Fourier **magnitude**. The B21 noisy observation **is signed** and contains negative samples; `NP` historically clips negatives internally, whereas DAPS/SITCOM use signed data. This audit's external residual is **not** a method-native objective/prior log likelihood. DAPS has no uniformly archived native-float comparable terminal; 96 SITCOM/NP native-float residuals are separately reported. All K comparisons charge generation of *every* candidate plus external FFT score calls; CUDA-event elapsed is a counter, **not** GPU utilization.

## 3. Main results: exact K4 (mean over eight exposed images)

| Portfolio | Oracle raw PSNR | Selected by signed | Signed mean regret | Selected by clipped | Clipped mean regret | Good25 signed / clipped | Sum generation wall per image |
|---|---:|---:|---:|---:|---:|---:|---:|
| DAPS K4 | 31.518 | 28.258 | 3.260 | **31.493** | 0.025 | 7 / 8 | 260.11 s |
| NP-LF K4 | 30.388 | **30.367** | 0.021 | 25.880 | 4.508 | 8 / 6 | 147.40 s |
| NP-S2 K4 | 30.222 | **30.203** | 0.019 | 25.914 | 4.308 | 8 / 6 | 147.17 s |
| SITCOM K4 | 27.439 | 27.423 | 0.016 | 27.404 | 0.035 | 8 / 8 | 122.48 s |
| NP LF/S2 mixed K8 | 30.392 | **30.253** | 0.140 | 23.451 | 6.941 | 8 / 5 | 294.58 s |

The NP mixed8 wall is the sum of all eight distinct NP trajectories: **~294.58 s/image**; mixed8 is **not** a new method with K4-equivalent cost. Its raw-PSNR oracle gain over NP-LF K4 is **~0.004 dB**, so the extra budget has little return on this exposed set. A1.2 recorded K1 single-root mean PSNR: DAPS 21.805, NP-LF 21.145, NP-S2 21.273, SITCOM 22.072; K4 generation substantially improves oracle tail outcomes, but the improvement need not be recovered by any GT-free selector.

Selected Good30 counts (out of 8) for signed/clipped K4: DAPS `7/8`; NP-LF `4/3`; NP-S2 `4/3`; SITCOM `0/0`. Signed DAPS K4 raw Good25 7/8 is driven by a 180° inversion that is indistinguishable to the Fourier magnitude. Per-family clipped scores must **not** be labeled a method-specific policy endorsed by this observed n=8 GT pattern.

## 4. Physics-informed failure examples and identification limits

**A. DAPS signed raw collapse is chiefly an unavoidable orientation ambiguity, not necessarily a wrong measurement basin.** For `68539`, DAPS rep2 selected by signed gives **6.364 dB raw / 31.871 dB rotated**, while DAPS rep3 selected by clipped gives **32.228 dB raw**. Their signed normalized residuals are almost tied (`0.142248` vs `0.142257`): signed prefers the rotated result by only ~`9×10^-6`. The source-validated forward model satisfies the 180° magnitude symmetry. Choosing raw-canonical orientation from that measurement alone is not identifiable without extra asymmetric information/prior. If the *selected signed* DAPS outputs are scored using the **GT-aware** larger of raw/rotated PSNR solely for diagnosis, their mean becomes **31.447 dB**, versus 31.518 dB raw oracle, with **0/8** symmetry-aware bad20 and an auxiliary mean gap ~0.071 dB. This is **not** a deployable rotation correction or replacement for raw PSNR primary.

**B. NP clipped misselection often survives rotation.** `66631 NP-LF K4`: clipped selects rep0 (**10.629 raw / 14.796 rotated**), although signed selects rep3 (**30.517 raw**). Yet clipped normalized residual of bad rep0 is `0.102627`, below good rep3 `0.102846`. `67174 NP-LF K4`: clipped selects rep2 (**13.140 raw / 14.629 rotated**) over signed rep3 (**29.109 raw**); clipped residuals `0.127079` versus `0.128074`. `67174 NP-S2 K4`: clipped chooses rep1 (**19.045 raw / 11.593 rotated**) instead of signed rep2 (**28.233 raw**). These are strong observed counterexamples to naïve clipped measurement-fit reranking. No topological basin claims follow from final-state pixels.

**C. The sign/clipping operation is quantitatively consequential.** Among 442,368 measurement entries per image, **36.57%–39.39%** are *negative noisy values*; clipping deletes the signed target on over a third of the measured spectrum. As a consequence the two normalized scores need not preserve candidate order. Across the 112 image×policy portfolios, **33** yielded different candidate IDs from signed versus clipped scoring. For `NP_MIXED_K8`, signed gives **30.253 dB (8/8 Good25)**, clipped only **23.451 dB (5/8)**. This is not simply a normalization detail: it changes the target and selection geometry. However, choosing signed/clipped by family now would use the exposed GT outcomes and cannot be called a prospectively validated rule.

**D. Cross-family common residual misroutes the whole panel.** Post-hoc exploratory pooling of **all 16 candidates/image** and minimizing either external residual picks a **SITCOM** candidate **8/8**, despite DAPS providing each image's best raw-PSNR candidate. Pooled signed-selected mean **27.423**, pooled clipped-selected mean **27.404**, pooled GT oracle mean **31.518 dB** (gaps ~4.095 and ~4.114 dB). SITCOM fits the same Fourier measurements more tightly yet gives substantially lower GT PSNR. Thus an uncalibrated physics residual is **unsuitable as a cross-family route-to-best-quality score** here. Pooled selection was NOT among the two preregistered within-method selectors and must remain clearly labeled **exploratory retrospective diagnostic**.

**E. Source-locked noise calibration cannot classify alternate basins uniquely.** Synthetic independent Gaussian RMSE reference: `sigma=.05`, 32 draws with seed `720611`, maximum RMSE `0.05016058`, conservative flag `3×max=0.15048175`. The predeclared taxonomy gives **86 RAW_GOOD25**, **3 ROTATION_EQUIVALENT_CANDIDATE**, **25 OTHER_MEASUREMENT_CONSISTENT_LOW_RAW**, **14 UNRESOLVED_LOW_RAW**, **0 POOR_MEASUREMENT_FIT**, over all 128 terminal candidates. Method-specific counts: DAPS `19 good,2 rotation,10 other-consistent,1 unresolved`; NP-LF `21,0,3,8`; NP-S2 `22,1,4,5`; SITCOM `24,0,8,0`. These are *heuristic descriptive labels* against a synthetic RMSE reference, not posterior-mode proofs. The absence of any `POOR_MEASUREMENT_FIT` flag means the 3x cutoff cannot be used as a sharp convergence detector on this panel.

**F. Unquantized-versus-PNG score incompatibility.** For the 96 archived SITCOM/NP native float outputs, NP float-minus-PNG signed-relative-residual discrepancies stay near `~2e-5` in absolute magnitude, but some SITCOM float/PNG residuals differ by as much as **0.012337** (e.g. `68539:SITCOM:3`). For uniform common reranking, this report uses **saved RGB8 PNG8 for all methods**; it does not rebrand historical SITCOM native-float losses as homogeneous PNG8 scores, and DAPS float endpoints are not archived comparably.

## 5. Cost, unsupported cells, and publication boundaries

The additional common PNG8 FFT score cost for K4 across all eight images was about **0.334 s total DAPS**, **0.335 s SITCOM**, **0.343 s NP-LF**, **0.346 s NP-S2** (roughly 0.042–0.043 s/image for four candidates in the observed CPU environment). These CPU times are *observed local wall measurements*, not an equivalent GPU benchmark and do not include hypothetical data movement/serialization cost for a deployed cross-device selector. The A1.2 candidate-generation wall sums in the table are charged for every K candidate. Native SITCOM step160 correction-norm selection across independent roots, NP trailing LF proposal score across independent roots, and native DAPS float common score remain **N/A/unsupported**: their appropriate comparable historical statistics were not archived for this analysis. Original B22 `60140` SITCOM selector failure is a **different cohort**; do not conflate it with the four A1.2 `60140:SITCOM` replicas.

**Limits:** n=8 exposed prior-development samples, not held-out, not independent method tuning; all within-method score observations are descriptive with no confidence intervals or generalization claims. No prospective K4 selector validated; no new samplers, model runs, OOD cohort, unarchived posterior trajectories or geometric basin transitions. No adaptive GT-fitting, cross-family classifier, visual candidate montage, feature-level candidate distance map or training/test calibration was performed. Published claims must retain raw RGB8 PSNR primary and disclose 180°-invariant physics and the GT-oracle nondeployability. Contact sheets/extra plots were not necessary to compute the frozen numerical results and are not included in the metadata-only ZIP.

## 6. PAC source/artifact index (all exact, do not guess alternative paths)

**Executor source checkout:** `/egr/research-pac/huang248/pr_diffusion_track_a_a13_candidate_audit`, execution HEAD `b55c5a47f010fad019270b77c4c9366a9c8b99cc`.

**Original A1 inputs and outputs (READ-ONLY):**

- `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016/a1_2_exposed_b22_inputs_v1`
- `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a1_oracle_budget_20261016/a1_2_exposed_b22_k4_v1`

**A1.3 sole new output namespace:** `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/a13_candidate_audit_20261020/`

- `a13_0_inventory_v1/A13_INVENTORY_COMPLETE.json`: PASS receipt.
- `a13_0_inventory_v1/A13_SOURCE_AND_INPUT_AUDIT.json`: 128 byte/receipt/source and operator checks, SHA256 `ca67a9b94da73464b6eee3795b6f21370c444e096592473bf0dc0739e70b4f9f`.
- `a13_1_cpu_forensics_v1/A13_CANDIDATE_METRICS.csv`: 128 full scored candidates.
- `a13_1_cpu_forensics_v1/A13_K_POLICY_ROWS.csv`: 224 image/policy/scorer rows.
- `a13_1_cpu_forensics_v1/A13_SUMMARY.csv`: 28 portfolio/scorer means.
- `a13_1_cpu_forensics_v1/A13_ORACLE_REPLAY_CHECKS.json`: 112 archived prefix-oracle exact matches.
- `a13_1_cpu_forensics_v1/A13_FAILURE_TAXONOMY.csv`, `A13_TAXONOMY_COUNTS.csv`, `A13_SCORE_RANKING.csv`, `A13_EXAMPLE_CASES.md`: counterexamples/classification.
- `a13_1_cpu_forensics_v1/A13_NOISE_REFERENCE.json`, `A13_SCORE_LEAKAGE_AUDIT.md`, `A13_UNSUPPORTED_CELLS.md`, `A13_CPU_RESOURCE.json`, `A13_SOURCE_AND_INPUT_AUDIT.json` and `.md`, `A13_SYNTHETIC_TESTS.log`, `A13_ANALYSIS_STDOUT_STDERR.log`: calibration, source ledger and run/test evidence.
- `A13_CPU_METADATA_RETURN_V1.zip`: compact *metadata-only* archive with all 16 Stage1 metadata files, SHA256 `6b889c5fdca0db20b58d1ffa4c44b339021af1ae4a387da44cdd5d8ccd7bb7bb`.

All per-candidate original file paths and SHA256 hashes are present inside the **A13 source audit JSON** and the **uploaded archive**, and are not to be replaced by guessed names. The 8 original observations and 8 GT tensor paths are likewise recorded exactly in the input checks. Reporting commits may advance GitHub branch HEAD, but the execution source and receipt stay pinned as above.

## 7. Planner decisions requested

1. **Accept A1.3 as a complete, internally verified retrospective static audit.** Retain A1.2 oracle/raw PNG8 quality-versus-charged-compute as the primary Track A publication evidence, with A1.3 selector pathology and physics limits as the secondary mechanism. No A1.3 source artifacts need rerunning; preserve frozen PAC output.
2. **Decide how to address orientation ambiguity.** It is scientifically important, but raw PSNR orientation failures remain practical failures under the current desired metric. A future method would need credible asymmetric prior/landmark or a separately stated equivariant evaluation protocol; a likelihood-only reranker cannot distinguish Fourier-indistinguishable rotations in principle.
3. **Do not endorse DAPS-clipped/NP-signed as a cross-method policy based on these GT-observed eight images.** Before any prospective test, choose one principled fixed score definition/source-backed native comparable statistic, or explicitly abandon GT-free selection as a practical claim. Treat method-specific data likelihood mismatches as explanatory, not a tuned intervention.
4. **Do not run a larger GPU campaign automatically.** A1.4 requires a concrete new phase-retrieval reliability/novelty hypothesis, truly eligible independent cohort, paper metric format frozen consistently across methods, matched K/wall/GPU-time budget and negative controls, and formal [DATA RESERVATION] + [RUN RESERVATION] in issue #41 coordinated with Track B. If approved, user's preference is to use up to four **safely reserved GPUs** for multi-hour independent sweeps while keeping single-candidate `batch=1` latency controls and reporting throughput separately. This is a resource preference, not an existing lease.
5. **Track B retains ownership of new basin-transition algorithms.** This executor classified static endpoints only. The planner should decide any follow-on independent desk review of GT-free phase-ambiguity scoring or novel benchmark framing without presupposing that broader OOD mismatch itself is novel.

**Formal executor state:** `A1.3_COMPLETED_SOURCE_VERIFIED_CPU_FORENSICS_128 / TRACK_A_PLANNER_REVIEW_REQUESTED / NO_NEW_GPU_OR_DATA_AUTHORIZED`.
