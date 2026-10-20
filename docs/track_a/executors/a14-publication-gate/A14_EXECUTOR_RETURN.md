# A1.4 scientific executor return — publication viability of Noise Picking
**2026-10-20 · dedicated branch `codex/prdiff-track-a-a14-publication-gate-20261020` · desk science complete.**

## Executive verdict

**REFOCUS TO A PHYSICS/RELIABILITY BENCHMARK ONLY IF THERE IS A NEW MECHANISTIC QUESTION; DO NOT START A NEW NP-ONLY METHOD PAPER OR LARGE GPU GRID ON PRESENT EVIDENCE.** Confidence **moderate-high** that *the current demonstrated* trade-off is inadequate for an independent NP paper; confidence **low** about complete exclusion of GT-free failure-detection novelty because Xue et al.'s full ISCAS2025 article was unavailable.

A1.2 remains the best defensible **offline GT-best-of-K** finding with *all* K runs charged. A1.3 explains failure of measurement-residual reranking; both are genuinely useful research evidence, but the strong NP-vs-SITCOM quality advantage and weaker/faster-vs-DAPS interpolation on eight already-exposed images is not a new standalone algorithmic contribution. A1.3's apparently good DAPS-clipped/NP-signed rules cannot be retrospectively combined into an independently validated policy.

**Most important NEW desk evidence:** **FAST-DIPS (ICLR2026)** is a directly relevant, recently accepted competitor with **100 FFHQ and 100 ImageNet** phase-retrieval measurements, its own runtime–quality plot, and public operator source matching the core A1 PR task. This elevates FAST-DIPS above an optional citation; it is a **mandatory new comparator** for any future NP quality/compute-frontier claim.

## 1. What was directly checked

- Read root `AGENTS.md`, metaplanner autonomy amendment, Track A A1.3 acceptance/A1.4 handoff, original Track A publication reframe, A0/A1/A1.3 source reports and current #41 coordinator. A1.3 actual PAC source was `b55c5a47f010fad019270b77c4c9366a9c8b99cc`, retrospective report `f002643fbaff0b0541886e625258f428cac3ea90`; no new execution has occurred in A1.4.
- Inspected FAST-DIPS **35-page original full paper including Table1, Fig3, App A.5–A.7**, paper visual table and runtime text; inspected official `ququlza/FAST-DIPS` `README.md`, `configs_fast_dips_cli.txt`, `forward_operators.py`, `misc.py`, `main.py`, `data.py` through GitHub.
- Located official abstracts and available details of DAPS (CVPR2025), SITCOM (ICML2025), DiffFPR (ICML2024), DDRM-PR (Applied Optics2025), I2I-PR (Digital Signal Processing2026), Hu et al. prior mismatch (arXiv2024), and ADMM-PnP AC-DC (ICLR2026).
- Verified the **existence**, authors, venue and DOI of Xue et al. ISCAS2025 Fourier-PR failure detection, but **did not obtain its full 5-page manuscript or official code**. Its detector inputs, algorithm, pretraining, GT use, fidelity/reliability and overlap with NP **remain UNKNOWN**. This is an explicitly unresolved literature dependency, not absence of prior art.

## 2. FAST-DIPS side-by-side: exact methodological compatibility and threat

| Item | A1.2 NP/DAPS/SITCOM | FAST-DIPS (published/original release) | Inference |
|---|---|---|---|
| Forward operator | 256 RGB model image `[-1,1]`, pad64, centered normalized Fourier magnitude, oversample2 | Source `PhaseRetrieval` implements *same formula* (2026 official GitHub `forward_operators.py`, `misc.py`) | **Source-level formula MATCH**, not yet a numeric pair test between compiled environments |
| Measurement noise | `|FFT| + Normal(0,.05^2)` stored **signed**; NP internally clips | `Operator.measure(x)` `A(x)+.05*torch.randn_like(y)` signed; `oversample=2.0` preset | Physics measurement type MATCH, per-image noise/seed tensors not identical |
| Pretrained prior | A1 FFHQ checkpoint SHA256 `81d53574...` | FFHQ DDPM download described in official README | **Actual checkpoint bytes NOT CHECKED**, potential confound |
| Quality/selection | A1 independent 4 roots/method, GT best-K RGB8 common saved PNG8 | Paper says **best of 4** for PR; `main.py` calculates GT metrics across `num_runs=4`; skimage metrics/per-run float pipeline | Both offline oracle, **not a deployed selector**. PSNR pipelines not proved identical |
| Runtime | A1 full K4 candidate subprocess wall / image | Paper App A.6 reports **mean per-run sampling-only**, batch1, excludes I/O and metrics | Must charge FAST full K4 `4×time` and separately disclose differing timing scopes |
| Test cohort | Eight *previously exposed B22* IDs `69073,68539,67948,60140,67293,67174,61101,66631` | Paper App A.6 lists FFHQ indices `00000..00099`, ImageNet `49000..49099` | **Not paired**; cannot equate 30.388 A1 and 29.253 published FFHQ |

**Original FAST paper Table1 (FFHQ, PR best4)**:
- DAPS **30.253 dB**, **122.100 seconds / run**, so **488.400 seconds / four runs**, *sampling-only*.
- SITCOM **28.512**, **37.425/run**, so **149.700 seconds / four runs**.
- FAST-DIPS **29.253**, **10.354/run**, so **41.416 seconds / four runs**.
The **four-run multiplication is a normalization to our candidate-set evaluation endpoint**, not a claim that FAST's authors hid its convention: they explicitly describe it in App A.6. In the paper's own matched-cohort performance, FAST trades ~1 dB vs DAPS for ~11.8× less sampled compute at best4 and outperforms SITCOM in PSNR with less time. This is a **strong quality–compute alternative to NP** but NOT a legitimate quantitative NP head-to-head, because NP was not run on those 100 cases. The paper's Fig3 also already evaluates compute–PSNR trends, making an unadorned Pareto scatter insufficient novelty.

**Official FAST runnable source:** PR `T=150`, 2 ADMM iterations per level, 5 x-update steps, `rho=200`, `epsilon=.05`, `num_runs=4`, `batch_size=1`, with one model sampling loop per root. Model weights, data normalization and exact GT PSNR need a dedicated uniform adapter before PAC execution. Release makes the main benchmarking compatibility plausible, not yet certified.

## 3. The three publishability hypotheses

| Claim | A1.4 assessment | Falsifier / missing deliverable |
|---|---|---|
| **H1: NP candidate generator dominates or improves frontier** | **CURRENT CLAIM STOP; conditional benchmark PIVOT** | FAST direct source compatibility, stronger published fast phase retrieval and A1.2 absence of a K4 Good25 edge make NP merely an intermediate GT-oracle cost point on exposed cases. A truly independent same-checkpoint/observation/cohort/hardware K4+budget-matched FAST/DAPS/SITCOM/NP comparison is needed; if no meaningful tail/mean advantage and causal NP proposal ablation, STOP NP-method-paper. |
| **H2: NP solves GT-free failure detection/selection** | **STOP** | A1.3 score sign/clipping reversals, non-rotation errors, cross-method miscalibration and 180° Fourier ambiguity; no independently validated native selectable confidence. ISCAS 2025 Xue is a high overlap threat but **method overlap unknown** until full paper read. Need additional asymmetry/prior information and predeclared independent calibration, NOT reuse 8 exposed GT examples. |
| **H3: NP is robust to prior mismatch** | **PIVOT to specific causal hypothesis only**, no experiment yet | Generic prior-distribution mismatch and patch-based remedy are covered by Hu et al. An NP-specific causal improvement in candidate-set success at matched K/GPU under quantified shift, distinct from ordinary hardness and learned-prior changes, must be derived and accompanied by eligible external data. No NP-OOD result exists. |

**Scientific decision:** At present no claim clears a **prospective run-authorization threshold**. The only possible limited paper path is **a well-designed failure/reliability benchmark** illuminating GT oracle nondeployability, signed noisy magnitude vs clipped target, phase-conjugate symmetry, and true K-charged efficiency, with strong 2026 baselines and an independently sourced test cohort. Even that requires independent contribution beyond FAST's published quality-vs-time plots. It may reasonably fail novelty and be stopped. Do not falsely present it as a newly validated NP method.

## 4. Proposed next action and stopping principle

**Priority P0 (zero GPU, no PAC):** retrieve Xue et al.'s full article legitimately (author manuscript, institution subscription, or source) and freeze method-overlap notes; if unavailable after reasonable efforts, mark the gap openly and do not center H2 publication claim. Verify FAST released checkpoint *provenance metadata* versus our pinned FFHQ checkpoint, common dataset preprocessing and exact timing/GT best4 semantics through GitHub source. The operator formula and signed measurement noise are already checked. Build **source-only compatibility table**; no need for a GPU smoke on a source mismatch.

**Priority P1:** Choose exactly ONE falsifiable research question for a true independent benchmark: can noise picking increase raw-GT candidate coverage in rare failures **relative to FAST at equal charged K/device-seconds**, with an explicit counterfactual NP proposal ablation? If no principled expected advantage remains after desk/source check, **STOP** the NP paper and do not spend PAC cycles. If yes, draft and freeze the limited pilot protocol in [A14_PROSPECTIVE_PROTOCOL_PROPOSAL.md](A14_PROSPECTIVE_PROTOCOL_PROPOSAL.md), then seek new independent data provenance and Track A planner/user authorization.

**Conditional later computational stage** (**not** today): use an 8–12-image *engineering-only* operator/model/timing smoke to establish adapter correctness, and only afterwards decide whether a ≥100-image independent inference panel is justified. The user prefers all 4 GPUs for long independent sweeps where available, but a GPU reservation must be explicit in issue #41 and Track B has priority. Batch1 per-root latency and four-way cluster throughput are distinct. No data/measurement seeds or PAC output path has been fabricated or reserved.

## 5. Branch artifacts and original source links

Artifacts committed on this **dedicated A1.4 branch only**:
1. [A14_PRIOR_ART_MATRIX.md](A14_PRIOR_ART_MATRIX.md) — papers, exact operator/noise/prior and K/reporting conventions, verified vs unknown.
2. [A14_CLAIM_KILL_MATRIX.md](A14_CLAIM_KILL_MATRIX.md) — H1/H2/H3 falsifiers and threshold proposals.
3. [A14_PROSPECTIVE_PROTOCOL_PROPOSAL.md](A14_PROSPECTIVE_PROTOCOL_PROPOSAL.md) — precisely specified conditional frozen-input, same-prior FAST-first candidate portfolio and cost/selection protocol, **NOT RUN AUTHORIZATION**.
4. **This** `A14_EXECUTOR_RETURN.md` — scientific choice for planner.

First-party anchor links:
- [FAST-DIPS original full paper (ICLR2026, Table1, Fig3, App A.5-A.7)](https://arxiv.org/pdf/2603.01591), [official repo](https://github.com/ququlza/FAST-DIPS), [operator source](https://github.com/ququlza/FAST-DIPS/blob/main/forward_operators.py), [task defaults](https://github.com/ququlza/FAST-DIPS/blob/main/misc.py), [training-free method](https://fast-dips.github.io/).
- [Xue et al. ISCAS2025 DOI](https://doi.org/10.1109/ISCAS56072.2025.11042933), [DBLP record](https://dblp.org/rec/conf/iscas/XueDZPZX25). Full paper **NOT FOUND**, no fabricated detector.
- [DiffFPR (ICML2024)](https://proceedings.mlr.press/v235/li24bj.html), [SITCOM](https://arxiv.org/abs/2410.04479), [DAPS](https://arxiv.org/abs/2407.01521), [Hu prior mismatch](https://arxiv.org/abs/2410.11730), [DDRM-PR](https://arxiv.org/abs/2501.03030), [I2I-PR (2026)](https://arxiv.org/abs/2507.09609).
- Provenance: [A1.3 accepted report](https://github.com/Cheng-HanHuang/pr_diffusion/blob/f002643fbaff0b0541886e625258f428cac3ea90/docs/track_a/executors/a13-candidate-audit/A13_EXECUTOR_RETURN.md), [Track A 2026-10-20 planner gate](https://github.com/Cheng-HanHuang/pr_diffusion/blob/37b872fef94090171b1afcb27359355eb76b513b/docs/planning/track_a/2026-10-20_A13_ACCEPTANCE_AND_A14_RESEARCH_GATE.md), [issue #41](https://github.com/Cheng-HanHuang/pr_diffusion/issues/41).

**Execution ledger:** `PAC_OPERATIONS=0`; `GPU=0`; `NEW_DIFFUSION=0`; `NEW_MEASUREMENT=0`; `NEW_IMAGES=0`; `B24_CONFIRMATION305_ACCESS=0`; `B21_B26_OR_A1_MUTATIONS=0`. No new source trees, checkpoints, tests or outputs on PAC.

**Formal return:** `A14_DESK_SCIENCE_COMPLETE / REFOCUS_REPRODUCIBILITY_RELIABILITY_OR_STOP / FAST_DIPS_MANDATORY / NO_PAC_GPU_AUTHORIZATION / PLANNER_REVIEW_REQUESTED`.
