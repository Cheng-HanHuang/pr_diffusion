# PRDiffusion shared metaplanner charter — 2026-10-15

> **UPDATED GOVERNANCE (2026-10-15):** The [autonomy amendment](2026-10-15_AUTONOMY_AMENDMENT.md) supersedes the earlier docs-only / no-delegation / metaplanner-per-run permission limits. Tracks A and B may autonomously research, implement scoped work, delegate executor chats, and propose PAC jobs under the shared [coordination issue #41](https://github.com/Cheng-HanHuang/pr_diffusion/issues/41). Frozen-data, namespace, and PAC user-execution restrictions remain.


**Authority:** user / metaplanner chat. **Roles:** B26 executor (frozen experiment completion); Track A scientific planner (Noise Picking publication evaluation); Track B scientific planner (new basin-aware method). This charter coordinates work without authorizing any fresh scientific GPU runs.

## Anchor and interpretation

Repository: https://github.com/Cheng-HanHuang/pr_diffusion
This charter is created from **frozen B26 pre-run commit** `5beb7f6485c5a7e44be19bcf4dc1cc5cd47603d2`, solely to share documented history. It does **not** modify or advance `codex/b26-np-conditional-correction` or its recorded PAC scientific commit. Neither track may modify, check out for execution, or retarget the other track's branch.
B26 exact resumption instruction is `docs/handoffs/PRDIFFUSION_B26_RESUME_EXECUTOR_2026-10-15.md` in branch `codex/planner-b26-resume-handoff-20261015`. The B26 executor owns B26 follow-up and receipts; planners may read **accepted** B26 results after release, but not drive interim tuning from a running experiment.
Repository root `AGENTS.md` and branch-specific historical safety boundaries apply, including B24 `STOP_B24_METHOD_REFINEMENT`, B24 confirmation305 lock, native source/measurement integrity, and no previously forbidden NP-to-SITCOM transplantation. Do not confuse published ground-truth best-of-4 with a deployable selector.

## Non-overlapping ownership ledger

| Item | B26 executor | Track A publication planner | Track B method planner |
|---|---|---|---|
| GitHub branch | `codex/b26-np-conditional-correction` | `codex/prdiff-track-a-np-paper-planner-20261015` | `codex/prdiff-track-b-basin-planner-20261015` |
| PAC checkout/worktree (proposed, verify nonexistence first) | `/egr/research-pac/huang248/pr_diffusion_b26` | `/egr/research-pac/huang248/pr_diffusion_track_a_np_paper` | `/egr/research-pac/huang248/pr_diffusion_track_b_basin` |
| PAC output root (reserved namespace only, no directories created by charter) | `/egr/research-pac/huang248/outputs/pr_diffusion/b26` | `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper` | `/egr/research-pac/huang248/outputs/pr_diffusion/track_b_basin` |
| New code/config/docs namespace | `scripts/b26/`, `configs/b26/`, `docs/b26/` | `scripts/track_a/`, `configs/track_a/`, `docs/track_a/` and `docs/planning/track_a/` | `scripts/track_b/`, `configs/track_b/`, `docs/track_b/` and `docs/planning/track_b/` |
| Project mission | frozen H/R and CPU estimator, exact 640+393216 work | honest NP paper viability and benchmark conventions | novel basin discovery/diversification transferable across native samplers |
| Access to other team's artifacts | read-only after authorized, immutable | read-only historical/accepted | read-only historical/accepted |
| New GPU experiments | only pre-authorized frozen B26 | none without metaplanner signoff | none without metaplanner signoff |

All paths above are **reservations**; PAC existence/availability is not verified from GitHub. The user executes all PAC commands. Both planners must check actual paths, source identities, read/write permissions, and running processes before proposing any creation or launch. Do not infer nonexistent local clones, cached measurements, or candidate reconstructions. The default Python is not trusted. Environments are under `/egr/research-pac/huang248/conda-envs` (historical: `prdiff_ffhq`, `daps`, `sitcom_ode_bw`). Never use `/home`.

## Scientific jurisdiction and anti-duplication

**Track A** owns forensic analysis of existing B22/B24 NP/DAPS/SITCOM benchmarks, historical run counts, native objectives versus external selectors, replay-free comparison tables, publication novelty, target venue/claim review, and any future NP-only or FAIR BASELINE comparisons if separately authorized. A must not propose a new phase-retrieval basin-guidance algorithm or mutate NP/B26 code to optimize an already seen evaluation panel.

**Track B** owns mathematical/mechanistic design for a **new plug-in basin-aware transition or candidate-diversification component** and its eventual cost-matched validation under at least two native parent families if feasible and principled. B must not run a separate NP best-of-K/selector grid, optimize B22 historical selectors, or produce an alternative NP-only publication. If a candidate general method requires an NP-specific correction/selection technique already investigated in A/B26, document the dependency; do not duplicate it.

Method-level innovation belongs to B; descriptive and fair benchmarking of historical NP belongs to A. Shared problems (candidate basins, protocol, native operator, how to score) are legitimate for *read-only* discussion by both but must have one execution owner after metaplanner approval. No planning branch is permitted to change `main`, B19–B26 experimental branches, frozen model/checkpoints, or historical outputs.

## Data/exposure and confirmation firewall

- B21 100-image prospective Fresh2 results and B22 100-image fixed-baseline benchmark are **already exposed** and valid for archival comparison/diagnostics, NOT for tuning and reusing as pristine confirmation.
- B24 `DEV80` is **already method-development exposed**, and B24's 7,424 screened records are exposed screening evidence (not an untouched test set).
- B24's **305 confirmation** registry/payloads remain locked. Do not read/open, hash/list payload paths, generate their new measurements, or attempt to reuse that confirmation cohort. Historical split membership may be used only where an existing gate specifically permits exclusion proof.
- Prior panels, hard-case images, and reported oracles may be used for *mechanistic diagnosis and paper retrospective* only, not for claiming new prospective test performance.
- A future shared, prospective evaluation ledger (image IDs, exposure/seed provenance, generated locked observations, division into A and B test populations) will be assigned by the **metaplanner only**, after discovery of eligible new images. No new observation or GT-derived screening by planners. If fully untouched images are scarce, distinguish new-measurement-on-exposed-image validation from image-level generalization; do not pretend independence.
- Track A and Track B future evaluation/cohorts must be disjoint from each other unless a separately preregistered *joint final benchmark* is authorized by metaplanner, with both methods frozen *before* outcomes are visible. Do not allow A to tune on B's final panel or vice versa.

## Baseline conventions and scoring invariants

B22 100 locked image/measurement panel:
- Fresh2 = **2 independent DAPS** trajectories; exact **operator-loss improvement margin 0.7** selects candidate 2 over 1. Mean PSNR `29.299`, Good25 `92/100`, 5th percentile `15.860`, GPU-s/image `312.6`.
- SITCOM-4S = **4 native SITCOM-ODE** trajectories; minimum `correction_norm` at `tau=0.8` selects executable result; mean `26.442`, Good25 `93/100`, q05 `23.415`, GPU-s/image `196.1`.
- NP-8-RS = **8 native NP** trajectories: 2 scoring configurations (LF and S2 preprojection lambda=0.01) × seeds 100,101,102,103; minimum `selector_post_winner_lf_mse_mean` selects output; mean `29.269`, Good25 `95/100`, q05 `25.146`, GPU-s/image `411.8`.
- B22 `SITCOM-oracle4` and `NP-oracle8` are **ground-truth PSNR oracles**, never deployable methods. DAPS and SITCOM papers' phase-retrieval best-of-four reporting is publication evaluation convention, *not the B22 clean-free executable 4S selector*.
- During optimization, DAPS uses a conditional posterior approximation with both learned prior and observation terms; SITCOM uses multiple consistency/regularization terms; NP uses candidate score/projection. This does NOT imply that simply summing native internal training/trajectory losses creates a meaningful across-run final selector. Any selector must be source-grounded, executable, non-GT, reproducible, and cost-accounted; do not grant NP a favorable comparison by imposing its preferred measurement-only selection onto all baselines.
- For any future matched comparison, report independently: one-run results; paper-style GT oracle best-of-K (diagnostic only); native executable selected result; carefully defined alternative clean-free selectors if prospectively frozen; time and operations/UNet calls/FLOPs (with unsupported operations itemized); mean/median/q05/min/Good25/bad20; bootstrap uncertainty at image level; raw PSNR as primary and symmetry-aware/LPIPS/SSIM as auxiliary. Candidate set and selection semantics must be explicit.

## GPU/storage/commit governance

B26 is first in GPU priority. Until metaplanner grants a scoped phase, both planners have **zero GPU, zero new FFHQ measurement, and zero pretrained model-inference authority**; read-only repository analysis and compact documentation are allowed. Do not launch CPU-intensive analysis against PAC datasets without permission either.
For any proposed run, submit a one-page authorization request with: hypothesis and falsifier; exact native source SHA+model SHA; eligible data/measurement split with no confirmation overlap; candidate counts and cost; GPU physical IDs/expected wall+GPU-hours/memory; output path (under the track-owned namespace); stop gate and resumability; frozen pre-run SHA and exact return format. Only user/metaplanner approval can turn a proposal into an experiment.
Never overwrite completed runs or reinterpret aborted attempts as zero-cost. Shared GPU collision avoidance via actual `nvidia-smi` and process identities, not fixed assumptions; no dynamic GPU use without explicit approval.
Code commits are docs-only initially, on each track's *own* planning branch. An implementation stage (separate later track-owned execution branch) requires metaplanner authorization and clean-source inventory. No auto-merges, rebases, force-pushes, retarget, or modification of historical PRs. Never `set -e`, `set -u`, `set -o pipefail`, `exit`, `logout`, broad `pkill`. Save full terminal output and provide bounded summaries.

## Communication contract and conflicts

Each planner's first deliverable is a concise, source-backed **scientific planning report**, not a GPU job:
- A: `TRACK_A_PLANNER_RETURN` — confirmed source and baseline conventions, paper viability/novelty, gaps, lowest-cost offline analyses, falsification/no-go criteria, one scoped next-request.
- B: `TRACK_B_PLANNER_RETURN` — novelty/prior-art matrix, 2–3 falsifiable basin-generation mechanisms, native-state compatibility and algorithm specification, budget/control design, disjoint exposure needs, one scoped next-request.
Each return reports branch/commit, source links, read-only artifacts, decisions, any dependencies on B26, and proposed resource/storage changes. Neither planner can approve its own GPU run. If proposed work overlaps, the metaplanner resolves ownership and data/compute before either proceeds.
