# PRDiffusion metaplanner autonomy amendment — 2026-10-15 (after three planner/executor responses)

**Effective immediately. This amendment supersedes the earlier `2026-10-15_SHARED_CHARTER.md` wherever that charter restricted Track A/B to docs-only analysis, prevented subplanner-created executor chats/branches, or required fresh metaplanner scientific signoff for every ordinary stage. All source integrity, held-out data, B24 confirmation305, no overlapping PAC worktree/output, honest accounting, and executor-PAC-access restrictions REMAIN.**

## 1. Revised roles and priorities

**Metaplanner (this chat)** sets broad shared ownership, resolves contested data/GPU/method ownership, accepts major stage returns, and occasionally prioritizes research. It should NOT micromanage local hypotheses, forbid exploratory pivots, or become a serial bottleneck for each small run.

**Track A autonomous research lead** owns the NP publication question, fair DAPS/SITCOM native/oracle/executable baseline comparison, scope/novelty analyses, local experiment portfolio, and a prospective benchmark if scientifically warranted. Its first GitHub scientific return is [Track A return](https://github.com/Cheng-HanHuang/pr_diffusion/blob/codex/prdiff-track-a-np-paper-planner-20261015/docs/planning/track_a/2026-10-15_TRACK_A_PLANNER_RETURN.md), initial assessment **conditional GO only** for an NP reliability study, not demonstrated superiority. Track A may choose to continue, narrow, or stop the paper project on evidence, and may spin up its own specialist **executor chats** and branch-specific workflows.

**Track B autonomous research lead**, given **higher scientific upside and first priority when GPU demand conflicts**, owns new basin-aware interventions, theory/synthetic proof-of-concept, native-source integration, and fair controlled development experiments. Its reported first candidate is a prior-aligned Fourier-phase bridge (candidate to test, not assumed validated), with conditional-energy saddle escape as fallback; symmetric-mode particles remain deferred. Initial recommended toy uses homometric sets A={0,1,4,10,12,17}, B={0,1,8,11,13,17}, comparing isotropic/random-phase/proposed bridge. Track B can pursue another stronger direction after novelty audit. Track B may create specialized executor chats independently.

**B26 is finished and CLOSED**, not blocking either track. Exact executor report is archived at [B26 GitHub report](https://github.com/Cheng-HanHuang/pr_diffusion/blob/codex/planner-b26-resume-handoff-20261015/docs/handoffs/B26_PLANNER_RETURN_2026-10-15.md), copied exactly from the executor's Drive report. Machine-complete PASS: 640 GPU root-arm trajectories, 393,216 synthetic CPU reverse trajectories; H Good25=55/80 vs signed-score R 53/80; R−H mean −0.2215 dB, wide bootstrap CI. CPU finite-likelihood results are suggestive in toy cases but do not establish FFHQ validity. Executor recommends `stop`; no more B26 work by default. B26 assets/results are read-only research evidence.

## 2. Freedom to work and delegate

Each track lead is **authorized now**, without another metaplanner approval, to:
- perform literature/source/code research, critique and replan within its scientific domain;
- write and push docs, code, tests, CPU toy solvers, plotting and small synthetic experiments to **its own** branches/namespaces;
- perform metadata-only retrospective audits on already-exposed existing historical artifacts and use CPU analyses where no locked/excluded GT or payload is touched; distinguish descriptive diagnosis from new held-out validation;
- choose concrete experiment designs, thresholds and stage gates for its own development work and decide STOP/CONTINUE on those development results;
- create, delegate to, and oversee **multiple executor chats**, including separate scoped GitHub subbranches, implementation plans, code reviews, PAC inventory and execution protocols, resource envelopes, and GitHub return requirements;
- evolve one track from planner to small internally coordinated **research program**, provided native source and held-out provenance are preserved.

A track planner can thus launch a subexecutor planning/code handoff without waiting for this metaplanner. It cannot literally operate PAC through the user or give an executor imaginary PAC access. **User execution of an explicitly specified PAC stage is its operational consent**; do not equate merely writing a GitHub plan with authorization to execute silently.

## 3. Git/GPU/storage concurrency — two tracks CAN coexist

The existing repository can accommodate two planners, multiple subexecutor branches, and independent output trees. **Do not sequentialize research simply because there is one repo**.

| Scope | Track A | Track B |
|---|---|---|
| Planner branch | `codex/prdiff-track-a-np-paper-planner-20261015` | `codex/prdiff-track-b-basin-planner-20261015` |
| Subexecutor branch prefix | `codex/prdiff-track-a-<study-id>` | `codex/prdiff-track-b-<study-id>` |
| PAC worktree prefix | `/egr/research-pac/huang248/pr_diffusion_track_a_*` | `/egr/research-pac/huang248/pr_diffusion_track_b_*` |
| Reserved planner PAC path | `.../pr_diffusion_track_a_np_paper` | `.../pr_diffusion_track_b_basin` |
| Exclusive PAC output root | `/egr/research-pac/huang248/outputs/pr_diffusion/track_a_np_paper/` | `/egr/research-pac/huang248/outputs/pr_diffusion/track_b_basin/` |
| Source namespace | `scripts/track_a/`, `configs/track_a/`, `docs/track_a/`, `docs/planning/track_a/` | `scripts/track_b/`, `configs/track_b/`, `docs/track_b/`, `docs/planning/track_b/` |
| Handoff source path | `docs/track_a/executors/<study-id>/` | `docs/track_b/executors/<study-id>/` |

Paths are **reserved namespaces only**, not verified PAC directories. Every executor first inspects actual PAC identities, existing processes and output paths. Each execution campaign uses a new ID under its track-owned output root; never write into `b21_solver/`, `b22_baselines/`, `b23/`, `b24/`, `b25/`, `b26/`, `main`, or the other track's tree. Maintain separate clean worktrees and do not checkout the same worktree simultaneously in two branches. Branches must not be auto-merged; proposed reusable code can be cherry-picked or reviewed at a later explicit integration gate.

**Canonical shared ledger:** [PRDiffusion coordination issue #41](https://github.com/Cheng-HanHuang/pr_diffusion/issues/41). All active subexecutors, data cohorts and GPU run reservations MUST be recorded there by comment before running, including source commit, owner, physical GPU IDs and resource envelope, path, stage and stop. Query live `nvidia-smi` and processes; reserve GPU and reject conflicts. Track B has priority on contested time, but Track A is not prohibited from concurrent use if hardware availability suffices.

**GPU consent distinction:** A/B may independently design and delegate GPU development pilots, preregister stage budgets and publish safe PAC launch commands *without preapproval from the metaplanner*. No code may start a PAC GPU run autonomously; the user explicitly runs the disclosed command. The first heavy/new-data run must include source SHA, frozen hypothesis, exact sample/measurements, GPU-time/VRAM estimate, smoke+stop gate, operational rollback and bounded outputs. Very large benchmark campaigns, access to a common prospective final-test cohort, or cross-track method transfers still need a deliberate joint decision, not merely a GitHub issue comment.

## 4. Evaluation firewall and shared science

B21/B22 final100 and B24 DEV80 are already exposed, allowed for historical diagnostics, **never fresh test set**. B24 confirmation305 is excluded and remains locked without exception. No recycled held-out cohort from the other track. Ground truth may be used after run only for offline diagnostics, not GT-aware execution/selection. For any new actual-image development or prospective benchmark, register a disjoint cohort/seed-namespace **before** generating new measurements in issue #41; consult the ledger to avoid clash. Wholly synthetic data created by Track B from first principles do not require actual-image cohort reservation.

Track A owns native selector fairness, existing NP paper and best-of-K performance; Track B owns novel basin transitions and cross-native algorithm prototypes. Both may share frozen baselines and theoretical ideas read-only. When both want the exact same new baseline run, they negotiate through issue #41 and write an agreed common frozen protocol rather than each running copies. Avoid leaking A's final-test outcomes into B's design, or vice versa.

DAPS published PR best-of-four and SITCOM published best-of-four are **GT oracle evaluation conventions**; do not pretend they are deployable selectors. B22 uses Clean-free Fresh2 two DAPS, SITCOM-4S four correction-norm-ranked SITCOM, NP-8-RS eight (2 configurations×4 seeds) LF-ranked NP. Method-native internal regularization is not automatically an appropriate cross-run final score.

## 5. Github-only durable handoffs/returns

**GitHub is the persistent decision log.** Chat output can accompany but must not be the only record. Each stage generates a `*_PLANNER_RETURN.md`, `*_EXECUTOR_RETURN.md` or `*_STOP_REPORT.md` committed in the owning branch with source SHA, operator/model/candidate budget, identity/seed/data exposure, results (including negative), resource costs, limitations, artifacts/sha256/Drive links, and next action. Full archives/tensors/datasets remain on PAC or Drive. Register each committed return in issue #41.

- Track A's first scientific return **is committed**: `docs/planning/track_a/2026-10-15_TRACK_A_PLANNER_RETURN.md`.
- B26 original complete return **is copied verbatim from Drive to a GitHub reporting branch**, keeping frozen execution code untouched: `docs/handoffs/B26_PLANNER_RETURN_2026-10-15.md` on `codex/planner-b26-resume-handoff-20261015`.
- Track B's preliminary recommendation is known from its conversation, but at the time of this amendment **no committed `TRACK_B_PLANNER_RETURN` exists yet on the Track B branch**. Track B MUST commit that first substantive return before delegating its first downstream executor, or commit an updated superseding return that includes those ideas.

This document is copied to metaplanner and both track-planner branches so each has the same immediate autonomy instructions. **The metaplanner branch's version and issue #41 are canonical for future edits; before allocating shared cohorts or GPU runs, fetch the current canonical content and issue status**.
