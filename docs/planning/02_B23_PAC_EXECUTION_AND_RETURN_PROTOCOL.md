# B23 PAC execution and planner-return protocol

Date: 2026-07-31

Status: operational supplement to the accepted B23 plan. This document does not authorize any
stage or GPU execution by itself.

Repository: `Cheng-HanHuang/pr_diffusion`

Planning branch: `codex/post-b22-reliability-plan`

Accepted scientific-plan snapshot: `ed4f46e8f116648eda76d387388d762d7cb8f3d7`

Frozen B22 scientific base: `ba78c06e0c5eac0c915263e4faed0b262d5e917a`

Planning PR: `#36`

## 1. Purpose and precedence

This document tells a fresh executor how to:

1. initialize work on PAC without relying on chat memory;
2. preserve the existing repositories, branches, local patches, and large artifacts;
3. separate small durable repository evidence from large PAC-only results;
4. return a compact, auditable checkpoint to the planner after every authorized stage.

Scientific scope, hypotheses, gates, and stop rules come from:

1. `docs/planning/2026-07-30_b23_final_research_plan.md`;
2. `docs/planning/2026-07-30_b23_supersession_ledger.md`;
3. `docs/planning/01_B23_EXECUTOR_START_HERE.md`;
4. this operational protocol.

If this document conflicts with the final research plan, the final research plan wins. If a PAC
inventory contradicts a path or identity below, the executor must record the contradiction and
stop before depending on it.

## 2. Roles and authorization

### Planner

The planner:

- decides scientific gates and plan amendments;
- reviews checkpoint reports and small evidence committed to GitHub;
- returns the next authorization or a required revision.

### Executor

The executor:

- implements exactly one authorized stage;
- writes code, tests, schemas, manifests, runbooks, and checkpoint reports;
- prepares safe PAC command blocks;
- does not silently repair scientific assumptions, relax gates, or expand scope;
- stops at every authorization boundary.

### PAC operator

Unless the executor has explicitly verified direct PAC access, the user is the PAC operator:

- the executor supplies commands;
- the user runs them on `pac`;
- the user returns the requested concise output or artifact path;
- the executor analyzes the result and updates the execution branch.

The executor must never imply that a GitHub connection also provides PAC access.

### Current authorization state

```text
B23 final plan: ACCEPTED
B23.0: REQUIRES EXPLICIT USER AUTHORIZATION
B23.1 GPU replay: REQUIRES B23.0 SIGN-OFF AND A SECOND AUTHORIZATION
B23.2+: NOT AUTHORIZED
Large GPU panels: NOT AUTHORIZED
```

The words “continue,” “start,” or “take over” do not override a stage gate unless the message also
identifies the authorized stage. When authorization is ambiguous, read and report only.

## 3. Repository and branch contract

### 3.1 Immutable identities

The executor must verify, not merely quote:

| Item | Expected identity |
|---|---|
| repository | `Cheng-HanHuang/pr_diffusion` |
| accepted scientific-plan snapshot | `ed4f46e8f116648eda76d387388d762d7cb8f3d7` |
| planning branch | `codex/post-b22-reliability-plan` |
| planning PR | `#36` |
| frozen B22 base | `ba78c06e0c5eac0c915263e4faed0b262d5e917a` |
| protected historical branch | `b19_solver_integration` |

The user's initialization message must also pin the later operational-handoff head. That head must
contain this file and have the accepted scientific-plan snapshot as an ancestor. If the remote
planning head differs from the pinned operational-handoff head, the executor must show both SHAs
and stop until the planner confirms which snapshot governs execution.

### 3.2 Default execution layout

The following is the default proposal, not permission to create it:

| Item | Proposed value |
|---|---|
| execution branch | `codex/b23-execution` |
| branch point | exact operational-handoff head pinned in the user's authorization |
| clean worktree | `/egr/research-pac/huang248/pr_diffusion_b23` |
| output root | `/egr/research-pac/huang248/outputs/pr_diffusion/b23` |
| return-capsule root | `/egr/research-pac/huang248/outputs/pr_diffusion/b23/returns` |

The executor may create this branch/worktree only when the user's authorization explicitly approves
the branch point and B23.0.

### 3.3 Protected state

The executor must:

- preserve the existing dirty PAC checkout and local DAPS modifications;
- not edit, reset, rebase, merge, delete, or force-push `b19_solver_integration`;
- not use the historical dirty checkout as the clean B23 execution worktree;
- not modify an external solver checkout during inventory;
- capture `git status`, exact commits, submodule state, dirty-diff patches, and hashes before reuse;
- never merge or retarget the planning or execution PR without explicit user authorization.

Small B23 changes may be committed and pushed only to the approved B23 execution branch when the
user explicitly authorizes repository writes. Every experiment must run from a recorded pre-run
commit; uncommitted scientific changes are a stop condition.

## 4. PAC environment contract

### 4.1 Storage and host

- Primary storage root: `/egr/research-pac/huang248`
- Never place B23 repositories, environments, caches, temporary files, models, data, or outputs
  under `/home`.
- PAC is a shared four-GPU server without SLURM. GPU visibility does not authorize GPU use.
- Do not recursively scan large data, model, cache, or output trees.
- Use targeted manifests, known paths, `find` depth limits, and file-count/size summaries.

Recommended storage environment variables, when relevant:

```text
HF_HOME=/egr/research-pac/huang248/cache/huggingface
CONDA_ENVS_PATH=/egr/research-pac/huang248/conda-envs
CONDA_PKGS_DIRS=/egr/research-pac/huang248/conda-pkgs
TMPDIR=/egr/research-pac/huang248/tmp/b23
```

Do not overwrite an existing user setting merely to match these recommendations. Inventory first.

### 4.2 Known paths to verify

| Role | Path | Treatment |
|---|---|---|
| current historical checkout | `/egr/research-pac/huang248/pr_diffusion_b19_solver` | inspect and preserve; do not assume clean |
| older checkout named in `AGENTS.md` | `/egr/research-pac/huang248/pr_diffusion_repo` | inventory only; do not assume active |
| proposed B23 clean worktree | `/egr/research-pac/huang248/pr_diffusion_b23` | create only after authorization |
| FFHQ data | `/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024` | targeted read only |
| FFHQ checkpoint | `/egr/research-pac/huang248/models/ffhq_10m.pt` | hash and read only |
| official SITCOM | `/egr/research-pac/huang248/external/SITCOM_ODE` | inventory and preserve |
| NP/SITCOM fork | `/egr/research-pac/huang248/external/SITCOM_ODE_npsitcom` | inventory and preserve |
| DiffFPR | `/egr/research-pac/huang248/external/DiffFPR` | inventory if present |
| B21.11 Fresh2 benchmark | `/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/B21_11_fresh2_final_val100_meas5401` | historical evidence; read only |
| B21 source snapshot | `/egr/research-pac/huang248/outputs/pr_diffusion/checkpoints/B21_source_snapshot_20260727_040208` | historical evidence; read only |
| proposed B23 outputs | `/egr/research-pac/huang248/outputs/pr_diffusion/b23` | new B23 artifacts only |

A missing path is evidence to report, not permission to silently substitute another source.

### 4.3 Known environments to verify

| Environment | Historical role |
|---|---|
| `daps` | in-repository DAPS/Fresh work and related analysis |
| `prdiff_ffhq` | NP code |
| `sitcom_ode_bw` | SITCOM and historical NP/SITCOM fork |

The default system Python is not usable for this project. Every Python command must either:

- explicitly activate the intended conda environment; or
- call the fully qualified Python executable after it has been inventoried.

Record the interpreter path, Python, PyTorch, CUDA, cuDNN, NumPy, SciPy, and relevant solver-package
versions. Do not upgrade or repair an environment during inventory without a separate proposal.

## 5. First interaction with a fresh executor

Before editing code or generating PAC commands, the executor must read the full order in
`docs/planning/01_B23_EXECUTOR_START_HERE.md`.

Its first response must contain:

1. a concise statement of the B23 goal and current authorized stage;
2. confirmation of the accepted planning SHA and frozen B22 base;
3. the proposed execution branch, worktree, and output root;
4. a list of identities that must be verified on PAC;
5. one consolidated, safe, no-GPU PAC inventory command block;
6. the exact files/logs that command block will produce;
7. any discrepancy that would cause an immediate stop.

The inventory command block must:

- write full output to a timestamped file below the B23 output root or `/tmp`;
- print only a short completion summary and the final artifact paths;
- avoid `exit`, `logout`, shell replacement, broad `pkill`, and commands that can close the user's
  interactive terminal;
- avoid dumping large recursive listings or full logs to the terminal;
- use a subshell or a committed script when strict shell options are useful;
- be safe to rerun or create a new timestamped result rather than overwrite prior evidence.

## 6. Future GPU command contract

This section applies only after the relevant GPU stage is explicitly authorized.

Before every GPU launch, the executor must provide:

1. scientific question and gate;
2. exact pre-run commit and clean/dirty status;
3. config and manifest paths;
4. dataset/image/measurement IDs and seed derivation;
5. expected model calls, raw operation counts, candidates, work-FRE, and estimated time-FRE;
6. requested GPU IDs and expected peak memory;
7. smoke command;
8. validation command and expected pass signature;
9. full command;
10. status, log-tail, graceful-stop, and failure-collection commands;
11. exact output and return-capsule paths.

Long jobs should use `nohup` with a dedicated log and PID/status record because the user's VS Code
SSH session may disconnect. Do not use `tmux` as the assumed workflow. A launcher must not choose
GPUs dynamically or start work merely by being imported or validated.

The user launches the command manually unless direct PAC execution has been separately and
explicitly granted.

## 7. What belongs in GitHub and what stays on PAC

### 7.1 Commit to the B23 execution branch

Commit small, reviewable, durable artifacts:

- source code and tests;
- configs, schemas, frozen manifests, and seed registries;
- runbooks and exact command templates;
- environment/source identity summaries and patch hashes;
- machine-readable summary CSV/JSON/TSV files;
- selected small diagnostic plots needed for a gate;
- checkpoint, decision, deviation, and correction ledgers;
- an artifact manifest containing absolute PAC paths and SHA-256 hashes.

Do not commit credentials, tokens, private keys, machine authentication files, datasets, model
weights, or unnecessarily identifying system dumps.

### 7.2 Keep on PAC

Keep large or reproducible artifacts under the B23 output root:

- raw reconstructions and complete image panels;
- trajectories, tensors, checkpoints, and optimizer states;
- full stdout/stderr logs;
- copied datasets, measurements, models, and caches;
- large repeated-run or microbenchmark tables when a compact summary is sufficient.

Every PAC-only artifact needed for a claim must appear in the committed artifact manifest with:

```text
experiment_id
artifact_role
absolute_path
size_bytes
sha256
producer_commit
config_path
manifest_path
retention_class
```

## 8. Evidence capsules and `.tar.gz` policy

A `.tar.gz` file is useful for transporting a compact return package, but it must not be the only
scientific record. Binary archives are opaque in code review and can bloat Git history.

After each authorized checkpoint, create an extracted evidence-capsule directory on PAC:

```text
B23_<stage>_return_<timestamp>/
  README.md
  EXECUTION_IDENTITY.json
  GATE_DECISION.json
  ARTIFACT_MANIFEST.tsv
  CHECKSUMS.sha256
  COMMANDS.sh
  STDOUT_TAIL.txt
  STDERR_TAIL.txt
  summaries/
  selected_diagnostics/
```

Then create:

```text
B23_<stage>_return_<timestamp>.tar.gz
B23_<stage>_return_<timestamp>.tar.gz.sha256
```

Rules:

- the capsule contains summaries and diagnostic evidence, not model/data copies or full raw runs;
- paths in `ARTIFACT_MANIFEST.tsv` point to any large PAC-only evidence;
- the uncompressed review-critical text files are committed to GitHub;
- do not commit the archive itself by default;
- an archive may be committed only after explicit user approval and only when it is genuinely
  small, contains no sensitive material, and adds information not already available as transparent
  repo files;
- when the planner needs raw inspection beyond GitHub, the user attaches the capsule to the planner
  chat.

Thus, the normal planner handoff is an exact GitHub SHA plus PAC paths and checksums. A `.tar.gz`
attachment is the exception, not the primary database.

## 9. Repository checkpoint cadence

For every authorized experimental unit:

1. **Pre-run freeze commit**
   - code, tests, config, manifest, seeds, operation-count expectations, and runbook;
   - no GPU launch before this commit is pushed.
2. **PAC execution**
   - run exactly the frozen command;
   - preserve complete logs and artifacts under the experiment ID.
3. **Post-run evidence commit**
   - compact summaries, manifest, hashes, deviations, gate calculation, and failure notes;
   - do not alter the frozen method to make the completed run pass.
4. **Planner return**
   - send the template in Section 10;
   - stop for the named decision.

Several tiny engineering checks may share one pre-run or post-run commit, but scientific stages and
gate decisions must remain separately identifiable.

Recommended experiment identifiers:

```text
B23_0_<artifact>
B23_1A_<parent>_<repeat_or_smoke>
B23_1B_<donor>_<adapter_check>
B23_2A_<screen_or_preregistration>
B23_2B_<schedule>_<control>
```

Do not reuse an experiment ID after a failed or partial run. Add a revision suffix and retain the
original failure record.

## 10. Required planner-return block

After committing the checkpoint report, the executor must send the following block in chat. Use
`NONE` explicitly rather than omitting a field.

```text
PLANNER_RETURN
project: Cheng-HanHuang/pr_diffusion
stage:
stage_verdict:
authorized_scope_received:
gpu_work_performed: YES|NO
repository_branch:
repository_head:
planning_head_used:
pre_run_commit:
post_run_evidence_commit:
dirty_state_at_run:
experiments_completed:
experiments_failed_or_partial:
primary_gate_results:
compute_and_candidate_audit:
replay_or_rng_audit:
repo_checkpoint_report:
repo_machine_readable_artifacts:
pac_output_roots:
evidence_capsule_path:
evidence_capsule_sha256:
deviations:
correction_ledger_entries:
unresolved_blockers:
requested_planner_decision:
END_PLANNER_RETURN
```

For B23.0, `primary_gate_results` must summarize all thirteen items in Section 6 of
`01_B23_EXECUTOR_START_HERE.md`. For GPU stages, it must include the exact practical, statistical,
compute, replay, and candidate-count gates relevant to that stage.

The chat return should be concise. Full evidence belongs in the cited repository files and PAC
capsule.

## 11. Immediate-stop return

Stop without improvising and send a partial `PLANNER_RETURN` if any of the following occurs:

- remote head or ancestry differs from the authorized identity;
- the only usable source exists as an unresolved dirty diff;
- model, operator, measurement, preprocessing, or seed identity is missing;
- pre-B23 exposure cannot be resolved for a proposed split;
- native and wrapper operation counts disagree;
- RNG draw reconciliation fails;
- replay exceeds its frozen tolerance;
- a hidden candidate, retry, branch, or model call is discovered;
- projected or measured compute exceeds the authorized cap;
- a launcher reaches a GPU during B23.0;
- NaN, nontermination, corrupted output, or missing rows occur;
- progress would require a prohibited schedule or an unapproved scientific amendment.

Record failures as data. Do not delete or overwrite them.

## 12. Executor initialization template

The user should send the following message to a fresh executor only when ready to authorize B23.0.
Replace the handoff-head placeholder with the exact head supplied by the planner:

> You are the execution lead for B23 in `Cheng-HanHuang/pr_diffusion`.
>
> The final B23 plan is accepted. I authorize **B23.0 only** and approve the accepted planning
> scientific-plan snapshot `ed4f46e8f116648eda76d387388d762d7cb8f3d7`. The exact
> operational-handoff head is `<PINNED_HANDOFF_HEAD_FROM_USER_MESSAGE>`; use that as the proposed
> execution branch point, subject to your read-only remote/PAC identity checks and confirmation
> that it descends from the accepted scientific-plan snapshot. You may create the clean B23 execution
> branch/worktree only if those checks match. You may commit and push small, scoped B23.0
> code/docs/evidence to that execution branch and maintain a draft PR. Do not merge, retarget,
> rebase, squash, or force-push.
>
> No GPU work is authorized. B23.1 GPU replay, B23.2 schedules, large panels, Track B, and Track C
> are not authorized.
>
> Start at `docs/planning/01_B23_EXECUTOR_START_HERE.md`. Follow its complete reading order, then
> read `docs/planning/02_B23_PAC_EXECUTION_AND_RETURN_PROTOCOL.md`. Do not rely on transcript
> memory or superseded plans.
>
> Treat PAC paths as part of the reproducibility contract. Use `/egr/research-pac/huang248`, never
> `/home`; preserve the dirty historical checkout, `b19_solver_integration`, and all external
> repositories. Explicitly activate the correct conda environment; never use default Python.
>
> First return your understanding, proposed branch/worktree/output layout, and one consolidated
> safe no-GPU PAC inventory block. The block must save full output to files and print only a short
> summary so it cannot flood or close my terminal. After I provide the PAC inventory, complete
> only B23.0, update the repo with the required transparent artifacts, send the `PLANNER_RETURN`
> block, and stop for planner/user sign-off.
