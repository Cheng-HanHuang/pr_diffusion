# AGENTS.md

This repository studies reliable diffusion-prior phase retrieval. The active planning stage is B23:
compatibility-gated, fixed-budget solver synthesis after the frozen B22 comparison.

## Start here

For B23, read in this order:

1. `docs/planning/01_B23_EXECUTOR_START_HERE.md`
2. the complete reading order inside that file
3. `docs/planning/02_B23_PAC_EXECUTION_AND_RETURN_PROTOCOL.md`

The single authoritative scientific plan is:

`docs/planning/2026-07-30_b23_final_research_plan.md`

The July 29 plan and the July 30 modular amendment are historical and superseded. Do not combine
their stage plans with the final plan.

Accepted scientific-plan snapshot:

`ed4f46e8f116648eda76d387388d762d7cb8f3d7`

## Current authorization boundary

- Do not run GPU jobs automatically.
- B23.0 is zero-GPU and requires explicit user authorization.
- B23.1 GPU replay requires B23.0 sign-off and a second explicit authorization.
- B23.2 schedules, large panels, Track B, and Track C are not authorized.
- If the user's message does not name the authorized stage, read and report only.

## Scientific invariants

- Preserve native parent semantics; do not flatten Fresh/DAPS, LF, NP, and SITCOM into one generic
  state or gradient step.
- Native replay and donor extraction are separate gates.
- Do not treat historical hybrid code as an accepted parent implementation.
- Do not revive direct NP-to-SITCOM continuation, relabel LF-v1 as a new schedule, add hidden
  candidates, or tune on pre-B23 exposure.
- Raw RGB PSNR and Good25 remain primary. Rotation-minimized PSNR is auxiliary.
- Count raw operations, candidate branches, work-FRE, and time-FRE. Do not launder compute through
  an uncounted proposal, retry, adapter, or terminal candidate.
- Stop on source, operator, measurement, replay, RNG, or semantic incompatibility instead of
  silently changing the protocol.

## Repository and branch safety

- Preserve the immutable B22 scientific base:
  `ba78c06e0c5eac0c915263e4faed0b262d5e917a`.
- Preserve the planning branch and draft PR unless the user explicitly authorizes integration.
- Do not edit, reset, rebase, merge, delete, or force-push `b19_solver_integration`.
- Preserve dirty PAC checkouts, local DAPS modifications, and external-repository diffs.
- Use a separately approved clean B23 branch/worktree.
- Make small, reviewable changes.
- Run experiments only from a recorded, pushed pre-run commit.
- Commit or push only when the user's authorization explicitly permits writes to the B23 execution
  branch. Never merge a PR without separate authorization.

## PAC storage and paths

Use `/egr/research-pac/huang248`, never `/home`.

Known paths must be inventoried rather than assumed:

- historical checkout:
  `/egr/research-pac/huang248/pr_diffusion_b19_solver`
- older checkout name retained in historical docs:
  `/egr/research-pac/huang248/pr_diffusion_repo`
- proposed clean B23 worktree:
  `/egr/research-pac/huang248/pr_diffusion_b23`
- proposed B23 output root:
  `/egr/research-pac/huang248/outputs/pr_diffusion/b23`
- FFHQ data:
  `/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024`
- FFHQ checkpoint:
  `/egr/research-pac/huang248/models/ffhq_10m.pt`
- official SITCOM:
  `/egr/research-pac/huang248/external/SITCOM_ODE`
- NP/SITCOM fork:
  `/egr/research-pac/huang248/external/SITCOM_ODE_npsitcom`
- DiffFPR:
  `/egr/research-pac/huang248/external/DiffFPR`

Do not recursively scan large output, data, model, environment, or cache trees. Use targeted paths,
manifests, depth limits, and concise summaries.

## Python environments

Historically relevant environments are:

- `daps` for DAPS/Fresh and related in-repository analysis;
- `prdiff_ffhq` for NP code;
- `sitcom_ode_bw` for SITCOM and the historical NP/SITCOM fork.

The default Python has no usable project packages. Explicitly activate the intended conda
environment or use its inventoried absolute interpreter path. Do not change dependencies during an
inventory stage.

## PAC interaction style

- The user normally runs PAC commands supplied by the executor.
- First provide one consolidated safe inventory block that saves full output to a file and prints
  only a short summary.
- Avoid terminal-flooding commands, `exit`, `logout`, shell replacement, broad `pkill`, or other
  commands that can close the user's terminal.
- For later authorized long jobs, prefer `nohup` with explicit log, PID/status, and stop commands.
- Do not assume `tmux` or SLURM.
- Before a GPU command, state the scientific purpose, expected compute/candidate budget, estimated
  runtime, GPU request, output path, smoke gate, and failure-return procedure.

## Evidence and planner returns

- Commit code, configs, schemas, manifests, compact summaries, checksums, and decision/correction
  reports.
- Keep raw reconstructions, trajectories, tensors, checkpoints, full logs, datasets, and model
  weights on PAC.
- Do not use a `.tar.gz` file as the only scientific record.
- Follow `docs/planning/02_B23_PAC_EXECUTION_AND_RETURN_PROTOCOL.md` for evidence capsules and the
  exact `PLANNER_RETURN` block.
- Stop at the end of every authorized stage for planner/user sign-off.

## Safe no-GPU checks

- `git status` and targeted `git diff`
- `rg` for targeted text search
- `sed`, `head`, and `tail` for bounded reads
- schema/config parsing
- `python -m py_compile ...` in the correct environment
- focused unit tests using synthetic tensors
- dry-run or help commands that cannot launch an experiment

