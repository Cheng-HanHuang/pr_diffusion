# B21 executor scaffold

This folder contains the B21 executor-facing plan, runbooks, registry, patches, and reports for solver-level reliability experiments in diffusion-prior phase retrieval.

Read order for any executor:

1. `../AGENTS.md` from the repo root.
2. `b21_agent_execution_protocol.md`.
3. `b21_experiment_runbooks.md`.
4. The specific report/runbook for the task being executed.

Current first tasks:

- `B21.0`: measurement integrity audit of the FFHQ100 panel.
- `B21.1`: capture and document the B20 LF guidance patch.

Repo-side helper scripts live under `scripts/b21/`. They are analysis/documentation helpers only; they do not launch GPU reconstruction jobs.
