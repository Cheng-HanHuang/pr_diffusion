# AGENTS.md

This repository is for diffusion-prior phase retrieval, with the current active focus on FFHQ-style experiments and reliability improvements.

## What to know first

- The core package lives in `prdiffusion/`.
- Active work is mainly in `scripts/npsitcom/` and `scripts/phase_retrieval_20260608/`.
- The project currently centers on NP/SITCOM hybrid work, especially Branch A candidate selection and Branch B sigma-handoff diagnostics.
- PAC paths matter and should not be changed casually.

## Safety rules for future Codex sessions

- Do not run GPU jobs automatically.
- Do not run shell scripts unless the user explicitly asks for it and the cost is clear.
- Do not scan, delete, or overwrite large `output/`, `data/`, or `model/` directories unless the user explicitly asks.
- Make small, reviewable edits.
- Explain the plan before any risky change.
- Stop before committing unless the user explicitly asks you to commit.

## Safe quick checks

- `git status`
- `rg` for targeted text search
- `sed`, `head`, `tail` for short file reads
- `python -m py_compile ...`
- Dry-run or help-style commands that do not launch experiments

## Commonly relevant PAC paths

- `/egr/research-pac/huang248/pr_diffusion_repo`
- `/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024`
- `/egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610`
- `/egr/research-pac/huang248/external/SITCOM_ODE`
- `/egr/research-pac/huang248/external/SITCOM_ODE_npsitcom`
- `/egr/research-pac/huang248/external/DiffFPR`
- `/egr/research-pac/huang248/models/ffhq_10m.pt`

## Good working style

- Prefer narrow, readable changes over broad refactors.
- Treat expensive experiment launchers as manual-only.
- When in doubt, inspect code and docs first, then ask before acting on anything that could consume significant compute.
