# Lab migration intake (from command transcript)

This note captures values and workflow details extracted from the provided PAC command transcript, plus decisions now confirmed for migration tidiness.

## Confirmed from your transcript

### PAC / lab profile (confirmed)

- `REPO_ROOT=/egr/research-pac/huang248/pr_diffusion_repo`
- `DATA_ROOT=/egr/research-pac/huang248/data/celeba_hq_256_stage` (canonical going forward)
- `RUN_ROOT=/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411` (kept)
- `SPLIT_DIR=$RUN_ROOT/splits`
- `CONDA_ENV=prdiff` (standard PAC default)
- `HF_HOME=/egr/research-pac/huang248/.cache/huggingface`
- `PYTHONPATH=$REPO_ROOT:$PYTHONPATH`

### Mechanism ablation run details

- Neutral wrappers (`pr_*`) are temporary copies of `neurips_*` scripts.
- Mechanism settings were fixed to:
  - `np_soft=5`
  - `np_hard=1`
  - `np_proj_start=400`
- Run command used:
  - `--mode mechanism`
  - `--radius 0.5`
  - `--methods noise_picking`
  - split `validation_10`

### Data staging flow observed

- Split files copied from institution machine to PAC split directory.
- `validation_25` and `test_20` image subsets packed as tarballs and moved to PAC.
- Extracted into staged data root:
  - `/egr/research-pac/huang248/data/celeba_hq_256_stage`

## Operating constraints (confirmed)

1. Lab machine flow is **not SLURM-based** for active runs.
2. Direct Python invocation is preferred.
3. Shell helpers are acceptable for repeatability.

## Next repo tidiness actions

1. Keep using `env/machine.lab.env` profile values as the single PAC source of truth.
2. Treat `pr_*` wrappers as temporary, and keep canonical logic in `neurips_*` scripts.
3. Keep campaign root stable at `phase_retrieval_20260411` for this active cycle.
4. Standardize PAC docs/examples to `CONDA_ENV=prdiff` and stage data root.
5. Do **not** commit local runtime env or bulky results artifacts; commit only summaries/manifests and reproducible command docs.
