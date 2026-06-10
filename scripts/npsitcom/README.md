# NP + SITCOM-ODE experiment scaffolding

This folder contains experiments for using NP together with the external
`SITCOM_ODE` solver.  The code is intentionally guarded: scripts here assume
that the local external solver exists at

```text
/egr/research-pac/huang248/external/SITCOM_ODE
```

and should be run only on the PAC/local clone where both repositories and data
are available.

## Branch A: engineering-light hybrid selection

Goal: test whether NP and SITCOM-ODE are complementary without changing either
solver.

The candidate pool should contain rows from:

- NP selector outputs, for example `run_level.csv` from
  `scripts/phase_retrieval_20260608/run_np_selector_ffhq_one_gpu.sh`.
- SITCOM-ODE baseline outputs converted to the same row schema.

Then run:

```bash
python scripts/npsitcom/mix_select_candidates.py \
  --candidate_csv np:/path/to/np/run_level.csv \
  --candidate_csv sitcom:/path/to/sitcom/run_level_standardized.csv \
  --outdir /egr/research-pac/huang248/outputs/pr_diffusion/npsitcom_20260610/branchA_mix
```

This produces:

- `candidate_level.csv`: all input candidates with source labels.
- `source_summary.csv`: source-level NP vs SITCOM summaries.
- `selected_image_level.csv`: selected candidate per image/noise/alignment.
- `selected_summary.csv`: method-level aggregate performance.

The diagnostic method `oracle_best_psnr_diagnostic` uses ground truth and is not
an executable selector.  The other selector methods use measurement-side scores
when available.

## Branch B: timestep-compatible handoff

Goal: avoid the invalid warm-start pattern of feeding a clean reconstruction as
an initial high-noise state.  Instead, export NP-selected reconstructions as
valid diffusion states at chosen timesteps:

```text
x_t = sqrt(alpha_bar_t) * x_NP + sqrt(1 - alpha_bar_t) * eps.
```

First export states:

```bash
bash scripts/npsitcom/run_branchB_export_np_handoff_ffhq_one_gpu.sh 0 handoff_smoke
```

This writes a `handoff_manifest.csv` and `.pt` state files.  Then use
`run_sitcom_template.py` with a SITCOM command template once the correct
SITCOM-ODE entrypoint/arguments are identified.

Example dry run:

```bash
python scripts/npsitcom/run_sitcom_template.py \
  --manifest /path/to/handoff_manifest.csv \
  --cmd_template 'python /path/to/sitcom_entry.py --init_state {state_path} --outdir {outdir}' \
  --outdir /path/to/sitcom_handoff_outputs \
  --dry_run
```

## Local preparation

```bash
bash scripts/npsitcom/prepare_npsitcom_20260610.sh
python scripts/npsitcom/inspect_sitcom_ode.py
```

The inspection script does not run SITCOM; it only lists likely entrypoints so
we can wire the command template safely.
