# PAC direct-Python workflow (no SLURM)

This guide is for the current lab/PAC operating mode where experiments are run directly via Python (often inside `tmux`) rather than through SLURM submission.

## 1) Load environment profile

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-prdiff}"
source env/machine.lab.env
```

Expected defaults in `env/machine.lab.env`:

- `CONDA_ENV=prdiff`
- `DATA_ROOT=/egr/research-pac/<user>/data/celeba_hq_256_stage`
- `RUN_ROOT=/egr/research-pac/<user>/outputs/pr_diffusion/phase_retrieval_20260411`

## 2) Mechanism ablation (direct Python)

```bash
cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT/mechanism_lab"

python scripts/neurips_grid_experiments.py \
  --mode mechanism \
  --data_root "$DATA_ROOT" \
  --image_list_file "$SPLIT_DIR/validation_10.txt" \
  --outdir "$RUN_ROOT/mechanism_lab" \
  --radius 0.5 \
  --methods noise_picking
```

## 3) Postprocess latest mechanism run

```bash
LATEST_RUN_DIR=$(ls -dt "$RUN_ROOT"/mechanism_lab/mechanism_* 2>/dev/null | head -n 1 || true)
if [ -n "$LATEST_RUN_DIR" ]; then
  python scripts/neurips_postprocess_grid.py --run_dir "$LATEST_RUN_DIR"
fi
```

## 4) Provenance policy

- Keep local run outputs on PAC storage.
- Commit only lightweight reproducibility artifacts:
  - command logs,
  - run configuration summaries,
  - CSV summaries (`run_level.csv`, `image_level.csv`, `split_summary.csv`) when appropriate.
- Do not commit large image dumps/checkpoints/tarballs.
