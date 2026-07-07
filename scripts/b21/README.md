# B21 helper scripts

These scripts support the B21 executor workflow. They are designed to be run from the PAC repo checkout:

```bash
cd /egr/research-pac/huang248/pr_diffusion_b19_solver
git checkout b19_solver_integration
conda activate daps
```

Initial helpers:

- `audit_measurement_integrity.py`: B21.0 measurement-payload and per-case PSNR integrity audit.
- `capture_lf_patch.py`: B21.1 diff capture and source-snippet extraction for the B20 LF guidance patch.
- `run_b21_0_measurement_integrity_audit.sh`: command wrapper for B21.0.
- `run_b21_1_lf_patch_capture.sh`: command wrapper for B21.1.

The wrappers do not launch phase-retrieval GPU jobs. They create analysis/report artifacts under `/egr/research-pac/huang248/outputs/pr_diffusion/b21_solver/` and repo docs under `docs/b21/`.
