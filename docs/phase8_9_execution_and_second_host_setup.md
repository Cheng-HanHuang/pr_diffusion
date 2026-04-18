# Phase 8/9 execution + second-host setup commands

This runbook operationalizes the plans in:

- `docs/progress_report.md`
- `docs/phase4_7_analysis_updated.md`
- `docs/phase8_plus_experiment_plan.md`
- `docs/phase10_second_host_literature_review.md`

Use the helper script:

```bash
bash scripts/pac_phase8_9_and_host_setup.sh print_plan
```

## Main command groups

### 1) Phase 8 pilot (validation_10, 5 seeds)

```bash
bash scripts/pac_phase8_9_and_host_setup.sh phase8_pilot
```

### 2) Phase 8 full (validation_25, 10 seeds)

```bash
bash scripts/pac_phase8_9_and_host_setup.sh phase8_full
```

### 3) Phase 9 pilot sweep (`mask_start` in `200,400,600`)

```bash
bash scripts/pac_phase8_9_and_host_setup.sh phase9_pilot
```

## Second host setup (before Phase 10+)

Clone and smoke-check recommended hosts:

```bash
bash scripts/pac_phase8_9_and_host_setup.sh setup_hosts
```

The script uses:

- Primary second host: DiffFPR
- Comparison host: RED-diff
- Fallback transferable pattern: DiffPIR

## Compatibility note

- Baseline SITCOM `unmasked` and `masked` runs are launched via `scripts/pr_canonical_compare.py`.
- Phase 8/9 late-mask schedule runs are launched via `scripts/pr_phase8_9_schedule.py`, which supports `--mask_mode`, `--mask_start`, and `--mask_radius`.
