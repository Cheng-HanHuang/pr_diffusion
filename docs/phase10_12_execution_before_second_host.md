# Phase 10/11/12 execution before second-host experiments

This runbook implements the current continuation order:

1. Phase 10 (Noise Picking mechanism decoupling)
2. Phase 11 (hard-early vs soft-then-hard in Noise Picking)
3. Phase 12 (stronger SITCOM surrogate)
4. **Only then** move to second-host experiments.

Use:

```bash
bash scripts/pac_phase10_12_before_second_host.sh print_plan
```

## Commands

### Phase 10

```bash
bash scripts/pac_phase10_12_before_second_host.sh phase10_pilot
bash scripts/pac_phase10_12_before_second_host.sh phase10_full
```

### Phase 11

```bash
bash scripts/pac_phase10_12_before_second_host.sh phase11_pilot
bash scripts/pac_phase10_12_before_second_host.sh phase11_full
```

### Phase 12

```bash
bash scripts/pac_phase10_12_before_second_host.sh phase12_pilot
bash scripts/pac_phase10_12_before_second_host.sh phase12_full
```

## Implemented method ladders

### Phase 10 (`scripts/pr_phase10_11_np_grid.py --phase phase10`)

- `np_canonical`
- `np_fixedk_lateproj`
- `np_fixedk_alwaysproj`
- `np_fixedk_noproj`
- `np_candidate_switch_only`
- `np_projection_only_switch`

### Phase 11 (`scripts/pr_phase10_11_np_grid.py --phase phase11`)

- `hard_from_start`
- `hard_late`
- `hard_never`
- `soft_only`
- `soft_then_hard`

### Phase 12 (`scripts/pr_phase8_9_schedule.py` via runbook wrapper)

- `sitcom_unmasked`
- `sitcom_weak_then_strong` (new weighted schedule mode)
- `sitcom_hard_from_start_masked`
- `sitcom_late_mask_proxy`

## Notes

- Default split files are read from `${SPLIT_DIR}` (`validation_10.txt`, `validation_25.txt`).
- Outputs are written under `${OUT_ROOT}` with per-phase subdirectories.
- The weighted SITCOM surrogate uses `--mask_mode weighted --early_meas_weight 0.25 --late_meas_weight 1.0`.
