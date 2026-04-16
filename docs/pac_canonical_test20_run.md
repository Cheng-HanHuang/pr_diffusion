# PAC canonical `test_20` pilot

This note records the intended direct-Python PAC command for the next canonical pilot run.

## Purpose

Run the first PAC main-comparison pilot on `test_20` using the currently frozen PAC Noise Picking setting and the paper's headline comparison policy:

- **Noise Picking masked**
- **SITCOM unmasked**

## PAC defaults for this pilot

- radius: `0.5`
- `num_candidates_soft = 5`
- `num_candidates_hard = 1`
- `proj_start = 400`
- seeds: `100,101,102,103,104,105,106,107,108,109`
- SITCOM variant: `unmasked`
- SITCOM learning rate override: `lr_inner = 0.02`

## Recommended command

```bash
source env/machine.lab.env
bash scripts/pac_run_canonical_test20_direct.sh
```

## Equivalent direct Python command

```bash
python scripts/neurips_canonical_compare.py \
  --data_root "$DATA_ROOT" \
  --outdir "$RUN_ROOT/canonical_test20_lab" \
  --image_list_file "$SPLIT_DIR/test_20.txt" \
  --radii 0.5 \
  --seeds "100,101,102,103,104,105,106,107,108,109" \
  --sitcom_variant unmasked \
  --sitcom_lr 0.02 \
  --np_num_candidates_soft 5 \
  --np_num_candidates_hard 1 \
  --np_proj_start 400
```

## Expected outputs

The canonical comparison script already writes:

- `run_level.csv`
- `image_level.csv`
- per-image config CSVs

No extra postprocess step is required for this pilot.

## Interpretation

This run is a **PAC operational and scientific checkpoint**, not a replacement for the later full held-out benchmark. It should be used to verify that:

1. PAC direct execution is stable on the larger staged subset,
2. the frozen PAC Noise Picking setting remains strong off the validation slice,
3. the main comparison against **unmasked SITCOM** is ready to scale up.
