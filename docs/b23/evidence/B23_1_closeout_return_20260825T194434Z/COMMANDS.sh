#!/usr/bin/env bash
# Evidence-only reproduction; CUDA must remain hidden.
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests/b23 -v
CUDA_VISIBLE_DEVICES='' python scripts/b23/validate_b23_0.py --repo . --output-json <validation.json>
# collect_b23_1_closeout.py validates an existing accepted capsule; it launches no scientific process.
