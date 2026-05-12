# Temporary runner record for FFHQ NP/SITCOM experiments

This file records temporary `/tmp` runners used during FFHQ-25 NP tuning.

## Main environment variables

```bash
ROOT=/egr/research-pac/huang248/outputs/pr_diffusion/phase_retrieval_20260411
DATA_ROOT=/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024
IMAGE_LIST_FILE=$ROOT/splits/ffhq_available25.txt
GUIDED_MODEL_PATH=/egr/research-pac/huang248/models/ffhq_10m.pt
GUIDED_DIFFUSION_DIR=/egr/research-pac/huang248/external/DiffFPR
REPO=/egr/research-pac/huang248/pr_diffusion_repo
```

## Practical NP setting

`score_radius=0.6`, `proj_radius=0.2`, `proj_start=300`, `soft=5`, `hard=1`, `oversample=2`.

## Runners

- `/tmp/run_np_noise_one.sh` for sigma sweep (`0.00, 0.01, 0.05, 0.10, 0.20, 0.50`).
- `/tmp/run_np_bestof2_candidates_one.sh` for doubled-candidate ablations.
- `/tmp/run_np_score_mode_one.sh` for score-mode S1-S4.
- `/tmp/run_np_s2_lambda_one.sh` for `prev_l2` lambda sweep.
- `/tmp/run_np_schedule_one.sh` for projection-radius schedules.

## Warning

Avoid stale-output reuse; prefer timestamped output directories for reruns.
