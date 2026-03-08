
#!/usr/bin/env python3

from __future__ import annotations



import argparse

import os

import random

import shutil

from pathlib import Path

from typing import List, Optional





def find_by_basename(root: Path, basename: str) -> Optional[Path]:

    for p in root.rglob(basename):

        if p.name == basename:

            return p

    return None





def list_all_jpgs(root: Path) -> List[Path]:

    return sorted([p for p in root.rglob("*.jpg") if p.is_file()])





def maybe_kagglehub_download() -> Path:

    """

    Downloads CelebA-HQ resized 256x256 via kagglehub.

    Requires Kaggle credentials configured for your account.

    """

    try:

        import kagglehub  # type: ignore

    except Exception as e:

        raise RuntimeError(

            "kagglehub not installed. Install it (pip install kagglehub) or provide --dataset_root "

            "pointing to an existing celeba_hq_256 folder."

        ) from e



    path = kagglehub.dataset_download("badasstechie/celebahq-resized-256x256")

    root = Path(path) / "celeba_hq_256"

    if not root.exists():

        raise RuntimeError(f"Expected celeba_hq_256 under {path}, but not found.")

    return root





def write_slurm(

    slurm_path: Path,

    *,

    data_root: Path,

    images: List[str],

    conda_env: str,

    out_root: Path,

    model_id: str,

    base_seed: int,

):

    # Inline images list in script

    imgs_bash = "\n  ".join([f'"{x}"' for x in images])



    content = f"""#!/bin/bash

#SBATCH --job-name=prdiff_compare_subset5

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=32G

#SBATCH --time=06:00:00

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --array=0-4

#SBATCH --output=logs/prdiff_compare_%A_%a.out

#SBATCH --error=logs/prdiff_compare_%A_%a.err



set -euo pipefail



cd "${{SLURM_SUBMIT_DIR}}"

mkdir -p logs



CONDA_ENV="{conda_env}"

DATA_ROOT="{data_root}"

OUT_ROOT="{out_root}"

MODEL_ID="{model_id}"

BASE_SEED="{base_seed}"



IMAGES=(

  {imgs_bash}

)



N=${{#IMAGES[@]}}

if [ "${{SLURM_ARRAY_TASK_ID:-999999}}" -ge "$N" ]; then

  echo "ERROR: task id $SLURM_ARRAY_TASK_ID out of range (N=$N)"

  exit 1

fi

IMAGE="${{IMAGES[$SLURM_ARRAY_TASK_ID]}}"



start_ts=$(date +%s)

echo "START: $(date) | job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID image=$IMAGE"

echo "HOST:  $(hostname)"



source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate "$CONDA_ENV"



python scripts/compare_methods_no_lowfreq.py \\

  --images "$IMAGE" \\

  --data_root "$DATA_ROOT" \\

  --outdir "$OUT_ROOT/$IMAGE" \\

  --model_id "$MODEL_ID" \\

  --n_runs 10 \\

  --base_seed "$BASE_SEED" \\

  --sitcom_outer_steps 20 \\

  --sitcom_inner_steps 20 \\

  --noise_picking_steps 1000



end_ts=$(date +%s)

elapsed=$((end_ts - start_ts))

echo "END: $(date) | ELAPSED_SECONDS=$elapsed"

"""

    slurm_path.write_text(content, encoding="utf-8")

    slurm_path.chmod(0o755)





def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_root", type=str, default=None,

                    help="Path to existing celeba_hq_256 folder. If omitted, tries kagglehub download.")

    ap.add_argument("--subset_dir", type=str, required=True,

                    help="Where to store the 5-image subset, e.g. $HOME/data/prdiff_subset5")

    ap.add_argument("--seed", type=int, default=123, help="Random seed for selecting the 3 extra images")

    ap.add_argument("--conda_env", type=str, default="dip")

    ap.add_argument("--model_id", type=str, default="google/ddpm-celebahq-256")

    ap.add_argument("--out_root", type=str, default=None,

                    help="Output root for experiments (default: $HOME/out_hpc_compare_no_lowfreq)")

    ap.add_argument("--base_seed", type=int, default=100)

    args = ap.parse_args()



    subset_dir = Path(os.path.expandvars(args.subset_dir)).expanduser().resolve()

    subset_dir.mkdir(parents=True, exist_ok=True)



    out_root = Path(os.path.expandvars(args.out_root)).expanduser().resolve() if args.out_root else \

        (Path.home() / "out_hpc_compare_no_lowfreq")



    # Locate or download dataset

    if args.dataset_root:

        dataset_root = Path(os.path.expandvars(args.dataset_root)).expanduser().resolve()

    else:

        dataset_root = maybe_kagglehub_download()



    if not dataset_root.exists():

        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")



    must = ["09375.jpg", "09671.jpg"]

    must_paths = []

    for m in must:

        p = find_by_basename(dataset_root, m)

        if p is None:

            raise FileNotFoundError(f"Could not find required image {m} under {dataset_root}")

        must_paths.append(p)



    all_jpgs = list_all_jpgs(dataset_root)

    # Exclude the required ones

    must_set = set([p.name for p in must_paths])

    candidates = [p for p in all_jpgs if p.name not in must_set]

    if len(candidates) < 3:

        raise RuntimeError("Not enough candidate images to sample 3 random extras.")



    random.seed(args.seed)

    extra = random.sample(candidates, 3)



    selected = must_paths + extra

    selected_names = [p.name for p in selected]



    # Copy into subset folder

    for p in selected:

        dst = subset_dir / p.name

        shutil.copy2(p, dst)



    # Write a manifest

    manifest = subset_dir / "subset_images.txt"

    manifest.write_text("\n".join(selected_names) + "\n", encoding="utf-8")



    # Generate a slurm script that uses this subset folder

    slurm_path = Path("scripts") / "slurm_compare_subset5.sh"

    write_slurm(

        slurm_path,

        data_root=subset_dir,

        images=selected_names,

        conda_env=args.conda_env,

        out_root=out_root,

        model_id=args.model_id,

        base_seed=args.base_seed,

    )



    print("Created subset folder:", subset_dir)

    print("Selected images:", selected_names)

    print("Wrote manifest:", manifest)

    print("Wrote slurm script:", slurm_path)





if __name__ == "__main__":

    main()

