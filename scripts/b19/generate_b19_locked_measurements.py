#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import torch


REPO = Path("/egr/research-pac/huang248/pr_diffusion_b19_solver")
BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")
DATA_ROOT = Path("/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024")
SITCOM_ROOT = Path("/egr/research-pac/huang248/external/SITCOM_ODE")

ALL25 = [
    "00000", "00004", "00005", "00007", "00008",
    "00009", "00010", "00011", "00012", "00013",
    "00014", "00015", "00016", "00017", "00018",
    "00019", "00020", "00025", "00027", "00028",
    "00029", "00032", "00034", "00037", "00039",
]


def load_branchA_helpers():
    path = REPO / "scripts/npsitcom/run_branchA_sitcom_trajectory_hard.py"
    spec = importlib.util.spec_from_file_location("branchA_sitcom_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_hard_image_folder, mod.load_external_sitcom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_ids", default=",".join(ALL25))
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--oversample", type=float, default=2.0)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--data_root", default=str(DATA_ROOT))
    ap.add_argument("--sitcom_root", default=str(SITCOM_ROOT))
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base = Path(args.base)
    meas_dir = base / "measurements"
    meas_dir.mkdir(parents=True, exist_ok=True)

    image_ids = [x.strip().zfill(5) for x in args.image_ids.split(",") if x.strip()]

    build_hard_image_folder, load_external_sitcom = load_branchA_helpers()

    torch.cuda.set_device(f"cuda:{args.gpu}")
    device = torch.device(f"cuda:{args.gpu}")

    old_cwd, get_dataset, _get_eval_fn, _Evaluator, get_operator, _get_model, _get_sampler = load_external_sitcom(Path(args.sitcom_root))

    try:
        for image_id in image_ids:
            out_path = meas_dir / f"ffhq{image_id}_phase_noise005_meas{args.seed}.pt"
            if out_path.exists() and not args.overwrite:
                print(f"[skip] {out_path}")
                continue

            # Reset the RNG per image to match the existing one-image locked-measurement convention.
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True

            outdir = base / f"B19_16D_locked_{image_id}_meas{args.seed}_measurement_probe_sitcom1S"
            image_root = outdir / "sitcom_images_hard5"
            manifest = build_hard_image_folder(Path(args.data_root), [image_id], image_root)

            data = get_dataset(
                name="image",
                root=str(image_root),
                resolution=args.resolution,
                device="cuda",
                start_id=0,
                end_id=1,
            )
            images = data.get_data(1, 0)
            operator = get_operator(name="phase_retrieval", sigma=args.noise, oversample=args.oversample)
            y = operator.measure(images)

            payload = {
                "measurement": y.detach().cpu(),
                "images": images.detach().cpu(),
                "manifest": manifest,
                "image_ids": [image_id],
                "noise": args.noise,
                "oversample": args.oversample,
                "resolution": args.resolution,
            }
            torch.save(payload, out_path)
            print(f"[write] {out_path} shape={tuple(y.shape)} first={float(y.flatten()[0]):.6g}")
            del images, y
            torch.cuda.empty_cache()
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
