#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import List


def list_candidate_images(data_root: str, max_id: int) -> List[str]:
    candidates: List[str] = []
    for path in Path(data_root).rglob("*.jpg"):
        name = path.name
        stem = path.stem
        if not stem.isdigit():
            continue
        idx = int(stem)
        if 0 <= idx <= max_id:
            candidates.append(name)
    return sorted(set(candidates))


def write_list(path: str, values: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for v in values:
            f.write(f"{v}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Create fixed NeurIPS experiment image splits.")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--outdir", type=str, default="docs/neurips_splits")
    p.add_argument("--seed", type=int, default=20260411)
    p.add_argument("--max_image_id", type=int, default=5401)
    p.add_argument("--dev_count", type=int, default=10)
    p.add_argument("--val_count", type=int, default=25)
    p.add_argument("--test_count", type=int, default=50)
    args = p.parse_args()

    imgs = list_candidate_images(args.data_root, args.max_image_id)
    need = args.dev_count + args.val_count + args.test_count
    if len(imgs) < need:
        raise ValueError(f"Need at least {need} images <= {args.max_image_id:05d}, found {len(imgs)}.")

    rng = random.Random(args.seed)
    rng.shuffle(imgs)

    dev = sorted(imgs[: args.dev_count])
    val = sorted(imgs[args.dev_count : args.dev_count + args.val_count])
    test = sorted(imgs[args.dev_count + args.val_count : need])

    write_list(os.path.join(args.outdir, "dev_10.txt"), dev)
    write_list(os.path.join(args.outdir, "validation_25.txt"), val)
    write_list(os.path.join(args.outdir, "test_50.txt"), test)
    write_list(
        os.path.join(args.outdir, "seed_list_10.txt"),
        ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109"],
    )

    print(f"Wrote splits under: {args.outdir}")


if __name__ == "__main__":
    main()
