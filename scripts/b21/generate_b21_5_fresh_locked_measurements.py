#!/usr/bin/env python3
"""Generate a frozen phase-retrieval measurement panel with the local DAPS operator.

The image preprocessing and operator match upstream DAPS:
  ToTensor -> Resize(256) -> CenterCrop(256) -> [-1,1]
  PhaseRetrieval(oversample=2.0, sigma=0.05).measure(x)

A deterministic per-image noise seed is derived from the declared panel seed and
image id so regeneration order does not change the measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

from PIL import Image
import torch
import torchvision.transforms as transforms


def parse_images(raw: str) -> list[str]:
    return [f"{int(x):05d}" for x in raw.replace(",", " ").split() if x.strip()]


def image_path(root: Path, image_id: str) -> Path:
    value = int(image_id)
    folder = f"{(value // 1000) * 1000:05d}"
    candidates = [
        root / folder / f"{value:05d}.png",
        root / "00000" / f"{value:05d}.png",
        root / f"{value:05d}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    hits = list(root.rglob(f"{value:05d}.png"))
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(f"image {image_id} not found under {root}; candidates={candidates}")


def derived_seed(panel_seed: int, image_id: str) -> int:
    digest = hashlib.sha256(f"B21.5-fresh-measurement:{panel_seed}:{image_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--panel-seed", type=int, required=True)
    parser.add_argument("--measurement-tag", type=int, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--oversample", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    daps_root = repo / "external/daps"
    if not daps_root.exists():
        raise FileNotFoundError(daps_root)
    sys.path.insert(0, str(daps_root))
    from forward_operator import PhaseRetrieval  # noqa: E402

    images = parse_images(args.images)
    if len(set(images)) != len(images):
        raise ValueError("duplicate image ids")
    for image_id in images:
        value = int(image_id)
        if not 60000 <= value <= 69999:
            raise ValueError(f"fresh validation image {image_id} is outside official FFHQ validation ids 60000--69999")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
    ])
    operator = PhaseRetrieval(
        oversample=args.oversample,
        resolution=args.resolution,
        sigma=args.sigma,
    )

    rows: list[dict[str, object]] = []
    for image_id in images:
        source = image_path(args.image_root.resolve(), image_id)
        destination = args.out_dir / (
            f"ffhq{image_id}_phase_noise{int(round(args.sigma * 100)):03d}_"
            f"meas{args.measurement_tag}.pt"
        )
        seed = derived_seed(args.panel_seed, image_id)

        if destination.exists() and not args.force:
            try:
                measurement = torch.load(destination, map_location="cpu", weights_only=True)
            except TypeError:
                measurement = torch.load(destination, map_location="cpu")
            if not torch.is_tensor(measurement):
                raise TypeError(f"existing measurement is not a tensor: {destination}")
            status = "existing"
        else:
            image = (transform(Image.open(source).convert("RGB")) * 2.0 - 1.0).unsqueeze(0).to(device)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                measurement = operator.measure(image).detach().cpu()
            torch.save(measurement, destination)
            status = "generated"

        if tuple(measurement.shape) != (1, 3, 384, 384):
            raise ValueError(f"unexpected measurement shape for {image_id}: {tuple(measurement.shape)}")
        if not torch.isfinite(measurement).all():
            raise ValueError(f"nonfinite measurement for {image_id}")

        row = {
            "image_id": image_id,
            "source_path": str(source),
            "measurement_path": str(destination.resolve()),
            "status": status,
            "panel_seed": args.panel_seed,
            "derived_noise_seed": seed,
            "measurement_tag": args.measurement_tag,
            "sigma": args.sigma,
            "oversample": args.oversample,
            "resolution": args.resolution,
            "shape": list(measurement.shape),
            "dtype": str(measurement.dtype),
            "finite": bool(torch.isfinite(measurement).all()),
            "min": float(measurement.min()),
            "max": float(measurement.max()),
            "mean": float(measurement.mean()),
            "sha256": tensor_sha256(measurement),
        }
        rows.append(row)
        print(
            f"[{status}] image={image_id} seed={seed} shape={tuple(measurement.shape)} "
            f"sha256={row['sha256']} path={destination}"
        )

    manifest = {
        "protocol": "DAPS ImageDataset preprocessing plus PhaseRetrieval.measure",
        "images": images,
        "panel_seed": args.panel_seed,
        "measurement_tag": args.measurement_tag,
        "resolution": args.resolution,
        "oversample": args.oversample,
        "sigma": args.sigma,
        "device": str(device),
        "rows": rows,
    }
    manifest_path = args.out_dir / f"fresh_measurement_manifest_meas{args.measurement_tag}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[write] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
