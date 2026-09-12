#!/usr/bin/env python3
"""Generate one B24 locked FFHQ phase-retrieval input on an assigned GPU.

This script performs no image selection. The caller supplies a manifest-frozen
image id and B24 measurement seed. It matches the accepted DAPS/B23 image
preprocessing and PhaseRetrieval(oversample=2, sigma=0.05) construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

DAPS = Path("/egr/research-pac/huang248/pr_diffusion_b19_solver/external/daps")
FFHQ = Path("/egr/research-pac/huang248/data/ffhq/ffhq-dataset/images1024x1024")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    h = hashlib.sha256(header + b"\0")
    h.update(value.numpy().tobytes(order="C"))
    return h.hexdigest()


def find_image(image_id: str) -> Path:
    number = int(image_id)
    folder = f"{(number // 1000) * 1000:05d}"
    # For IDs below 1000, ``folder`` is already ``00000``.  Preserve the
    # historical explicit ``00000`` fallback, but deduplicate equal paths
    # before testing cardinality so one real source cannot count as two hits.
    candidates = list(dict.fromkeys([
        FFHQ / folder / f"{number:05d}.png",
        FFHQ / "00000" / f"{number:05d}.png",
        FFHQ / f"{number:05d}.png",
    ]))
    hits = [p for p in candidates if p.is_file()]
    if len(hits) != 1:
        raise RuntimeError(f"expected one FFHQ source for {image_id}; candidates={candidates}; hits={hits}")
    return hits[0].resolve()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-id", required=True)
    ap.add_argument("--measurement-seed", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    image_id = f"{int(args.image_id):05d}"
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    if not torch.cuda.is_available():
        raise RuntimeError("B24.2 locked-input generation requires assigned CUDA GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    sys.path.insert(0, str(DAPS))
    from forward_operator import PhaseRetrieval  # pylint: disable=import-outside-toplevel

    source = find_image(image_id)
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Resize(256), transforms.CenterCrop(256)
    ])
    ground_truth = (transform(Image.open(source).convert("RGB")) * 2.0 - 1.0).unsqueeze(0)
    if tuple(ground_truth.shape) != (1, 3, 256, 256) or not bool(torch.isfinite(ground_truth).all()):
        raise RuntimeError("invalid ground-truth tensor")

    operator = PhaseRetrieval(oversample=2.0, resolution=256, sigma=0.05)
    torch.manual_seed(int(args.measurement_seed))
    torch.cuda.manual_seed_all(int(args.measurement_seed))
    with torch.no_grad():
        measurement = operator.measure(ground_truth.to(device)).detach().cpu()
    if tuple(measurement.shape) != (1, 3, 384, 384) or measurement.dtype != torch.float32:
        raise RuntimeError(f"unexpected measurement tensor schema: {measurement.shape}/{measurement.dtype}")
    if not bool(torch.isfinite(measurement).all()):
        raise RuntimeError("nonfinite measurement")

    gt_path = args.output / "ground_truth.pt"
    meas_path = args.output / "measurement.pt"
    torch.save({"ground_truth": ground_truth}, gt_path)
    torch.save(measurement, meas_path)

    item = {
        "schema_version": "b24.locked-input.v1",
        "image_id": image_id,
        "measurement_id": f"B24:{image_id}",
        "measurement_seed": int(args.measurement_seed),
        "ground_truth_source_path": str(source),
        "ground_truth_source_sha256": sha256_file(source),
        "ground_truth_tensor_path": str(gt_path.resolve()),
        "ground_truth_tensor_sha256": tensor_sha256(ground_truth),
        "measurement_path": str(meas_path.resolve()),
        "measurement_file_sha256": sha256_file(meas_path),
        "measurement_tensor_sha256": tensor_sha256(measurement),
        "measurement_shape": list(measurement.shape),
        "measurement_dtype": str(measurement.dtype),
        "problem": {"resolution": 256, "oversample": 2.0, "sigma_y": 0.05},
    }
    (args.output / "input_manifest.json").write_text(
        json.dumps(item, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS", "image_id": image_id,
        "measurement_seed": int(args.measurement_seed),
        "measurement_tensor_sha256": item["measurement_tensor_sha256"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
