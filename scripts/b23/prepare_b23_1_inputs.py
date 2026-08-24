#!/usr/bin/env python3
"""Generate exactly the five preregistered B23.1 locked inputs.

This is an authorized GPU input-generation step, not an image-selection step.
The signed rows must already exist and pass the CPU preregistration validator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(header + b"\0")
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def find_image(root: Path, image_id: str) -> Path:
    folder = f"{(int(image_id) // 1000) * 1000:05d}"
    candidates = (root / folder / f"{image_id}.png", root / f"{image_id}.png")
    hits = [path for path in candidates if path.is_file()]
    if not hits:
        hits = list(root.rglob(f"{image_id}.png"))
    if len(hits) != 1:
        raise RuntimeError(f"expected one FFHQ image {image_id}, found {len(hits)}")
    return hits[0].resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite input root: {output}")
    sys.path.insert(0, str(repo))

    import torch
    from PIL import Image
    from torchvision import transforms

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("B23.1 locked input generation requires the authorized CUDA device")
    config = json.loads((repo / "configs/b23/b23_1a_b_execution.yaml").read_text())
    paths = json.loads((repo / "configs/b23/pac_paths.yaml").read_text())
    registry = repo / config["registry"]["combined"]
    with registry.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5:
        raise RuntimeError("B23.1 input generation is bounded to exactly five signed rows")

    daps_root = Path(paths["historical_checkout"]) / "external/daps"
    sys.path.insert(0, str(daps_root))
    from forward_operator import PhaseRetrieval  # pylint: disable=import-outside-toplevel

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    operator = PhaseRetrieval(oversample=2.0, resolution=256, sigma=0.05)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(256), transforms.CenterCrop(256)]
    )
    output.mkdir(parents=True)
    manifest_rows = []
    for row in rows:
        image_id = row["image_id"]
        row_dir = output / f"{row['split'].replace('.', '_')}_row{int(row['row_id']):02d}_{image_id}"
        row_dir.mkdir()
        source = find_image(Path(paths["ffhq_data"]), image_id)
        ground_truth = (transform(Image.open(source).convert("RGB")) * 2.0 - 1.0).unsqueeze(0)
        if tuple(ground_truth.shape) != (1, 3, 256, 256) or not torch.isfinite(ground_truth).all():
            raise RuntimeError(f"invalid ground truth for {image_id}")
        measurement_seed = int(row["measurement_seed"])
        torch.manual_seed(measurement_seed)
        torch.cuda.manual_seed_all(measurement_seed)
        with torch.no_grad():
            measurement = operator.measure(ground_truth.to(device)).detach().cpu()
        if tuple(measurement.shape) != (1, 3, 384, 384) or not torch.isfinite(measurement).all():
            raise RuntimeError(f"invalid measurement for {image_id}")
        gt_path = row_dir / "ground_truth.pt"
        measurement_path = row_dir / "measurement.pt"
        torch.save({"ground_truth": ground_truth}, gt_path)
        torch.save(measurement, measurement_path)
        item = {
            **row,
            "ground_truth_source_path": str(source),
            "ground_truth_source_sha256": sha256_file(source),
            "ground_truth_tensor_path": str(gt_path),
            "ground_truth_tensor_sha256": tensor_sha256(ground_truth),
            "measurement_path": str(measurement_path),
            "measurement_file_sha256": sha256_file(measurement_path),
            "measurement_tensor_sha256": tensor_sha256(measurement),
            "measurement_shape": list(measurement.shape),
            "measurement_dtype": str(measurement.dtype),
        }
        (row_dir / "input_manifest.json").write_text(
            json.dumps(item, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest_rows.append(item)
    combined = {
        "schema_version": "b23.b23-1-inputs.v1",
        "status": "PASS",
        "gpu_work_performed": True,
        "selection_performed_during_generation": False,
        "registry_sha256": sha256_file(registry),
        "rows": manifest_rows,
    }
    (output / "INPUTS.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "rows": 5, "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
