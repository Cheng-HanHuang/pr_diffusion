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


def torch_load(path: Path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_existing(repo: Path, input_root: Path, output_json: Path | None) -> int:
    import torch

    config = json.loads((repo / "configs/b23/b23_1a_b_execution.yaml").read_text())
    required_root = Path(config["execution"]["required_reuse_input_root"]).resolve()
    if input_root != required_root:
        raise RuntimeError(
            f"reuse-input root mismatch: observed={input_root} required={required_root}"
        )
    registry = repo / config["registry"]["combined"]
    with registry.open(newline="", encoding="utf-8") as handle:
        signed_rows = list(csv.DictReader(handle))
    combined_path = input_root / "INPUTS.json"
    if not combined_path.is_file():
        raise FileNotFoundError(combined_path)
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    rows = combined.get("rows")
    if (
        combined.get("schema_version") != "b23.b23-1-inputs.v1"
        or combined.get("status") != "PASS"
        or combined.get("selection_performed_during_generation") is not False
        or combined.get("registry_sha256") != sha256_file(registry)
        or not isinstance(rows, list)
        or len(rows) != 5
        or len(signed_rows) != 5
    ):
        raise RuntimeError("existing B23.1 input-set identity or cardinality is invalid")
    signed_by_key = {(row["split"], row["row_id"]): row for row in signed_rows}
    observed_keys = set()
    for item in rows:
        key = (item["split"], str(item["row_id"]))
        if key in observed_keys or key not in signed_by_key:
            raise RuntimeError(f"unexpected or duplicate existing input row: {key}")
        observed_keys.add(key)
        signed = signed_by_key[key]
        for field in signed:
            if str(item[field]) != signed[field]:
                raise RuntimeError(f"existing input row changed signed field {field}: {key}")
        source = Path(item["ground_truth_source_path"])
        gt_path = Path(item["ground_truth_tensor_path"])
        measurement_path = Path(item["measurement_path"])
        for path in (source, gt_path, measurement_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        if sha256_file(source) != item["ground_truth_source_sha256"]:
            raise RuntimeError(f"ground-truth source file changed: {source}")
        if sha256_file(measurement_path) != item["measurement_file_sha256"]:
            raise RuntimeError(f"measurement file changed: {measurement_path}")
        gt_payload = torch_load(gt_path)
        ground_truth = gt_payload.get("ground_truth") if isinstance(gt_payload, dict) else None
        measurement_payload = torch_load(measurement_path)
        measurement = (
            measurement_payload.get("measurement")
            if isinstance(measurement_payload, dict)
            else measurement_payload
        )
        if not torch.is_tensor(ground_truth) or not torch.is_tensor(measurement):
            raise RuntimeError(f"existing input tensors are missing: {key}")
        if tensor_sha256(ground_truth) != item["ground_truth_tensor_sha256"]:
            raise RuntimeError(f"ground-truth tensor changed: {key}")
        if tensor_sha256(measurement) != item["measurement_tensor_sha256"]:
            raise RuntimeError(f"measurement tensor changed: {key}")
        if list(measurement.shape) != item["measurement_shape"] or str(measurement.dtype) != item["measurement_dtype"]:
            raise RuntimeError(f"measurement tensor schema changed: {key}")
        if not torch.isfinite(ground_truth).all() or not torch.isfinite(measurement).all():
            raise RuntimeError(f"existing input contains a nonfinite tensor: {key}")
    if observed_keys != set(signed_by_key):
        raise RuntimeError("existing input set omitted a signed row")
    result = {
        "schema_version": "b23.b23-1-reused-input-validation.v1",
        "status": "PASS",
        "gpu_work_performed": False,
        "input_generation_performed": False,
        "reuse_input_root": str(input_root),
        "inputs_json_sha256": sha256_file(combined_path),
        "registry_sha256": sha256_file(registry),
        "rows": len(rows),
        "measurement_file_sha256s": [item["measurement_file_sha256"] for item in rows],
        "measurement_tensor_sha256s": [item["measurement_tensor_sha256"] for item in rows],
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output-root", type=Path)
    destination.add_argument("--validate-existing", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.validate_existing:
        return validate_existing(repo, args.validate_existing.resolve(), args.output_json)
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
