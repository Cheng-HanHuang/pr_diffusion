#!/usr/bin/env python3
"""Validate B22.1 smoke outputs and emit compact return-bundle metadata."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch

from b22_smoke_common import (
    load_ground_truth,
    metric_pair,
    read_json,
    tensor_content_sha256,
    validate_reconstruction,
    write_json,
)


def load_reconstruction(path: str | Path) -> torch.Tensor:
    payload = torch.load(str(path), map_location="cpu")
    if isinstance(payload, torch.Tensor):
        value = payload
    elif isinstance(payload, dict) and isinstance(payload.get("reconstruction"), torch.Tensor):
        value = payload["reconstruction"]
    else:
        raise TypeError(f"No reconstruction tensor found in {path}")
    value = value.to(dtype=torch.float32)
    validate_reconstruction(value)
    return value


def require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise RuntimeError(f"{label}: observed {observed!r}, expected {expected!r}")


def require_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(float(observed), float(expected), rel_tol=1.0e-7, abs_tol=1.0e-7):
        raise RuntimeError(f"{label}: observed {observed!r}, expected {expected!r}")


def validate_method(
    name: str,
    result: Dict[str, Any],
    expected_method: str,
    expected_config: Dict[str, Any],
    manifest: Dict[str, Any],
    gt: torch.Tensor,
) -> Dict[str, Any]:
    require_equal(result["schema_version"], 1, f"{name} schema")
    require_equal(result["method"], expected_method, f"{name} method name")
    require_equal(result["image_id"], manifest["image_id"], f"{name} image ID")
    require_equal(
        result["measurement_file_sha256"],
        manifest["measurement_file_sha256"],
        f"{name} measurement file SHA",
    )
    require_equal(
        result["measurement_raw_content_sha256"],
        manifest["measurement_content_sha256"],
        f"{name} raw measurement content SHA",
    )
    require_equal(
        result["ground_truth_tensor_content_sha256"],
        manifest["ground_truth_tensor_content_sha256"],
        f"{name} ground-truth SHA",
    )
    require_equal(result["model_sha256"], manifest["model_sha256"], f"{name} model SHA")
    require_equal(result["finite_output"], True, f"{name} finite-output flag")
    require_equal(result["config"], expected_config, f"{name} frozen config")

    tensor_path = Path(result["reconstruction_tensor_path"])
    png_path = Path(result["reconstruction_png_path"])
    if not tensor_path.is_file():
        raise FileNotFoundError(f"{name} reconstruction tensor missing: {tensor_path}")
    if not png_path.is_file():
        raise FileNotFoundError(f"{name} reconstruction PNG missing: {png_path}")

    reconstruction = load_reconstruction(tensor_path)
    require_equal(
        tensor_content_sha256(reconstruction),
        result["reconstruction_content_sha256"],
        f"{name} reconstruction content SHA",
    )

    recomputed = metric_pair(reconstruction, gt)
    for key, value in recomputed.items():
        require_close(value, result["metrics"][key], f"{name} {key}")

    timing = result["timing"]
    for key in ("model_load_s", "reconstruction_s", "total_observed_s"):
        value = float(timing[key])
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"{name} invalid timing {key}={value}")

    memory = result["memory"]
    if int(memory["peak_allocated_bytes"]) <= 0:
        raise RuntimeError(f"{name} did not record positive peak allocated GPU memory")
    if int(memory["peak_reserved_bytes"]) <= 0:
        raise RuntimeError(f"{name} did not record positive peak reserved GPU memory")

    return {
        "method": expected_method,
        "image_id": result["image_id"],
        **recomputed,
        "good25_raw": recomputed["psnr_raw"] >= 25.0,
        "good25_ambiguity_aware": recomputed["psnr_ambiguity_aware"] >= 25.0,
        "model_load_s": float(timing["model_load_s"]),
        "reconstruction_s": float(timing["reconstruction_s"]),
        "total_observed_s": float(timing["total_observed_s"]),
        "peak_allocated_bytes": int(memory["peak_allocated_bytes"]),
        "peak_reserved_bytes": int(memory["peak_reserved_bytes"]),
        "device_name": result["device_name"],
        "reconstruction_png_path": str(png_path),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No comparison rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_root", required=True)
    args = parser.parse_args()

    config = read_json(args.config)
    run_root = Path(args.run_root).resolve()
    manifest = read_json(run_root / "input" / "input_manifest.json")
    gt = load_ground_truth(manifest["ground_truth_tensor_path"], device="cpu")

    sitcom = read_json(run_root / "sitcom1" / "result.json")
    np1 = read_json(run_root / "np1" / "result.json")

    rows = [
        validate_method("sitcom1", sitcom, "SITCOM-1", config["sitcom1"], manifest, gt),
        validate_method("np1", np1, "NP-1", config["np1"], manifest, gt),
    ]

    require_equal(
        sitcom["measurement_used_content_sha256"],
        manifest["measurement_content_sha256"],
        "SITCOM must use the raw locked measurement",
    )
    require_equal(sitcom["measurement_preprocessing"], "none", "SITCOM preprocessing")
    require_equal(
        np1["measurement_preprocessing"],
        "clamp_min_zero_in_memory",
        "NP preprocessing",
    )
    if np1["measurement_used_content_sha256"] == manifest["measurement_content_sha256"]:
        raise RuntimeError(
            "NP clipped-measurement content hash unexpectedly equals the raw hash"
        )
    if int(np1["measurement_negative_entries_clipped"]) <= 0:
        raise RuntimeError("NP reports zero clipped entries on a negative-valued tensor")

    comparison_path = run_root / "comparison.csv"
    write_csv(comparison_path, rows)

    validation = {
        "schema_version": 1,
        "status": "PASS",
        "gate": "B22.1 one-image fixed-baseline smoke",
        "image_id": manifest["image_id"],
        "selection_rule": manifest["selection_rule"],
        "selection_uses_method_outcome": manifest["selection_uses_method_outcome"],
        "measurement_file_sha256": manifest["measurement_file_sha256"],
        "repo_head": manifest["repo_head"],
        "validated_methods": ["SITCOM-1", "NP-1"],
        "comparison_csv": str(comparison_path),
        "checks": {
            "locked_measurement_identity": True,
            "shared_ground_truth_identity": True,
            "finite_outputs": True,
            "offline_metric_recomputation": True,
            "runtime_recording": True,
            "gpu_memory_recording": True,
            "frozen_config_exact_match": True,
            "sitcom_raw_measurement": True,
            "np_in_memory_clipping": True,
        },
        "full_panel_authorized": False,
        "next_gate": "Return archive for execution-lead review and explicit B22.1 sign-off",
    }
    write_json(run_root / "validation.json", validation)

    (run_root / "RETURN_MANIFEST.txt").write_text(
        "\n".join(
            [
                "Attach the sibling .tar.gz archive to the execution-lead chat.",
                f"run_root={run_root}",
                f"image_id={manifest['image_id']}",
                f"measurement_sha256={manifest['measurement_file_sha256']}",
                "validation=PASS",
                "full_panel_authorized=0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "PASS",
                "image_id": manifest["image_id"],
                "methods": ["SITCOM-1", "NP-1"],
                "full_panel_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
