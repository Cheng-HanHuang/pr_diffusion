#!/usr/bin/env python3
"""Strict CPU validator and aggregator for B22.2 smoke/full stages."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from b22_smoke_common import (
    metric_pair,
    read_json,
    tensor_content_sha256,
    validate_reconstruction,
    write_json,
)

POLICY_ORDER = [
    "Fresh1",
    "Fresh2",
    "SITCOM-1",
    "SITCOM-4S",
    "NP-1",
    "NP-8-RS",
    "SITCOM-oracle4",
    "NP-oracle8",
]


def load_gt(path: Path, resolution: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
        ]
    )
    return (transform(Image.open(path).convert("RGB")) * 2.0 - 1.0).unsqueeze(0)


def load_png(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0


def load_reconstruction(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    value = payload["reconstruction"] if isinstance(payload, dict) else payload
    value = value.float()
    validate_reconstruction(value)
    return value


def close(a: float, b: float, tol: float = 1.0e-5) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def validate_selected_entry(
    entry: dict[str, Any], gt: torch.Tensor, label: str
) -> tuple[dict[str, float], str]:
    tensor_path = Path(entry["tensor_path"])
    png_path = Path(entry["png_path"])
    if not tensor_path.is_file() or not png_path.is_file():
        raise FileNotFoundError(f"{label} selected artifact missing")
    reconstruction = load_reconstruction(tensor_path)
    observed_hash = tensor_content_sha256(reconstruction)
    if observed_hash != entry["reconstruction_content_sha256"]:
        raise RuntimeError(f"{label} selected reconstruction hash mismatch")
    metrics = metric_pair(reconstruction, gt)
    for key, value in metrics.items():
        if not close(value, entry["metrics"][key]):
            raise RuntimeError(
                f"{label} {key} mismatch: recomputed {value}, stored {entry['metrics'][key]}"
            )
    return metrics, str(png_path)


def validate_candidate_result(
    result_path: Path, gt: torch.Tensor, family: str, image_id: str
) -> dict[str, Any]:
    result = read_json(result_path)
    if result.get("status") != "PASS" or result.get("method_family") != family:
        raise RuntimeError(f"Invalid {family} candidate result: {result_path}")
    tensor_path = Path(result["reconstruction_tensor_path"])
    png_path = Path(result["reconstruction_png_path"])
    if not tensor_path.is_file() or not png_path.is_file():
        raise FileNotFoundError(f"Candidate artifacts missing: {result_path}")
    reconstruction = load_reconstruction(tensor_path)
    if tensor_content_sha256(reconstruction) != result["reconstruction_content_sha256"]:
        raise RuntimeError(f"Candidate hash mismatch: {result_path}")
    metrics = metric_pair(reconstruction, gt)
    for key, value in metrics.items():
        if not close(value, result["metrics"][key]):
            raise RuntimeError(
                f"Candidate metric mismatch {family}/{image_id}/{result_path.parent.name}/{key}"
            )
    timing = float(result["timing"]["reconstruction_s"])
    if not math.isfinite(timing) or timing <= 0:
        raise RuntimeError(f"Invalid candidate timing: {result_path}")
    memory = result["memory"]
    if int(memory["peak_allocated_bytes"]) <= 0 or int(memory["peak_reserved_bytes"]) <= 0:
        raise RuntimeError(f"Invalid candidate GPU memory record: {result_path}")
    if family == "SITCOM":
        value = float(result["selector"]["correction_norm"])
    else:
        value = float(result["selector_stats"]["selector_post_winner_lf_mse_mean"])
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite executable selector statistic: {result_path}")
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    policies = sorted({str(r["policy"]) for r in rows}, key=lambda x: POLICY_ORDER.index(x))
    for policy in policies:
        rs = [r for r in rows if r["policy"] == policy]
        raw = [float(r["psnr_raw"]) for r in rs]
        amb = [float(r["psnr_ambiguity_aware"]) for r in rs]
        times = [float(r["policy_gpu_seconds"]) for r in rs]
        out.append(
            {
                "policy": policy,
                "diagnostic_only": bool(rs[0]["diagnostic_only"]),
                "n_images": len(rs),
                "raw_psnr_mean": mean(raw),
                "raw_psnr_median": median(raw),
                "raw_psnr_min": min(raw),
                "raw_psnr_q05": quantile(raw, 0.05),
                "raw_psnr_q10": quantile(raw, 0.10),
                "raw_good25": sum(x >= 25.0 for x in raw),
                "ambiguity_psnr_mean": mean(amb),
                "ambiguity_good25": sum(x >= 25.0 for x in amb),
                "mean_policy_gpu_seconds": mean(times),
                "sum_policy_gpu_seconds": sum(times),
            }
        )
    return out


def read_fresh_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100:
        raise RuntimeError(f"Expected 100 Fresh2 rows, found {len(rows)}")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        image_id = f"{int(row['image_id']):05d}"
        if image_id in result:
            raise RuntimeError(f"Duplicate Fresh row {image_id}")
        result[image_id] = row
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    args = parser.parse_args()

    config = read_json(args.config)
    run_root = Path(args.run_root).resolve()
    stage_root = run_root / args.stage
    manifest = read_json(stage_root / "manifest.json")
    rows = manifest["rows"]
    expected = (
        int(config["gate"]["smoke_required_rows_per_method"])
        if args.stage == "smoke"
        else int(config["gate"]["full_required_rows_per_method"])
    )
    if len(rows) != expected:
        raise RuntimeError(f"Stage has {len(rows)} manifest rows, expected {expected}")

    for method in ("sitcom", "np"):
        shard_count = int(config["execution"][f"{method}_shards"])
        for shard_id in range(shard_count):
            worker = read_json(stage_root / "workers" / f"{method}_shard{shard_id}.json")
            if worker["status"] != "PASS" or worker["assigned_rows"] != worker["completed_rows"]:
                raise RuntimeError(f"Incomplete {method} shard {shard_id}: {worker}")

    fresh = read_fresh_rows(Path(config["pac_paths"]["fresh2_rows_csv"])) if args.stage == "full" else {}
    paired: list[dict[str, Any]] = []
    candidate_audit: list[dict[str, Any]] = []

    for row in rows:
        image_id = row["image_id"]
        gt = load_gt(Path(row["ground_truth_path"]), int(config["problem"]["resolution"]))
        if tensor_content_sha256(gt) != row["ground_truth_content_sha256"]:
            raise RuntimeError(f"Ground truth hash mismatch for {image_id}")

        sitcom_path = stage_root / "sitcom" / f"row{int(row['row_id']):03d}_{image_id}" / "policy.json"
        np_path = stage_root / "np" / f"row{int(row['row_id']):03d}_{image_id}" / "policy.json"
        sitcom = read_json(sitcom_path)
        np_policy = read_json(np_path)

        sit_candidate_paths = sorted((sitcom_path.parent / "candidates").glob("*/result.json"))
        np_candidate_paths = sorted((np_path.parent / "candidates").glob("*/result.json"))
        if len(sit_candidate_paths) != 4 or len(np_candidate_paths) != 8:
            raise RuntimeError(
                f"Candidate result count mismatch for {image_id}: "
                f"SITCOM={len(sit_candidate_paths)}, NP={len(np_candidate_paths)}"
            )
        sit_candidate_results = [
            validate_candidate_result(path, gt, "SITCOM", image_id)
            for path in sit_candidate_paths
        ]
        np_candidate_results = [
            validate_candidate_result(path, gt, "NP", image_id)
            for path in np_candidate_paths
        ]

        if sitcom["status"] != "PASS" or int(sitcom["candidate_count"]) != 4:
            raise RuntimeError(f"Invalid SITCOM policy for {image_id}")
        if np_policy["status"] != "PASS" or int(np_policy["candidate_count"]) != 8:
            raise RuntimeError(f"Invalid NP policy for {image_id}")
        if sorted(int(r["run_index"]) for r in sit_candidate_results) != [0, 1, 2, 3]:
            raise RuntimeError(f"SITCOM candidate indices mismatch for {image_id}")
        np_keys = sorted((int(r["config_index"]), int(r["seed"])) for r in np_candidate_results)
        expected_np_keys = sorted(
            (cfg_index, int(seed))
            for cfg_index, _ in enumerate(config["np"]["configs"])
            for seed in config["np"]["seeds"]
        )
        if np_keys != expected_np_keys:
            raise RuntimeError(f"NP candidate identities mismatch for {image_id}")
        if sitcom["measurement_content_sha256"] != row["measurement_content_sha256"]:
            raise RuntimeError(f"SITCOM measurement mismatch for {image_id}")
        if np_policy["measurement_raw_content_sha256"] != row["measurement_content_sha256"]:
            raise RuntimeError(f"NP measurement mismatch for {image_id}")

        sit_values = sitcom["selector_values"]
        expected_sit = min(
            sit_values,
            key=lambda r: (float(r["correction_norm"]), int(r["run_index"])),
        )
        selected_sit = sitcom["policies"]["SITCOM-4S"]
        if int(selected_sit["candidate_run_index"]) != int(expected_sit["run_index"]):
            raise RuntimeError(f"SITCOM selector mismatch for {image_id}")
        if int(sitcom["policies"]["SITCOM-1"]["candidate_run_index"]) != 0:
            raise RuntimeError(f"SITCOM-1 is not candidate 0 for {image_id}")

        np_values = np_policy["selector_values"]
        expected_np = min(
            np_values,
            key=lambda r: (
                float(r["selector_post_winner_lf_mse_mean"]),
                int(r["config_index"]),
                int(r["seed"]),
            ),
        )
        selected_np = np_policy["policies"]["NP-8-RS"]
        if (
            selected_np["candidate_config_tag"] != expected_np["config_tag"]
            or int(selected_np["candidate_seed"]) != int(expected_np["seed"])
        ):
            raise RuntimeError(f"NP selector mismatch for {image_id}")
        np1 = np_policy["policies"]["NP-1"]
        if np1["candidate_config_tag"] != "lf" or int(np1["candidate_seed"]) != 100:
            raise RuntimeError(f"NP-1 identity mismatch for {image_id}")

        if args.stage == "smoke" and image_id == config["smoke_replay"]["image_id"]:
            observed_sitcom_hash = sitcom["policies"]["SITCOM-1"][
                "reconstruction_content_sha256"
            ]
            observed_np_hash = np_policy["policies"]["NP-1"][
                "reconstruction_content_sha256"
            ]
            if observed_sitcom_hash != config["smoke_replay"][
                "sitcom1_reconstruction_content_sha256"
            ]:
                raise RuntimeError(
                    f"B22.1 SITCOM-1 replay hash mismatch: {observed_sitcom_hash}"
                )
            if observed_np_hash != config["smoke_replay"][
                "np1_reconstruction_content_sha256"
            ]:
                raise RuntimeError(f"B22.1 NP-1 replay hash mismatch: {observed_np_hash}")

        policy_specs = [
            ("SITCOM-1", sitcom["policies"]["SITCOM-1"], False, 1),
            ("SITCOM-4S", sitcom["policies"]["SITCOM-4S"], False, 4),
            ("SITCOM-oracle4", sitcom["policies"]["SITCOM-oracle4"], True, 4),
            ("NP-1", np_policy["policies"]["NP-1"], False, 1),
            ("NP-8-RS", np_policy["policies"]["NP-8-RS"], False, 8),
            ("NP-oracle8", np_policy["policies"]["NP-oracle8"], True, 8),
        ]
        for policy_name, entry, diagnostic, candidate_count in policy_specs:
            metrics, png_path = validate_selected_entry(entry, gt, f"{policy_name}/{image_id}")
            if policy_name.startswith("SITCOM"):
                all_seconds = float(sitcom["sum_candidate_reconstruction_s"])
            else:
                all_seconds = float(np_policy["sum_candidate_reconstruction_s"])
            policy_seconds = (
                float(entry["reconstruction_s"])
                if candidate_count == 1
                else all_seconds
            )
            paired.append(
                {
                    "row_id": row["row_id"],
                    "image_id": image_id,
                    "policy": policy_name,
                    "diagnostic_only": diagnostic,
                    **metrics,
                    "good25_raw": metrics["psnr_raw"] >= 25.0,
                    "good25_ambiguity_aware": metrics["psnr_ambiguity_aware"] >= 25.0,
                    "native_candidate_count": candidate_count,
                    "policy_gpu_seconds": policy_seconds,
                    "selected_png_path": png_path,
                    "measurement_content_sha256": row["measurement_content_sha256"],
                }
            )

        for item in sit_values:
            candidate_audit.append(
                {"row_id": row["row_id"], "image_id": image_id, "family": "SITCOM", **item}
            )
        for item in np_values:
            candidate_audit.append(
                {"row_id": row["row_id"], "image_id": image_id, "family": "NP", **item}
            )

        if args.stage == "full":
            source = fresh.get(image_id)
            if source is None:
                raise RuntimeError(f"Fresh rows missing image {image_id}")
            fresh_specs = [
                (
                    "Fresh1",
                    Path(source["arm1_sample_path"]),
                    float(source["arm1_wall_seconds"]),
                    1,
                ),
                (
                    "Fresh2",
                    Path(
                        source["arm1_sample_path"]
                        if source["fresh2_selected_variant"] == "base_full"
                        else source["arm2_sample_path"]
                    ),
                    float(source["fresh2_total_wall_seconds"]),
                    2,
                ),
            ]
            for policy_name, sample_path, seconds, count in fresh_specs:
                if not sample_path.is_file():
                    raise FileNotFoundError(sample_path)
                metrics = metric_pair(load_png(sample_path), gt)
                paired.append(
                    {
                        "row_id": row["row_id"],
                        "image_id": image_id,
                        "policy": policy_name,
                        "diagnostic_only": False,
                        **metrics,
                        "good25_raw": metrics["psnr_raw"] >= 25.0,
                        "good25_ambiguity_aware": metrics["psnr_ambiguity_aware"] >= 25.0,
                        "native_candidate_count": count,
                        "policy_gpu_seconds": seconds,
                        "selected_png_path": str(sample_path),
                        "measurement_content_sha256": row["measurement_content_sha256"],
                    }
                )

    expected_policies = 6 if args.stage == "smoke" else 8
    if len(paired) != len(rows) * expected_policies:
        raise RuntimeError(
            f"Paired row count {len(paired)} != {len(rows)}*{expected_policies}"
        )
    for policy in ({r["policy"] for r in paired}):
        if sum(r["policy"] == policy for r in paired) != len(rows):
            raise RuntimeError(f"Policy {policy} is incomplete")

    summaries = summary_rows(paired)
    failures = [r for r in paired if not r["diagnostic_only"] and not r["good25_raw"]]
    write_csv(stage_root / "paired_rows.csv", paired)
    write_csv(stage_root / "summary.csv", summaries)
    write_csv(stage_root / "candidate_selector_audit.csv", candidate_audit)
    if failures:
        write_csv(stage_root / "failures_raw25.csv", failures)
    else:
        (stage_root / "failures_raw25.csv").write_text(
            "row_id,image_id,policy\n", encoding="utf-8"
        )

    validation = {
        "schema_version": 1,
        "status": "PASS",
        "stage": args.stage,
        "n_images": len(rows),
        "paired_rows": len(paired),
        "policies": [r["policy"] for r in summaries],
        "checks": {
            "worker_shards_complete": True,
            "locked_measurement_identity": True,
            "ground_truth_identity": True,
            "candidate_counts_exact": True,
            "all_candidate_output_hashes": True,
            "all_candidate_metrics_recomputed": True,
            "all_candidate_runtime_memory_records": True,
            "selected_output_hashes": True,
            "offline_metrics_recomputed": True,
            "sitcom_selector_recomputed": True,
            "np_selector_recomputed": True,
            "no_policy_row_omission": True,
            "b22_1_exact_replay_hashes": args.stage == "smoke",
            "fresh_outputs_reused_not_rerun": args.stage == "full",
        },
        "automatic_full_launch_machine_gate": args.stage == "smoke",
        "automatic_full_launch_allowed": args.stage == "smoke",
        "full_panel_complete": args.stage == "full",
        "full_panel_scientific_signoff_pending": args.stage == "full",
    }
    write_json(stage_root / "validation.json", validation)
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
