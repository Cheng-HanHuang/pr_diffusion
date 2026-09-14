#!/usr/bin/env python3
"""Validate and freeze the realized cumulative B24 7424 baseline case universe.

Zero-GPU audit only.  This script never generates a measurement, loads a model,
or executes a reconstruction.  It cross-checks the deterministic cumulative
7424 manifest against the actual atomic IMAGE_COMPLETE.json records from the
64 -> 256 -> 2048 -> 6144 -> 7424 campaign.

Per row the registry freezes:
- deterministic screen/image identity and B24_CLASS_RANK_V1 rank key;
- measurement seed and realized measurement file/tensor SHA-256;
- all four DAPS and four SITCOM solver seeds;
- A/B/C/D label and best-of-four baseline PSNRs;
- exact source completion path and content SHA-256.

No development/held-out role is assigned here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prdiffusion.b24_protocol import (  # noqa: E402
    class_rank_key,
    classify_good25,
    render_screen_manifest,
    validate_pre_b24,
)

ROOT = Path("/egr/research-pac/huang248")
OUTROOT = ROOT / "outputs/pr_diffusion/b24"
DEFAULT_64 = OUTROOT / "B24_2_64_20260826T040303Z"
DEFAULT_256 = OUTROOT / "B24_2_256_extension_20260827T013232Z"
DEFAULT_2048 = OUTROOT / "B24_2_2048_extension_20260827T073255Z"
DEFAULT_6144 = OUTROOT / "B24_2_6144_extension_20260829T174423Z"
LATEST_7424_POINTER = OUTROOT / "B24_2_7424_LATEST_RUN.txt"
EXPECTED_EXTENSION_PER_SHARD = {64: 16, 256: 48, 2048: 448, 6144: 1024, 7424: 320}
EXPECTED_STAGE = {
    64: "B24.2_64",
    256: "B24.2_256_EXTENSION",
    2048: "B24.2_2048_EXTENSION",
    6144: "B24.2_6144_EXTENSION",
    7424: "B24.2_7424_EXTENSION",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("refusing empty case freeze")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def require_equal(label: str, observed, expected, path: Path) -> None:
    if observed != expected:
        raise RuntimeError(
            f"{label} mismatch in {path}: observed={observed!r} expected={expected!r}"
        )


def resolve_latest_7424() -> Path:
    if not LATEST_7424_POINTER.is_file():
        raise FileNotFoundError(f"missing latest 7424 pointer: {LATEST_7424_POINTER}")
    text = LATEST_7424_POINTER.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"empty latest 7424 pointer: {LATEST_7424_POINTER}")
    return Path(text).resolve()


def source_root_for_row(row_index: int, roots: dict[int, Path]) -> Path:
    if row_index < 64:
        return roots[64]
    if row_index < 256:
        return roots[256]
    if row_index < 2048:
        return roots[2048]
    if row_index < 6144:
        return roots[6144]
    return roots[7424]


def source_count_for_row(row_index: int) -> int:
    if row_index < 64:
        return 64
    if row_index < 256:
        return 256
    if row_index < 2048:
        return 2048
    if row_index < 6144:
        return 6144
    return 7424


def find_completion(root: Path, shard: int, row_index: int, image_id: str) -> Path:
    shard_root = root / f"shard{shard}"
    if not shard_root.is_dir():
        raise FileNotFoundError(shard_root)
    hits = sorted(shard_root.glob(f"row*_{image_id}/IMAGE_COMPLETE.json"))
    exact: list[Path] = []
    for path in hits:
        try:
            value = read_json(path)
        except Exception:
            continue
        if (
            int(value.get("row_index", -1)) == row_index
            and str(value.get("image_id", "")).zfill(5) == image_id
        ):
            exact.append(path)
    if len(exact) != 1:
        raise RuntimeError(
            f"expected one atomic completion for row={row_index} image={image_id} "
            f"under {shard_root}, got {len(exact)}: {exact}"
        )
    return exact[0]


def verify_final_7424_shards(root: Path, manifest_sha: str) -> None:
    for shard in range(4):
        summary_path = root / f"shard{shard}/SHARD_COMPLETE.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = read_json(summary_path)
        require_equal("7424 shard status", summary.get("status"), "PASS", summary_path)
        require_equal(
            "7424 shard stage", summary.get("stage"), EXPECTED_STAGE[7424], summary_path
        )
        require_equal(
            "7424 shard completed",
            int(summary.get("completed", -1)),
            EXPECTED_EXTENSION_PER_SHARD[7424],
            summary_path,
        )
        require_equal(
            "7424 shard row_count",
            int(summary.get("row_count", -1)),
            EXPECTED_EXTENSION_PER_SHARD[7424],
            summary_path,
        )
        require_equal(
            "7424 shard manifest file SHA",
            summary.get("manifest_file_sha256"),
            manifest_sha,
            summary_path,
        )
        actual = sum(1 for _ in (root / f"shard{shard}").glob("row*/IMAGE_COMPLETE.json"))
        require_equal(
            "7424 atomic completion count",
            actual,
            EXPECTED_EXTENSION_PER_SHARD[7424],
            summary_path,
        )
        print(
            f"FINAL7424_SHARD_PASS|shard={shard}|completed={actual}/320|summary={summary_path}",
            flush=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run64", type=Path, default=DEFAULT_64)
    ap.add_argument("--run256", type=Path, default=DEFAULT_256)
    ap.add_argument("--run2048", type=Path, default=DEFAULT_2048)
    ap.add_argument("--run6144", type=Path, default=DEFAULT_6144)
    ap.add_argument("--run7424", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    roots = {
        64: args.run64.resolve(),
        256: args.run256.resolve(),
        2048: args.run2048.resolve(),
        6144: args.run6144.resolve(),
        7424: (args.run7424.resolve() if args.run7424 is not None else resolve_latest_7424()),
    }
    for n, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"missing cumulative run root {n}: {root}")

    manifest_path = roots[7424] / "B24_2_baseline_7424.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 7424:
        raise RuntimeError("expected exact cumulative 7424-row manifest")
    require_equal(
        "row_index sequence",
        [int(r["row_index"]) for r in rows],
        list(range(7424)),
        manifest_path,
    )

    # Recompute the deterministic cumulative screen from the signed exposure freeze.
    exposure = REPO_ROOT / "manifests/b24/PRE_B24_EXPOSURE.csv"
    exposed = validate_pre_b24(exposure)
    deterministic_rows = render_screen_manifest(exposed_ids=exposed, count=7424)
    require_equal("deterministic 7424 manifest rows", rows, deterministic_rows, manifest_path)

    manifest_file_sha = sha256_file(manifest_path)
    manifest_payload_sha = manifest.get("manifest_sha256")
    if not isinstance(manifest_payload_sha, str) or len(manifest_payload_sha) != 64:
        raise RuntimeError("missing/invalid manifest payload SHA")

    verify_final_7424_shards(roots[7424], manifest_file_sha)

    registry: list[dict] = []
    counts: Counter[str] = Counter()
    measurement_file_hashes: set[str] = set()
    measurement_tensor_hashes: set[str] = set()

    for row in rows:
        idx = int(row["row_index"])
        image_id = str(row["image_id"]).zfill(5)
        shard = int(row["shard_id"])
        require_equal("manifest shard", shard, idx % 4, manifest_path)
        require_equal("manifest gpu", int(row["gpu_id"]), shard, manifest_path)

        source_count = source_count_for_row(idx)
        source_root = source_root_for_row(idx, roots)
        completion_path = find_completion(source_root, shard, idx, image_id)
        completion = read_json(completion_path)
        require_equal("completion status", completion.get("status"), "PASS", completion_path)
        require_equal("row_index", int(completion.get("row_index", -1)), idx, completion_path)
        require_equal(
            "image_id", str(completion.get("image_id", "")).zfill(5), image_id, completion_path
        )
        require_equal("shard_id", int(completion.get("shard_id", -1)), shard, completion_path)
        require_equal("gpu_id", int(completion.get("gpu_id", -1)), shard, completion_path)
        require_equal(
            "measurement_seed",
            int(completion.get("measurement_seed", -1)),
            int(row["measurement_seed"]),
            completion_path,
        )

        daps_seeds = [int(x) for x in row["daps_solver_seeds"]]
        sitcom_seeds = [int(x) for x in row["sitcom_solver_seeds"]]
        require_equal(
            "daps_solver_seeds",
            [int(x) for x in completion.get("daps_solver_seeds", [])],
            daps_seeds,
            completion_path,
        )
        require_equal(
            "sitcom_solver_seeds",
            [int(x) for x in completion.get("sitcom_solver_seeds", [])],
            sitcom_seeds,
            completion_path,
        )

        measurement_file_sha = completion.get("measurement_file_sha256")
        measurement_tensor_sha = completion.get("measurement_tensor_sha256")
        if not isinstance(measurement_file_sha, str) or len(measurement_file_sha) != 64:
            raise RuntimeError(f"invalid measurement_file_sha256: {completion_path}")
        if not isinstance(measurement_tensor_sha, str) or len(measurement_tensor_sha) != 64:
            raise RuntimeError(f"invalid measurement_tensor_sha256: {completion_path}")
        measurement_file_hashes.add(measurement_file_sha)
        measurement_tensor_hashes.add(measurement_tensor_sha)

        daps_psnr = float(completion["daps_best_psnr_raw_rgb_db"])
        sitcom_psnr = float(completion["sitcom_best_psnr_raw_rgb_db"])
        if not math.isfinite(daps_psnr) or not math.isfinite(sitcom_psnr):
            raise RuntimeError(f"non-finite baseline PSNR: {completion_path}")
        recomputed_label = classify_good25(daps_psnr, sitcom_psnr)
        label = str(completion.get("class_label", "")).upper()
        require_equal("Good25 class recomputation", label, recomputed_label, completion_path)
        if label not in {"A", "B", "C", "D"}:
            raise RuntimeError(f"invalid class label in {completion_path}: {label}")
        counts[label] += 1

        registry.append(
            {
                "row_index": idx,
                "image_id": image_id,
                "screen_rank_sha256": str(row["screen_rank_sha256"]),
                "class_label": label,
                "class_rank_sha256": class_rank_key(label, image_id),
                "measurement_seed": int(row["measurement_seed"]),
                "measurement_file_sha256": measurement_file_sha,
                "measurement_tensor_sha256": measurement_tensor_sha,
                "daps_solver_seed_0": daps_seeds[0],
                "daps_solver_seed_1": daps_seeds[1],
                "daps_solver_seed_2": daps_seeds[2],
                "daps_solver_seed_3": daps_seeds[3],
                "sitcom_solver_seed_0": sitcom_seeds[0],
                "sitcom_solver_seed_1": sitcom_seeds[1],
                "sitcom_solver_seed_2": sitcom_seeds[2],
                "sitcom_solver_seed_3": sitcom_seeds[3],
                "daps_best_psnr_raw_rgb_db": daps_psnr,
                "sitcom_best_psnr_raw_rgb_db": sitcom_psnr,
                "source_cumulative_stage": source_count,
                "source_stage": str(completion.get("stage", "")),
                "source_completion_sha256": sha256_file(completion_path),
                "source_completion": str(completion_path),
            }
        )

    if len(registry) != 7424:
        raise RuntimeError(f"registry row-count drift: {len(registry)}")
    if len({r["image_id"] for r in registry}) != 7424:
        raise RuntimeError("duplicate image IDs in realized 7424 registry")
    if sum(counts.values()) != 7424:
        raise RuntimeError(f"class-count drift: {dict(counts)}")

    out_dir = (args.out_dir or (roots[7424] / "case_freeze")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "B24_7424_CASE_UNIVERSE.csv"
    write_csv_atomic(csv_path, registry)
    csv_sha = sha256_file(csv_path)

    by_class = {
        label: sorted(
            [r for r in registry if r["class_label"] == label],
            key=lambda r: r["class_rank_sha256"],
        )
        for label in "ABCD"
    }
    ranked_path = out_dir / "B24_7424_CLASS_RANKED.json"
    write_json_atomic(
        ranked_path,
        {
            "schema_version": "b24.7424-class-ranked.v1",
            "manifest_file_sha256": manifest_file_sha,
            "manifest_payload_sha256": manifest_payload_sha,
            "class_rank_domain": "B24_CLASS_RANK_V1",
            "classes": {
                label: [
                    {
                        "rank": rank,
                        "row_index": r["row_index"],
                        "image_id": r["image_id"],
                        "class_rank_sha256": r["class_rank_sha256"],
                    }
                    for rank, r in enumerate(values, start=1)
                ]
                for label, values in by_class.items()
            },
        },
    )
    ranked_sha = sha256_file(ranked_path)

    quota = {label: counts[label] >= 100 for label in "ABC"}
    summary = {
        "schema_version": "b24.7424-case-universe-freeze.v1",
        "status": "PASS" if all(quota.values()) else "FAIL_CLASS_QUOTA",
        "gpu_work_performed": False,
        "measurement_generation_performed": False,
        "reconstruction_performed": False,
        "method_execution_performed": False,
        "role_assignment_performed": False,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": manifest_file_sha,
        "manifest_payload_sha256": manifest_payload_sha,
        "row_count": 7424,
        "class_counts": {label: counts[label] for label in "ABCD"},
        "abc_at_least_100": quota,
        "unique_measurement_file_sha256_count": len(measurement_file_hashes),
        "unique_measurement_tensor_sha256_count": len(measurement_tensor_hashes),
        "case_universe_csv": str(csv_path),
        "case_universe_csv_sha256": csv_sha,
        "class_ranked_json": str(ranked_path),
        "class_ranked_json_sha256": ranked_sha,
        "source_runroots": {str(k): str(v) for k, v in roots.items()},
        "next": "FREEZE_FIRST_100_A_B_C_BY_B24_CLASS_RANK_V1_IF_ALL_QUOTAS_PASS",
    }
    summary_path = out_dir / "B24_7424_CASE_FREEZE_SUMMARY.json"
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))
    if not all(quota.values()):
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
