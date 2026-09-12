#!/usr/bin/env python3
"""Validate and freeze the realized cumulative B24 6144 baseline case universe.

This is a zero-GPU audit.  It does not regenerate measurements or reconstructions.
It cross-checks the deterministic 6144 manifest against the realized atomic
IMAGE_COMPLETE.json records from the cumulative 64 -> 256 -> 2048 -> 6144
campaign and emits a compact case registry suitable for a later prospective
DEV/HELDOUT role freeze.

The registry freezes, per row:
- FFHQ image identity and deterministic class rank;
- measurement seed plus realized measurement file/tensor SHA-256;
- all four DAPS and all four SITCOM solver seeds;
- baseline A/B/C/D class and baseline best-of-four PSNRs;
- exact source completion record.

No role assignment is made here.  In particular, this script does not alter the
historical "first 100 development, next 100 held-out" rule.  A role amendment,
if any, must be frozen separately before running project methods on held-out
cases.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prdiffusion.b24_protocol import class_rank_key  # noqa: E402

ROOT = Path("/egr/research-pac/huang248")
OUTROOT = ROOT / "outputs/pr_diffusion/b24"
DEFAULT_64 = OUTROOT / "B24_2_64_20260826T040303Z"
DEFAULT_256 = OUTROOT / "B24_2_256_extension_20260827T013232Z"
DEFAULT_2048 = OUTROOT / "B24_2_2048_extension_20260827T073255Z"
DEFAULT_6144 = OUTROOT / "B24_2_6144_extension_20260829T174423Z"


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
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def source_root_for_row(row_index: int, roots: dict[int, Path]) -> Path:
    if row_index < 64:
        return roots[64]
    if row_index < 256:
        return roots[256]
    if row_index < 2048:
        return roots[2048]
    return roots[6144]


def find_completion(root: Path, shard: int, row_index: int, image_id: str) -> Path:
    shard_root = root / f"shard{shard}"
    if not shard_root.is_dir():
        raise FileNotFoundError(shard_root)
    hits = sorted(shard_root.glob(f"row*_{image_id}/IMAGE_COMPLETE.json"))
    exact = []
    for p in hits:
        try:
            v = read_json(p)
        except Exception:
            continue
        if int(v.get("row_index", -1)) == row_index and str(v.get("image_id", "")).zfill(5) == image_id:
            exact.append(p)
    if len(exact) != 1:
        raise RuntimeError(
            f"expected exactly one completion for row={row_index} image={image_id} "
            f"under {shard_root}, got {len(exact)}: {exact}"
        )
    return exact[0]


def require_equal(label: str, observed, expected, path: Path) -> None:
    if observed != expected:
        raise RuntimeError(
            f"{label} mismatch in {path}: observed={observed!r} expected={expected!r}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run64", type=Path, default=DEFAULT_64)
    ap.add_argument("--run256", type=Path, default=DEFAULT_256)
    ap.add_argument("--run2048", type=Path, default=DEFAULT_2048)
    ap.add_argument("--run6144", type=Path, default=DEFAULT_6144)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    roots = {
        64: args.run64.resolve(),
        256: args.run256.resolve(),
        2048: args.run2048.resolve(),
        6144: args.run6144.resolve(),
    }
    for n, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"missing cumulative run root {n}: {root}")

    manifest_path = roots[6144] / "B24_2_baseline_6144.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 6144:
        raise RuntimeError("expected exact cumulative 6144-row manifest")
    require_equal(
        "row_index sequence",
        [int(r["row_index"]) for r in rows],
        list(range(6144)),
        manifest_path,
    )

    manifest_file_sha = sha256_file(manifest_path)
    manifest_payload_sha = manifest.get("manifest_sha256")
    if not isinstance(manifest_payload_sha, str) or len(manifest_payload_sha) != 64:
        raise RuntimeError("missing/invalid manifest payload SHA")

    # Require the completed 6144 extension summaries to agree on the manifest.
    for shard in range(4):
        summary_path = roots[6144] / f"shard{shard}/SHARD_COMPLETE.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = read_json(summary_path)
        require_equal("6144 shard status", summary.get("status"), "PASS", summary_path)
        require_equal("6144 shard completed", int(summary.get("completed", -1)), 1024, summary_path)
        require_equal(
            "6144 shard manifest file SHA",
            summary.get("manifest_file_sha256"),
            manifest_file_sha,
            summary_path,
        )

    registry: list[dict] = []
    counts = Counter()
    measurement_file_hashes: set[str] = set()
    measurement_tensor_hashes: set[str] = set()

    for row in rows:
        idx = int(row["row_index"])
        image_id = str(row["image_id"]).zfill(5)
        shard = int(row["shard_id"])
        require_equal("manifest shard", shard, idx % 4, manifest_path)
        require_equal("manifest gpu", int(row["gpu_id"]), shard, manifest_path)

        source_root = source_root_for_row(idx, roots)
        completion_path = find_completion(source_root, shard, idx, image_id)
        c = read_json(completion_path)
        require_equal("completion status", c.get("status"), "PASS", completion_path)
        require_equal("row_index", int(c.get("row_index", -1)), idx, completion_path)
        require_equal("image_id", str(c.get("image_id", "")).zfill(5), image_id, completion_path)
        require_equal("shard_id", int(c.get("shard_id", -1)), shard, completion_path)
        require_equal("gpu_id", int(c.get("gpu_id", -1)), shard, completion_path)
        require_equal(
            "measurement_seed",
            int(c.get("measurement_seed", -1)),
            int(row["measurement_seed"]),
            completion_path,
        )
        daps_seeds = [int(x) for x in row["daps_solver_seeds"]]
        sitcom_seeds = [int(x) for x in row["sitcom_solver_seeds"]]
        require_equal(
            "daps_solver_seeds",
            [int(x) for x in c.get("daps_solver_seeds", [])],
            daps_seeds,
            completion_path,
        )
        require_equal(
            "sitcom_solver_seeds",
            [int(x) for x in c.get("sitcom_solver_seeds", [])],
            sitcom_seeds,
            completion_path,
        )

        mfile = c.get("measurement_file_sha256")
        mtensor = c.get("measurement_tensor_sha256")
        if not isinstance(mfile, str) or len(mfile) != 64:
            raise RuntimeError(f"invalid measurement_file_sha256: {completion_path}")
        if not isinstance(mtensor, str) or len(mtensor) != 64:
            raise RuntimeError(f"invalid measurement_tensor_sha256: {completion_path}")
        measurement_file_hashes.add(mfile)
        measurement_tensor_hashes.add(mtensor)

        label = str(c.get("class_label", "")).upper()
        if label not in {"A", "B", "C", "D"}:
            raise RuntimeError(f"invalid class label in {completion_path}: {label}")
        counts[label] += 1

        registry.append({
            "row_index": idx,
            "image_id": image_id,
            "class_label": label,
            "class_rank_sha256": class_rank_key(label, image_id),
            "measurement_seed": int(row["measurement_seed"]),
            "measurement_file_sha256": mfile,
            "measurement_tensor_sha256": mtensor,
            "daps_solver_seed_0": daps_seeds[0],
            "daps_solver_seed_1": daps_seeds[1],
            "daps_solver_seed_2": daps_seeds[2],
            "daps_solver_seed_3": daps_seeds[3],
            "sitcom_solver_seed_0": sitcom_seeds[0],
            "sitcom_solver_seed_1": sitcom_seeds[1],
            "sitcom_solver_seed_2": sitcom_seeds[2],
            "sitcom_solver_seed_3": sitcom_seeds[3],
            "daps_best_psnr_raw_rgb_db": float(c["daps_best_psnr_raw_rgb_db"]),
            "sitcom_best_psnr_raw_rgb_db": float(c["sitcom_best_psnr_raw_rgb_db"]),
            "source_stage": str(c.get("stage", "")),
            "source_completion": str(completion_path),
        })

    if len(registry) != 6144:
        raise RuntimeError(f"registry row-count drift: {len(registry)}")
    if len({r["image_id"] for r in registry}) != 6144:
        raise RuntimeError("duplicate image IDs in realized 6144 registry")
    if sum(counts.values()) != 6144:
        raise RuntimeError(f"class-count drift: {dict(counts)}")

    out_dir = (args.out_dir or (roots[6144] / "case_freeze")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "B24_6144_CASE_UNIVERSE.csv"
    write_csv_atomic(csv_path, registry)
    csv_sha = sha256_file(csv_path)

    by_class = {
        label: sorted(
            [r for r in registry if r["class_label"] == label],
            key=lambda r: r["class_rank_sha256"],
        )
        for label in "ABCD"
    }
    ranked_path = out_dir / "B24_6144_CLASS_RANKED.json"
    write_json_atomic(
        ranked_path,
        {
            "schema_version": "b24.6144-class-ranked.v1",
            "manifest_file_sha256": manifest_file_sha,
            "manifest_payload_sha256": manifest_payload_sha,
            "classes": {
                label: [
                    {
                        "rank": rank,
                        "row_index": r["row_index"],
                        "image_id": r["image_id"],
                        "class_rank_sha256": r["class_rank_sha256"],
                    }
                    for rank, r in enumerate(vals, start=1)
                ]
                for label, vals in by_class.items()
            },
        },
    )
    ranked_sha = sha256_file(ranked_path)

    summary = {
        "schema_version": "b24.6144-case-universe-freeze.v1",
        "status": "PASS",
        "gpu_work_performed": False,
        "measurement_generation_performed": False,
        "reconstruction_performed": False,
        "role_assignment_performed": False,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": manifest_file_sha,
        "manifest_payload_sha256": manifest_payload_sha,
        "row_count": 6144,
        "class_counts": {label: counts[label] for label in "ABCD"},
        "unique_measurement_file_sha256_count": len(measurement_file_hashes),
        "unique_measurement_tensor_sha256_count": len(measurement_tensor_hashes),
        "case_universe_csv": str(csv_path),
        "case_universe_csv_sha256": csv_sha,
        "class_ranked_json": str(ranked_path),
        "class_ranked_json_sha256": ranked_sha,
        "source_runroots": {str(k): str(v) for k, v in roots.items()},
        "next": "PLANNER_FREEZE_DEV_HELDOUT_POLICY_BEFORE_PROJECT_METHOD_EXECUTION",
    }
    summary_path = out_dir / "B24_6144_CASE_FREEZE_SUMMARY.json"
    write_json_atomic(summary_path, summary)

    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
