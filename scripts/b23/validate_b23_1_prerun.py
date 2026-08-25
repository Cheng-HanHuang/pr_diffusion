#!/usr/bin/env python3
"""Fail-closed CPU validation for the authorized B23.1A/B pre-run freeze."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PARENTS = ("Fresh1", "LF-v1", "NP-1", "SITCOM-1")
EXPECTED_IMAGES = {
    "B23.1-SMOKE-1": ("65082",),
    "B23.1-SMOKE-4": ("61492", "62959", "66821", "68142"),
}
STRATA = ((60000, 62500), (62500, 65000), (65000, 67500), (67500, 70000))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def typed_registry_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        value: dict[str, Any] = dict(row)
        for field in ("row_id", "measurement_seed", "solver_base_seed"):
            value[field] = int(value[field])
        for field in ("assigned_before_run", "pre_b23_exposure_checked"):
            if value[field] not in {"true", "false"}:
                raise ValueError(f"invalid registry boolean {field}={value[field]!r}")
            value[field] = value[field] == "true"
        result.append(value)
    return result


def ranked_pick(tag: str, pool: list[str]) -> str:
    return min(pool, key=lambda value: hashlib.sha256(f"{tag}:{value}".encode()).hexdigest())


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def git_diff_sha256(repo: Path) -> str:
    payload = subprocess.check_output(["git", "-C", str(repo), "diff", "--binary"])
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--expected-head")
    parser.add_argument("--pac", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from prdiffusion.b23_protocol import derive_seed, validate_future_split_registry

    config_path = repo / "configs/b23/b23_1a_b_execution.yaml"
    exposure_path = repo / "manifests/b23/PRE_B23_EXPOSURE.csv"
    registry_path = repo / "manifests/b23/b23_1_signed_registry.csv"
    one_path = repo / "manifests/b23/b23_1_one_image_smoke.signed.csv"
    four_path = repo / "manifests/b23/b23_1_four_image_smoke.signed.csv"
    for path in (config_path, exposure_path, registry_path, one_path, four_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = read_json(config_path)
    auth = config["authorization"]
    if config["status"] != "AUTHORIZED_PRERUN_FROZEN":
        raise ValueError("B23.1 execution config is not frozen and authorized")
    if auth["authorized_stages"] != ["B23.1A", "B23.1B"] or not auth["gpu_work_authorized"]:
        raise ValueError("B23.1A/B GPU authorization is incomplete")
    forbidden = {"B23.2", "large panels", "B24", "adaptive schedules"}
    if set(auth["unauthorized_work"]) != forbidden:
        raise ValueError("closed authorization boundary changed")
    execution = config["execution"]
    if tuple(execution["parent_order"]) != PARENTS:
        raise ValueError("native replay parent order changed")
    if execution["native_repeats"] < 3 or execution["max_parent_trajectories"] != 32:
        raise ValueError("bounded replay count changed")
    if execution["expected_replay_parent_trajectories"] != 16:
        raise ValueError("replay trajectory count must be 4 parents x (3 native + 1 wrapper)")
    if execution["expected_smoke_parent_trajectories"] != 16:
        raise ValueError("smoke trajectory count must be 4 images x 4 parents")
    if config["stop_after"] != "B23.1_RETURN_PENDING_PLANNER_REVIEW":
        raise ValueError("B23.1 stop boundary changed")

    exposure_sha = sha256_file(exposure_path)
    if exposure_sha != config["registry"]["pre_b23_exposure_sha256"]:
        raise ValueError("PRE_B23_EXPOSURE identity changed after smoke selection")
    exposed = {row["image_id"] for row in read_csv(exposure_path)}
    rows = read_csv(registry_path)
    validate_future_split_registry(typed_registry_rows(rows), exposed)
    if len(rows) != 5:
        raise ValueError(f"expected exactly five signed rows, found {len(rows)}")
    if read_csv(one_path) != [row for row in rows if row["split"] == "B23.1-SMOKE-1"]:
        raise ValueError("one-image signed registry does not project from combined registry")
    if read_csv(four_path) != [row for row in rows if row["split"] == "B23.1-SMOKE-4"]:
        raise ValueError("four-image signed registry does not project from combined registry")

    by_split: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_split.setdefault(row["split"], []).append(row)
        if row["source_manifest_sha256"] != exposure_sha:
            raise ValueError("signed row carries the wrong exposure-manifest identity")
        if row["assigned_before_run"] != "true" or row["pre_b23_exposure_checked"] != "true":
            raise ValueError("signed row lacks preregistration assertions")
    for split, expected in EXPECTED_IMAGES.items():
        observed = tuple(row["image_id"] for row in by_split.get(split, []))
        if observed != expected:
            raise ValueError(f"{split} selection changed: {observed} != {expected}")

    eligible = [f"{value:05d}" for value in range(60000, 70000) if f"{value:05d}" not in exposed]
    if ranked_pick("B23.1A:replay:v1", eligible) != "65082":
        raise ValueError("replay selection is not reproducible")
    for index, ((low, high), expected) in enumerate(zip(STRATA, EXPECTED_IMAGES["B23.1-SMOKE-4"])):
        pool = [value for value in eligible if low <= int(value) < high and value != "65082"]
        if ranked_pick(f"B23.1A:smoke4:stratum{index}:v1", pool) != expected:
            raise ValueError(f"smoke stratum {index} selection is not reproducible")

    native_seed_records = []
    for index, row in enumerate(rows):
        expected_measurement = derive_seed(
            23100,
            stream_name="measurement_noise",
            image_id=row["image_id"],
            measurement_id=row["measurement_id"],
            parent_id="B23.1-REGISTRY",
        )
        expected_solver = derive_seed(
            23200,
            stream_name="native_start_noise",
            image_id=row["image_id"],
            measurement_id=row["measurement_id"],
            parent_id="B23.1-REGISTRY",
        )
        if int(row["measurement_seed"]) != expected_measurement:
            raise ValueError(f"row {index} measurement seed is not derived by b23-sha256-v1")
        if int(row["solver_base_seed"]) != expected_solver:
            raise ValueError(f"row {index} solver seed is not derived by b23-sha256-v1")
        adapted = []
        for parent in PARENTS:
            canonical_seed = derive_seed(
                int(row["solver_base_seed"]),
                stream_name="native_start_noise",
                image_id=row["image_id"],
                measurement_id=row["measurement_id"],
                parent_id=parent,
                branch_id="root",
                draw_index=0,
            )
            native_seed = canonical_seed % (2 ** 32)
            if not 0 <= native_seed <= (2 ** 32 - 1):
                raise ValueError("native seed adapter produced a value outside uint32")
            adapted.append(native_seed)
            native_seed_records.append({
                "split": row["split"],
                "row_id": int(row["row_id"]),
                "image_id": row["image_id"],
                "parent_id": parent,
                "canonical_parent_seed": canonical_seed,
                "native_entrypoint_seed": native_seed,
            })
        if len(set(adapted)) != len(PARENTS):
            raise ValueError(f"adapted native parent-seed collision in signed row {index}")

    head = git(repo, "rev-parse", "HEAD")
    branch = git(repo, "branch", "--show-current")
    dirty = git(repo, "status", "--porcelain=v1")
    if args.expected_head and head != args.expected_head:
        raise ValueError(f"head mismatch: observed={head} expected={args.expected_head}")
    result = {
        "schema_version": "b23.b23-1-prerun-validation.v1",
        "status": "PASS",
        "gpu_work_performed": False,
        "branch": branch,
        "head": head,
        "worktree_clean": dirty == "",
        "authorization": "B23.1A/B_ONLY",
        "unauthorized": sorted(forbidden),
        "parents": list(PARENTS),
        "native_repeats": execution["native_repeats"],
        "signed_rows": len(rows),
        "replay_image": "65082",
        "smoke_images": list(EXPECTED_IMAGES["B23.1-SMOKE-4"]),
        "exposed_images": len(exposed),
        "image_level_disjoint": not bool({row["image_id"] for row in rows} & exposed),
        "pre_b23_exposure_sha256": exposure_sha,
        "max_parent_trajectories": execution["max_parent_trajectories"],
        "native_seed_adapter": "canonical_parent_seed modulo 2**32",
        "native_seed_records": native_seed_records,
        "next_stop": config["stop_after"],
    }
    if args.pac:
        paths = read_json(repo / "configs/b23/pac_paths.yaml")
        source_checks = {
            "daps": (
                Path(paths["historical_checkout"]) / "external/daps",
                "e7a77d094167084faed19b599b96673b7bb11447",
                "fbb5b42369ecf0d3b9b67f8fc162053bc40ec32aed41dbd92a67e8d81dcfad69",
            ),
            "difffpr": (
                Path(paths["difffpr"]),
                "a45ffe58f18fed8a63d3446600424e2b08733524",
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            "official_sitcom": (
                Path(paths["official_sitcom"]),
                "275ab67efbd8146bffca20155171ba6be1169c09",
                "a9f0076d6f852b6898000142c19a09131ffc49ceba0e3d935cd465e85df26e6e",
            ),
        }
        observed_sources = {}
        for name, (path, expected_source_head, expected_diff) in source_checks.items():
            if not path.is_dir():
                raise FileNotFoundError(path)
            observed_head = git(path, "rev-parse", "HEAD")
            observed_diff = git_diff_sha256(path)
            if observed_head != expected_source_head or observed_diff != expected_diff:
                raise ValueError(
                    f"{name} source identity mismatch: head={observed_head} diff={observed_diff}"
                )
            observed_sources[name] = {
                "path": str(path), "head": observed_head, "tracked_diff_sha256": observed_diff
            }
        model = Path(paths["ffhq_model"])
        if sha256_file(model) != config["problem"]["model_sha256"]:
            raise ValueError("FFHQ checkpoint identity mismatch")
        sitcom_checkpoint = Path(paths["official_sitcom"]) / "checkpoint/ffhq256.pt"
        if sha256_file(sitcom_checkpoint) != config["problem"]["model_sha256"]:
            raise ValueError("SITCOM checkpoint identity mismatch")
        if not Path(paths["ffhq_data"]).is_dir():
            raise FileNotFoundError(paths["ffhq_data"])
        hardware_text = subprocess.check_output(
            [
                "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        hardware = [", ".join(part.strip() for part in line.split(",")) for line in hardware_text.splitlines() if line.strip()]
        expected_hardware = ["NVIDIA RTX PRO 6000 Blackwell Server Edition, 580.126.20, 97887"] * 4
        if hardware != expected_hardware:
            raise ValueError(f"PAC hardware identity changed: {hardware}")
        result["pac_identity"] = {
            "status": "PASS",
            "sources": observed_sources,
            "model_sha256": sha256_file(model),
            "sitcom_checkpoint_sha256": sha256_file(sitcom_checkpoint),
            "hardware": hardware,
        }
    if not result["worktree_clean"]:
        result["status"] = "PASS_WITH_EXPECTED_PRERUN_SOURCE_CHANGES"
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
