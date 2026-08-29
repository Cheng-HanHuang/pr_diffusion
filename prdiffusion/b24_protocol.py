"""B24 pure-stdlib protocol primitives.

This module intentionally performs no model load and imports neither torch nor CUDA.
B24.0 uses it only for deterministic registries, validation, sharding, and atomic
completion metadata.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

B24_BASE_HEAD = "27505e6328157ac9296c95dc5e611cbeef80de98"
PRE_B23_SHA256 = "a513cb4e3b79b39700ff1d623cb4b2eaf496bc2d6d0fe58bd963709e6a56d288"
PRE_B23_UNIQUE_IMAGES = 328
REQUIRED_B23_1_IDS = ("65082", "61492", "62959", "66821", "68142")

EXPOSURE_COLUMNS = (
    "image_id",
    "measurement_id",
    "dataset_split",
    "first_project_stage",
    "roles_seen",
    "ground_truth_inspected",
    "artifacts",
    "exclusion_reason",
    "source_evidence",
)

GPU_UUIDS = {
    0: "GPU-8c9c6250-7b65-20d8-5c81-d6cb618810c3",
    1: "GPU-883c037a-34d2-48c4-467f-9a352fd8fdff",
    2: "GPU-c381c0f4-1dbc-004f-7d3a-1d7f7794dffe",
    3: "GPU-7d65c050-d7e8-5a6b-ee38-1d72d7a5696a",
}
WORKER_HARD_CEILING_MIB = 52_452
WORKER_TARGET_MIB = 48_000
DEVICE_RESERVE_MIB = 4_096
MIN_FREE_BEFORE_LAUNCH_MIB = WORKER_TARGET_MIB + DEVICE_RESERVE_MIB


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _norm_image_id(value: str | int) -> str:
    text = str(value).strip()
    if not text.isdigit():
        raise ValueError(f"invalid numeric FFHQ image id: {value!r}")
    number = int(text)
    if number < 0 or number >= 70_000:
        raise ValueError(f"FFHQ image id out of range: {number}")
    return f"{number:05d}"


def _domain_hash(domain: str, *parts: str | int) -> bytes:
    material = "|".join([domain, B24_BASE_HEAD, *[str(p) for p in parts]])
    return hashlib.sha256(material.encode("utf-8")).digest()


def canonical_seed(domain: str, *parts: str | int) -> int:
    """Return a deterministic non-negative 63-bit seed."""
    return int.from_bytes(_domain_hash(domain, *parts)[:8], "big") & ((1 << 63) - 1)


def global_allocation(image_id: str | int) -> str:
    image = _norm_image_id(image_id)
    bucket = int.from_bytes(_domain_hash("B24_FFHQ_GLOBAL_V1", image)[:8], "big") % 100
    return "B24_SCREEN_ELIGIBLE" if bucket <= 79 else "FUTURE_RESERVE"


def screen_order_key(image_id: str | int) -> str:
    image = _norm_image_id(image_id)
    return _domain_hash("B24_SCREEN_ORDER_V1", image).hex()


def class_rank_key(class_label: str, image_id: str | int) -> str:
    label = class_label.strip().upper()
    if label not in {"A", "B", "C", "D"}:
        raise ValueError(f"invalid class: {class_label}")
    return _domain_hash("B24_CLASS_RANK_V1", label, _norm_image_id(image_id)).hex()


def a_audit_keep(image_id: str | int) -> bool:
    bucket = int.from_bytes(_domain_hash("B24_A_AUDIT_V1", _norm_image_id(image_id))[:8], "big") % 10
    return bucket == 0


def seed_row(image_id: str | int) -> dict[str, Any]:
    image = _norm_image_id(image_id)
    measurement = canonical_seed("B24_MEASUREMENT_SEED_V1", image)
    methods: dict[str, list[int]] = {}
    all_solver: list[int] = []
    for method in ("DAPS", "SITCOM"):
        values = [canonical_seed("B24_SOLVER_SEED_V1", image, method, rep) for rep in range(4)]
        if len(set(values)) != 4:
            raise RuntimeError(f"within-method seed collision for {image}/{method}")
        methods[method] = values
        all_solver.extend(values)
    if len(set(all_solver)) != 8:
        raise RuntimeError(f"cross-method solver seed collision for {image}")
    if measurement in set(all_solver):
        raise RuntimeError(f"measurement/solver seed collision for {image}")
    return {"image_id": image, "measurement_seed": measurement, "solver_seeds": methods}


def classify_good25(daps_best_psnr_db: float, sitcom_best_psnr_db: float) -> str:
    daps = float(daps_best_psnr_db) >= 25.0
    sitcom = float(sitcom_best_psnr_db) >= 25.0
    if daps and sitcom:
        return "A"
    if (not daps) and sitcom:
        return "B"
    if daps and (not sitcom):
        return "C"
    return "D"


def _read_exposure(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    return rows, columns


def validate_pre_b23(path: Path) -> set[str]:
    if sha256_file(path) != PRE_B23_SHA256:
        raise ValueError(f"PRE_B23 SHA-256 mismatch: {path}")
    rows, columns = _read_exposure(path)
    if columns != EXPOSURE_COLUMNS:
        raise ValueError(f"PRE_B23 columns mismatch: {columns}")
    images = {_norm_image_id(row["image_id"]) for row in rows if row.get("image_id", "").strip()}
    if len(images) != PRE_B23_UNIQUE_IMAGES:
        raise ValueError(f"PRE_B23 unique image count mismatch: {len(images)}")
    return images


def build_pre_b24(pre_b23: Path, out: Path) -> dict[str, Any]:
    inherited = validate_pre_b23(pre_b23)
    rows, columns = _read_exposure(pre_b23)
    assert columns == EXPOSURE_COLUMNS
    normalized: dict[str, dict[str, str]] = {}
    for row in rows:
        image = _norm_image_id(row["image_id"])
        row = dict(row)
        row["image_id"] = image
        normalized[image] = row

    evidence = "docs/b23/evidence/B23_1_closeout_return_20260825T194434Z/"
    for raw in REQUIRED_B23_1_IDS:
        image = _norm_image_id(raw)
        if image not in normalized:
            normalized[image] = {
                "image_id": image,
                "measurement_id": "UNKNOWN_ALL_MEASUREMENTS",
                "dataset_split": "PRE_B24_EXCLUDED",
                "first_project_stage": "B23.1",
                "roles_seen": "B23.1 replay/smoke exposure",
                "ground_truth_inspected": "TRUE",
                "artifacts": evidence,
                "exclusion_reason": "B23.1 image-wide exposure; all measurements excluded",
                "source_evidence": evidence,
            }

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPOSURE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for image in sorted(normalized, key=int):
            writer.writerow(normalized[image])

    images = validate_pre_b24(out)
    return {
        "inherited_unique_images": len(inherited),
        "pre_b24_unique_images": len(images),
        "sha256": sha256_file(out),
        "path": str(out),
    }


def validate_pre_b24(path: Path) -> set[str]:
    rows, columns = _read_exposure(path)
    if columns != EXPOSURE_COLUMNS:
        raise ValueError(f"PRE_B24 columns mismatch: {columns}")
    images = [_norm_image_id(row["image_id"]) for row in rows if row.get("image_id", "").strip()]
    unique = set(images)
    if len(images) != len(unique):
        raise ValueError("PRE_B24 contains duplicate image rows")
    if len(unique) < 333:
        raise ValueError(f"PRE_B24 must contain at least 333 unique images, got {len(unique)}")
    missing = {_norm_image_id(x) for x in REQUIRED_B23_1_IDS} - unique
    if missing:
        raise ValueError(f"PRE_B24 missing B23.1 exposures: {sorted(missing)}")
    return unique


def render_screen_manifest(
    *, exposed_ids: Iterable[str | int], count: int, image_ids: Iterable[str | int] = range(70_000)
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    exposed = {_norm_image_id(x) for x in exposed_ids}
    eligible = []
    for raw in image_ids:
        image = _norm_image_id(raw)
        if image in exposed:
            continue
        allocation = global_allocation(image)
        if allocation == "B24_SCREEN_ELIGIBLE":
            eligible.append((screen_order_key(image), image))
    eligible.sort()
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} eligible images for requested count={count}")
    rows = []
    for position, (rank, image) in enumerate(eligible[:count]):
        seeds = seed_row(image)
        rows.append(
            {
                "row_index": position,
                "image_id": image,
                "allocation": "B24_SCREEN_ELIGIBLE",
                "screen_rank_sha256": rank,
                "measurement_seed": seeds["measurement_seed"],
                "daps_solver_seeds": seeds["solver_seeds"]["DAPS"],
                "sitcom_solver_seeds": seeds["solver_seeds"]["SITCOM"],
                "shard_id": position % 4,
                "gpu_id": position % 4,
                "gpu_uuid": GPU_UUIDS[position % 4],
            }
        )
    return rows


def shard_rows(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[Mapping[str, Any]]]:
    shards: dict[int, list[Mapping[str, Any]]] = {i: [] for i in range(4)}
    seen = set()
    for expected_position, row in enumerate(rows):
        position = int(row["row_index"])
        if position != expected_position:
            raise ValueError(f"non-contiguous row_index at {expected_position}: {position}")
        image = _norm_image_id(row["image_id"])
        if image in seen:
            raise ValueError(f"duplicate image in run manifest: {image}")
        seen.add(image)
        shard = position % 4
        if int(row["shard_id"]) != shard or int(row["gpu_id"]) != shard:
            raise ValueError(f"shard/gpu mismatch at row {position}")
        if row["gpu_uuid"] != GPU_UUIDS[shard]:
            raise ValueError(f"GPU UUID mismatch at row {position}")
        shards[shard].append(row)
    if sum(len(v) for v in shards.values()) != len(rows):
        raise RuntimeError("shard union mismatch")
    return shards


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    try:
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass


def completion_is_reusable(path: Path, expected_identity: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "COMPLETE":
        return False
    identity = value.get("identity")
    return isinstance(identity, dict) and canonical_json_sha256(identity) == canonical_json_sha256(dict(expected_identity))
