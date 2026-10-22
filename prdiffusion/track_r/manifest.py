"""R0 candidate-manifest contract and fail-closed source adapters.

Only exposed-source paths are accepted. The PAC inventory determines exact nested
native output layouts before any source-specific record is emitted.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

SOURCES = ("S1_b22", "S2_b24_dev80", "S3_a12", "S4_a15", "S5_r0_fast")
REQUIRED = ("uid","source","family","image_id","run_index","run_label",
            "candidate_path","candidate_format","gt_path","gt_format",
            "measurement_path","measurement_sha256",
            "recorded_psnr_db","recorded_psnr_convention",
            "native_selector_name","native_selector_value",
            "recorded_cost_seconds","recorded_unet_forwards","notes")
FORBIDDEN = ("confirmation", "confirm305")


def assert_no_forbidden(path):
    raw = str(path)
    if any(x in raw.lower() for x in FORBIDDEN):
        raise RuntimeError("STOP_PROTECTED_COHORT_PATH")
    if raw and (raw.startswith("/") or raw.startswith(".")):
        resolved = str(Path(raw).resolve(strict=False))
        if any(x in resolved.lower() for x in FORBIDDEN):
            raise RuntimeError("STOP_PROTECTED_COHORT_SYMLINK")
    return path


def uid_for(source, family, image_id, run_label):
    if source not in SOURCES:
        raise ValueError("unknown source")
    pieces = [source, family, str(image_id), run_label]
    if any(not s or "|" in s for s in pieces):
        raise ValueError("UID components must be nonempty and pipe-free")
    return "|".join(pieces)


def validate_row(row):
    missing = sorted(set(REQUIRED)-set(row))
    if missing:
        raise ValueError("missing fields: " + ",".join(missing))
    if row["source"] not in SOURCES:
        raise ValueError("unknown source")
    if str(row["uid"]) != uid_for(row["source"], str(row["family"]),
                                  str(row["image_id"]), str(row["run_label"])):
        raise ValueError("UID mismatch")
    if not isinstance(row["run_index"], int) or row["run_index"] < 0:
        raise ValueError("invalid run order")
    for name in ("candidate_path","gt_path","measurement_path"):
        if row[name] is not None:
            assert_no_forbidden(row[name])
    for name in ("candidate_path", "measurement_path"):
        if not isinstance(row[name],str) or not row[name].startswith("/"):
            raise ValueError("missing absolute " + name)
    if row["candidate_format"] not in ("pt","png") or row["gt_format"] not in ("pt","png"):
        raise ValueError("bad image format")
    for key in ("recorded_psnr_db","native_selector_value","recorded_cost_seconds"):
        if row[key] not in (None,""):
            v = float(row[key])
            if not math.isfinite(v):
                raise ValueError("nonfinite " + key)
    if row["measurement_sha256"] and (len(row["measurement_sha256"]) != 64):
        raise ValueError("measurement hash malformed")
    return dict(row)


def write_manifest(rows, path):
    """Atomic create; never overwrite an existing manifest."""
    assert_no_forbidden(path)
    if Path(path).exists():
        raise FileExistsError(path)
    uids = set()
    with Path(path).open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED)
        writer.writeheader()
        for row in rows:
            r = validate_row(row)
            if r["uid"] in uids:
                raise ValueError("duplicate candidate UID")
            uids.add(r["uid"])
            writer.writerow({k:r[k] for k in REQUIRED})


def read_manifest(path):
    assert_no_forbidden(path)
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("recorded_psnr_db","native_selector_value",
                  "recorded_cost_seconds","recorded_unet_forwards"):
            r[k] = None if r[k] == "" else float(r[k])
        r["run_index"] = int(r["run_index"])
        validate_row(r)
    return rows


def _source_index(paths, key):
    """Access only the explicit exposed indices named by the planner."""
    p = paths["sources"][key]
    if key == "S2_b24_dev80":
        return p["manifest_in_repo"]
    if key == "S3_a12":
        return p["index_in_repo"]
    return None


def load_source_records(source, paths, verified_inventory, repo_root):
    """Fail closed until R0.0 resolves source layout and known SHA mismatches.

    The returned inventory is the sole authority for whether existing records
    are safe to parse. Candidate-materialization is deferred to R0.2, after
    its exact filename mapping has been reconciled with returned PAC samples.
    """
    if source not in SOURCES:
        raise ValueError("unknown source")
    for item in verified_inventory.get("checks", {}).values():
        if isinstance(item,dict) and item.get("status")=="MISMATCH":
            raise RuntimeError("STOP_R0_0_MISMATCH")
    target = _source_index(paths, source)
    if target:
        assert_no_forbidden(target)
        path = Path(repo_root) / target
        if not path.is_file():
            raise RuntimeError("STOP_SOURCE_INDEX_MISSING")
        return json.loads(path.read_text())
    raise RuntimeError("STOP_SOURCE_LAYOUT_PENDING_R0_0 " + source)
