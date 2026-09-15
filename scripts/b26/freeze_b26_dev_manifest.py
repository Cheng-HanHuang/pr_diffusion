#!/usr/bin/env python3
"""Freeze the exact B26 DEV80/native-NP provenance from accepted B24 artifacts.

This is a zero-GPU manifest builder.  It reads confirmation registry rows only
for the DEV/confirmation disjointness proof and never follows confirmation
payload paths.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCREEN_SHA = "b516c8154cbbb790d8a3592b86736bb0d4bd47d0833d85ecf3d6a9d710e950ba"
ROLE_CSV_SHA = "7c05dd67cc39263db74e58b7255297d014e7d674da0c0bebdee317bcbff809bf"
DEV80_MANIFEST_SHA = "3945c2ed26e10766a64ab4e7fb64ea673fa2f478fbf080ea8d1d2c85584f6d5f"
MODEL_SHA = "81d535743156ec6be34d8668e6920da94f0614074d7793a16c8fa9e306237faa"
RANK_DOMAIN = "B26_DEV16_RANK_V1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def readj(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def writej_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def rank_key(label: str, image_id: str) -> str:
    material = "|".join((RANK_DOMAIN, SCREEN_SHA, label, image_id))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def one_input_manifest(task: dict[str, Any], task_root: Path) -> Path:
    source = task.get("source_input_manifest")
    if source:
        p = Path(source).resolve()
        if p.is_file():
            return p
    direct = task_root / "input" / "input_manifest.json"
    if direct.is_file():
        return direct.resolve()
    candidates = sorted(task_root.glob("*/input_manifest.json")) + sorted(task_root.glob("*/*/input_manifest.json"))
    candidates = [p.resolve() for p in candidates if p.is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"cannot uniquely resolve input manifest under {task_root}: {candidates}")
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role-dir", type=Path, required=True)
    ap.add_argument("--dev80-run", type=Path, required=True)
    ap.add_argument("--closeout", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("manifest freeze requires CUDA_VISIBLE_DEVICES='' exactly")

    role_dir = args.role_dir.resolve()
    devrun = args.dev80_run.resolve()
    closeout = args.closeout.resolve()
    for d in (role_dir, devrun, closeout):
        if not d.is_dir():
            raise FileNotFoundError(d)

    role_csv = role_dir / "B24_METHOD_IMAGE_ROLES.csv"
    dev_manifest = devrun / "DEV80_MANIFEST.json"
    hard_subset = closeout / "HARD_SUBSET.csv"
    for p in (role_csv, dev_manifest, hard_subset, args.model.resolve()):
        if not p.is_file():
            raise FileNotFoundError(p)
    if sha256_file(role_csv) != ROLE_CSV_SHA:
        raise RuntimeError("role CSV SHA drift")
    if sha256_file(dev_manifest) != DEV80_MANIFEST_SHA:
        raise RuntimeError("DEV80 manifest SHA drift")
    if sha256_file(args.model.resolve()) != MODEL_SHA:
        raise RuntimeError("model SHA drift")

    with role_csv.open(newline="", encoding="utf-8") as f:
        roles = [dict(r) for r in csv.DictReader(f)]
    dev_roles = {str(r["image_id"]).zfill(5): r for r in roles if str(r.get("method_role", "")).upper() == "DEVELOPMENT"}
    conf_ids = {str(r["image_id"]).zfill(5) for r in roles if str(r.get("method_role", "")).upper() == "CONFIRMATION"}
    if len(dev_roles) != 80 or len(conf_ids) != 305 or set(dev_roles) & conf_ids:
        raise RuntimeError(f"role split/disjointness drift: DEV={len(dev_roles)} CONF={len(conf_ids)} overlap={len(set(dev_roles)&conf_ids)}")
    counts = {c: sum(str(r["class_label"]) == c for r in dev_roles.values()) for c in "ABCD"}
    if counts != {"A": 20, "B": 20, "C": 20, "D": 20}:
        raise RuntimeError(f"DEV strata drift: {counts}")

    with hard_subset.open(newline="", encoding="utf-8") as f:
        hard_rows = [dict(r) for r in csv.DictReader(f)]
    shared_ids = {str(r.get("image_id", "")).zfill(5) for r in hard_rows if r.get("image_id")}
    if len(shared_ids) != 10 or not shared_ids <= set(dev_roles):
        raise RuntimeError(f"hard subset drift: {len(shared_ids)}")

    task_files = sorted(devrun.glob("assignments/gpu*/task*.json"))
    if len(task_files) != 80:
        raise RuntimeError(f"expected 80 B24 task JSONs, got {len(task_files)}")
    tasks: dict[str, tuple[dict[str, Any], Path]] = {}
    for p in task_files:
        t = readj(p)
        iid = str(t["image_id"]).zfill(5)
        if iid in tasks:
            raise RuntimeError(f"duplicate task {iid}")
        tasks[iid] = (t, p.resolve())
    if set(tasks) != set(dev_roles):
        raise RuntimeError("task/DEV80 identity mismatch")

    rows: list[dict[str, Any]] = []
    for iid in sorted(dev_roles):
        role = dev_roles[iid]
        task, task_path = tasks[iid]
        if str(task.get("class_label")) != str(role["class_label"]):
            raise RuntimeError(f"stratum mismatch {iid}")
        if str(task.get("method_role", "")).upper() != "DEVELOPMENT":
            raise RuntimeError(f"non-DEV task {iid}")
        task_root = devrun / "workers" / f"gpu{int(task['assigned_gpu'])}" / str(task["output_subdir"])
        complete_path = task_root / "IMAGE_COMPLETE.json"
        if not complete_path.is_file():
            raise FileNotFoundError(complete_path)
        complete = readj(complete_path)
        if complete.get("status") != "PASS" or str(complete.get("image_id", "")).zfill(5) != iid:
            raise RuntimeError(f"bad IMAGE_COMPLETE {iid}")
        if bool(complete.get("confirmation_exposed", True)):
            raise RuntimeError(f"confirmation exposure flag {iid}")
        np_result_path = Path(complete["np_results"]["NP4_INDEPENDENT"]).resolve()
        np_result = readj(np_result_path)
        if np_result.get("status") != "PASS" or np_result.get("arm") != "NP4_INDEPENDENT":
            raise RuntimeError(f"bad accepted NP4 result {iid}")
        if int(np_result.get("total_unet_evals", -1)) != 8800 or len(np_result.get("terminals", [])) != 4:
            raise RuntimeError(f"NP4 work/terminal drift {iid}")

        inp_path = one_input_manifest(task, task_root)
        inp = readj(inp_path)
        if str(inp.get("image_id", "")).zfill(5) != iid:
            raise RuntimeError(f"input identity drift {iid}")
        if int(inp["measurement_seed"]) != int(role["dev_measurement_seed"]):
            raise RuntimeError(f"measurement seed drift {iid}")
        meas = Path(inp["measurement_path"]).resolve()
        gt = Path(inp["ground_truth_tensor_path"]).resolve()
        if not meas.is_file() or not gt.is_file():
            raise FileNotFoundError(f"missing locked payload {iid}")
        if sha256_file(meas) != inp["measurement_file_sha256"]:
            raise RuntimeError(f"measurement file SHA drift {iid}")

        roots = [int(role[f"np_root_seed_{j}"]) for j in range(4)]
        if len(set(roots)) != 4:
            raise RuntimeError(f"root collision {iid}")
        terminals = []
        for j, trow in enumerate(np_result["terminals"]):
            if int(trow["terminal_index"]) != j:
                raise RuntimeError(f"terminal order drift {iid}/{j}")
            terminals.append({
                "root_index": j,
                "root_seed": roots[j],
                "reconstruction_sha256": trow["reconstruction_sha256"],
                "selector_plus_mean": float(trow["selector_post_winner_lf_mse_mean"]),
                "rng_sha256": trow.get("rng_sha256"),
                "selected_noise_sha256": trow.get("selected_noise_sha256"),
                "accepted_psnr_raw_db": float(trow["psnr_raw_db"]),
                "accepted_good25": bool(trow["good25"]),
            })
        rows.append({
            "image_id": iid,
            "screening_stratum": str(role["class_label"]),
            "method_role": "DEVELOPMENT",
            "pilot16": str(role.get("pilot16", "")).upper() == "TRUE",
            "dev_measurement_seed": int(role["dev_measurement_seed"]),
            "measurement_path": str(meas),
            "measurement_file_sha256": inp["measurement_file_sha256"],
            "measurement_tensor_sha256": inp["measurement_tensor_sha256"],
            "ground_truth_tensor_path": str(gt),
            "ground_truth_tensor_sha256": inp["ground_truth_tensor_sha256"],
            "input_manifest_path": str(inp_path),
            "input_manifest_sha256": sha256_file(inp_path),
            "b24_task_path": str(task_path),
            "b24_image_complete_path": str(complete_path.resolve()),
            "accepted_np4_result_path": str(np_result_path),
            "accepted_np4_result_sha256": sha256_file(np_result_path),
            "np_roots": terminals,
            "shared_failure_10": iid in shared_ids,
            "fresh_daps4_best_psnr_raw_rgb_db": complete.get("fresh_daps4_best_psnr_raw_rgb_db"),
            "fresh_sitcom4_best_psnr_raw_rgb_db": complete.get("fresh_sitcom4_best_psnr_raw_rgb_db"),
            "fresh2_context": "available only through accepted corrected B24 closeout; no B26 rerun",
            "rank_key": rank_key(str(role["class_label"]), iid),
        })

    rows.sort(key=lambda r: (r["screening_stratum"], r["rank_key"], r["image_id"]))
    dev16: list[str] = []
    smoke: list[str] = []
    for c in "ABCD":
        group = [r for r in rows if r["screening_stratum"] == c]
        if len(group) != 20:
            raise RuntimeError(c)
        chosen = group[:4]
        dev16.extend(r["image_id"] for r in chosen)
        smoke.append(chosen[0]["image_id"])
        for rank, r in enumerate(group):
            r["stratum_hash_rank"] = rank
            r["dev16"] = rank < 4
            r["smoke_image"] = rank == 0

    payload = {
        "schema_version": "b26.dev80-manifest.v1",
        "status": "PASS",
        "source_screen_manifest_sha256": SCREEN_SHA,
        "source_role_csv": str(role_csv.resolve()),
        "source_role_csv_sha256": ROLE_CSV_SHA,
        "source_dev80_manifest": str(dev_manifest.resolve()),
        "source_dev80_manifest_sha256": DEV80_MANIFEST_SHA,
        "source_corrected_closeout": str(closeout),
        "development_count": 80,
        "confirmation_registry_count": 305,
        "development_confirmation_overlap": 0,
        "confirmation_payloads_accessed": False,
        "screening_strata": counts,
        "dev16_rank_domain": RANK_DOMAIN,
        "dev16_image_ids": dev16,
        "smoke_image_ids": smoke,
        "shared_failure_10_ids": sorted(shared_ids),
        "rows": rows,
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    writej_atomic(args.output.resolve(), payload)
    print(json.dumps({
        "status": "PASS", "output": str(args.output.resolve()), "development": 80,
        "confirmation_registry": 305, "dev16": dev16, "smoke": smoke,
        "manifest_sha256": sha256_file(args.output.resolve()), "confirmation_payloads_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
