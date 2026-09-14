#!/usr/bin/env python3
"""Recover the interrupted B24.2 GPU-2 256->2048 handoff only.

This supervisor is intentionally narrow. It:
1. removes only exact B24-owned DAPS scratch artifacts for incomplete shard-2 rows;
2. resumes the existing cumulative-256 shard 2 in-process with the calibrated
   baseline admission gate;
3. verifies the parent shard reaches PASS 48/48; and
4. launches only cumulative-2048 shard 2 in the already-existing 2048 run.

GPUs/shards 0, 1, and 3 are never launched, stopped, or modified here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_DEFAULT = Path("/egr/research-pac/huang248/pr_diffusion_b24")
DAPS = Path("/egr/research-pac/huang248/pr_diffusion_b19_solver/external/daps")
PY = Path("/egr/research-pac/huang248/conda-envs/prdiff_ffhq/bin/python")
GPU = 2
SAFE_NAME = re.compile(r"^[a-z0-9._-]+$")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def completion_exists(root: Path, row_index: int, image_id: str, width: int) -> bool:
    return (root / f"row{row_index:0{width}d}_{image_id}/IMAGE_COMPLETE.json").is_file()


def cleanup_daps_scratch(data_name: str) -> None:
    """Delete only an exact B24-owned temporary DAPS dataset/config pair."""
    if not SAFE_NAME.fullmatch(data_name):
        raise RuntimeError(f"unsafe DAPS scratch name: {data_name!r}")
    if not (data_name.startswith("b24-256-") or data_name.startswith("b24-2048-")):
        raise RuntimeError(f"refusing non-B24 DAPS scratch cleanup: {data_name}")

    dataset_root = (DAPS / "dataset").resolve()
    config_root = (DAPS / "configs/data").resolve()
    data_dir = DAPS / "dataset" / data_name
    config_path = DAPS / "configs/data" / f"{data_name}.yaml"

    # Parent containment checks are deliberately explicit before any deletion.
    if data_dir.parent.resolve() != dataset_root:
        raise RuntimeError(f"unsafe dataset path: {data_dir}")
    if config_path.parent.resolve() != config_root:
        raise RuntimeError(f"unsafe config path: {config_path}")

    removed = []
    if config_path.exists() or config_path.is_symlink():
        config_path.unlink()
        removed.append(str(config_path))
    if data_dir.exists() or data_dir.is_symlink():
        if data_dir.is_symlink():
            data_dir.unlink()
        else:
            shutil.rmtree(data_dir)
        removed.append(str(data_dir))
    if removed:
        print(f"STALE_DAPS_SCRATCH_REMOVED|name={data_name}|paths={'|'.join(removed)}", flush=True)


def clean_incomplete_parent_rows(parent256: Path) -> int:
    manifest = read_json(parent256 / "B24_2_baseline_256.json")
    rows = [
        r for r in manifest["rows"]
        if int(r["gpu_id"]) == GPU and int(r["row_index"]) >= 64
    ]
    if len(rows) != 48:
        raise RuntimeError(f"expected 48 parent shard-2 rows, got {len(rows)}")
    incomplete = 0
    shard_root = parent256 / "shard2"
    for row in rows:
        idx = int(row["row_index"])
        image_id = str(row["image_id"]).zfill(5)
        if completion_exists(shard_root, idx, image_id, 3):
            continue
        incomplete += 1
        data_name = f"b24-256-{parent256.name}-s2-r{idx:03d}-{image_id}".lower()
        cleanup_daps_scratch(data_name)
    print(f"PARENT256_INCOMPLETE_ROWS|shard=2|count={incomplete}", flush=True)
    return incomplete


def clean_incomplete_2048_rows(run2048: Path) -> int:
    manifest = read_json(run2048 / "B24_2_baseline_2048.json")
    rows = [
        r for r in manifest["rows"]
        if int(r["gpu_id"]) == GPU and int(r["row_index"]) >= 256
    ]
    if len(rows) != 448:
        raise RuntimeError(f"expected 448 2048 shard-2 rows, got {len(rows)}")
    shard_root = run2048 / "shard2"
    incomplete = 0
    for row in rows:
        idx = int(row["row_index"])
        image_id = str(row["image_id"]).zfill(5)
        if completion_exists(shard_root, idx, image_id, 4):
            continue
        incomplete += 1
        data_name = f"b24-2048-{run2048.name}-s2-r{idx:04d}-{image_id}".lower()
        cleanup_daps_scratch(data_name)
    print(f"B24_2048_INCOMPLETE_ROWS|shard=2|count={incomplete}", flush=True)
    return incomplete


def run_parent256(repo: Path, parent256: Path, gate_mib: int) -> None:
    summary = parent256 / "shard2/SHARD_COMPLETE.json"
    if summary.is_file():
        value = read_json(summary)
        if value.get("status") == "PASS" and int(value.get("completed", -1)) == 48:
            print("PARENT256_ALREADY_COMPLETE|shard=2|completed=48", flush=True)
            return
        raise RuntimeError(f"bad existing parent shard summary: {summary}")

    clean_incomplete_parent_rows(parent256)
    launch = read_json(parent256 / "LAUNCH.json")
    parent64 = Path(launch["parent64_runroot"]).resolve() / "B24_2_baseline_64.json"
    manifest256 = parent256 / "B24_2_baseline_256.json"
    worker = load_module(
        repo / "scripts/b24/run_b24_2_256_extension_shard.py",
        "b24_parent256_gpu2_recovery",
    )

    # The 256 worker predates the calibrated baseline gate. Override only this
    # recovery instance, including the B24.1 helper it imports at runtime.
    worker.MIN_FREE_BEFORE_LAUNCH_MIB = gate_mib
    original_loader = worker.load_module

    def calibrated_loader(path: Path, name: str):
        module = original_loader(path, name)
        if name == "b24_smoke_reuse_256":
            module.MIN_FREE_MIB = gate_mib
        return module

    worker.load_module = calibrated_loader
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(repo / "scripts/b24/run_b24_2_256_extension_shard.py"),
            "--manifest", str(manifest256),
            "--parent-manifest", str(parent64),
            "--shard", "2",
            "--gpu", "2",
            "--repo", str(repo),
            "--output-root", str(parent256 / "shard2"),
            "--resume",
        ]
        rc = worker.main()
    finally:
        sys.argv = old_argv
    if rc != 0:
        raise RuntimeError(f"parent-256 shard-2 recovery returned rc={rc}")

    value = read_json(summary)
    if value.get("status") != "PASS" or int(value.get("completed", -1)) != 48:
        raise RuntimeError(f"parent-256 shard 2 did not close PASS 48/48: {value}")
    print("PARENT256_RECOVERY_PASS|shard=2|completed=48", flush=True)


def launch_2048_shard2(repo: Path, parent256: Path, run2048: Path, gate_mib: int) -> int:
    launch = read_json(run2048 / "LAUNCH.json")
    if Path(launch["parent256_runroot"]).resolve() != parent256.resolve():
        raise RuntimeError("2048 launch parent does not match recovered parent256 run")
    manifest = Path(launch["manifest_path"]).resolve()
    if manifest != (run2048 / "B24_2_baseline_2048.json").resolve():
        raise RuntimeError(f"unexpected 2048 manifest path: {manifest}")

    clean_incomplete_2048_rows(run2048)
    shard_root = run2048 / "shard2"
    command = [
        str(PY), str(repo / "scripts/b24/run_b24_2_2048_extension_shard.py"),
        "--manifest", str(manifest),
        "--parent-runroot", str(parent256),
        "--shard", "2", "--gpu", "2", "--repo", str(repo),
        "--output-root", str(shard_root),
    ]
    if shard_root.exists():
        command.append("--resume")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TORCH_HOME"] = "/egr/research-pac/huang248/models/torch_cache"
    env["B24_BASELINE_MIN_FREE_MIB"] = str(gate_mib)

    log_path = run2048 / "logs/gpu2.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        cwd=repo,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    (run2048 / "pids").mkdir(exist_ok=True)
    (run2048 / "pids/gpu2.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
    print(
        f"GPU2_2048_LAUNCHED|pid={proc.pid}|runroot={run2048}|gate_mib={gate_mib}|resume={shard_root.exists()}",
        flush=True,
    )
    return proc.pid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent256", type=Path, required=True)
    ap.add_argument("--run2048", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    ap.add_argument("--gate-mib", type=int, default=10240)
    args = ap.parse_args()
    if args.gate_mib < 8192:
        raise RuntimeError(f"refusing gate below 8192 MiB: {args.gate_mib}")

    repo = args.repo.resolve()
    parent256 = args.parent256.resolve()
    run2048 = args.run2048.resolve()
    if not parent256.is_dir() or not run2048.is_dir():
        raise FileNotFoundError((parent256, run2048))

    run_parent256(repo, parent256, args.gate_mib)
    launch_2048_shard2(repo, parent256, run2048, args.gate_mib)
    print("GPU2_HANDOFF_COMPLETE|parent256=PASS|2048_shard2=LAUNCHED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
