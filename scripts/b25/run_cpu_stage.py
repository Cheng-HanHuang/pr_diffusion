#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import resource
import subprocess
import time
from pathlib import Path


def writej(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resource-json", type=Path, required=True)
    ap.add_argument("--max-rss-gib", type=float, required=True)
    ap.add_argument("--max-wall-seconds", type=float, required=True)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise RuntimeError("missing wrapped command")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("B25 CPU stage requires CUDA_VISIBLE_DEVICES=''")

    started = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(command, check=False, timeout=float(args.max_wall_seconds))
        returncode = int(proc.returncode)
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = 124
    wall = time.perf_counter() - started
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    max_rss_gib = float(usage.ru_maxrss) / (1024.0 * 1024.0)
    payload = {
        "schema_version": "b25.cpu-stage-resource.v1",
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": wall,
        "ru_maxrss_raw": usage.ru_maxrss,
        "ru_maxrss_note": "Linux ru_maxrss is KiB",
        "max_rss_gib": max_rss_gib,
        "user_cpu_seconds": float(usage.ru_utime),
        "system_cpu_seconds": float(usage.ru_stime),
        "max_rss_gib_limit": float(args.max_rss_gib),
        "max_wall_seconds_limit": float(args.max_wall_seconds),
        "resource_pass": (
            not timed_out
            and max_rss_gib <= args.max_rss_gib
            and wall <= args.max_wall_seconds + 1.0
        ),
        "gpu_work_performed": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    writej(args.resource_json, payload)
    if returncode != 0:
        print(json.dumps({"status": "FAILED_OR_TIMED_OUT", **payload}, sort_keys=True))
        return returncode
    if not payload["resource_pass"]:
        print(json.dumps({"status": "RESOURCE_LIMIT_EXCEEDED", **payload}, sort_keys=True))
        return 96
    print(json.dumps({"status": "PASS", **payload}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
