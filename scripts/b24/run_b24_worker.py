#!/usr/bin/env python3
"""B24 worker control plane.

B24.0 supports only --dry-run. Scientific backend dispatch remains deliberately
disabled until a separately authorized B24.1 commit wires and validates the
terminal-only DAPS/SITCOM implementations.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from prdiffusion.b24_protocol import GPU_UUIDS, MIN_FREE_BEFORE_LAUNCH_MIB, shard_rows


def _query_gpu(gpu_id: int) -> tuple[str, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=uuid,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    uuid, free = [part.strip() for part in output.split(",", 1)]
    return uuid, int(free)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=range(4), required=True)
    parser.add_argument("--gpu", type=int, choices=range(4), required=True)
    parser.add_argument("--mode", choices=("serial", "batched"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("B24.0", "B24.1", "B24.2"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    value = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise SystemExit("STOP: manifest rows missing")
    shards = shard_rows(rows)
    if args.gpu != args.shard:
        raise SystemExit("STOP: B24 uses fixed shard_id == physical gpu_id")
    expected_uuid = GPU_UUIDS[args.gpu]
    selected = list(shards[args.shard])

    rendered = {
        "stage": args.stage,
        "mode": args.mode,
        "dry_run": args.dry_run,
        "shard": args.shard,
        "gpu_id": args.gpu,
        "gpu_uuid": expected_uuid,
        "row_count": len(selected),
        "image_ids": [row["image_id"] for row in selected],
        "output_root": str(args.output_root),
        "scientific_backend_enabled": False,
    }

    if args.dry_run:
        print(json.dumps(rendered, sort_keys=True))
        return 0

    if args.stage == "B24.0":
        raise SystemExit("STOP: B24.0 is zero-GPU; non-dry worker execution is forbidden")

    observed_uuid, free_mib = _query_gpu(args.gpu)
    if observed_uuid != expected_uuid:
        raise SystemExit(f"STOP: GPU UUID mismatch for physical id {args.gpu}: {observed_uuid}")
    if free_mib < MIN_FREE_BEFORE_LAUNCH_MIB:
        raise SystemExit(
            f"STOP: GPU {args.gpu} free={free_mib} MiB < required pre-launch {MIN_FREE_BEFORE_LAUNCH_MIB} MiB"
        )

    raise SystemExit(
        "STOP: scientific backend dispatch is intentionally disabled in the B24.0 head; "
        "a separately authorized B24.1 commit must wire serial/batched terminal-only backends "
        "and pass equivalence/memory gates before B24.2 baseline screening"
    )


if __name__ == "__main__":
    raise SystemExit(main())
