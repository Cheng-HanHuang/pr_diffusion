#!/usr/bin/env python3
"""Render deterministic B24 baseline-screen manifests without loading a model or GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prdiffusion.b24_protocol import (
    canonical_json_sha256,
    render_screen_manifest,
    shard_rows,
    validate_pre_b24,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exposure", type=Path, default=Path("manifests/b24/PRE_B24_EXPOSURE.csv"))
    parser.add_argument("--count", type=int, choices=(64, 256), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    exposed = validate_pre_b24(args.exposure.resolve())
    rows = render_screen_manifest(exposed_ids=exposed, count=args.count)
    shards = shard_rows(rows)
    value = {
        "schema_version": 1,
        "stage": "B24.2_FUTURE_BASELINE_SCREEN_NOT_AUTHORIZED_BY_RENDER",
        "scientific_execution_authorized": False,
        "count": len(rows),
        "rows": rows,
    }
    value["manifest_sha256"] = canonical_json_sha256(value)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "out": str(args.out.resolve()),
        "manifest_sha256": value["manifest_sha256"],
        "count": len(rows),
        "shard_counts": {str(k): len(v) for k, v in shards.items()},
        "gpu_work": False,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
