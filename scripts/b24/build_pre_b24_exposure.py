#!/usr/bin/env python3
"""Build PRE_B24_EXPOSURE.csv from the populated, hash-frozen PAC B23 registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prdiffusion.b24_protocol import build_pre_b24


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pre-b23",
        type=Path,
        default=Path("/egr/research-pac/huang248/pr_diffusion_b23/manifests/b23/PRE_B23_EXPOSURE.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("manifests/b24/PRE_B24_EXPOSURE.csv"),
    )
    args = parser.parse_args()
    result = build_pre_b24(args.pre_b23.resolve(), args.out.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
