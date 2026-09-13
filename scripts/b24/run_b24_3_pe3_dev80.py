#!/usr/bin/env python3
"""Run exactly the two frozen PE3 arms on one existing DEV80 measurement.

This wrapper reuses the tested generic DEV80 NP execution path and changes only
its installed arm registry/spec identity.  It generates no measurement and
accepts only frozen DEVELOPMENT rows.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEV80_PATH = REPO / "scripts" / "b24" / "run_b24_3_dev80_np.py"
PE3_PATH = REPO / "scripts" / "b24" / "run_b24_3_pe3_refinement.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


dev = load_module("b24_3_dev80_np_pe3_host", DEV80_PATH)
pe3 = load_module("b24_3_pe3_impl_dev80", PE3_PATH)
pe3.install_pe3(dev.base)

ARM_ORDER = ("NP_PE3_SCORE", "NP_PE3_RANDOM")
dev.ARM_ORDER = ARM_ORDER
dev.base.ARM_ORDER = ARM_ORDER


def main() -> int:
    return dev.main()


if __name__ == "__main__":
    raise SystemExit(main())
