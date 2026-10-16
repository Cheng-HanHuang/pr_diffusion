#!/usr/bin/env python3
"""A0 forensic SITCOM source diff identity check (stdout JSON only, never writes).

Runs only git read commands with optional Git locks disabled. Does not import,
execute or rewrite historical Python, model checkpoints or measurements.
"""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
from pathlib import Path

SITCOM = Path("/egr/research-pac/huang248/external/SITCOM_ODE")
HISTORICAL_HEAD = "275ab67efbd8146bffca20155171ba6be1169c09"
HISTORICAL_DIFF = "a9f0076d6f852b6898000142c19a09131ffc49ceba0e3d935cd465e85df26e6e"

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def pycache(path: str) -> bool:
    return "__pycache__" in Path(path).parts and path.endswith(".pyc")

def source_only_names(names: list[str]) -> list[str]:
    return [x for x in names if not pycache(x)]

def read_git(root: Path, *args: str) -> bytes:
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    run = subprocess.run(["git", "-C", str(root), *args],
                         check=False, env=env, capture_output=True, timeout=40)
    if run.returncode or len(run.stdout) > 10_000_000:
        raise RuntimeError("git read failed: " + " ".join(args)
                           + " " + run.stderr.decode("utf-8", "replace")[:300])
    return run.stdout

def inspect(root: Path = SITCOM) -> dict:
    if not root.is_dir():
        raise RuntimeError("SITCOM root does not exist")
    head = read_git(root, "rev-parse", "HEAD").decode().strip()
    full = read_git(root, "diff")
    tracked = read_git(root, "diff", "--name-only").decode().splitlines()
    scoped = read_git(root, "diff", "--",
                      ":(exclude)**/__pycache__/**")
    source_names = source_only_names(tracked)
    source_hashes = {}
    for name in source_names:
        # Only known tracked source files, no image/model/dataset/payload enumeration.
        if not (name.endswith(".py") or name.endswith(".yaml")
                or name.endswith(".json") or name.endswith(".sh")):
            raise RuntimeError("unexpected modified tracked non-source file: " + name)
        p = root / name
        if not p.is_file() or p.stat().st_size > 1_000_000:
            raise RuntimeError("unavailable/oversized tracked source: " + name)
        source_hashes[name] = digest(p.read_bytes())
    result = {
        "stage": "A0.0_source_reconciliation",
        "head": head,
        "historical_head": HISTORICAL_HEAD,
        "historical_diff_sha256": HISTORICAL_DIFF,
        "full_diff_sha256": digest(full),
        "source_only_diff_sha256": digest(scoped),
        "tracked_changed_files": source_names,
        "pycache_changed_count": sum(pycache(p) for p in tracked),
        "tracked_source_file_sha256": source_hashes,
        "head_matches": head == HISTORICAL_HEAD,
        "source_only_diff_matches": digest(scoped) == HISTORICAL_DIFF,
        "full_diff_matches": digest(full) == HISTORICAL_DIFF,
        "no_unexpected_source_changes": source_names == ["posterior_sample.py"],
    }
    result["status"] = ("SOURCE_ONLY_RECONCILED_BUT_BYTECODE_DRIFT"
                        if result["head_matches"] and result["source_only_diff_matches"]
                        and result["no_unexpected_source_changes"]
                        else "STOP_SOURCE_IDENTITY_NOT_RECONCILED")
    return result

if __name__ == "__main__":
    # Use no output file: shell may redirect to an EXCLUSIVE A0 path if desired.
    try:
        print(json.dumps(inspect(), sort_keys=True, indent=2))
    except Exception as error:
        print(json.dumps({"status":"STOP_SOURCE_READ_FAILED",
                          "error":str(error)},sort_keys=True))
        raise SystemExit(2)
