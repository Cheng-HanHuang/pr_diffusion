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
    # Frozen B24 HISTORICAL_DIFF is a digest of the FULL tracked diff,
    # including historical tracked pyc modifications. It must NEVER be
    # compared to the current source-only diff digest.
    source_patch = read_git(root, "diff", "--", "posterior_sample.py")
    expected_b22_patch = (
        "diff --git a/posterior_sample.py b/posterior_sample.py\n"
        "index 788d4cb..cf91b11 100644\n"
        "--- a/posterior_sample.py\n"
        "+++ b/posterior_sample.py\n"
        "@@ -41,7 +41,7 @@ def safe_dir(dir):\n"
        "         get (or create) a directory\n"
        '     """\n'
        "     if not Path(dir).exists():\n"
        "-        Path(dir).mkdir()\n"
        "+        Path(dir).mkdir(parents=True, exist_ok=True)\n"
        "     return Path(dir)\n"
    ).encode("utf-8")
    result = {
        "stage": "A0.0_source_reconciliation",
        "head": head,
        "historical_head": HISTORICAL_HEAD,
        "historical_full_diff_sha256": HISTORICAL_DIFF,
        "current_full_diff_sha256": digest(full),
        "current_source_only_diff_sha256": digest(scoped),
        "current_posterior_sample_diff_sha256": digest(source_patch),
        "b22_archived_posterior_sample_diff_sha256": digest(expected_b22_patch),
        "b22_archived_source_patch_equal": source_patch == expected_b22_patch,
        "tracked_changed_files": source_names,
        "pycache_changed_count": sum(pycache(p) for p in tracked),
        "tracked_source_file_sha256": source_hashes,
        "head_matches": head == HISTORICAL_HEAD,
        "full_diff_matches_b24": digest(full) == HISTORICAL_DIFF,
        "no_unexpected_source_changes": source_names == ["posterior_sample.py"],
        "bounded_source_patch": source_patch.decode("utf-8", "replace")
                                 if len(source_patch) <= 12000 else None,
        "not_comparable": "historical full tracked diff SHA versus current source-only SHA",
    }
    result["status"] = ("MATCH_B24_FULL_DIFF"
                        if result["head_matches"] and result["full_diff_matches_b24"]
                        and result["no_unexpected_source_changes"]
                        else "AWAITING_B24_HISTORICAL_SOURCE_PATCH_RECONCILIATION")
    return result

if __name__ == "__main__":
    # Use no output file: shell may redirect to an EXCLUSIVE A0 path if desired.
    try:
        print(json.dumps(inspect(), sort_keys=True, indent=2))
    except Exception as error:
        print(json.dumps({"status":"STOP_SOURCE_READ_FAILED",
                          "error":str(error)},sort_keys=True))
        raise SystemExit(2)
