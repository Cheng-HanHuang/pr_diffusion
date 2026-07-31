#!/usr/bin/env python3
"""Checksum and package one compact B23.0 evidence capsule."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import tarfile
from pathlib import Path


REQUIRED = (
    "README.md", "EXECUTION_IDENTITY.json", "GATE_DECISION.json",
    "ARTIFACT_MANIFEST.tsv", "COMMANDS.sh", "STDOUT_TAIL.txt", "STDERR_TAIL.txt",
    "ZERO_GPU_STEP_RESULTS.tsv", "CORRECTION_LEDGER.json",
    "summaries/PAC_FREEZE.json", "summaries/EXPOSURE_COVERAGE.json",
    "summaries/PRE_B23_EXPOSURE.csv", "summaries/B23_0_CHECKPOINT_REPORT.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--max-uncompressed-mib", type=int, default=20)
    args = parser.parse_args()
    capsule = args.capsule.resolve()
    missing = [name for name in REQUIRED if not (capsule / name).is_file()]
    if missing:
        raise ValueError(f"capsule is missing required files: {missing}")

    files = sorted(
        path for path in capsule.rglob("*")
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    )
    if any(path.is_symlink() for path in capsule.rglob("*")):
        raise ValueError("capsules may not contain symbolic links")
    total = sum(path.stat().st_size for path in files)
    cap = args.max_uncompressed_mib * 1024 * 1024
    if total > cap:
        raise ValueError(f"capsule exceeds uncompressed cap: {total} > {cap}")
    if len(files) > 500:
        raise ValueError("capsule unexpectedly contains more than 500 files")

    checksum_path = capsule / "CHECKSUMS.sha256"
    checksum_path.write_text(
        "".join(
            f"{sha256_file(path)}  ./{path.relative_to(capsule).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    files.append(checksum_path)
    files.sort()

    archive = capsule.with_suffix(".tar.gz")
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for path in files:
                    relative = Path(capsule.name) / path.relative_to(capsule)
                    info = tar.gettarinfo(str(path), arcname=relative.as_posix())
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.mode = 0o755 if path.name == "COMMANDS.sh" else 0o644
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
    archive_hash = sha256_file(archive)
    sidecar = Path(str(archive) + ".sha256")
    sidecar.write_text(f"{archive_hash}  {archive.name}\n", encoding="utf-8")
    print(
        f"status=PACKAGED archive={archive} sha256={archive_hash} "
        f"bytes={archive.stat().st_size} files={len(files)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
