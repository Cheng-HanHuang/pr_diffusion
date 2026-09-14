#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    run = args.run.resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    complete = run / "WORKER_COMPLETE.json"
    if not complete.is_file():
        raise FileNotFoundError(complete)
    value = json.loads(complete.read_text(encoding="utf-8"))
    if value.get("status") != "PASS":
        raise RuntimeError("refusing to package non-PASS B25 run")

    archive = run.with_suffix(".tar.gz")
    sidecar = Path(str(archive) + ".sha256")
    if archive.exists() or sidecar.exists():
        raise FileExistsError(f"archive or sidecar already exists: {archive}")

    files = []
    for path in sorted(run.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"refusing symlink in B25 capsule: {path}")
        if path.is_file() and path.name != "SHA256SUMS.txt":
            files.append(path)
    sums = run / "SHA256SUMS.txt"
    with sums.open("w", encoding="utf-8") as f:
        for path in files:
            rel = path.relative_to(run)
            if rel.is_absolute() or ".." in rel.parts:
                raise RuntimeError(f"unsafe relative path: {rel}")
            f.write(f"{sha256_file(path)}  {rel.as_posix()}\n")
    files.append(sums)

    with tarfile.open(archive, "w:gz") as tf:
        for path in files:
            rel = path.relative_to(run)
            arcname = Path(run.name) / rel
            tf.add(path, arcname=arcname.as_posix(), recursive=False)

    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        if not members:
            raise RuntimeError("empty B25 archive")
        for member in members:
            p = Path(member.name)
            if p.is_absolute() or ".." in p.parts:
                raise RuntimeError(f"unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"link member forbidden: {member.name}")

    digest = sha256_file(archive)
    sidecar.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    if sha256_file(archive) != digest:
        raise RuntimeError("archive digest changed after sidecar write")
    print(json.dumps({
        "status": "PASS",
        "run": str(run),
        "archive": str(archive),
        "archive_sha256": digest,
        "sidecar": str(sidecar),
        "file_count": len(files),
        "archive_member_count": len(members),
        "archive_safe": True,
        "internal_checksums": str(sums),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
