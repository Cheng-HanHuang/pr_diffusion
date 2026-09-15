#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_stream(handle) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
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
    manifest: dict[str, str] = {}
    with sums.open("w", encoding="utf-8") as f:
        for path in files:
            rel = path.relative_to(run)
            if rel.is_absolute() or ".." in rel.parts:
                raise RuntimeError(f"unsafe relative path: {rel}")
            digest = sha256_file(path)
            manifest[rel.as_posix()] = digest
            f.write(f"{digest}  {rel.as_posix()}\n")
    files.append(sums)

    with tarfile.open(archive, "w:gz") as tf:
        for path in files:
            rel = path.relative_to(run)
            arcname = Path(run.name) / rel
            tf.add(path, arcname=arcname.as_posix(), recursive=False)

    verified = 0
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        if not members:
            raise RuntimeError("empty B25 archive")
        by_name = {m.name: m for m in members}
        for member in members:
            p = Path(member.name)
            if p.is_absolute() or ".." in p.parts:
                raise RuntimeError(f"unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"link member forbidden: {member.name}")
            if not member.isfile():
                raise RuntimeError(f"non-file archive member forbidden: {member.name}")
        for rel, expected in manifest.items():
            name = (Path(run.name) / rel).as_posix()
            member = by_name.get(name)
            if member is None:
                raise RuntimeError(f"archive missing checksummed member: {name}")
            handle = tf.extractfile(member)
            if handle is None:
                raise RuntimeError(f"cannot read archive member: {name}")
            observed = sha256_stream(handle)
            if observed != expected:
                raise RuntimeError(f"internal checksum mismatch: {name}: {observed} != {expected}")
            verified += 1
        sums_name = (Path(run.name) / "SHA256SUMS.txt").as_posix()
        if sums_name not in by_name:
            raise RuntimeError("archive missing SHA256SUMS.txt")

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
        "internal_checksums_verified": verified,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
