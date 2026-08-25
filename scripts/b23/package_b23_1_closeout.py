#!/usr/bin/env python3
"""Create a deterministic compact B23.1A/B closeout archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import tarfile
from pathlib import Path


REQUIRED = (
    "README.md", "EXECUTION_IDENTITY.json", "GATE_DECISION.json",
    "ARTIFACT_MANIFEST.tsv", "COMMANDS.sh", "STDOUT_TAIL.txt", "STDERR_TAIL.txt",
    "ZERO_GPU_STEP_RESULTS.tsv", "CORRECTION_LEDGER.json",
    "summaries/RUN_SUMMARY.json", "summaries/REPLAY_SUMMARY.json",
    "summaries/COMPUTE_SUMMARY.json", "summaries/DONOR_COMPATIBILITY.json",
    "summaries/B23_1_CHECKPOINT_REPORT.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(capsule: Path, max_bytes: int = 5 * 1024 * 1024) -> tuple[Path, str]:
    capsule = capsule.resolve()
    missing = [relative for relative in REQUIRED if not (capsule / relative).is_file()]
    if missing:
        raise ValueError(f"compact capsule is missing required files: {missing}")
    paths = list(capsule.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("compact capsule may not contain symbolic links")
    files = sorted(path for path in paths if path.is_file() and path.name != "CHECKSUMS.sha256")
    if len(files) > 100 or sum(path.stat().st_size for path in files) > max_bytes:
        raise ValueError("compact capsule exceeds its review bound")
    checksum = capsule / "CHECKSUMS.sha256"
    checksum.write_text("".join(f"{sha256_file(path)}  ./{path.relative_to(capsule).as_posix()}\n" for path in files), encoding="utf-8")
    files = sorted([*files, checksum])
    archive = capsule.with_suffix(".tar.gz")
    if archive.exists() or Path(str(archive) + ".sha256").exists():
        raise FileExistsError("refusing to overwrite compact closeout transport")
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for path in files:
                    arcname = Path(capsule.name) / path.relative_to(capsule)
                    info = tar.gettarinfo(str(path), arcname=arcname.as_posix())
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o755 if path.name == "COMMANDS.sh" else 0o644
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
    digest = sha256_file(archive)
    Path(str(archive) + ".sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return archive, digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsule", type=Path, required=True)
    args = parser.parse_args()
    archive, digest = package(args.capsule)
    print(f"status=PACKAGED archive={archive} sha256={digest} bytes={archive.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
