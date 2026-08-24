#!/usr/bin/env python3
"""Create a transportable B23.1 summary capsule while leaving large raw traces on PAC."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    run_root = args.run_root.resolve()
    capsule = args.capsule.resolve()
    archive = Path(str(capsule) + ".tar.gz")
    sidecar = Path(str(archive) + ".sha256")
    if capsule.exists() or archive.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite B23.1 return artifacts")
    capsule.mkdir(parents=True)

    for relative in (
        "configs/b23/b23_1a_b_execution.yaml",
        "manifests/b23/b23_1_signed_registry.csv",
        "manifests/b23/b23_1_one_image_smoke.signed.csv",
        "manifests/b23/b23_1_four_image_smoke.signed.csv",
        "manifests/b23/PRE_B23_EXPOSURE.csv",
    ):
        copy(repo / relative, capsule / relative)

    artifact_index = []
    for source in sorted(run_root.rglob("*.json")):
        relative = source.relative_to(run_root)
        copy(source, capsule / "run" / relative)
    for source in sorted(run_root.rglob("RUN.json")):
        run = json.loads(source.read_text(encoding="utf-8"))
        reconstruction = Path(run["reconstruction_path"])
        if not reconstruction.is_file():
            raise FileNotFoundError(reconstruction)
        relative_run = source.parent.relative_to(run_root)
        destination = capsule / "terminal_artifacts" / relative_run / (
            "reconstruction" + reconstruction.suffix.lower()
        )
        copy(reconstruction, destination)
        artifact_index.append(
            {
                "parent_id": run["parent_id"],
                "mode": run["mode"],
                "image_id": run["image_id"],
                "pac_path": str(reconstruction),
                "capsule_path": destination.relative_to(capsule).as_posix(),
                "bytes": reconstruction.stat().st_size,
                "sha256": sha256(reconstruction),
            }
        )
    (capsule / "TERMINAL_ARTIFACT_INDEX.json").write_text(
        json.dumps({"schema_version": "b23.terminal-artifact-index.v1", "artifacts": artifact_index}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksums = []
    for path in sorted(capsule.rglob("*")):
        if path.is_file() and path.name != "CHECKSUMS.sha256":
            checksums.append(f"{sha256(path)}  {path.relative_to(capsule).as_posix()}")
    (capsule / "CHECKSUMS.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(capsule, arcname=capsule.name, recursive=True)
    sidecar.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS", "capsule": str(capsule), "archive": str(archive),
        "sha256": sha256(archive), "terminal_artifacts": len(artifact_index),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
