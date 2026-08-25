#!/usr/bin/env python3
"""Create a transportable B23.1 summary capsule while leaving large raw traces on PAC."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


EXPECTED_FINAL_STEPS = (
    "prerun_validate",
    "reuse_inputs_validate",
    "replay_fresh1_native_0_recovered",
    "replay_lf_v1_native_0",
    "replay_np_1_native_0",
    "replay_sitcom_1_native_0",
    "replay_fresh1_native_1",
    "replay_lf_v1_native_1",
    "replay_np_1_native_1",
    "replay_sitcom_1_native_1",
    "replay_fresh1_native_2",
    "replay_lf_v1_native_2",
    "replay_np_1_native_2",
    "replay_sitcom_1_native_2",
    "freeze_fresh1",
    "replay_fresh1_wrapper",
    "analyze_fresh1",
    "freeze_lf_v1",
    "replay_lf_v1_wrapper",
    "analyze_lf_v1",
    "freeze_np_1",
    "replay_np_1_wrapper",
    "analyze_np_1",
    "freeze_sitcom_1",
    "replay_sitcom_1_wrapper",
    "analyze_sitcom_1",
    "compute_microbench",
    "four_image_smoke",
    "donor_classification",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def validate_final_status(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"final prerequisite ledger is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["step", "status", "rc"]:
            raise ValueError(f"unexpected FINAL_STATUS.tsv header: {reader.fieldnames}")
        rows = list(reader)
    observed = tuple(row["step"] for row in rows)
    if observed != EXPECTED_FINAL_STEPS:
        raise ValueError(
            "FINAL_STATUS.tsv prerequisite sequence mismatch: "
            f"observed={observed} expected={EXPECTED_FINAL_STEPS}"
        )
    failed = [row for row in rows if row["status"] != "PASS" or row["rc"] != "0"]
    if failed:
        raise ValueError(f"FINAL_STATUS.tsv contains non-PASS prerequisites: {failed}")
    return rows


def git_identity(repo: Path) -> dict[str, object]:
    branch = subprocess.run(
        ["git", "-C", str(repo), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    if branch != "codex/b23-execution" or dirty:
        raise ValueError(f"unsafe packaging repository identity: branch={branch} dirty={dirty}")
    return {"branch": branch, "head": head, "worktree_clean": True}


def transport_recovered_evidence(
    *, repo: Path, run_root: Path, capsule: Path
) -> dict[str, object]:
    config = json.loads((repo / "configs/b23/b23_1a_b_execution.yaml").read_text())
    recovery = config["execution"]
    expected_source = Path(recovery["required_recovery_source"]).resolve()
    expected = recovery["recovery_identifiers"]
    accepted_run_path = run_root / "replay/fresh1/native_0/RUN.json"
    accepted_run = json.loads(accepted_run_path.read_text(encoding="utf-8"))
    source = Path(accepted_run.get("recovered_from_partial_run", "")).resolve()
    if source != expected_source:
        raise ValueError(f"recovery source mismatch: observed={source} expected={expected_source}")
    if accepted_run.get("gpu_work_performed_in_current_run") is not False:
        raise ValueError("Fresh1 native_0 is not marked as an evidence-only recovery")

    required = {
        "input/input_manifest.json": expected["input_manifest_sha256"],
        "cuda_timing.json": expected["timing_sha256"],
        "native_trace.json": expected["trace_sha256"],
    }
    transported = []
    destination_root = capsule / "run/replay/fresh1/native_0"
    for relative, expected_sha in required.items():
        source_path = source / relative
        observed_sha = sha256(source_path)
        if observed_sha != expected_sha:
            raise ValueError(
                f"recovery artifact mismatch for {relative}: "
                f"observed={observed_sha} expected={expected_sha}"
            )
        destination = destination_root / relative
        copy(source_path, destination)
        transported.append(
            {
                "source_pac_path": str(source_path),
                "capsule_path": destination.relative_to(capsule).as_posix(),
                "bytes": source_path.stat().st_size,
                "sha256": observed_sha,
            }
        )

    for relative in ("frozen_runtime_config.json", "parent.log"):
        source_path = source / relative
        if source_path.is_file():
            destination = destination_root / relative
            copy(source_path, destination)
            transported.append(
                {
                    "source_pac_path": str(source_path),
                    "capsule_path": destination.relative_to(capsule).as_posix(),
                    "bytes": source_path.stat().st_size,
                    "sha256": sha256(source_path),
                }
            )

    raw_trace = Path(accepted_run["raw_trace_path"])
    if raw_trace.stat().st_size != expected["raw_trajectory_bytes"]:
        raise ValueError("recovered raw trajectory size changed before packaging")
    raw_sha = sha256(raw_trace)
    if raw_sha != accepted_run["raw_trace_sha256"]:
        raise ValueError("recovered raw trajectory hash changed before packaging")
    return {
        "schema_version": "b23.recovery-artifact-index.v1",
        "gpu_work_performed_during_recovery_packaging": False,
        "transported_artifacts": transported,
        "external_large_artifact": {
            "reason": "943 MB raw trajectory intentionally remains on PAC",
            "pac_path": str(raw_trace),
            "bytes": raw_trace.stat().st_size,
            "sha256": raw_sha,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--supersedes-archive", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    run_root = args.run_root.resolve()
    capsule = args.capsule.resolve()
    archive = Path(str(capsule) + ".tar.gz")
    sidecar = Path(str(archive) + ".sha256")
    if capsule.exists() or archive.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite B23.1 return artifacts")
    identity = git_identity(repo)
    final_rows = validate_final_status(run_root / "FINAL_STATUS.tsv")
    capsule.mkdir(parents=True)

    for relative in (
        "configs/b23/b23_1a_b_execution.yaml",
        "manifests/b23/b23_1_signed_registry.csv",
        "manifests/b23/b23_1_one_image_smoke.signed.csv",
        "manifests/b23/b23_1_four_image_smoke.signed.csv",
        "manifests/b23/b23_1_correction_ledger.json",
        "manifests/b23/PRE_B23_EXPOSURE.csv",
    ):
        copy(repo / relative, capsule / relative)

    artifact_index = []
    for source in sorted(run_root.rglob("*.json")):
        relative = source.relative_to(run_root)
        copy(source, capsule / "run" / relative)
    copy(run_root / "FINAL_STATUS.tsv", capsule / "run/FINAL_STATUS.tsv")
    recovery_index = transport_recovered_evidence(
        repo=repo, run_root=run_root, capsule=capsule
    )
    (capsule / "RECOVERY_ARTIFACT_INDEX.json").write_text(
        json.dumps(recovery_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    run_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(run_root.rglob("RUN.json"))
    ]
    if len(run_records) != 32 or any(run.get("status") != "PASS" for run in run_records):
        raise ValueError("packaging requires exactly 32 PASS parent trajectories")
    execution_heads = sorted({run["acceptance_repo_head"] for run in run_records})
    if len(execution_heads) != 1:
        raise ValueError(f"multiple acceptance repository heads: {execution_heads}")
    superseded = None
    if args.supersedes_archive is not None:
        superseded_path = args.supersedes_archive.resolve()
        if not superseded_path.is_file():
            raise FileNotFoundError(superseded_path)
        superseded = {
            "path": str(superseded_path),
            "bytes": superseded_path.stat().st_size,
            "sha256": sha256(superseded_path),
        }
    (capsule / "PACKAGING_IDENTITY.json").write_text(
        json.dumps(
            {
                "schema_version": "b23.packaging-identity.v1",
                "status": "PASS",
                "execution_acceptance_repo_head": execution_heads[0],
                "packaging_repository": identity,
                "final_prerequisite_rows": len(final_rows),
                "gpu_work_performed_during_packaging": False,
                "superseded_archive": superseded,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
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
