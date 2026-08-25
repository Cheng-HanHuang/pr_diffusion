#!/usr/bin/env python3
"""Commit transparent B23.0 evidence and a bounded approved capsule archive."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


BRANCH = "codex/b23-execution"
REQUIRED_ZERO_GPU_STEPS = (
    "unit_tests",
    "repository_validation",
    "b23_1_dry_render",
    "pac_evidence_collection",
)


def run(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command, cwd=cwd, check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    return completed.stdout.strip()


def git(repo: Path, *args: str) -> str:
    return run(["git", "-C", str(repo), *args])


def validate_archive(archive: Path, max_bytes: int) -> None:
    if archive.stat().st_size > max_bytes:
        raise ValueError(f"archive exceeds Git review cap: {archive.stat().st_size} > {max_bytes}")
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
    if len(members) > 500:
        raise ValueError("archive contains too many members")
    for member in members:
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise ValueError(f"unsafe archive member: {member.name}")


def validate_step_results(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != ("step", "status", "return_code"):
            raise ValueError(f"unexpected zero-GPU step-result columns: {reader.fieldnames}")
        rows = list(reader)
    observed = tuple(row["step"] for row in rows)
    if observed != REQUIRED_ZERO_GPU_STEPS:
        raise ValueError(
            f"zero-GPU step order mismatch: observed={observed} expected={REQUIRED_ZERO_GPU_STEPS}"
        )
    failed = [
        row for row in rows
        if row["status"] != "PASS" or row["return_code"] != "0"
    ]
    if failed:
        raise ValueError(f"refusing publication with failed zero-GPU steps: {failed}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--commit-archive", action="store_true")
    parser.add_argument("--max-archive-mib", type=int, default=5)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    capsule = args.capsule.resolve()
    archive = capsule.with_suffix(".tar.gz")
    sidecar = Path(str(archive) + ".sha256")
    if not args.commit_archive:
        raise ValueError("user-approved archive publication requires --commit-archive")
    if not archive.is_file() or not sidecar.is_file():
        raise ValueError("package and checksum sidecar must exist before publication")
    validate_step_results(capsule / "ZERO_GPU_STEP_RESULTS.tsv")
    validate_archive(archive, args.max_archive_mib * 1024 * 1024)
    decision = json.loads((capsule / "GATE_DECISION.json").read_text(encoding="utf-8"))
    if decision["verdict"] != "PASS_RECOMMEND_PLANNER_REVIEW":
        raise ValueError(f"refusing publication of failed evidence: {decision['verdict']}")
    if git(repo, "branch", "--show-current") != BRANCH:
        raise ValueError(f"publication requires branch {BRANCH}")
    if git(repo, "status", "--porcelain"):
        raise ValueError("execution worktree must be clean before evidence publication")
    pre_run_head = git(repo, "rev-parse", "HEAD")
    remote_lines = git(repo, "ls-remote", "--heads", "origin", f"refs/heads/{BRANCH}").splitlines()
    if len(remote_lines) != 1:
        raise ValueError(f"remote branch identity missing or ambiguous: {remote_lines}")
    remote_head = remote_lines[0].split()[0]
    if remote_head != pre_run_head:
        raise ValueError(f"remote execution head changed: remote={remote_head} local={pre_run_head}")
    if not git(repo, "config", "user.name") or not git(repo, "config", "user.email"):
        raise ValueError("git user.name and user.email must be configured")

    evidence_root = repo / "docs/b23/evidence"
    extracted_destination = evidence_root / capsule.name
    archive_root = evidence_root / "capsules"
    archive_destination = archive_root / archive.name
    if extracted_destination.exists() or archive_destination.exists():
        raise ValueError("timestamped evidence destination already exists")
    archive_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(capsule, extracted_destination)
    shutil.copy2(archive, archive_destination)
    shutil.copy2(sidecar, archive_root / sidecar.name)
    shutil.copy2(
        capsule / "summaries/PRE_B23_EXPOSURE.csv",
        repo / "manifests/b23/PRE_B23_EXPOSURE.csv",
    )
    shutil.copy2(
        capsule / "summaries/B23_0_CHECKPOINT_REPORT.md",
        repo / "docs/b23/B23_0_CHECKPOINT_REPORT.md",
    )

    relative_paths = [
        str(extracted_destination.relative_to(repo)),
        str(archive_destination.relative_to(repo)),
        str((archive_root / sidecar.name).relative_to(repo)),
        "manifests/b23/PRE_B23_EXPOSURE.csv",
        "docs/b23/B23_0_CHECKPOINT_REPORT.md",
    ]
    git(repo, "add", "--", *relative_paths)
    staged = git(repo, "diff", "--cached", "--name-only").splitlines()
    allowed_prefixes = (
        f"docs/b23/evidence/{capsule.name}/",
        "docs/b23/evidence/capsules/",
        "manifests/b23/PRE_B23_EXPOSURE.csv",
        "docs/b23/B23_0_CHECKPOINT_REPORT.md",
    )
    unexpected = [path for path in staged if not path.startswith(allowed_prefixes)]
    if unexpected:
        raise ValueError(f"unexpected staged paths: {unexpected}")
    git(repo, "commit", "-m", "B23.0 evidence: freeze PAC identities and exposure")
    evidence_head = git(repo, "rev-parse", "HEAD")
    if args.push:
        git(repo, "push", "origin", f"HEAD:refs/heads/{BRANCH}")
    result = {
        "status": "PUBLISHED" if args.push else "COMMITTED_NOT_PUSHED",
        "branch": BRANCH,
        "pre_run_commit": pre_run_head,
        "post_run_evidence_commit": evidence_head,
        "archive_repo_path": str(archive_destination),
        "archive_pac_path": str(archive),
        "archive_sha256_sidecar": str(sidecar),
        "remote_push_performed": bool(args.push),
    }
    result_path = capsule.parent / f"{capsule.name}_PUBLISH_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**result, "result_path": str(result_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
