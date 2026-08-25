#!/usr/bin/env python3
"""Commit and non-force-push compact B23.1A/B evidence, not its transport archive."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


BRANCH = "codex/b23-execution"
STEPS = ("unit_tests", "repository_validation", "accepted_evidence_validation")


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def validate_steps(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != ("step", "status", "return_code"):
            raise ValueError("invalid zero-GPU step ledger header")
        rows = list(reader)
    if tuple(row["step"] for row in rows) != STEPS or any(row["status"] != "PASS" or row["return_code"] != "0" for row in rows):
        raise ValueError("compact evidence publication requires exactly three PASS/0 steps")


def validate_archive(archive: Path) -> None:
    if archive.stat().st_size > 5 * 1024 * 1024:
        raise ValueError("compact archive exceeds 5 MiB")
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
    if len(members) > 100:
        raise ValueError("compact archive has too many members")
    for member in members:
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise ValueError(f"unsafe compact archive member: {member.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    repo, capsule = args.repo.resolve(), args.capsule.resolve()
    archive = capsule.with_suffix(".tar.gz")
    sidecar = Path(str(archive) + ".sha256")
    validate_steps(capsule / "ZERO_GPU_STEP_RESULTS.tsv")
    validate_archive(archive)
    decision = json.loads((capsule / "GATE_DECISION.json").read_text(encoding="utf-8"))
    if decision.get("verdict") != "PASS_RECOMMEND_B23_1A_B_SIGNOFF_ONLY" or decision.get("cross_family_h0") != "FAIL" or decision.get("qualified_np_sitcom_adapters") != 0 or decision.get("b23_2_authorized") is not False:
        raise ValueError("compact gate decision does not preserve the failed H0 and closed B23.2 boundary")
    if git(repo, "branch", "--show-current") != BRANCH or git(repo, "status", "--porcelain"):
        raise ValueError("publication requires the clean B23 execution branch")
    pre_run = git(repo, "rev-parse", "HEAD")
    remote = git(repo, "ls-remote", "--heads", "origin", f"refs/heads/{BRANCH}").splitlines()
    if len(remote) != 1 or remote[0].split()[0] != pre_run:
        raise ValueError("remote execution head changed before publication")
    destination = repo / "docs/b23/evidence" / capsule.name
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(capsule, destination)
    shutil.copy2(capsule / "summaries/B23_1_CHECKPOINT_REPORT.md", repo / "docs/b23/B23_1_CHECKPOINT_REPORT.md")
    git(repo, "add", "--", str(destination.relative_to(repo)), "docs/b23/B23_1_CHECKPOINT_REPORT.md")
    staged = git(repo, "diff", "--cached", "--name-only").splitlines()
    allowed = (f"docs/b23/evidence/{capsule.name}/", "docs/b23/B23_1_CHECKPOINT_REPORT.md")
    if any(not path.startswith(allowed) for path in staged):
        raise ValueError(f"unexpected staged paths: {staged}")
    git(repo, "commit", "-m", "B23.1 evidence: close replay and donor compatibility")
    evidence_head = git(repo, "rev-parse", "HEAD")
    if args.push:
        git(repo, "push", "origin", f"HEAD:refs/heads/{BRANCH}")
    result = {
        "status": "PUBLISHED" if args.push else "COMMITTED_NOT_PUSHED", "branch": BRANCH,
        "pre_run_commit": pre_run, "post_run_evidence_commit": evidence_head,
        "compact_capsule_pac_path": str(capsule), "compact_archive_pac_path": str(archive),
        "compact_archive_sha256_sidecar": str(sidecar), "archive_committed_to_git": False,
        "gpu_work_performed_during_correction": False, "cross_family_h0": "FAIL",
        "qualified_np_sitcom_adapters": 0, "b23_2_authorized": False,
        "remote_push_performed": bool(args.push),
    }
    result_path = capsule.parent / f"{capsule.name}_PUBLISH_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**result, "result_path": str(result_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
