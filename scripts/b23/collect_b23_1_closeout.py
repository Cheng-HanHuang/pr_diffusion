#!/usr/bin/env python3
"""Build compact B23.1A/B closeout evidence from the accepted immutable capsule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from prdiffusion.b23_protocol import validate_compute_ledger, validate_replay_report


PARENTS = ("Fresh1", "LF-v1", "NP-1", "SITCOM-1")
SCIENTIFIC_HEAD = "3ffb237818e1bfa4921b3f4f8bc9a3bd24b7e406"
PACKAGING_HEAD = "fad055d40d5bd0eaf4c9471359177c321958d2d7"
SOURCE_CAPSULE = "B23_1_return_20260825T184922Z"
SOURCE_ARCHIVE_SHA256 = "5731e6b0c20be940ae8a1e8b1326b3668111f2291550280bfeb0013408257469"
EXPECTED_FINAL_STEPS = (
    "prerun_validate", "reuse_inputs_validate", "replay_fresh1_native_0_recovered",
    "replay_lf_v1_native_0", "replay_np_1_native_0", "replay_sitcom_1_native_0",
    "replay_fresh1_native_1", "replay_lf_v1_native_1", "replay_np_1_native_1",
    "replay_sitcom_1_native_1", "replay_fresh1_native_2", "replay_lf_v1_native_2",
    "replay_np_1_native_2", "replay_sitcom_1_native_2", "freeze_fresh1",
    "replay_fresh1_wrapper", "analyze_fresh1", "freeze_lf_v1",
    "replay_lf_v1_wrapper", "analyze_lf_v1", "freeze_np_1",
    "replay_np_1_wrapper", "analyze_np_1", "freeze_sitcom_1",
    "replay_sitcom_1_wrapper", "analyze_sitcom_1", "compute_microbench",
    "four_image_smoke", "donor_classification",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_internal_checksums(source: Path) -> int:
    checksum_file = source / "CHECKSUMS.sha256"
    rows = checksum_file.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    for row in rows:
        expected, relative = row.split("  ", 1)
        relative = relative.removeprefix("./")
        target = source / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or relative in seen:
            raise ValueError(f"unsafe or duplicate checksum path: {relative}")
        if not target.is_file() or target.is_symlink() or sha256_file(target) != expected:
            raise ValueError(f"source capsule checksum mismatch: {relative}")
        seen.add(relative)
    actual = {
        path.relative_to(source).as_posix() for path in source.rglob("*")
        if path.is_file() and path.name != "CHECKSUMS.sha256"
    }
    if seen != actual:
        raise ValueError(f"source checksum coverage mismatch: missing={sorted(actual-seen)} extra={sorted(seen-actual)}")
    return len(rows)


def validate_final_status(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["step", "status", "rc"]:
            raise ValueError(f"invalid final status header: {reader.fieldnames}")
        rows = list(reader)
    if tuple(row["step"] for row in rows) != EXPECTED_FINAL_STEPS:
        raise ValueError("accepted scientific prerequisite sequence changed")
    if any(row["status"] != "PASS" or row["rc"] != "0" for row in rows):
        raise ValueError("accepted scientific prerequisite ledger contains a failure")
    return len(rows)


def summarize_runs(source: Path) -> tuple[dict, list[dict]]:
    records = [read_json(path) for path in sorted((source / "run").rglob("RUN.json"))]
    if len(records) != 32:
        raise ValueError(f"expected 32 accepted trajectories, found {len(records)}")
    if any(record.get("status") != "PASS" for record in records):
        raise ValueError("every accepted trajectory must be PASS")
    if {record.get("acceptance_repo_head") for record in records} != {SCIENTIFIC_HEAD}:
        raise ValueError("trajectory acceptance head changed")
    if any(record.get("b23_2_authorized") is not False or record.get("adaptive_schedule_used") is not False for record in records):
        raise ValueError("accepted trajectories crossed the B23.2/adaptive boundary")
    if any(record.get("terminal_candidates") != 1 for record in records):
        raise ValueError("accepted trajectory has a non-singleton terminal candidate set")
    modes = Counter(str(record["mode"]) for record in records)
    expected_modes = {"native": 12, "wrapper": 4, "smoke": 16}
    if dict(modes) != expected_modes:
        raise ValueError(f"trajectory mode counts changed: {dict(modes)}")
    parents = Counter(str(record["parent_id"]) for record in records)
    if dict(parents) != {parent: 8 for parent in PARENTS}:
        raise ValueError(f"parent trajectory counts changed: {dict(parents)}")
    current_gpu = sum(record.get("gpu_work_performed_in_current_run") is True for record in records)
    recovered = sum(record.get("gpu_work_performed_in_current_run") is False for record in records)
    if (current_gpu, recovered) != (31, 1):
        raise ValueError(f"recovery accounting changed: current={current_gpu} recovered={recovered}")
    images = sorted({str(record["image_id"]) for record in records})
    summary = {
        "schema_version": "b23.run-summary.v1", "status": "PASS",
        "total_trajectories": 32, "mode_counts": expected_modes,
        "parent_counts": {parent: parents[parent] for parent in PARENTS},
        "physical_trajectories_in_accepted_run": current_gpu,
        "recovered_accepted_trajectories": recovered, "terminal_candidates_per_run": 1,
        "images": images, "b23_2_authorized": False, "adaptive_schedules_used": False,
    }
    return summary, records


def summarize_replay(source: Path) -> dict:
    rows = []
    for parent, directory in zip(PARENTS, ("fresh1", "lf_v1", "np_1", "sitcom_1")):
        report_path = source / f"run/replay/{directory}/REPLAY_REPORT.json"
        report = read_json(report_path)
        validate_replay_report(report)
        if report.get("parent_id") != parent or report.get("verdict") != "PASS" or report.get("eligibility") != "BITWISE":
            raise ValueError(f"{parent} replay is not BITWISE/PASS")
        if len(set(report.get("native_run_ids", []))) != 3:
            raise ValueError(f"{parent} replay lacks three unique native runs")
        comparison = report["wrapper_comparison"]
        numeric = ["max_abs_err", "mean_abs_err", "relative_l2_err", "trace_max_abs_err", "measurement_loss_delta", "raw_psnr_delta"]
        if not comparison.get("tensor_hash_equal") or any(comparison[name] != 0 for name in numeric):
            raise ValueError(f"{parent} BITWISE comparison is nonzero")
        rows.append({
            "parent_id": parent, "eligibility": "BITWISE", "verdict": "PASS",
            "native_runs": 3, "wrapper_runs": 1, "all_declared_deltas_zero": True,
            "report_sha256": sha256_file(report_path),
        })
    return {"schema_version": "b23.replay-summary.v1", "status": "PASS", "reports": rows}


def summarize_compute(source: Path) -> dict:
    rows = []
    hardware = read_json(source / "run/compute/HARDWARE_IDENTITY.json")
    expected_inventory = "e0f7f472ad4c31e9858d23889ec1c4acb82d71ba5e3bcf7955599c398df4bf94"
    if hardware.get("inventory_sha256") != expected_inventory:
        raise ValueError("calibration hardware identity changed")
    for parent, file_id in zip(PARENTS, ("Fresh1", "LF_v1", "NP_1", "SITCOM_1")):
        path = source / f"run/compute/COMPUTE_LEDGER_{file_id}.json"
        ledger = read_json(path)
        validate_compute_ledger(ledger)
        fre = ledger["fre"]
        if fre.get("status") != "CALIBRATED" or ledger.get("parent_or_policy_id") != parent:
            raise ValueError(f"invalid calibrated compute ledger for {parent}")
        rows.append({
            "parent_id": parent, "work_FRE": fre["work_FRE"], "time_FRE": fre["time_FRE"],
            "claim_FRE": fre["claim_FRE"], "gpu_active_seconds": ledger["timing"]["gpu_active_seconds"],
            "wall_seconds": ledger["timing"]["wall_seconds"], "ledger_sha256": sha256_file(path),
        })
    return {
        "schema_version": "b23.compute-summary.v1", "status": "PASS",
        "hardware_identity": hardware, "ledgers": rows,
    }


def validate_donor(source: Path) -> dict:
    donor = read_json(source / "run/DONOR_COMPATIBILITY.json")
    qualified = donor.get("cross_family_adapter_qualified_donor_count")
    if donor.get("status") != "PASS" or qualified != 0:
        raise ValueError("donor classification or qualified-adapter count changed")
    if donor.get("b23_2_authorized") is not False or donor.get("adaptive_schedules_authorized") is not False:
        raise ValueError("donor record crosses the authorization boundary")
    donor["cross_family_h0"] = "FAIL"
    donor["h0_failure_reason"] = "Zero NP/SITCOM cross-family adapters qualified."
    return donor


def artifact_manifest(source: Path, archive: Path, records: list[dict]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.append({
        "experiment_id": "B23.1A/B", "artifact_role": "accepted_full_capsule_archive",
        "absolute_path": str(archive), "size_bytes": archive.stat().st_size,
        "sha256": sha256_file(archive), "producer_commit": PACKAGING_HEAD,
        "config_path": "configs/b23/b23_1a_b_execution.yaml",
        "manifest_path": "manifests/b23/b23_1_signed_registry.csv", "retention_class": "PAC_RETAIN",
    })
    terminal_index = read_json(source / "TERMINAL_ARTIFACT_INDEX.json")["artifacts"]
    if len(terminal_index) != 32:
        raise ValueError("terminal artifact index no longer contains 32 entries")
    for index, item in enumerate(terminal_index):
        # RUN.json freezes the reconstruction *content* hash, while the terminal index freezes the
        # transported file-byte hash. They are intentionally different for serialized tensors.
        matching = [record for record in records if record["reconstruction_path"] == item["pac_path"]]
        if len(matching) != 1 or item["bytes"] <= 0:
            raise ValueError(f"terminal artifact {index} does not reconcile to one run")
        record = matching[0]
        rows.append({
            "experiment_id": f"B23.1-{record['mode']}-{record['parent_id']}-{record['image_id']}",
            "artifact_role": "terminal_reconstruction", "absolute_path": item["pac_path"],
            "size_bytes": item["bytes"], "sha256": item["sha256"],
            "producer_commit": record.get("recovered_execution_repo_head") or SCIENTIFIC_HEAD,
            "config_path": "configs/b23/b23_1a_b_execution.yaml",
            "manifest_path": "manifests/b23/b23_1_signed_registry.csv", "retention_class": "PAC_RETAIN",
        })
    recovered = read_json(source / "RECOVERY_ARTIFACT_INDEX.json")["external_large_artifact"]
    rows.append({
        "experiment_id": "B23.1A-Fresh1-65082-native-0", "artifact_role": "recovered_raw_trajectory",
        "absolute_path": recovered["pac_path"], "size_bytes": recovered["bytes"],
        "sha256": recovered["sha256"], "producer_commit": "45c6b6107e2bf2b100eac6b771ea0d1004f19a20",
        "config_path": "configs/b23/b23_1a_b_execution.yaml",
        "manifest_path": "manifests/b23/b23_1_one_image_smoke.signed.csv", "retention_class": "PAC_RETAIN_LARGE",
    })
    return rows


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ("experiment_id", "artifact_role", "absolute_path", "size_bytes", "sha256", "producer_commit", "config_path", "manifest_path", "retention_class")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-capsule", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--pre-run-head", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise ValueError("evidence closeout requires CUDA_VISIBLE_DEVICES to be empty")
    repo, source, archive, capsule = (path.resolve() for path in (args.repo, args.source_capsule, args.source_archive, args.capsule))
    if source.name != SOURCE_CAPSULE or not source.is_dir() or not archive.is_file():
        raise ValueError("accepted source capsule/archive identity is missing")
    if sha256_file(archive) != SOURCE_ARCHIVE_SHA256:
        raise ValueError("accepted source archive SHA-256 changed")
    if capsule.exists():
        raise FileExistsError(capsule)
    packaging = read_json(source / "PACKAGING_IDENTITY.json")
    if packaging.get("execution_acceptance_repo_head") != SCIENTIFIC_HEAD or packaging.get("packaging_repository", {}).get("head") != PACKAGING_HEAD:
        raise ValueError("reviewed scientific or packaging head changed")
    checksum_count = validate_internal_checksums(source)
    status_count = validate_final_status(source / "run/FINAL_STATUS.tsv")
    run_summary, records = summarize_runs(source)
    replay_summary = summarize_replay(source)
    compute_summary = summarize_compute(source)
    donor = validate_donor(source)
    manifest_rows = artifact_manifest(source, archive, records)

    capsule.mkdir(parents=True)
    summaries = capsule / "summaries"
    summaries.mkdir()
    write_json(summaries / "RUN_SUMMARY.json", run_summary)
    write_json(summaries / "REPLAY_SUMMARY.json", replay_summary)
    write_json(summaries / "COMPUTE_SUMMARY.json", compute_summary)
    write_json(summaries / "DONOR_COMPATIBILITY.json", donor)
    shutil.copy2(repo / "manifests/b23/b23_1_correction_ledger.json", capsule / "CORRECTION_LEDGER.json")
    write_manifest(capsule / "ARTIFACT_MANIFEST.tsv", manifest_rows)
    identity = {
        "schema_version": "b23.closeout-execution-identity.v1", "stage": "B23.1A/B",
        "reviewed_scientific_head": SCIENTIFIC_HEAD, "reviewed_packaging_head": PACKAGING_HEAD,
        "closeout_pre_run_commit": args.pre_run_head, "source_capsule": source.name,
        "source_archive_path": str(archive), "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_internal_checksums_verified": checksum_count, "scientific_final_steps_verified": status_count,
        "gpu_work_performed_during_scientific_execution": True,
        "gpu_work_performed_during_evidence_closeout": False,
    }
    write_json(capsule / "EXECUTION_IDENTITY.json", identity)
    decision = {
        "schema_version": "b23.closeout-gate-decision.v1", "verdict": "PASS_RECOMMEND_B23_1A_B_SIGNOFF_ONLY",
        "scientific_execution_accepted": True, "cross_family_h0": "FAIL",
        "qualified_np_sitcom_adapters": 0,
        "recommendation": "CONTINUE DAPS-NATIVE ONLY UNDER NARROWED CLAIM",
        "b23_2_authorized": False, "b24_authorized": False,
        "large_panels_authorized": False, "adaptive_schedules_authorized": False,
        "gpu_work_performed_during_evidence_closeout": False,
    }
    write_json(capsule / "GATE_DECISION.json", decision)
    report = f"""# B23.1A/B checkpoint report

Status: **PASS_RECOMMEND_B23_1A_B_SIGNOFF_ONLY**

GPU work performed during evidence closeout: **NO**

The planner accepted the scientific execution at `{SCIENTIFIC_HEAD}` and the repaired packaging at
`{PACKAGING_HEAD}`. This zero-GPU closeout revalidated the immutable `{SOURCE_CAPSULE}` archive,
all {checksum_count} internal checksums, and all {status_count} scientific prerequisite rows without
launching a parent process, generating a measurement, or reconstructing an image.

The accepted bounded execution contains 32 trajectories: 12 native repeats, four wrapper runs, and
16 heterogeneous smoke runs. One completed Fresh1 native trajectory was recovered rather than
rerun. All four replay reports are BITWISE/PASS, all four compute ledgers are calibrated, and all
four smoke images completed under all four parents.

The cross-family H0 **failed**: zero NP/SITCOM cross-family adapters qualified. NP-1 and SITCOM-1
remain baseline-only across family boundaries. The evidence supports only the narrowed statement
`CONTINUE DAPS-NATIVE ONLY UNDER NARROWED CLAIM`; it does not authorize a schedule.

B23.2, B24 execution, large panels, and adaptive schedules remain **NOT AUTHORIZED**. Full raw
scientific artifacts remain on PAC and are identified by absolute path, byte size, producer commit,
and SHA-256 in `ARTIFACT_MANIFEST.tsv`.
"""
    (summaries / "B23_1_CHECKPOINT_REPORT.md").write_text(report, encoding="utf-8")
    (capsule / "README.md").write_text(
        "# B23.1A/B compact evidence closeout\n\n"
        "This protocol-shaped capsule summarizes the accepted scientific run. It contains no raw "
        "trajectory or reconstruction payload. `ARTIFACT_MANIFEST.tsv` locates and hashes those "
        "immutable PAC artifacts. Cross-family H0 failed; B23.2 remains closed.\n",
        encoding="utf-8",
    )
    (capsule / "COMMANDS.sh").write_text(
        "#!/usr/bin/env bash\n# Evidence-only reproduction; CUDA must remain hidden.\n"
        "CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests/b23 -v\n"
        "CUDA_VISIBLE_DEVICES='' python scripts/b23/validate_b23_0.py --repo . --output-json <validation.json>\n"
        "# collect_b23_1_closeout.py validates an existing accepted capsule; it launches no scientific process.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "capsule": str(capsule), "trajectories": 32, "artifact_manifest_rows": len(manifest_rows), "cross_family_h0": "FAIL", "gpu_work_performed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
