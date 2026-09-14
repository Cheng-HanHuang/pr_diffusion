#!/usr/bin/env python3
"""Reporting-only successor for the completed B24.3 zero-GPU closeout.

This script does not re-evaluate any reconstruction, operator loss, measurement, or
model. It reads the already-published closeout capsule, preserves valid artifacts,
and corrects only the complementarity reporting scope:

- all-DEV80 uniqueness is reported separately;
- shared-failure-subset success and exclusivity are reported inside that subset;
- Fresh2 <= DAPS2 oracle <= DAPS4 oracle is checked per image;
- HARD_SUBSET.csv membership/counts are checked against recomputation.

The source capsule is never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

GOOD25 = 25.0
TOL = 1e-6
HERE = Path(__file__).resolve()
SPEC_PATH = HERE.parents[2] / "configs" / "b24" / "b24_3_zero_gpu_reporting_correction.json"


def readj(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def writej(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def unique_good25(rows: list[dict[str, str]], methods: list[str]) -> dict[str, list[str]]:
    return {
        method: [
            str(row["image_id"]).zfill(5)
            for row in rows
            if f(row, method) >= GOOD25
            and all(f(row, other) < GOOD25 for other in methods if other != method)
        ]
        for method in methods
    }


def shared_failure_subset(rows: list[dict[str, str]], methods: list[str]) -> dict[str, Any]:
    subset = [
        row for row in rows
        if f(row, "DAPS4_ORACLE") < GOOD25 and f(row, "SITCOM4_ORACLE") < GOOD25
    ]
    successes = {
        method: [str(row["image_id"]).zfill(5) for row in subset if f(row, method) >= GOOD25]
        for method in methods
    }
    exclusive = {
        method: [
            str(row["image_id"]).zfill(5)
            for row in subset
            if f(row, method) >= GOOD25
            and all(f(row, other) < GOOD25 for other in methods if other != method)
        ]
        for method in methods
    }
    return {
        "definition": "DAPS4_ORACLE < 25 dB and SITCOM4_ORACLE < 25 dB",
        "count": len(subset),
        "image_ids": [str(row["image_id"]).zfill(5) for row in subset],
        "good25_success_image_ids": successes,
        "good25_success_counts": {k: len(v) for k, v in successes.items()},
        "exclusive_good25_image_ids_among_compared_executable_methods": exclusive,
        "exclusive_good25_counts_among_compared_executable_methods": {k: len(v) for k, v in exclusive.items()},
    }


def check_nested_oracles(rows: list[dict[str, str]]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    max_fresh_minus_d2 = float("-inf")
    max_d2_minus_d4 = float("-inf")
    for row in rows:
        image = str(row["image_id"]).zfill(5)
        fresh = f(row, "FRESH2_SELECTED")
        d2 = f(row, "DAPS2_ORACLE")
        d4 = f(row, "DAPS4_ORACLE")
        max_fresh_minus_d2 = max(max_fresh_minus_d2, fresh - d2)
        max_d2_minus_d4 = max(max_d2_minus_d4, d2 - d4)
        if fresh > d2 + TOL or d2 > d4 + TOL:
            violations.append({
                "image_id": image,
                "fresh2": fresh,
                "daps2_oracle": d2,
                "daps4_oracle": d4,
            })
    return {
        "pass": not violations,
        "tolerance_db": TOL,
        "violation_count": len(violations),
        "violations": violations,
        "max_fresh2_minus_daps2_oracle_db": max_fresh_minus_d2,
        "max_daps2_oracle_minus_daps4_oracle_db": max_d2_minus_d4,
    }


def assert_expected_counts(actual: dict[str, int], expected: dict[str, Any], label: str) -> None:
    exp = {str(k): int(v) for k, v in expected.items()}
    if actual != exp:
        raise RuntimeError(f"{label} mismatch: expected={exp} actual={actual}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    spec = readj(SPEC_PATH)
    if spec.get("gpu_work_authorized") is not False or spec.get("confirmation_exposed") is not False:
        raise RuntimeError("reporting-correction authorization drift")

    source = args.source.resolve()
    output = args.output.resolve()
    if str(source) != str(Path(spec["source_capsule"]["run_root"]).resolve()):
        raise RuntimeError(f"source capsule drift: {source}")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    required = [
        "FRESH2_PER_IMAGE.csv",
        "FRESH2_SUMMARY.json",
        "DEV80_CLOSEOUT_PER_IMAGE.csv",
        "PAIRWISE_EXECUTABLE.csv",
        "HARD_SUBSET.csv",
        "COMPLEMENTARITY.json",
        "COMPUTE_CLOSEOUT.json",
        "B24_3_DEV_CLOSEOUT.md",
        "B24_3_DEV_CLOSEOUT.json",
        "SHA256SUMS.txt",
        "LAUNCH_IDENTITY.txt",
    ]
    for name in required:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)

    source_closeout = readj(source / "B24_3_DEV_CLOSEOUT.json")
    source_comp = readj(source / "COMPLEMENTARITY.json")
    fresh2_summary = readj(source / "FRESH2_SUMMARY.json")
    compute_closeout = readj(source / "COMPUTE_CLOSEOUT.json")
    if source_closeout.get("decision") != "STOP_B24_METHOD_REFINEMENT":
        raise RuntimeError("source stop verdict drift")
    if source_closeout.get("confirmation_exposed") is not False:
        raise RuntimeError("source confirmation exposure drift")

    rows = read_csv(source / "DEV80_CLOSEOUT_PER_IMAGE.csv")
    if len(rows) != 80 or len({str(r["image_id"]).zfill(5) for r in rows}) != 80:
        raise RuntimeError("DEV80 closeout row identity drift")
    methods = [str(x) for x in spec["executable_methods"]]

    nested = check_nested_oracles(rows)
    if not nested["pass"]:
        raise RuntimeError(f"nested-oracle inequality violated: {nested['violations']}")

    all_unique = unique_good25(rows, methods)
    all_unique_counts = {k: len(v) for k, v in all_unique.items()}
    assert_expected_counts(all_unique_counts, spec["all_dev80_expected_unique_good25_counts"], "all-DEV80 uniqueness")

    shared = shared_failure_subset(rows, methods)
    if shared["count"] != int(spec["shared_failure_subset"]["expected_count"]):
        raise RuntimeError(f"shared-failure subset count drift: {shared['count']}")
    assert_expected_counts(
        shared["good25_success_counts"],
        spec["shared_failure_subset"]["expected_good25_success_counts"],
        "shared-failure successes",
    )
    assert_expected_counts(
        shared["exclusive_good25_counts_among_compared_executable_methods"],
        spec["shared_failure_subset"]["expected_exclusive_good25_counts"],
        "shared-failure exclusivity",
    )

    hard_rows = read_csv(source / "HARD_SUBSET.csv")
    hard_ids = sorted(str(r["image_id"]).zfill(5) for r in hard_rows)
    recomputed_ids = sorted(shared["image_ids"])
    if hard_ids != recomputed_ids:
        raise RuntimeError(f"HARD_SUBSET.csv membership drift: file={hard_ids} recomputed={recomputed_ids}")

    # Logical implication of nested DAPS candidate sets: Fresh2 cannot be Good25
    # if the four-trajectory DAPS oracle is below Good25.
    fresh2_shared = shared["good25_success_image_ids"]["FRESH2_SELECTED"]
    if fresh2_shared:
        raise RuntimeError(f"Fresh2 unexpectedly succeeds inside DAPS4-fail subset: {fresh2_shared}")

    known = spec["known_reporting_correction"]
    fresh_unique = all_unique["FRESH2_SELECTED"]
    if fresh_unique != [str(known["fresh2_global_unique_image_id"]).zfill(5)]:
        raise RuntimeError(f"Fresh2 all-DEV80 unique image drift: {fresh_unique}")
    if fresh_unique[0] in set(shared["image_ids"]):
        raise RuntimeError("Fresh2 global unique image unexpectedly lies in shared-failure subset")
    pe3_successes = shared["good25_success_image_ids"]["PE3_SCORE_SELECTED"]
    if pe3_successes != [str(known["pe3_score_shared_failure_success_image_id"]).zfill(5)]:
        raise RuntimeError(f"PE3 score shared-failure success drift: {pe3_successes}")
    if pe3_successes[0] in set(shared["exclusive_good25_image_ids_among_compared_executable_methods"]["PE3_SCORE_SELECTED"]):
        raise RuntimeError("PE3 score shared-failure success unexpectedly exclusive")

    # Preserve valid source artifacts byte-for-byte. Ambiguous/wrong reporting files
    # are regenerated below. The original capsule itself remains untouched.
    copy_names = [
        "FRESH2_PER_IMAGE.csv",
        "FRESH2_SUMMARY.json",
        "DEV80_CLOSEOUT_PER_IMAGE.csv",
        "PAIRWISE_EXECUTABLE.csv",
        "HARD_SUBSET.csv",
        "COMPUTE_CLOSEOUT.json",
    ]
    for name in copy_names:
        shutil.copy2(source / name, output / name)
    shutil.copy2(source / "SHA256SUMS.txt", output / "SOURCE_CAPSULE_SHA256SUMS.txt")
    shutil.copy2(source / "LAUNCH_IDENTITY.txt", output / "SOURCE_LAUNCH_IDENTITY.txt")

    complementarity = {
        "schema_version": "b24.dev80-complementarity.v2",
        "status": "PASS",
        "n": 80,
        "good25_threshold_db": GOOD25,
        "executable_methods": methods,
        "diagnostic_oracles": source_comp.get("diagnostic_oracles", []),
        "method_distributions": source_comp.get("method_distributions", {}),
        "pairwise_executable": source_comp.get("pairwise_executable", {}),
        "failure_sets": source_comp.get("failure_sets", {}),
        "all_dev80": {
            "definition": "uniqueness among compared executable methods over all 80 DEV images",
            "unique_good25_image_ids": all_unique,
            "unique_good25_counts": all_unique_counts,
        },
        "shared_failure_subset": shared,
        "checks": {
            "nested_oracle_inequality": nested,
            "hard_subset_csv_membership_matches_recomputation": True,
            "fresh2_success_count_inside_shared_failure_subset": 0,
        },
        "confirmation_exposed": False,
        "interpretation_guard": "All-DEV80 uniqueness and shared-failure-subset success/exclusivity are distinct quantities. Pairwise unions/oracles diagnose complementarity only and are not executable selectors unless explicitly labeled as such.",
    }
    writej(output / "COMPLEMENTARITY.json", complementarity)

    checks = {
        "schema_version": "b24.reporting-correction-checks.v1",
        "status": "PASS",
        "decision_preserved": source_closeout.get("decision") == "STOP_B24_METHOD_REFINEMENT",
        "confirmation_exposed": False,
        "gpu_work_performed": False,
        "measurement_generation_performed": False,
        "nested_oracle_inequality": nested,
        "hard_subset_csv_membership_matches_recomputation": True,
        "shared_failure_subset_count": shared["count"],
        "shared_failure_success_counts": shared["good25_success_counts"],
        "shared_failure_exclusive_counts": shared["exclusive_good25_counts_among_compared_executable_methods"],
        "all_dev80_unique_good25_counts": all_unique_counts,
    }
    writej(output / "CORRECTION_CHECKS.json", checks)

    metadata = {
        "schema_version": "b24.reporting-correction-metadata.v1",
        "status": "PASS",
        "correction_type": "ZERO_GPU_REPORTING_ONLY",
        "source_capsule_run_root": str(source),
        "source_capsule_archive_sha256": spec["source_capsule"]["archive_sha256"],
        "source_scientific_run_head": spec["source_capsule"]["scientific_run_head"],
        "source_capsule_preserved_unchanged": True,
        "decision": "STOP_B24_METHOD_REFINEMENT",
        "confirmation_exposed": False,
        "gpu_work_performed": False,
        "measurement_generation_performed": False,
        "fresh2_metrics_recomputed": False,
        "reconstruction_performed": False,
    }
    writej(output / "CORRECTION_METADATA.json", metadata)

    closeout = dict(source_closeout)
    closeout["schema_version"] = "b24.dev-closeout.v2"
    closeout["status"] = "PASS"
    closeout["decision"] = "STOP_B24_METHOD_REFINEMENT"
    closeout["confirmation_exposed"] = False
    closeout["gpu_work_performed"] = False
    closeout["measurement_generation_performed"] = False
    closeout["reporting_correction"] = metadata
    closeout["complementarity_summary"] = {
        "all_dev80_unique_good25_counts": all_unique_counts,
        "shared_failure_subset": {
            "count": shared["count"],
            "image_ids": shared["image_ids"],
            "good25_success_counts": shared["good25_success_counts"],
            "exclusive_good25_counts_among_compared_executable_methods": shared["exclusive_good25_counts_among_compared_executable_methods"],
        },
        "nested_oracle_inequality_pass": True,
    }
    closeout["next"] = "B24_FINAL_SIGNOFF_READY; method refinement stopped; confirmation remains locked"
    writej(output / "B24_3_DEV_CLOSEOUT.json", closeout)

    success_counts = shared["good25_success_counts"]
    exclusive_counts = shared["exclusive_good25_counts_among_compared_executable_methods"]
    lines = [
        "# B24 development closeout — corrected reporting successor",
        "",
        "## Frozen decision",
        "",
        "`STOP_B24_METHOD_REFINEMENT` remains binding. No confirmation image was exposed.",
        "",
        "This successor changes reporting only. It does not recompute Fresh2 metrics, run reconstruction, generate measurements, or perform GPU work. The original capsule is preserved unchanged.",
        "",
        "## Historical Fresh2 on DEV80",
        "",
        f"Fresh2 selected Good25 remains **{fresh2_summary['selected_distribution']['good25_count']}/80**, mean **{fresh2_summary['selected_distribution']['mean_db']:.4f} dB**, median **{fresh2_summary['selected_distribution']['median_db']:.4f} dB**.",
        "Fresh2 remains a historical descriptive comparator, not a post-hoc B24 advancement candidate.",
        "",
        "## All-DEV80 uniqueness",
        "",
        "Unique Good25 successes among the compared executable methods over all 80 DEV images:",
    ]
    for method in methods:
        ids = all_unique[method]
        lines.append(f"- `{method}`: **{len(ids)}**" + (f" ({', '.join(ids)})" if ids else ""))
    lines += [
        "",
        "These counts are over all DEV80 and must not be described as shared-failure-subset uniqueness.",
        "",
        "## Shared-failure subset",
        "",
        f"The subset where both `DAPS4_ORACLE < 25 dB` and `SITCOM4_ORACLE < 25 dB` contains **{shared['count']}** images.",
        "",
        "Good25 successes / exclusive successes among the compared executable methods inside this 10-case subset:",
    ]
    for method in methods:
        lines.append(f"- `{method}`: **{success_counts[method]} / {exclusive_counts[method]}**")
    lines += [
        "",
        "`PE3_SCORE_SELECTED` succeeds on image `34587`, but that success is shared with `NP4_SELECTED` and is therefore not exclusive. Fresh2 has **0** successes in this subset. Its globally unique success `17146` lies outside the subset.",
        "",
        "## Nested-oracle check",
        "",
        "The successor verifies per image:",
        "",
        "`FRESH2_SELECTED <= DAPS2_ORACLE <= DAPS4_ORACLE`.",
        "",
        "The check passes on all 80 DEV images. Consequently Fresh2 cannot rescue a failure of the same four-trajectory DAPS candidate set.",
        "",
        "## Compute interpretation",
        "",
        "The previously reported dispatch-supported FLOP figures remain unchanged. The `0.492x` Fresh2/DAPS2 figure is a dispatch-supported-FLOP ratio only, not exact total computation or runtime. Unsupported Fourier/custom forward/backward work remains explicitly outside that exact-equivalence claim.",
        "",
        "## Scope guard",
        "",
        "B24 closes as a negative development result. Confirmation remains locked. Any Fresh2 follow-up requires a separate prospective scientific question.",
    ]
    (output / "B24_3_DEV_CLOSEOUT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifacts = sorted(p for p in output.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    with (output / "SHA256SUMS.txt").open("w", encoding="utf-8") as fsum:
        for path in artifacts:
            fsum.write(f"{sha256_file(path)}  {path.name}\n")

    print(json.dumps({
        "status": "PASS",
        "decision": "STOP_B24_METHOD_REFINEMENT",
        "source_capsule_preserved": True,
        "nested_oracle_inequality_pass": True,
        "shared_failure_subset_count": shared["count"],
        "shared_failure_success_counts": success_counts,
        "shared_failure_exclusive_counts": exclusive_counts,
        "all_dev80_unique_good25_counts": all_unique_counts,
        "output": str(output),
        "confirmation_exposed": False,
        "gpu_work_performed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
