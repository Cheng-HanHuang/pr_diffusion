#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "correct_b24_3_zero_gpu_closeout.py"
spec = importlib.util.spec_from_file_location("b24_reporting_correction_tested", TARGET)
if spec is None or spec.loader is None:
    raise RuntimeError(TARGET)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_nested_oracle_pass() -> None:
    rows = [
        {"image_id": "00001", "FRESH2_SELECTED": "20", "DAPS2_ORACLE": "21", "DAPS4_ORACLE": "22"},
        {"image_id": "00002", "FRESH2_SELECTED": "30", "DAPS2_ORACLE": "30", "DAPS4_ORACLE": "31"},
    ]
    got = mod.check_nested_oracles(rows)
    assert got["pass"] is True
    assert got["violation_count"] == 0


def test_nested_oracle_violation() -> None:
    rows = [
        {"image_id": "00001", "FRESH2_SELECTED": "22", "DAPS2_ORACLE": "21", "DAPS4_ORACLE": "23"},
    ]
    got = mod.check_nested_oracles(rows)
    assert got["pass"] is False
    assert got["violation_count"] == 1


def test_shared_failure_success_and_exclusivity() -> None:
    methods = ["FRESH2_SELECTED", "NP4_SELECTED", "EPP321_SELECTED", "PE3_SCORE_SELECTED"]
    rows = [
        {
            "image_id": "00001", "DAPS4_ORACLE": "20", "SITCOM4_ORACLE": "20",
            "FRESH2_SELECTED": "20", "NP4_SELECTED": "30", "EPP321_SELECTED": "20", "PE3_SCORE_SELECTED": "30",
        },
        {
            "image_id": "00002", "DAPS4_ORACLE": "20", "SITCOM4_ORACLE": "20",
            "FRESH2_SELECTED": "20", "NP4_SELECTED": "20", "EPP321_SELECTED": "30", "PE3_SCORE_SELECTED": "20",
        },
        {
            "image_id": "00003", "DAPS4_ORACLE": "30", "SITCOM4_ORACLE": "20",
            "FRESH2_SELECTED": "30", "NP4_SELECTED": "20", "EPP321_SELECTED": "20", "PE3_SCORE_SELECTED": "20",
        },
    ]
    got = mod.shared_failure_subset(rows, methods)
    assert got["count"] == 2
    assert got["good25_success_counts"] == {
        "FRESH2_SELECTED": 0,
        "NP4_SELECTED": 1,
        "EPP321_SELECTED": 1,
        "PE3_SCORE_SELECTED": 1,
    }
    assert got["exclusive_good25_counts_among_compared_executable_methods"] == {
        "FRESH2_SELECTED": 0,
        "NP4_SELECTED": 0,
        "EPP321_SELECTED": 1,
        "PE3_SCORE_SELECTED": 0,
    }


def test_all_dev80_uniqueness_is_not_subset_uniqueness() -> None:
    methods = ["FRESH2_SELECTED", "NP4_SELECTED"]
    rows = [
        {"image_id": "17146", "FRESH2_SELECTED": "30", "NP4_SELECTED": "20", "DAPS4_ORACLE": "30", "SITCOM4_ORACLE": "30"},
        {"image_id": "34587", "FRESH2_SELECTED": "20", "NP4_SELECTED": "30", "DAPS4_ORACLE": "20", "SITCOM4_ORACLE": "20"},
    ]
    all_unique = mod.unique_good25(rows, methods)
    shared = mod.shared_failure_subset(rows, methods)
    assert all_unique["FRESH2_SELECTED"] == ["17146"]
    assert shared["good25_success_counts"]["FRESH2_SELECTED"] == 0
    assert "17146" not in shared["image_ids"]


def main() -> int:
    test_nested_oracle_pass()
    test_nested_oracle_violation()
    test_shared_failure_success_and_exclusivity()
    test_all_dev80_uniqueness_is_not_subset_uniqueness()
    print("B24_3_ZERO_GPU_REPORTING_CORRECTION_TESTS_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
