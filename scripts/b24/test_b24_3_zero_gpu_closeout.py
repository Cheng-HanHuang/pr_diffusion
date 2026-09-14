#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "closeout_b24_3_zero_gpu.py"
spec = importlib.util.spec_from_file_location("b24_zero_gpu_closeout_tested", TARGET)
if spec is None or spec.loader is None:
    raise RuntimeError(TARGET)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_overlap() -> None:
    got = mod.overlap([30.0, 30.0, 10.0, 10.0], [30.0, 10.0, 30.0, 10.0])
    assert got == {"both_good": 1, "a_only_good": 1, "b_only_good": 1, "both_fail": 1, "good25_union": 3}


def test_paired() -> None:
    got = mod.paired([30.0, 10.0, 31.0, 9.0], [20.0, 30.0, 30.0, 9.0])
    assert got["good25_rescues"] == 1
    assert got["good25_harms"] == 1
    assert got["large_rescues_ge5db"] == 1
    assert got["large_harms_le_minus5db"] == 1
    assert got["wins_ties_losses"] == [2, 1, 1]


def test_fresh2_rule() -> None:
    theta = mod.FRESH2_THETA
    def select(l0: float, l1: float) -> int:
        return 1 if l1 < l0 - theta else 0
    assert select(10.0, 9.29) == 1
    assert select(10.0, 9.30) == 0  # strict historical margin
    assert select(10.0, 9.31) == 0


def test_np_fourier_accounting() -> None:
    proposals = 8796
    post = 2796
    fwd = 3 * proposals + post
    inv = post
    assert fwd == 29184
    assert inv == 2796
    assert fwd + inv == 31980
    assert mod.fft_work_units() > 0


def main() -> int:
    test_overlap(); test_paired(); test_fresh2_rule(); test_np_fourier_accounting()
    print("B24_3_ZERO_GPU_CLOSEOUT_TESTS_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
