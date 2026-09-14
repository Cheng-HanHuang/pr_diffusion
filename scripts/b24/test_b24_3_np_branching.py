#!/usr/bin/env python3
"""Zero-GPU structural tests for the frozen B24.3 NP branching budget."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_b24_3_np_branching.py"

spec = importlib.util.spec_from_file_location("b24_3_runner_tested", RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError(RUNNER)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class BudgetTests(unittest.TestCase):
    def test_np1_np4(self):
        np1_prop = 300 * 5 + 699
        self.assertEqual(np1_prop, 2199)
        self.assertEqual(1 + np1_prop, 2200)
        self.assertEqual(4 * np1_prop, 8796)
        self.assertEqual(4 + 4 * np1_prop, 8800)
        self.assertEqual(mod.EXPECTED_COUNTS["NP1"]["total_unet_evals"], 2200)
        self.assertEqual(mod.EXPECTED_COUNTS["NP4_INDEPENDENT"]["total_unet_evals"], 8800)

    def test_epp_exact_compute_match(self):
        proposals = 4 * 72 * 5 + 2 * 76 * 10 + 152 * 20 + 4 * 699
        self.assertEqual(proposals, 8796)
        self.assertEqual(4 + proposals, 8800)
        self.assertEqual(mod.EXPECTED_COUNTS["NP_EPP"]["proposal_evals"], proposals)
        self.assertEqual(mod.EXPECTED_COUNTS["NP_EPP_RANDOM_PRUNE"]["proposal_evals"], proposals)

    def test_epp_no_reallocation_is_lower_work(self):
        proposals = 4 * 72 * 5 + 2 * 76 * 5 + 152 * 5 + 4 * 699
        self.assertEqual(proposals, 5756)
        self.assertEqual(4 + proposals, 5760)
        self.assertEqual(mod.EXPECTED_COUNTS["NP_EPP_NO_REALLOCATION"]["total_unet_evals"], 5760)

    def test_dps_exact_proposal_match(self):
        proposals = (
            72 * 5
            + 5 + 5 * 75 * 5
            + 5 + 5 * 75 * 5
            + 5 + 5 * 75 * 5
            + 4 * 699
        )
        self.assertEqual(proposals, 8796)
        self.assertEqual(1 + proposals, 8797)
        self.assertEqual(mod.EXPECTED_COUNTS["NP_DPS"]["total_unet_evals"], 8797)

    def test_seed_domains_are_deterministic_and_separated(self):
        a = mod.seed63("B24_METHOD_DPS_BRANCH_V1", "12345", "root0", 72, 0)
        b = mod.seed63("B24_METHOD_DPS_BRANCH_V1", "12345", "root0", 72, 0)
        c = mod.seed63("B24_METHOD_DPS_BRANCH_V1", "12345", "root0", 72, 1)
        d = mod.seed63("B24_METHOD_EPP_HARD_FORK_V1", "12345", "root0", 0)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)
        self.assertGreaterEqual(a, 0)
        self.assertLess(a, 1 << 63)

    def test_arm_order_and_ceiling(self):
        self.assertEqual(set(mod.ARM_ORDER), set(mod.EXPECTED_COUNTS))
        self.assertEqual(mod.HARD_CEILING_MIB, 52452)
        self.assertEqual(mod.TRAILING_WINDOW, 32)
        self.assertEqual(mod.PROJ_START, 300)


if __name__ == "__main__":
    unittest.main()
