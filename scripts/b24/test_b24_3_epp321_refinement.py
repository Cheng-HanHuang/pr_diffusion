#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts" / "b24" / "run_b24_3_epp321_refinement.py"
SPEC = REPO / "configs" / "b24" / "b24_3_epp321_refinement.json"


def load_runner():
    spec = importlib.util.spec_from_file_location("b24_3_epp321_test_target", RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(RUNNER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestEPP321(unittest.TestCase):
    def test_main_preprojection_budget_exact(self):
        self.assertEqual(72 * 4 * 5 + 76 * 3 * 8 + 152 * 2 * 9, 6000)

    def test_main_total_budget_exact_np4(self):
        proposals = 6000 + 699 * 4
        self.assertEqual(proposals, 8796)
        self.assertEqual(proposals + 4, 8800)

    def test_no_reallocation_budget(self):
        pre = 72 * 4 * 5 + 76 * 3 * 5 + 152 * 2 * 5
        self.assertEqual(pre, 4100)
        self.assertEqual(pre + 699 * 4, 6896)
        self.assertEqual(pre + 699 * 4 + 4, 6900)

    def test_transition_partition(self):
        covered = list(range(0, 72)) + list(range(72, 148)) + list(range(148, 300))
        self.assertEqual(covered, list(range(300)))
        self.assertEqual(len(covered), len(set(covered)))

    def test_runner_registry_and_counts(self):
        mod = load_runner()
        mod.install_refinement()
        self.assertEqual(tuple(mod.base.ARM_ORDER), mod.ARM_ORDER)
        for arm, expected in mod.EXPECTED_COUNTS.items():
            self.assertIn(arm, mod.base.RUNNERS)
            self.assertEqual(mod.base.EXPECTED_COUNTS[arm], expected)

    def test_frozen_scope_and_compute_policy(self):
        cfg = json.loads(SPEC.read_text(encoding="utf-8"))
        self.assertEqual(cfg["execution_scope"]["images"], 16)
        self.assertFalse(cfg["execution_scope"]["remaining_64_development_images_authorized"])
        self.assertFalse(cfg["execution_scope"]["confirmation_images_authorized"])
        self.assertEqual(cfg["arms"]["NP_EPP_321"]["estimated_total_unet_evals_including_initial"], 8800)
        self.assertEqual(cfg["arms"]["NP_EPP_321_RANDOM_PRUNE"]["estimated_total_unet_evals_including_initial"], 8800)
        self.assertEqual(cfg["arms"]["NP_EPP_321_NO_REALLOCATION"]["estimated_total_unet_evals_including_initial"], 6900)
        self.assertIn("FLOPs", cfg["compute_policy"]["cross_family_policy"])


if __name__ == "__main__":
    unittest.main()
