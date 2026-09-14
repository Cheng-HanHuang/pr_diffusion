#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PE3_PATH = REPO / "scripts" / "b24" / "run_b24_3_pe3_refinement.py"
SPEC_PATH = REPO / "configs" / "b24" / "b24_3_pe3_final_dev.json"

spec = importlib.util.spec_from_file_location("pe3_test_impl", PE3_PATH)
assert spec and spec.loader
pe3 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pe3
spec.loader.exec_module(pe3)


class FakeBranch:
    def __init__(self, lineage, score):
        self.lineage = lineage
        self._score = score
    def trailing_score(self):
        return self._score


class FakeBase:
    SPEC_PATH = None
    ARM_ORDER = ()
    EXPECTED_COUNTS = {}
    RUNNERS = {}
    @staticmethod
    def domain_hash(domain, image_id, lineage):
        # Deterministic test ordering independent of measurement score.
        rank = {"root0": "d", "root1": "a", "root2": "c", "root3": "b"}[lineage]
        return rank
    @staticmethod
    def checkpoint_event(branches, survivors, checkpoint, ctx, rule):
        return {
            "checkpoint_before_transition": checkpoint,
            "rule": rule,
            "survivor_lineages": [b.lineage for b in survivors],
        }


class Ctx:
    image_id = "00001"


class PE3ContractTests(unittest.TestCase):
    def test_exact_work_budget(self):
        proposals = 72 * 4 * 5 + 228 * (10 + 5 + 5) + 699 * 4
        self.assertEqual(proposals, 8796)
        self.assertEqual(proposals + 4, 8800)

    def test_score_role_assignment(self):
        fake = FakeBase()
        pe3.base = fake
        branches = [
            FakeBranch("root0", 0.40), FakeBranch("root1", 0.10),
            FakeBranch("root2", 0.30), FakeBranch("root3", 0.20),
        ]
        events = []
        explorer, protected = pe3.role_assignment(Ctx(), branches, "score", events)
        self.assertEqual(explorer.lineage, "root1")
        self.assertEqual([b.lineage for b in protected], ["root3", "root2"])
        self.assertEqual(events[0]["dropped_lineages"], ["root0"])
        self.assertTrue(events[0]["role_assignment_uses_measurement"])

    def test_random_role_assignment_is_score_independent(self):
        fake = FakeBase()
        pe3.base = fake
        a = [FakeBranch(f"root{i}", float(i)) for i in range(4)]
        b = [FakeBranch(f"root{i}", float(10 - i)) for i in range(4)]
        ea, eb = [], []
        xa, pa = pe3.role_assignment(Ctx(), a, "random", ea)
        xb, pb = pe3.role_assignment(Ctx(), b, "random", eb)
        self.assertEqual(xa.lineage, xb.lineage)
        self.assertEqual([x.lineage for x in pa], [x.lineage for x in pb])
        self.assertFalse(ea[0]["role_assignment_uses_measurement"])

    def test_install_contract(self):
        fake = FakeBase()
        pe3.install_pe3(fake)
        self.assertEqual(fake.ARM_ORDER, ("NP_PE3_SCORE", "NP_PE3_RANDOM"))
        self.assertEqual(fake.EXPECTED_COUNTS["NP_PE3_SCORE"]["total_unet_evals"], 8800)
        self.assertEqual(fake.EXPECTED_COUNTS["NP_PE3_RANDOM"]["total_unet_evals"], 8800)
        self.assertEqual(Path(fake.SPEC_PATH), SPEC_PATH)

    def test_frozen_gate(self):
        cfg = json.loads(SPEC_PATH.read_text())
        gate = cfg["advancement_gate_vs_np4"]
        self.assertEqual(gate["median_paired_delta_db_min"], 0.0)
        self.assertTrue(gate["good25_count_at_least_np4"])
        self.assertTrue(gate["good25_rescues_at_least_harms"])
        self.assertTrue(gate["large_5db_rescues_at_least_harms"])
        self.assertEqual(gate["if_neither_passes"], "STOP_B24_METHOD_REFINEMENT")
        self.assertFalse(cfg["execution_scope"]["confirmation_images_authorized"])


if __name__ == "__main__":
    unittest.main()
