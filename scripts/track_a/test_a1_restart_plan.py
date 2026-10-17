"""A1.0d synthetic-only restart-plan tests; no external data or GPU."""
import copy
import unittest

from scripts.track_a.a1_protocol import ProtocolStop
from scripts.track_a.a1_restart_plan import (
    build_synthetic_plan, synthetic_summary, verify_synthetic_plan,
)


class A1RestartPlanTest(unittest.TestCase):
    def test_budget_and_seed_fairness(self):
        result = build_synthetic_plan(["10000", "10001"])
        self.assertEqual(result["candidate_total"], 32)
        self.assertEqual(result["np_mixed8_extra_runs"], 0)
        self.assertEqual(result["status"], "DRYRUN_NOT_LAUNCHABLE")
        self.assertFalse(result["checks"]["native_commands_executable"])
        for item in result["images"]:
            self.assertEqual(item["unique_native_candidates"], 16)
            self.assertEqual(len(item["mixed4_reuses"]), 4)
            self.assertEqual(len(item["mixed8_reuses"]), 8)
        for r in result["candidates"]:
            self.assertEqual(r["native_command"], "NOT_READY_SOURCE_PREFLIGHT_PENDING")
            self.assertTrue(r["independent_process_required"])
            self.assertEqual(r["gt_runtime_access"],
                             "FORBIDDEN_EXCEPT_OFFLINE_POSTRUN_EVALUATOR")
        self.assertEqual(synthetic_summary(), synthetic_summary())

    def test_fail_closed_on_changed_measurement_or_seed(self):
        result = build_synthetic_plan(["10000"])
        altered = copy.deepcopy(result["candidates"])
        altered[0]["measurement_seed32"] ^= 1
        with self.assertRaisesRegex(ProtocolStop, "measurement"):
            verify_synthetic_plan(altered, result["images"])
        altered = copy.deepcopy(result["candidates"])
        altered[0]["native_seed32"] = altered[1]["native_seed32"]
        with self.assertRaisesRegex(ProtocolStop, "seed collision"):
            verify_synthetic_plan(altered, result["images"])

    def test_fail_closed_on_extra_gpu_command_and_duplicate(self):
        result = build_synthetic_plan(["10000"])
        altered = copy.deepcopy(result["candidates"])
        altered[0]["native_command"] = "python run_reconstruction.py"
        with self.assertRaisesRegex(ProtocolStop, "no executable"):
            verify_synthetic_plan(altered, result["images"])
        altered = copy.deepcopy(result["candidates"])
        altered[0]["candidate_id"] = altered[1]["candidate_id"]
        with self.assertRaisesRegex(ProtocolStop, "candidate ID reused"):
            verify_synthetic_plan(altered, result["images"])

    def test_invalid_image_ids_or_missing_image_fail(self):
        for ids in ([], ["999"], ["10000", "10000"], ["10000"] * 13):
            with self.assertRaises(ProtocolStop):
                build_synthetic_plan(ids)

    def test_np_portfolio_reference_tamper_stops(self):
        result = build_synthetic_plan(["10000"])
        shared = copy.deepcopy(result["images"])
        shared[0]["mixed8_reuses"][2] = "99999:NP_LF:1"
        with self.assertRaisesRegex(ProtocolStop, "unknown run"):
            verify_synthetic_plan(result["candidates"], shared)


if __name__ == "__main__":
    unittest.main()
