"""A23 CPU-only CP4 draft is a truthful, incomplete supplement placeholder."""
import json
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DRAFT=ROOT/"docs/track_r/executors/xp/XP_A23_CP4_SUPPLEMENT_DRAFT_PENDING.json"

class A23CP4Draft(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d=json.loads(DRAFT.read_text())
    def test_human_gpu_pause_is_hard(self):
        d=self.d
        self.assertTrue(d["gpu_resource_hold"])
        self.assertFalse(d["native_GPU_work_authorized_now"])
        self.assertTrue(d["not_a_CP4_completion_receipt"])
        self.assertTrue(d["no_protocol_v2"])
        self.assertTrue(d["no_new_selector"])
        self.assertTrue(d["no_matched_compute_claim"])
    def test_preserved_accepted_historical_returns(self):
        a=self.d["accepted"]
        self.assertEqual(a["tab3"]["commit"],"df6e8cd5c62f506343f2ff8fe6d493c4d032ea68")
        self.assertEqual(a["g7_c2_clip"]["commit"],"9090a50a4d4ffe922bdb43c5ee9e877c059a7dd1")
        self.assertEqual(a["g7_c2_clip"]["n_c2_pairs"],10080)
    def test_g7_total_budget_not_reset(self):
        g=self.d["g7"]
        self.assertAlmostEqual(g["published_c2_gpu_occupancy_h"] +
                               g["remaining_gpu_h_before_new_charges"],
                               g["A23_total_gpu_h_cap"])
        self.assertEqual(len(g["unrun_arms"]),4)
        self.assertIsNone(g["GT_upright_retention_table"])
    def test_pending_xpk_and_cp4_ledger_no_claims(self):
        x=self.d["xp_k"]
        self.assertIsNone(x["result"])
        self.assertIsNone(x["mechanism_v1_forecast_comparison"])
        self.assertEqual(len(x["family_order"]),3)
        self.assertEqual(x["gates"]["two_historical_B22_roots_per_family_original_parameter_identity_replay"],
                         "NOT_RUN_GPU_HOLD")
        self.assertAlmostEqual(self.d["ledger"]["project_gpu_h_checkpoint"]+
                               self.d["ledger"]["project_headroom_before_new_charges"],
                               self.d["ledger"]["project_gpu_h_cap"])
        self.assertIsNone(self.d["ledger"]["final_gpu_h"])
    def test_optional_A21_replicate_explicitly_unrun(self):
        notrun=self.d["not_run"]
        self.assertTrue(any("A21.10" in z["experiment"] and
                            z["status"]=="OPTIONAL_NOT_RUN_USER_EXCLUDED" for z in notrun))

if __name__=="__main__":
    unittest.main()
