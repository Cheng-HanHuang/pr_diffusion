"""A1.2 pure-CPU planning tests; no FFHQ image, native solver or GPU access."""
import unittest
from collections import Counter
from datetime import datetime,timezone

from scripts.track_a.a1_2_k4_exposed_b22 import (
    IDS, METHODS, K, make_plan, validate_input_manifest, select_best_by_psnr,
    EXPECTED_SELECTION_SHA, EXPECTED_INPUT_FREEZE_SHA, EXECUTION_ENABLED,
    RUN_WINDOW_UTC, RESERVATION_ID, MAX_TOTAL_WALL_S,
)
from scripts.track_a.a1_protocol import seed32_numpy

def fake_inputs():
    return [{
      "image_id":i, "split":"A1_2_B22_EXPOSED_ENGINEERING_ONLY",
      "gt_data_source":"B22_ALREADY_EXPOSED_FINAL100_NOT_NEW_HELDOUT",
      "measurement_dtype":"torch.float32", "measurement_shape":[1,3,384,384],
      "measurement_tensor_sha256":f"{n:064x}",
      "ground_truth_tensor_sha256":f"{n+20:064x}",
    } for n,i in enumerate(IDS)]

class A12K4ContractTest(unittest.TestCase):
    def test_k4_exactly_128_unique_native_seeds(self):
        p=make_plan(fake_inputs(),"test-head",{"DAPS":"frozen"})
        self.assertFalse(p["gpu_run_reserved"])
        self.assertEqual(p["candidate_count"],128)
        self.assertEqual(p["per_image_candidates"],16)
        self.assertEqual(len(p["candidate_registry"]),128)
        self.assertEqual(Counter(c["method"] for c in p["candidate_registry"]),Counter({m:32 for m in METHODS}))
        self.assertEqual(len({c["candidate_id"] for c in p["candidate_registry"]}),128)
        self.assertEqual(len({(c["image_id"],c["native_seed32"]) for c in p["candidate_registry"]}),128)
        for c in p["candidate_registry"]:
            self.assertEqual(c["native_seed32"],seed32_numpy(c["method"],c["image_id"],c["rep"]))
            self.assertTrue(0<=c["native_seed32"]<2**32)
        self.assertEqual(p["np_mixed8_additional_trajectories"],0)
        self.assertTrue(p["not_prospective_heldout"])
        self.assertEqual(EXECUTION_ENABLED,True)
        self.assertTrue(RESERVATION_ID.startswith("TRACK_A_A1_2_EXPOSED"))
        self.assertEqual(MAX_TOTAL_WALL_S,10800)
        self.assertFalse(p["gpu_run_reserved"])
        self.assertTrue(p["execution_enabled_only_with_gpu3_and_reservation_ack"])

    def test_lease_is_dated_and_bounded(self):
        early,late=RUN_WINDOW_UTC
        self.assertEqual(early,datetime(2026,10,20,2,30,tzinfo=timezone.utc))
        self.assertEqual(late,datetime(2026,10,20,5,0,tzinfo=timezone.utc))
        self.assertLess((late-early).total_seconds(),MAX_TOTAL_WALL_S)
        self.assertNotEqual(RESERVATION_ID,"")
        self.assertFalse(RESERVATION_ID.endswith("NOT_RESERVED"))

    def test_wrong_number_or_order_of_exposed_images_fails(self):
        for rows in (fake_inputs()[:7],list(reversed(fake_inputs()))):
            with self.assertRaises(RuntimeError):make_plan(rows,"head",{})

    def test_offline_oracle_charge_and_tie_break(self):
        rows=[{"offline_raw_png_psnr_db":psnr,
               "offline_ambiguity_aware_png_psnr_db":rot,
               "rep":rep,"candidate_id":f"x:DAPS:{rep}",
               "gpu_active_seconds_event":10+rep,
               "wall_seconds_subprocess":20+rep}
              for rep,(psnr,rot) in enumerate(((24,24),(28,30),(28,31),(25,25)))]
        got=select_best_by_psnr(rows,[0,1,2,3])
        self.assertEqual(got["best_candidate_id"],"x:DAPS:1")
        self.assertEqual(got["best_psnr_raw_db"],28)
        self.assertTrue(got["good25"])
        self.assertEqual(got["candidate_count"],4)
        self.assertEqual(got["charged_gpu_event_seconds"],46)
        self.assertEqual(got["charged_wall_seconds"],86)
        self.assertEqual(got["best_psnr_rot180_aware_db_on_raw_winner"],30)

    def test_manifest_provenance_failclosed(self):
        d={"schema":"prdiff.a1.2.exposed-input-freeze.v1",
           "status":"PASS_EXPOSED_B22_INPUTS_FROZEN_NOT_HELDOUT",
           "scope":"EXPOSED_ENGINEERING_NO_PUBLICATION_GENERALIZATION_INFERENCE",
           "selected_id_list_sha256":EXPECTED_SELECTION_SHA,
           "image_ids":list(IDS),"candidate_method_runs":0,
           "protected_confirmation_access":False,
           "no_new_measurements":True,"no_gpu_work":True,"rows":fake_inputs()}
        # This test intentionally does not open any files; fail before filesystem checks.
        for key,val in (("status","PRISTINE"),("scope","PROSPECTIVE"),
                        ("selected_id_list_sha256","0"*64),
                        ("image_ids",list(reversed(IDS))),
                        ("protected_confirmation_access",True),
                        ("no_new_measurements",False)):
            altered=dict(d,**{key:val})
            with self.assertRaises(RuntimeError):validate_input_manifest(altered)
        self.assertEqual(len(EXPECTED_INPUT_FREEZE_SHA),64)
        self.assertEqual(K,4)

if __name__=="__main__": unittest.main()
