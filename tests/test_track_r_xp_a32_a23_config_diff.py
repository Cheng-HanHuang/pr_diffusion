"""Pure CPU test: prereg A23 config diff screen never authorizes GPU work."""
import math
import unittest

from scripts.track_r.xp_a32_a23_config_diff_cpu import (
    FAMILIES, STATUS, check_exact_one_change,
)

class A23OneParameterTest(unittest.TestCase):
    def basic(self, family, original, proposed):
        pre={"solver":{"tau":original,"euler_step":.012,
                       "optimizer":{"steps":200,"lr":.005}},
             "seeds":[0,1,2,3], "source_sha":"frozen"}
        new={"solver":{"tau":proposed,"euler_step":.012,
                       "optimizer":{"steps":200,"lr":.005}},
             "seeds":[0,1,2,3], "source_sha":"frozen"}
        return pre,new

    def test_all_registered_calibrations_diff_one_leaf(self):
        for family,spec in FAMILIES.items():
            p,q=self.basic(family,spec["original"],spec["target"])
            with self.subTest(family=family):
                rec=check_exact_one_change(p,q,family,"solver.tau")
                self.assertEqual(rec["status"],STATUS)
                self.assertFalse(rec["future_GPU_generation_authorized"])
                self.assertFalse(rec["complete_native_source_diff_gate_passed"])
                self.assertFalse(rec["historical_two_root_identity_replay_passed"])
                self.assertTrue(rec["GPU_resource_hold"])
                self.assertEqual(rec["changed_key_count"],1)
    def test_euler_step_must_not_change(self):
        p,q=self.basic("SITCOM",.01,.05)
        q["solver"]["euler_step"]=.013
        with self.assertRaisesRegex(RuntimeError,"NOT_EXACTLY_ONE_PARAMETER"):
            check_exact_one_change(p,q,"SITCOM","solver.tau")
    def test_key_and_structure_changes_rejected(self):
        p,q=self.basic("SITCOM",.01,.05)
        q["seeds"].append(4)
        with self.assertRaisesRegex(RuntimeError,"KEYS_OR_STRUCTURES_CHANGED"):
            check_exact_one_change(p,q,"SITCOM","solver.tau")
    def test_wrong_key_path_rejected(self):
        p,q=self.basic("SITCOM",.01,.05)
        with self.assertRaisesRegex(RuntimeError,"CHANGED_KEY_NOT_FROZEN_NATIVE_BINDING"):
            check_exact_one_change(p,q,"SITCOM","solver.euler_step")
    def test_wrong_old_or_target_value_rejected(self):
        p,q=self.basic("SITCOM",.02,.05)
        with self.assertRaisesRegex(RuntimeError,"HISTORICAL_BASELINE_VALUE"):
            check_exact_one_change(p,q,"SITCOM","solver.tau")
        p,q=self.basic("SITCOM",.01,.049)
        with self.assertRaisesRegex(RuntimeError,"REGISTERED_CALIBRATION_VALUE"):
            check_exact_one_change(p,q,"SITCOM","solver.tau")
    def test_numeric_bool_and_nonfinite_not_accepted(self):
        p,q=self.basic("FAST_DIPS",.05,33.26)
        p["solver"]["tau"]=False
        with self.assertRaisesRegex(RuntimeError,"NUMERIC_PARAMETER"):
            check_exact_one_change(p,q,"FAST_DIPS","solver.tau")
        p,q=self.basic("FAST_DIPS",.05,math.nan)
        with self.assertRaisesRegex(RuntimeError,"NONFINITE_CONFIG"):
            check_exact_one_change(p,q,"FAST_DIPS","solver.tau")
    def test_fake_identity_snapshot_rejected(self):
        p,q=self.basic("DAPS_FFHQ",.01,math.sqrt(2.)*.05)
        q["source_sha"]="changed"
        with self.assertRaisesRegex(RuntimeError,"NOT_EXACTLY_ONE_PARAMETER"):
            check_exact_one_change(p,q,"DAPS_FFHQ","solver.tau")

if __name__=="__main__":
    unittest.main()
