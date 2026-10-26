"""Synthetic tests for R0-N.2 A4.3 policies (no PAC data)."""
import unittest
from scripts.track_r.xp_r0n2_ladder_cpu import per_image,bootstrap_image_rate

class LadderCpu(unittest.TestCase):
    def setUp(self):
        self.rows=[
            {"uid":"a","run_index":0,"rho_signed_raw":0.2,
             "psnr_raw":24.0,"psnr_repaired":27.0,"psnr_ufd":25.0,
             "psnr_orbit_best":27.0,"rho_signed_repaired":0.2, "S_repaired":0.01, "tau_signed_good_q95":0.5,"psnr_by_orbit":{"000":24.0,"111":27.0}},
            {"uid":"b","run_index":1,"rho_signed_raw":0.1,
             "psnr_raw":18.0,"psnr_repaired":18.0,"psnr_ufd":18.0,
             "psnr_orbit_best":18.0,"rho_signed_repaired":0.1, "S_repaired":0.9, "tau_signed_good_q95":0.5,"psnr_by_orbit":{"000":18.0,"111":17.0}}]
    def test_argmin_not_oracle(self):
        m=per_image(self.rows,{"a":"111","b":"000"},2)
        self.assertEqual(m["psnr"]["E0"],18)
        self.assertEqual(m["psnr"]["E1_CLIP"],18)
        self.assertEqual(m["psnr"]["O_orbit"],27)
        self.assertEqual(m["psnr"]["E2"],27)
        self.assertEqual(m["anatomy"],"selection_miss")
    def test_initial_run_and_clip_repair(self):
        m=per_image(self.rows,{"a":"111","b":"000"},1)
        self.assertEqual(m["psnr"]["E1_CLIP"],27)
        self.assertEqual(m["fix"],1)
        self.assertEqual(m["psnr"]["mean_of_K"],24)
    def test_oracle_unavailable(self):
        rows=[{**r,"psnr_raw":19.0,"psnr_repaired":19.0,
                "psnr_orbit_best":20.0,"psnr_ufd":19.0,
                "psnr_by_orbit":{"000":19.0,"111":20.0}} for r in self.rows]
        m=per_image(rows,{"a":"000","b":"000"},2)
        self.assertEqual(m["anatomy"],"generation_failure")
    def test_bootstrap_determinism(self):
        r1=bootstrap_image_rate([0.0,1.0,0.5],"synthetic")
        r2=bootstrap_image_rate([0.0,1.0,0.5],"synthetic")
        self.assertEqual(r1,r2)
        self.assertAlmostEqual(r1[0],0.5)
if __name__=="__main__":
    unittest.main()
