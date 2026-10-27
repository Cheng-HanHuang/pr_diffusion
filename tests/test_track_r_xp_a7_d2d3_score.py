"""A7 image K4 scorer policy tests, independent of GT and GPU."""
import unittest
from scripts.track_r.xp_a7_d2d3_score_cpu import selectors,COMPARE_DAPS

class A7Selectors(unittest.TestCase):
    def test_first_e0_and_each_metric_oracle_distinct(self):
        rho=[.2,.1,.4,.3]
        psnr=[20.,21.,26.,24.]
        ssim=[.70,.71,.69,.78]
        lpips=[.35,.40,.23,.25]
        runs=[]
        for i in range(4):
            runs.append(dict(rep=i,rho_signed=rho[i],
               psnr_db=psnr[i],ssim=ssim[i],lpips=lpips[i],
               orbit={"psnr_db":[psnr[i]+j*.1 for j in range(8)],
                      "ssim":[ssim[i]+j*.001 for j in range(8)],
                      "lpips":[lpips[i]-j*.001 for j in range(8)]}))
        out=selectors(runs)
        self.assertEqual(out["psnr_db"]["first"],20.)
        self.assertEqual(out["psnr_db"]["E0_signed_rho"],21.)
        self.assertEqual(out["psnr_db"]["oracle_K4_per_metric"],26.)
        self.assertEqual(out["ssim"]["oracle_K4_per_metric"],.78)
        self.assertEqual(out["lpips"]["oracle_K4_per_metric"],.23)
        self.assertEqual(out["psnr_db"]["O_orbit_raw_maximum"],26.7)
        self.assertEqual(out["lpips"]["E1_CLIP_G1"],"PENDING_XP_C_CLIP_D1_WORKER")
    def test_requires_four_runs_and_prospective_first(self):
        with self.assertRaisesRegex(RuntimeError,"STOP_A7_SELECT_RUNS"):
            selectors([])
        r=[dict(rep=j) for j in (1,0,2,3)]
        with self.assertRaisesRegex(RuntimeError,"STOP_A7_SELECT_RUNS"):
            selectors(r)
    def test_reference_values_are_labels_not_success_gates(self):
        self.assertEqual(COMPARE_DAPS,dict(psnr_db=25.78,ssim=.743,lpips=.254))
if __name__=="__main__":unittest.main()
