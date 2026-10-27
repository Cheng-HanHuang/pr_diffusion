"""Synthetic only D1 P1 theory-input clearance tests."""
import unittest
from scripts.track_r.xp_d1p1_theory_publish_cpu import validated
from scripts.track_r.xp_night_publish_cpu import GEN_HEAD,EXPECTED_MANIFEST_HASHES

def fixture():
    rows=[]
    for i in range(100):
        rows.append({
          "image_id":f"image_{i:03d}.png","source":"XP_D1_P1","family":"FAST",
          "n_runs":"4","n_GOOD":"2","n_ORBIT_global":"1",
          "n_ORBIT_partial":"0","n_CONSISTENT":"0",
          "n_INCONSISTENT":"1",
          "run_labels":"GGOI","e0_rank_labels":"GGOI",
          "e0_rank_order":"0,1,2,3",
          "rho_signed_by_run":"0.01;0.02;0.03;0.04",
          "tau_signed_good_q95":"0.0195",
          "orbit_best_member":"..7."
        })
    receipt={"status":"XP_D1_P1_K4_EVAL_PASS",
          "native_generation_head":GEN_HEAD,
          "stage_manifest_sha256":EXPECTED_MANIFEST_HASHES["D1_P1"],
          "n_images":100,"n_runs":400,"run_indices":[0,1,2,3],
          "primary_good_psnr_db":20.,
          "family":"FAST","dataset":"imagenet",
          "sealed_D1_P2_scored":False,
          "sealed_A64_scored":False,
          "sealed_XP_A16_scored":False,
          "label_counts":{"G":200,"O":100,"I":100},
          "use_for_v2_prediction_freeze":True}
    return rows,receipt

class TheoryRelease(unittest.TestCase):
    def test_synthetic_clearance(self):
        a,b=fixture()
        self.assertTrue(validated(a,b))
    def test_no_embargo_circumvention(self):
        a,b=fixture()
        b["sealed_D1_P2_scored"]=True
        with self.assertRaisesRegex(RuntimeError,"RECEIPT"):
            validated(a,b)
    def test_rank_strings_must_match_run_index(self):
        a,b=fixture()
        a[0]["e0_rank_labels"]="IGGO"
        with self.assertRaisesRegex(RuntimeError,"PER_IMAGE"):
            validated(a,b)
    def test_tau_constant(self):
        a,b=fixture()
        a[-1]["tau_signed_good_q95"]="0.03"
        with self.assertRaisesRegex(RuntimeError,"IDENTITY"):
            validated(a,b)

if __name__=="__main__": unittest.main()
