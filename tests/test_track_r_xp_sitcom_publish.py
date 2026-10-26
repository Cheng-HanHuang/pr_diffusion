"""Synthetic fail-closed SITCOM exact-replay publication assertions; CPU only."""
from __future__ import annotations
import copy,unittest
from scripts.track_r.xp_sitcom_replay_publish_cpu import verify,EXPECTED_MANIFEST_SHA,SOURCE_RUNNER_HEAD

def sample():
    gpu={
      "status":"XP_SITCOM_REPLAY_GENERATION_PASS",
      "runner_head":SOURCE_RUNNER_HEAD,
      "prep_head":"c055423234441fe907fef8dfda825815632ca15f",
      "manifest_sha256":EXPECTED_MANIFEST_SHA,
      "n_roots":8,
      "GT_free_generation":True,
      "timing_clean_eligible":True,
      "gpu_hours":0.10778908291249537,
      "sample_wall_s_median":31.592317841015756,
      "timing":{
          "timing_class":"CLEAN","timing_valid":True,
          "n_samples":48,"foreign_pids":[],"reasons":[]
      }
    }
    status={
      "state":"PASS","reason":None,"completed_roots":8,
      "head":SOURCE_RUNNER_HEAD,"gpu_hours":gpu["gpu_hours"]
    }
    score={
      "status":"XP_SITCOM_REPLAY_PASS",
      "head":SOURCE_RUNNER_HEAD,
      "manifest_sha256":EXPECTED_MANIFEST_SHA,
      "n_roots":8,"n_label_match":8,"n_exact_historical":8,
      "max_abs_delta_db":1.971301713865614e-6,
      "no_gpu_scoring":True,
      "generation_gt_free":True,"original_outputs_immutable":True,
      "results":[{"uid":f"candidate_{i}","pass":True,
                  "exact_historical":True,"replay_label":"GOOD",
                  "historical_label":"GOOD","abs_delta_db":1e-6}
                 for i in range(8)]
    }
    return status,gpu,score

class PublicationGate(unittest.TestCase):
    def test_clean_exact_receipt_accepted(self):
        self.assertTrue(verify(*sample()))
    def test_foreign_process_or_unverified_timing_stops(self):
        s,g,r=sample()
        g["timing"]["foreign_pids"]=[987]
        with self.assertRaisesRegex(RuntimeError,"STOP_SITCOM_PUBLICATION_TIMING"):
            verify(s,g,r)
        s,g,r=sample()
        g["timing"]["n_samples"]=47
        with self.assertRaisesRegex(RuntimeError,"STOP_SITCOM_PUBLICATION_TIMING"):
            verify(s,g,r)
    def test_replay_mismatch_fails_even_if_status_claims_pass(self):
        s,g,r=sample()
        r["results"][1]["replay_label"]="ORBIT_ERROR"
        with self.assertRaisesRegex(RuntimeError,"STOP_SITCOM_PUBLICATION_ROOTS"):
            verify(s,g,r)
        s,g,r=sample()
        r["results"][3]["exact_historical"]=False
        with self.assertRaisesRegex(RuntimeError,"STOP_SITCOM_PUBLICATION_ROOTS"):
            verify(s,g,r)
    def test_missing_results_duplicate_uid_and_budget_stops(self):
        s,g,r=sample()
        r["results"][-1]["uid"]=r["results"][0]["uid"]
        with self.assertRaisesRegex(RuntimeError,"STOP_SITCOM_PUBLICATION_DUPLICATE_UID"):
            verify(s,g,r)
        s,g,r=sample()
        g["gpu_hours"]=2.01
        with self.assertRaisesRegex(RuntimeError,"STOP_SITCOM_PUBLICATION_BUDGET"):
            verify(s,g,r)
        s,g,r=sample()
        s["state"]="CPU_OFFLINE_SCORING"
        with self.assertRaisesRegex(RuntimeError,"STOP_SITCOM_PUBLICATION_NOT_PASS"):
            verify(s,g,r)

if __name__=="__main__":
    unittest.main()
