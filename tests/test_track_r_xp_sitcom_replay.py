"""Synthetic CPU-only SITCOM B22/B24 replay provenance/selection checks."""
from __future__ import annotations
import unittest
from scripts.track_r.xp_sitcom_replay_prepare_cpu import selection,settings

class SitcomReplayPreflightTests(unittest.TestCase):
    def test_select_two_first_ids_and_two_runs_per_cohort(self):
        rows=[]
        for source,n in (("S1_b22",100),("S2_b24_dev80",80)):
            for j in range(n):
                for rep in range(4):
                    rows.append({
                        "uid":f"{source}|SITCOM|{j:05d}|r{rep}",
                        "source":source,"family":"SITCOM",
                        "image_id":f"{j:05d}","run_index":rep,
                    })
        rows.extend([{
            "uid":f"filler{i}","source":"UNRELATED","family":"UNRELATED",
            "image_id":"other","run_index":0
        } for i in range(2520)])
        got=selection(rows)
        self.assertEqual(len(got),8)
        self.assertEqual([(r["source"],r["image_id"],r["run_index"]) for r in got],
            [(src,f"{j:05d}",rep) for src in ("S1_b22","S2_b24_dev80")
             for j in range(2) for rep in (0,1)])
    def test_missing_historical_run_rejected(self):
        rows=[]
        for src,n in (("S1_b22",100),("S2_b24_dev80",80)):
            for j in range(n):
                for rep in range(4):
                    rows.append({"uid":f"{src}-{j}-{rep}",
                                 "source":src,"family":"SITCOM",
                                 "image_id":f"{j:05d}","run_index":rep})
        rows.extend({"uid":f"f{i}","source":"other","family":"other","image_id":"-"}
                    for i in range(2520))
        rows[0]["run_index"]=2
        with self.assertRaisesRegex(RuntimeError,"STOP_XP_SITCOM_HISTORICAL_FOUR_RUNS"):
            selection(rows)
    def test_b22_native_contract_locked(self):
        m={
          "anneal_schedule":"linear","anneal_sigma_max":100.0,
          "anneal_sigma_min":0.1,"anneal_steps":200,
          "diff_schedule":"linear","diff_sigma_min":0.01,"diff_steps":5,
          "lgvd_lr":5e-5,"lgvd_lr_min_ratio":0.01,"lgvd_steps":100,
          "lgvd_tau":0.01,"timestep":"poly-7",
          "model_path_relative_to_sitcom_root":"checkpoint/ffhq256.pt",
          "measurement_preprocessing":"none",
          "master_seed":43,"trajectory_count":4,"selector_step":160,
        }
        c={"sitcom":m}
        self.assertEqual(settings(c)["anneal_steps"],200)
        c["sitcom"]["master_seed"]=44
        with self.assertRaisesRegex(RuntimeError,"STOP_XP_SITCOM_B22_STREAM_OR_SELECTOR"):
            settings(c)
if __name__=="__main__":
    unittest.main()
