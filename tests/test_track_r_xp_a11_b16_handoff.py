"""Synthetic zero-GPU tests for B16 400-root -> 2160-root autonomous handoff."""
from __future__ import annotations
import hashlib
import unittest
from unittest.mock import patch
from scripts.track_r import xp_a11_b16_prod_prepare_cpu as prep
from scripts.track_r.xp_a11_b16_handoff_autopilot import (
    B22_SMOKE,B24_SMOKE,REFERENCE_SEC)

class TestB16Production(unittest.TestCase):
    def test_frozen_seed_domain_matches_historical_algorithm(self):
        image="01113"
        label="B"
        expected=[]
        for r in range(4):
            content="|".join(["B24_DEV80_SITCOM_SOLVER_V1",
                    prep.B24_SCREEN_SHA,label,image,str(r)])
            expected.append(int(hashlib.sha256(content.encode()).hexdigest()[:16],16)&(
                       (1<<63)-1))
        self.assertEqual([prep.seed63(label,image,r) for r in range(4)],expected)
        self.assertEqual(prep.unique_stored_seed(image,
               [x%(2**32) for x in expected]),label)
        with self.assertRaisesRegex(RuntimeError,"B24_SEED_DOMAIN"):
            prep.unique_stored_seed(image,[1,2,3,4])

    def test_build_exact_2160_new_roots_and_no_outcomes(self):
        b22=[]
        for n in range(100):
            iid=f"{60000+n:05d}"
            for rep in range(4):
                b22.append(dict(uid=f"S1_b22|SITCOM|{iid}|run{rep}",
                  source="S1_b22",family="SITCOM",image_id=iid,
                  rep=rep,seed=43,rng_mode="sequential_post_model",
                  measurement_source="S1_b22",
                  measurement_path=f"/fake/{iid}.pt",
                  measurement_sha256="y"))
        catalog=[]
        for n in range(80):
            iid=f"{n+1000:05d}"
            for rep in range(4):
                catalog.append(dict(uid=f"S2_b24_dev80|SITCOM|{iid}|rep{rep}",
                    source="S2_b24_dev80",family="SITCOM",
                    image_id=iid,run_index=rep,
                    measurement_path=f"/fake/{iid}.pt",
                    measurement_sha256="y"))
        def historical_seed(row,mode):
            assert mode=="independent_model_seed"
            return prep.seed63("C",row["image_id"],int(row["run_index"]))%(2**32),{}
        with patch.object(prep,"native_seed_metadata",side_effect=historical_seed):
            b22groups,b24groups=prep.build(b22,catalog)
        self.assertEqual(len(b22groups),100)
        self.assertEqual(len(b24groups),960)
        self.assertEqual(sum(x["new_roots"] for x in b22groups+b24groups),2160)
        self.assertEqual([r["rep"] for r in b22groups[0]["rows"]],
                         list(range(16)))
        self.assertEqual(len(set(r["uid"] for x in b22groups
                                 for r in x["rows"][4:])),1200)
        self.assertEqual(len(set(x["rows"][0]["uid"] for x in b24groups)),960)
        self.assertTrue(all("gt_path" not in r and
                            "historical_candidate_path" not in r
                            for x in b22groups+b24groups for r in x["rows"]))
        self.assertEqual(b22groups[0]["rows"][4]["seed"],43)
        self.assertEqual(b24groups[0]["rows"][0]["seed"],
                         prep.seed63("C","01000",4)%(2**32))

    def test_smoke_sets_and_ffhq_timing_frozen(self):
        self.assertEqual(B22_SMOKE,{0,1})
        self.assertEqual(B24_SMOKE,{100,101,112,113})
        self.assertAlmostEqual(REFERENCE_SEC,32.104482267051935)
        self.assertTrue(16.05 < 22.8 < 1.5*REFERENCE_SEC)

if __name__=="__main__":
    unittest.main()
