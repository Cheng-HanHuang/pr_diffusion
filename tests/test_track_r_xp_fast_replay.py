"""CPU-only safety checks of FAST replay root selection and gate identity."""
import unittest
from collections import Counter
from scripts.track_r.xp_fast_replay_cpu import frozen_selection,validate_historical_identity

def synthetic_sources():
    rows=[]
    for source,n in (("S1_b22",100),("S2_b24_dev80",80)):
        for iid in range(n):
            for rep in range(4):
                rows.append({"uid":f"{source}|{iid:05d}|{rep}",
                  "measurement_source":source,"image_id":f"{iid:05d}",
                  "rep":rep, "seed":iid*10+rep})
    return rows

class ReplaySelection(unittest.TestCase):
    def test_historical_scored_schema_uses_catalog_y_sha(self):
        row={"uid":"S5_r0_fast|FAST|S1_b22_00007|rep0",
             "measurement_source":"S1_b22","image_id":"00007","rep":0,
             "measurement_path":"/egr/research-pac/huang248/outputs/y.pt",
             "measurement_sha256":"1"*64}
        history={"uid":row["uid"],"source":"S5_r0_fast","family":"FAST",
                 "image_id":"00007","run_index":0,"measurement_source":"S1_b22",
                 "candidate_path":"/candidate.png","gt_path":"/groundtruth.png",
                 "psnr_raw":28.1,"raw_label":"GOOD","tau_signed_good_q95":0.03}
        cat={"uid":row["uid"],"source":"S5_r0_fast","family":"FAST",
             "image_id":"00007","run_index":0,"measurement_source":"S1_b22",
             "candidate_path":"/candidate.png","gt_path":"/groundtruth.png",
             "gt_format":"png","measurement_path":row["measurement_path"],
             "measurement_sha256":row["measurement_sha256"],
             "candidate_sha256":"2"*64}
        # Crucial regression: historical scored JSONL omits measurement_sha256.
        self.assertNotIn("measurement_sha256",history)
        self.assertTrue(validate_historical_identity(row,history,cat))
        with self.assertRaisesRegex(RuntimeError,"STOP_REPLAY_R07_IDENTITY"):
            validate_historical_identity(row,history,dict(cat,measurement_sha256="0"*64))
        with self.assertRaisesRegex(RuntimeError,"STOP_REPLAY_R07_CATALOG_SCHEMA"):
            broken=dict(cat);broken.pop("measurement_sha256")
            validate_historical_identity(row,history,broken)

    def test_scored_candidate_and_gt_join_must_match(self):
        row={"uid":"u","measurement_source":"S2_b24_dev80","image_id":"02763","rep":1,
             "measurement_path":"/y.pt","measurement_sha256":"a"*64}
        scored={"uid":"u","source":"S5_r0_fast","family":"FAST",
                "image_id":"02763","run_index":1,"measurement_source":"S2_b24_dev80",
                "candidate_path":"/a.png","gt_path":"/gt.png",
                "psnr_raw":12.0,"raw_label":"NONORBIT_INCONSISTENT",
                "tau_signed_good_q95":0.02}
        cat=dict(scored,gt_format="png",measurement_path="/y.pt",
                 measurement_sha256="a"*64,candidate_sha256="b"*64)
        self.assertTrue(validate_historical_identity(row,scored,cat))
        with self.assertRaisesRegex(RuntimeError,"STOP_REPLAY_R07_IDENTITY"):
            validate_historical_identity(row,scored,dict(cat,candidate_path="/b.png"))

    def test_eight_replay_with_seed_order(self):
        selected=frozen_selection(synthetic_sources())
        self.assertEqual(len(selected),8)
        self.assertEqual([(x["measurement_source"],x["image_id"],x["rep"])
          for x in selected],
          [(src,img,rep)
           for src in ("S1_b22","S2_b24_dev80")
           for img in ("00000","00001")
           for rep in (0,1)])
        self.assertEqual(len({r["uid"] for r in selected}),8)

    def test_requires_frozen_180(self):
        rows=[r for r in synthetic_sources() if
              r["image_id"]!="00099" or r["measurement_source"]!="S1_b22"]
        with self.assertRaisesRegex(RuntimeError,"STOP_FROZEN_180_SOURCE_IDS"):
            frozen_selection(rows)

    def test_missing_rep_detected(self):
        rows=[r for r in synthetic_sources() if
              not (r["measurement_source"]=="S1_b22" and
                   r["image_id"]=="00000" and r["rep"]==1)]
        with self.assertRaisesRegex(RuntimeError,"STOP_MISSING_FROZEN_RUN"):
            frozen_selection(rows)

if __name__=="__main__":
    unittest.main()
