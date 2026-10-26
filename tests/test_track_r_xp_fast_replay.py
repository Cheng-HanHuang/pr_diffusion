"""CPU-only safety checks of FAST replay root selection and gate identity."""
import unittest
from collections import Counter
from scripts.track_r.xp_fast_replay_cpu import frozen_selection

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
