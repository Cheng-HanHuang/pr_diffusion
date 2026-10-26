"""Synthetic XP-T FAST frozen task order and seed regression (no GPU)."""
import unittest
from scripts.track_r.xp_t_fast_prepare_cpu import xp_seed,make_rows

def frozen():
    originals=[]
    for source,n in (("S1_b22",100),("S2_b24_dev80",80)):
        for j in range(n):
            iid="%05d"%j
            for rep in range(4):
                originals.append({
                    "uid":"S5_r0_fast|FAST|%s_%s|rep%d"%(source,iid,rep),
                    "source":"S5_r0_fast","measurement_source":source,
                    "image_id":iid,"rep":rep,"seed":j*100+rep,
                    "measurement_path":"/data/"+source+"/"+iid,
                    "measurement_sha256":"a"*64
                })
    return {"schema":"prdiff.track_r.r0.5.gt_free_tasks.v1",
            "status":"R0_5_TASKS_FROZEN",
            "head":"c3170587e58e9610c97c6f4ae48964eab2db1b46",
            "fast_head":"a9758b38906b7c7950fa04491f37f6634948db26",
            "n_roots":720,"n_images":180,
            "gt_paths_supplied":False,"rows":originals}

class TimingBatch(unittest.TestCase):
    def test_seed_rule_is_stable_distinct(self):
        self.assertEqual(xp_seed("B22_00000",0),xp_seed("B22_00000",0))
        self.assertNotEqual(xp_seed("B22_00000",0),xp_seed("B22_00000",1))
        self.assertNotEqual(xp_seed("B22_00000",0),xp_seed("B22_00001",0))
        self.assertTrue(0<=xp_seed("B22_00000",0)<2**31)

    def test_frozen_first_three_ids_and_9_tasks(self):
        from unittest.mock import patch
        from scripts.track_r import xp_t_fast_prepare_cpu as p
        with patch.object(p,"sha",return_value="a"*64):
            rows=make_rows(frozen())
        self.assertEqual(len(rows),9)
        self.assertEqual([(r["image_id"],r["rep"]) for r in rows],
             [(i,r) for i in ("00000","00001","00002") for r in range(3)])
        self.assertTrue(all(r["measurement_source"]=="S1_b22" for r in rows))
        self.assertEqual(len({r["seed"] for r in rows}),9)

    def test_bad_historical_measurement_rejected(self):
        from unittest.mock import patch
        from scripts.track_r import xp_t_fast_prepare_cpu as p
        r=frozen()
        r["rows"][0]["measurement_sha256"]="b"*64
        with patch.object(p,"sha",return_value="a"*64):
            with self.assertRaisesRegex(RuntimeError,"STOP_XPT_HISTORICAL_Y_CONFLICT"):
                make_rows(r)

if __name__=="__main__":
    unittest.main()
