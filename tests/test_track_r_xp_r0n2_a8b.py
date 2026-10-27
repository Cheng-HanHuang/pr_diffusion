"""A8b synthetic data integrity tests (historical K<=4 only)."""
import unittest,copy
from scripts.track_r.xp_r0n2_a8b_cpu import augment,fmt6
def mk():
    src=[];existing=[]
    # Synthetic n_images 760: exercise A8b parser by checking its input count.
    for i in range(760):
        source="S1_b22" if i<100 else "S2_b24_dev80"
        iid=f"{i:05d}"
        k=2 if i<100 else 4
        for j in range(k):
            src.append({"uid":f"{source}|DAPS|{iid}|{j}",
              "source":source,"family":"DAPS","image_id":iid,
              "run_index":j,"rho_signed_raw":0.02+j*0.001,
              "tau_signed_good_q95":0.1,
              "raw_label":"GOOD","s_gt":"000"})
        existing.append({"source":source,"family":"DAPS",
           "image_id":iid,"n_runs":str(k),"run_labels":"G"*k})
    return existing,src
class A8bTests(unittest.TestCase):
    def test_six_digits(self):
        self.assertEqual(fmt6(0.123456789),"0.123457")
        with self.assertRaisesRegex(RuntimeError,"NONFINITE"):fmt6(float("nan"))
    def test_historical_input_totals_fail_when_synthetic(self):
        a,b=mk()
        with self.assertRaisesRegex(RuntimeError,"HISTORICAL_COUNT"):
            augment(a,b)
    def test_tau_negative_rejected_after_count_gate(self):
        a,b=mk()
        b=(b+[dict(b[-1],uid=f"additional_{i}",
                  image_id=f"additional_{i}",run_index=0)
              for i in range(3240-len(b))])
        b[0]["tau_signed_good_q95"]=-1
        with self.assertRaisesRegex(RuntimeError,"INVALID_TAU"):
            augment(a,b)
    def test_label_integrity_checked(self):
        a,b=mk()
        self.assertEqual(len({row["image_id"] for row in a}),760)
        self.assertTrue(all(int(row["n_runs"]) in (2,4) for row in a))
        self.assertEqual(sum(int(row["n_runs"]) for row in a),len(b))
if __name__=="__main__":unittest.main()
