"""A9.2(a) synthetic classification and exchangeability tests, CPU only."""
import math
import unittest

from scripts.track_r.xp_a9_exchangeability_cpu import (
    CLASSES,FROZEN_TAU,COHORTS,encode,statistics,two_proportion
)

class TestA9Exchangeability(unittest.TestCase):
    def test_all_five_classes(self):
        zero={format(i,"03b"):10.0 for i in range(8)}
        cases=[
            (dict(zero,**{"000":25.1}),0.06,"G"),
            (dict(zero,**{"111":25.1}),0.06,"O_g"),
            (dict(zero,**{"001":25.1}),0.06,"O_p"),
            (zero,FROZEN_TAU-1e-5,"C"),
            (zero,FROZEN_TAU+1e-5,"I"),
        ]
        for scores,rho,target in cases:
            self.assertEqual(encode(scores,rho),target)

    def test_two_proportion_symmetry_and_zero_variance(self):
        value=two_proportion(80,100,120,150)
        self.assertAlmostEqual(value["z"],0.0)
        self.assertAlmostEqual(value["p"],1.0)
        z=two_proportion(0,50,0,50)
        self.assertEqual(z["p"],1.0)
        self.assertAlmostEqual(
            two_proportion(10,100,80,100)["p"],
            two_proportion(80,100,10,100)["p"])

    def test_full_strata_coverage_and_no_drift(self):
        images={}
        for cohort,n in (("S1_b22",100),("S2_b24_dev80",80)):
            for j in range(n):
                label=CLASSES[j%len(CLASSES)]
                images[(cohort,str(j))]={
                    r:dict(run=r,rho=0.05,label=label) for r in range(64)}
        result=statistics(images)
        self.assertEqual(len(result),30)
        self.assertEqual(
            {(r["cohort"],r["class_label"],
              r["earlier_stage"],r["later_stage"]) for r in result}
            .__len__(),30)
        for row in result:
            self.assertAlmostEqual(row["rate_difference"],0.0)
            self.assertEqual(row["two_proportion_p"],1.0)
            self.assertEqual(row["paired_image_signflip_p"],1.0)
            self.assertEqual(row["two_proportion_holm_p"],1.0)

if __name__=="__main__":
    unittest.main()
