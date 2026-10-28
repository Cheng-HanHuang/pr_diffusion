"""A8 FAST v1 panel semantics and forecast comparison (synthetic, CPU only)."""
import unittest
from scripts.track_r.xp_a8_fast_v1_cpu import compare

class A8FASTPanels(unittest.TestCase):
    def fake(self):
        result={}
        for cohort,n in (("S1_b22",100),("S2_b24_dev80",80)):
            for i in range(n):
                val={}
                for rep in range(64):
                    val[rep]=dict(run=rep,rho=.05+rep*.00001,
                        good_raw=(rep==0 or rep==63),
                        good_orbit=(rep in (0,1,63)))
                result[(cohort,str(i))]=val
        return result
    def test_all_panels_and_four_variants(self):
        fc={}
        for cohort in ("B22","DEV80","POOLED"):
            for panel,n,k in (("primary",64,8),("primary",64,64),
                               ("secondary",16,8),("secondary",16,16)):
                for variant in ("race","S2","tail","race_tau1"):
                    for policy in ("E0","O_raw","O_orbit"):
                        key=("FAST",cohort,panel,n,variant,k,policy)
                        fc[key]=dict(family="FAST",cohort=cohort,panel=panel,
                          panel_n_max=n,model_variant=variant,K=k,
                          policy=policy,point=.5,lo95=0.,hi95=1.,
                          interval_type="predictive_realized_panel")
        rows,summary=compare(self.fake(),fc)
        self.assertEqual(len(rows),3*4*4*3)
        self.assertEqual(len(summary),3*2*4*3)
        for r in rows:
            self.assertTrue(0<=r["observed"]<=1)
        samples=[r for r in rows if r["cohort"]=="POOLED" and
                 r["K"]==64 and r["policy"]=="O_raw"]
        self.assertEqual(len(samples),4)
        self.assertTrue(all(abs(v["observed"]-1)<1e-12 for v in samples))
    def test_bad_cohort_size_stops(self):
        d=self.fake()
        del d[("S1_b22","0")]
        with self.assertRaisesRegex(RuntimeError,"COHORT_SIZE"):
            compare(d,{})
if __name__=="__main__":unittest.main()
