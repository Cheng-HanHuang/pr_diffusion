"""A7 postfreeze U-stat kernel and forecast interval tests; no PAC inputs."""
import unittest,itertools,math,random
from scripts.track_r import xp_a7_postfreeze_kernel_cpu as k

class A7PostFreezeKernel(unittest.TestCase):
    def test_exhaustive_small_k_with_ties(self):
        rng=random.Random(1227)
        for n in range(1,6):
            for _ in range(70):
                runs=[]
                for j in range(n):
                    g=bool(rng.randrange(2))
                    runs.append(dict(run=j,rho=rng.choice([.1,.2,.2,.3]),
                        good_raw=g,good_orbit=g or bool(rng.randrange(2))))
                for kk in range(1,n+1):
                    exact=k.exact_panel(runs,kk)
                    brute=k.brute(runs,kk)
                    for policy in k.POLS:
                        self.assertAlmostEqual(exact[policy],brute[policy],places=12)
    def test_oracle_dominance(self):
        rows=[dict(run=i,rho=.1+i/10,good_raw=i==3,
                   good_orbit=i in (1,3)) for i in range(4)]
        for kk in range(1,5):
            got=k.exact_panel(rows,kk)
            self.assertLessEqual(got["E0"],got["O_raw"]+1e-12)
            self.assertLessEqual(got["O_raw"],got["O_orbit"]+1e-12)
    def test_predictive_interval_agreement(self):
        good=dict(point=.5,lo95=.4,hi95=.6,
                  interval_type="predictive_panel")
        self.assertTrue(k.compare_interval(.6,good)["inside"])
        self.assertFalse(k.compare_interval(.601,good)["inside"])
        self.assertAlmostEqual(k.compare_interval(.5,good)["standardized_residual"],0)
    def test_parameter_only_combined_uncertainty(self):
        f=dict(point=.5,lo95=.49,hi95=.51,
               interval_type="parameter_only",se_observed=.15)
        self.assertTrue(k.compare_interval(.78,f)["inside"])
        with self.assertRaisesRegex(ValueError,"SE_obs"):
            k.compare_interval(.5,{k:v for k,v in f.items() if k!="se_observed"})
    def test_freeze_rejects_unregistered(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as temp:
            path=Path(temp)/"registry.md"
            path.write_text("| Freeze | Branch | Commit | Path | SHA-256 | Time | Seen |\n"
                            "|---|---|---|---|---|---|---|\n"
                            "| v1 | — | — | pending | — | — | — |\n")
            with self.assertRaisesRegex(RuntimeError,"BAD_REGISTRY"):
                k.verify_freeze(".",path,"v1","badsha",
                                "docs/pred.json","abc","out")
    def test_no_explicit_k_gt4_preview(self):
        self.assertEqual(k.PANEL["ffhq"]["FAST"],(64,16))
        self.assertEqual(k.PANEL["ffhq"]["DAPS"],(8,))
        self.assertEqual(k.PANEL["imagenet"]["FAST"],(16,64))
if __name__=="__main__":unittest.main()
