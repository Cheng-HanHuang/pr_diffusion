"""CPU-only regression tests for the independent prospective A8 audit."""
import itertools
import math
import unittest
import torch

from prdiffusion.track_r.physics import magnitude,to_rgb8
from scripts.track_r.xp_a8_fast_v1_audit_cpu import exact_probs,independently_label


class TestFastA8IndependentAudit(unittest.TestCase):
    def test_exact_combinatorial_probs_against_subsets(self):
        group={}
        for j,(rho,raw,orbit) in enumerate([
            (.1,False,False),(.2,True,True),(.2,False,True),
            (.4,True,True),(.5,False,False)
        ]):
            group[j]=dict(run=j,rho=rho,good_raw=raw,good_orbit=orbit)
        for n in range(1,6):
            for k in range(1,n+1):
                actual=exact_probs(group,n,k)
                sums={"E0":0,"O_raw":0,"O_orbit":0}
                subsets=list(itertools.combinations(range(n),k))
                for idx in subsets:
                    selected=[group[i] for i in idx]
                    winner=min(selected,key=lambda r:(r["rho"],r["run"]))
                    sums["E0"]+=int(winner["good_raw"])
                    sums["O_raw"]+=int(any(r["good_raw"] for r in selected))
                    sums["O_orbit"]+=int(any(r["good_orbit"] for r in selected))
                for policy in sums:
                    self.assertAlmostEqual(actual[policy],
                                           sums[policy]/len(subsets),
                                           places=12)

    def test_independent_orbit_and_residual(self):
        torch.manual_seed(20261028)
        x=torch.rand((1,3,256,256),dtype=torch.float32)*2-1
        y=magnitude(x)
        g8=to_rgb8(torch.flip(x,(-2,-1)))
        row=independently_label(x,g8,y)
        self.assertTrue(row["good_orbit"])
        self.assertLess(row["rho"],1e-5)
        self.assertGreaterEqual(row["rho"],0)
        self.assertTrue(math.isfinite(row["rho"]))

if __name__=="__main__":
    unittest.main()
