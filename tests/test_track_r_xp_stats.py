"""XP pure-CPU synthetic tests: ties, probabilities, embargo, root safeguards."""
import itertools,unittest
from math import comb
from prdiffusion.track_r.xp_stats import (argmin_weights,argmax_weights,
    success_oracle,expected_pick,freeze_gate)
class XPStats(unittest.TestCase):
    def test_min_exact_enumeration(self):
        vals=[7,4,4,1,9]
        indices=[2,0,1,3,4]
        for k in range(1,6):
            ws=argmin_weights(vals,k,indices)
            brute=[0.]*len(vals)
            for subset in itertools.combinations(range(len(vals)),k):
                j=min(subset,key=lambda j:(vals[j],indices[j]))
                brute[j]+=1/comb(len(vals),k)
            for a,b in zip(ws,brute):self.assertAlmostEqual(a,b)
    def test_max_exact_enumeration(self):
        vals=[7,4,4,1,9]
        for k in range(1,6):
            ws=argmax_weights(vals,k)
            brute=[0.]*len(vals)
            for subset in itertools.combinations(range(len(vals)),k):
                j=max(subset,key=lambda j:(vals[j],j))
                brute[j]+=1/comb(len(vals),k)
            for a,b in zip(ws,brute):self.assertAlmostEqual(a,b)
    def test_oracle_pass_at_k(self):
        n=8;g=3
        for k in range(1,n+1):
            outcomes=list(itertools.combinations(range(n),k))
            brute=sum(bool(set(s)&set(range(g))) for s in outcomes)/len(outcomes)
            self.assertAlmostEqual(success_oracle(n,g,k),brute)
    def test_embargo(self):
        self.assertFalse(freeze_gate("No predictions registered","FAST","ffhq",[4]))
        self.assertTrue(freeze_gate("No predictions registered","FAST","ffhq",[0,1,2,3]))
        self.assertFalse(freeze_gate("No predictions registered","DAPS","ffhq",[2]))
        self.assertTrue(freeze_gate("v1 registered","DAPS","ffhq",[2]))
        self.assertFalse(freeze_gate("v1 registered","FAST","imagenet",[4]))
        self.assertTrue(freeze_gate("v2 registered","FAST","imagenet",[4]))
    def test_invalid(self):
        with self.assertRaises(ValueError):argmin_weights([1,2],3)
        with self.assertRaises(ValueError):argmax_weights([1,2],0)
        with self.assertRaises(ValueError):success_oracle(4,5,2)
if __name__=="__main__":unittest.main()
