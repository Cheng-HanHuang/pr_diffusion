"""A47 deterministic CPU tests; no PAC hardware required."""
import unittest
from scripts.track_r.xp_a47_xpk_gpu_ledger_cpu import (
    charge,remaining,interval_union_seconds)
def ev(id,kind,gpu,t,**kw):
    return dict(id=id,kind=kind,gpu=gpu,wall_unix=t,**kw)
class A47Ledger(unittest.TestCase):
    def test_overlap_same_gpu_union(self):
        rows=[ev("a","START",0,100,family="SITCOM",phase="historical_replay"),
              ev("b","START",0,110,family="FAST",phase="pilot"),
              ev("a","END",0,130),ev("b","END",0,140)]
        self.assertEqual(charge(rows,200)["physical_gpu_seconds"]["0"],40)
    def test_parallel_different_gpu_additive(self):
        rows=[ev("a","START",0,100,family="SITCOM",phase="pilot"),
              ev("b","START",1,100,family="FAST",phase="pilot"),
              ev("a","END",0,130),ev("b","END",1,130)]
        self.assertEqual(charge(rows,200)["physical_gpu_h"],60/3600)
    def test_crashed_worker_charged_to_now(self):
        rows=[ev("a","START",2,100,family="DAPS",phase="calibrated")]
        r=charge(rows,160)
        self.assertEqual(r["unfinished_workers"],["a"])
        self.assertEqual(r["physical_gpu_seconds"]["2"],60)
        self.assertFalse(remaining(rows,160,240,reserve_h=.01)["can_admit_full_reserved_stage"])

    def test_live_worker_does_not_mean_budget_exhausted(self):
        rows=[ev("a","START",0,100,family="SITCOM",phase="historical_replay")]
        r=remaining(rows,160,247.8,0,reserve_h=4.0)
        self.assertTrue(r["can_fit_reserve_under_caps"])
        self.assertFalse(r["can_admit_full_reserved_stage"])
        self.assertEqual(r["unfinished_workers"],["a"])

    def test_exact_boundary_reserve(self):
        self.assertTrue(remaining([],100,399.5,29.5,reserve_h=.5)["can_admit_full_reserved_stage"])
        self.assertFalse(remaining([],100,399.5,29.5,reserve_h=.5001)["can_admit_full_reserved_stage"])
    def test_no_double_charge_on_resume(self):
        rows=[ev("a","START",0,100,family="SITCOM",phase="pilot"),ev("a","END",0,130)]
        self.assertEqual(charge(rows,500)["physical_gpu_seconds"]["0"],30)
        self.assertEqual(charge(rows,999)["physical_gpu_seconds"]["0"],30)
    def test_fail_closed_invalid_events(self):
        base=[ev("a","START",0,100,family="SITCOM",phase="pilot")]
        for rows in ([*base,base[0]],
                     [ev("a","END",0,120)],
                     [*base,ev("a","END",1,120)],
                     [ev("a","START",0,float("nan"),family="SITCOM",phase="pilot")],
                     [ev("a","START",0,200,family="SITCOM",phase="pilot"),ev("a","END",0,100)]):
            with self.subTest(rows=rows),self.assertRaises(ValueError):
                charge(rows,300)
    def test_union_disjoint_touching_nested(self):
        self.assertEqual(interval_union_seconds([(0,10),(2,3),(10,15),(20,30)]),25)
    def test_exceeded_budget_never_authorized(self):
        with self.assertRaisesRegex(ValueError,"XPK_BUDGET_EXCEEDED"):
            remaining([],100,250,30.01)
        with self.assertRaisesRegex(ValueError,"PROJECT_BUDGET_EXCEEDED"):
            remaining([],100,400.01)
if __name__=="__main__":unittest.main()
