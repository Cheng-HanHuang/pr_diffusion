#!/usr/bin/env python3
"""No-GPU structural and numerical sanity tests for A16 NP/FAST autopilot."""
from __future__ import annotations
import unittest

from scripts.track_a.a16_np_hybrid_core import (
    Arm, fast_effective_correction, gradient_correction, k_for_step, should_correct
)


class TestA16Policies(unittest.TestCase):
    def test_original_native_proposal_and_exact_nfe(self):
        a=Arm(name="native",steps=1000)
        a.validate()
        self.assertEqual(sum(k_for_step(a,i,300) for i in range(999))+1,2200)
        self.assertEqual(k_for_step(a,0,300),5)
        self.assertEqual(k_for_step(a,300,300),1)

    def test_sparse_and_adaptive(self):
        for policy in ("sparse","adaptive","beam2","budget2"):
            a=Arm(name=policy,proposal=policy)
            a.validate()
            self.assertGreaterEqual(min(k_for_step(a,i,300) for i in range(999)),1)
        native=sum(k_for_step(Arm(name="baseline"),i,300) for i in range(999))
        for policy in ("sparse","adaptive","beam2"):
            a=Arm(name=policy,proposal=policy)
            self.assertLess(sum(k_for_step(a,i,300) for i in range(999)),native)

    def test_correction_window(self):
        a=Arm(name="active",correction="fast",placement="post",stride=40)
        self.assertFalse(should_correct(a,5))
        self.assertFalse(should_correct(a,200))
        self.assertTrue(should_correct(a,320))
        self.assertFalse(should_correct(a,970))

    def test_bad_arms_rejected(self):
        with self.assertRaises(AssertionError):
            Arm(name="bad",correction="none",placement="post").validate()
        with self.assertRaises(AssertionError):
            Arm(name="bad",ranking="corrected_lf",correction="fast",placement="post").validate()

    def test_fast_native_k2_identity_gradient(self):
        import torch
        p=torch.randn(1,1,5,5)
        A=lambda x: x
        v=A(p)
        gamma=.75
        rho=200.
        first_gradient=(p-p)/gamma+rho*(A(p)-v)
        self.assertTrue(torch.equal(first_gradient,torch.zeros_like(p)))
        # This is the correct dual-compensated effective second target.
        y=torch.zeros_like(p)
        r=A(p)-y
        eps=.05
        alpha=min(1.,eps/float(r.norm()))
        v1=y+alpha*r
        u1=A(p)-v1
        b2=v1-u1
        g2=rho*(A(p)-b2)
        expected=2*rho*(1-alpha)*(A(p)-y)
        self.assertTrue(torch.allclose(g2,expected,atol=3e-5))

    def test_cpu_fast_and_gradient_no_nan(self):
        import torch
        p=torch.rand(1,1,8,8)
        y=torch.zeros_like(p)
        counts={"fast_updates":0,"gradient_updates":0}
        f=fast_effective_correction(lambda x:x,p,y,gamma=.5,steps=2,stats=counts)
        g=gradient_correction(lambda x:x,p,y,gamma=.5,rho=200.,steps=2,stats=counts)
        self.assertTrue(torch.isfinite(f).all())
        self.assertTrue(torch.isfinite(g).all())
        self.assertEqual(counts["fast_updates"],1)
        self.assertEqual(counts["gradient_updates"],1)


if __name__=="__main__":
    unittest.main()
