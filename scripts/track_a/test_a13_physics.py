#!/usr/bin/env python3
"""Pure synthetic CPU-only A1.3 tests. No PAC real data or native model."""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import unittest
from scripts.track_a import a13_physics as P

class TestA13Physics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch=P.require_cpu()

    def test_source_impulse_and_scale(self):
        r=P.source_witness()
        self.assertEqual(r["pad"],64)
        self.assertLess(r["impulse_max_absolute_error"],2e-7)

    def test_rotation_symmetry(self):
        self.assertTrue(P.symmetry_witness()["rot180_operator_symmetry_reconciled"])

    def test_residual_perfect_signed_and_clipping(self):
        t=self.torch
        x=t.full((1,3,256,256),-.5,dtype=t.float32)
        y=P.magnitude(x)
        r=P.residuals(x,y)
        self.assertLess(r["signed"]["relative_l2"],1e-8)
        self.assertEqual(r["negative_count"],0)
        y2=y.clone()
        y2[0,0,0,0]=-2.0
        r2=P.residuals(x,y2)
        self.assertEqual(r2["negative_count"],1)
        self.assertGreater(r2["signed"]["relative_l2"],0)
        self.assertGreater(r2["signed"]["sse"],r2["clipped"]["sse"])
        self.assertEqual(r2["fft_forward_calls"],1)

    def test_png_psnr_peak_and_rotation(self):
        t=self.torch
        gt=t.full((1,3,256,256),-1.,dtype=t.float32)
        gt[:,:,10,15]=1.
        q=P.png8_gt(gt)
        self.assertAlmostEqual(P.psnr_png8(q,q),120.0,places=5)
        altered=q.clone()
        altered[:,:,10,15]=0
        expected=float(-10*t.log10((altered-q).square().mean().clamp_min(1e-12)))
        self.assertAlmostEqual(P.psnr_png8(altered,q),expected,places=5)
        self.assertNotEqual(P.psnr_png8(t.rot90(q,2,(-2,-1)),q),
                            P.psnr_png8(q,q))

    def test_min_max_and_stable_ties(self):
        rows=[dict(rep=0,r_signed=0.9,psnr_raw_rgb8_db=32.),
              dict(rep=1,r_signed=0.05,psnr_raw_rgb8_db=8.),
              dict(rep=2,r_signed=0.05,psnr_raw_rgb8_db=12.)]
        self.assertEqual(P.select_min(rows,"r_signed")["rep"],1)
        self.assertEqual(P.select_oracle(rows)["rep"],0)
        self.assertEqual(P.select_min(rows[:1],"r_signed")["rep"],0)

    def test_synthetic_noise_reference_is_frozen(self):
        a=P.synthetic_noise_reference()
        b=P.synthetic_noise_reference()
        self.assertEqual(a,b)
        self.assertAlmostEqual(a["sigma"],0.05)
        self.assertGreater(a["noise_rmse_min"],0.049)
        self.assertLess(a["noise_rmse_max"],0.051)
        self.assertEqual(a["poor_fit_cutoff_rmse"],3*a["consistent_cutoff_rmse"])

    def test_nan_missing_and_gt_leakage_are_forbidden(self):
        with self.assertRaises(ValueError):
            P.select_min([{"r_signed":float("nan")}],"r_signed")
        with self.assertRaises(ValueError):
            P.select_min([{"r_signed":float("inf")}],"r_signed")
        with self.assertRaises(ValueError):
            P.select_min([{"psnr_raw_rgb8_db":35.}],"r_signed")
        with self.assertRaises(ValueError):
            P.select_min([],"r_signed")
        # Oracle always requires explicitly supplied post-selection GT evaluation.
        with self.assertRaises(ValueError):
            P.select_oracle([{"rep":0,"r_signed":0.1}])

if __name__=="__main__":
    unittest.main(verbosity=2)
