"""CPU tests for A1 procedural engineering input (no native DAPS import)."""
import hashlib
import unittest

import numpy as np
import torch

from scripts.track_a.a1_synthetic_probe_input import (
    N, procedural_rgb_png, model_range_tensor, tensor_content_sha256,
)


class A1SyntheticProbeTest(unittest.TestCase):
    def test_fixed_procedural_pattern_and_tensor(self):
        a=procedural_rgb_png()
        b=procedural_rgb_png()
        self.assertEqual(a.shape,(N,N,3))
        self.assertEqual(a.dtype,np.uint8)
        self.assertTrue(np.array_equal(a,b))
        self.assertGreater(len(np.unique(a[:,:,0])),100)
        gt=model_range_tensor(a)
        self.assertEqual(tuple(gt.shape),(1,3,N,N))
        self.assertEqual(gt.dtype,torch.float32)
        self.assertTrue(torch.isfinite(gt).all().item())
        self.assertGreaterEqual(float(gt.min()),-1)
        self.assertLessEqual(float(gt.max()),1)
        self.assertEqual(tensor_content_sha256(gt),tensor_content_sha256(model_range_tensor(b)))

    def test_dtype_shape_checks_are_fail_closed(self):
        for b in (np.ones((N,N,3),dtype=np.float32),
                  np.ones((N,N,1),dtype=np.uint8)):
            with self.assertRaises(ValueError):
                model_range_tensor(b)

    def test_dtype_shape_are_hashed(self):
        t=torch.arange(12,dtype=torch.float32).reshape(1,3,2,2)
        s=t.double()
        self.assertNotEqual(tensor_content_sha256(t),tensor_content_sha256(s))
        self.assertNotEqual(tensor_content_sha256(t),tensor_content_sha256(t.reshape(1,2,3,2)))
        self.assertEqual(len(tensor_content_sha256(t)),64)

if __name__=="__main__":
    unittest.main()
