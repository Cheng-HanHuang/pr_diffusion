"""B16 v1+v3 prospective frozen-registry and SHA admission unit contracts."""
import ast
import inspect
import unittest
from pathlib import Path
from scripts.track_r import xp_b16_v1_v3_admission_cpu as m

class TestB16Admission(unittest.TestCase):
    def test_forecast_pins(self):
        self.assertEqual(len(m.FREEZES),2)
        self.assertEqual(m.FREEZES["v1"][3],976)
        self.assertEqual(m.FREEZES["v3"][3],496)
        self.assertEqual(m.FREEZES["v3"][2],
            "9feeef5f5b28593122d45983ca38e99d95d9bfdee45fdfcbfee8a2fc66d5794d")
        self.assertEqual(m.REGISTRY_REF,
            "356ec2a87f73eea905cb2930909ff3053b93e18b")
    def test_b16_complete_contract(self):
        self.assertEqual(m.B16_SHA,
            "778ce4fa7d5ef9299e2530bbb32e8a28b3512e5abe97f0569bb3ec74e00247a9")
        self.assertEqual(m.B16_RECEIPT_SHA,
            "5a3426464fb1164d98138bfbe52212fe38036f4401cf24c1379b6a283ac9e88d")
        self.assertIn("verify_task(",inspect.getsource(m.source))
        self.assertIn("range(1060)",inspect.getsource(m.source))
    def test_forecast_before_sources(self):
        s=inspect.getsource(m.main)
        self.assertLess(s.index("forecasts=freeze()"),
                        s.index("provenance=source()"))
        self.assertIn('CUDA_VISIBLE_DEVICES',s)
        self.assertNotIn('gt_source',inspect.getsource(m.source))
    def test_syntax(self):
        ast.parse(Path(m.__file__).read_text())
if __name__=="__main__":
    unittest.main()
