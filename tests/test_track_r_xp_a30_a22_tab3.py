"""A22 CPU-only exact fixed-K E1 selection and registration tests."""
from __future__ import annotations
import ast
import unittest
from pathlib import Path
from scripts.track_r.xp_a30_a22_tab3_e1_cpu import g1_select,E1_KEYS,E1_AGG_KEYS

ROOT=Path(__file__).resolve().parents[1]

class A22Tab3(unittest.TestCase):
 def test_e0_min_signed_residual_before_orientation(self):
  runs=[dict(run=0,rho=.4,g1_choice="000",PSNR=20.),
        dict(run=1,rho=.1,g1_choice="111",PSNR=24.),
        dict(run=2,rho=.2,g1_choice="000",PSNR=23.),
        dict(run=3,rho=.3,g1_choice="111",PSNR=21.)]
  self.assertEqual(g1_select(runs,1)["run"],0)
  self.assertEqual(g1_select(runs,2)["run"],1)
  self.assertEqual(g1_select(runs,4)["run"],1)
 def test_tie_lowest_run(self):
  a=[dict(run=j,rho=.1) for j in (3,0,2,1)]
  self.assertEqual(g1_select(a,4)["run"],0)
 def test_no_out_of_range_K(self):
  a=[dict(run=j,rho=.1) for j in range(4)]
  with self.assertRaises(RuntimeError):g1_select(a,8)
 def test_exact_table3_schema(self):
  self.assertEqual(len(E1_KEYS),10)
  self.assertEqual(len(E1_AGG_KEYS),8)
  self.assertIn("VGG_LPIPS",E1_AGG_KEYS)
  self.assertIn("g1_choice",E1_KEYS)
 def test_no_generator_and_source_constraints(self):
  s=(ROOT/"scripts/track_r/xp_a30_a22_tab3_e1_cpu.py").read_text()
  ast.parse(s)
  for v in ("read_stage(\"G5\")","load_a7(\"D2\")",
            "load_a7(\"D3\")","clip_new","metrics_pretrained",
            "A7_LOCAL_ZIP_HASH","A7_D2_G5_NATIVE_RECON_HASH",
            "SITCOM_SHARED_MEASUREMENT_SHA","no_GT_in_selection"):
   self.assertIn(v,s)
  self.assertNotIn("torch.cuda.",s)
  self.assertNotIn("torch.compile(",s)

if __name__=="__main__":
 unittest.main()
