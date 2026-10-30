"""CPU regression and prereg checks for A15 CP2/CP3."""
import ast
import inspect
import unittest
from pathlib import Path
from scripts.track_r import xp_a15_cp2_d4_cpu as cp2
from scripts.track_r import xp_a15_cp2_audit_cpu as audit
from scripts.track_r import xp_a15_cp3_score_cpu as cp3
from scripts.track_r import xp_a15_cp3_protocol_cpu as protocol

ROOT=Path(cp2.__file__).resolve().parents[2]
class A15Tests(unittest.TestCase):
 def test_python_syntax(self):
  for name in ("xp_a15_cp2_d4_cpu.py","xp_a15_cp2_audit_cpu.py",
               "xp_a15_cp3_score_cpu.py","xp_a15_cp3_protocol_cpu.py"):
   p=ROOT/"scripts/track_r"/name
   ast.parse(p.read_text(),filename=str(p))
 def test_frozen_sources(self):
  self.assertEqual(cp2.PREREG_HEAD,
   "988aaed5fb61099ca43cbe950a181a6c38ab74c7")
  self.assertEqual(cp2.V4[2],
   "b0ff25bc98ef7ba294db3f33c4df4f1f9835951890ebd879afedf807c9d93da3")
  self.assertEqual(cp2.PROTO[2],
   "d5f973fe4ef82243d6b687981dd4c193822e30634254853adfe8d483b3920658")
  s=inspect.getsource(cp2.provenance)
  self.assertLess(s.index("artifacts=frozen(wt)"),s.index("g1a=jread("))
  self.assertIn("validate_complete",s)
  # GPU G1a execution head differs intentionally from frozen manifest
  # producer head; never authenticate native root against producer SHA.
  self.assertIn('g1a["source_head"]',s)
  self.assertIn('g1b["source_head"]',s)
  self.assertIn('A12_G1A_FROZEN_PRODUCER',s)
  self.assertNotIn('16,16,mf_a["head"]',s)
  self.assertNotIn('32,32,mf_b["head"]',s)
  self.assertIn('"CP2_PRODUCER_VS_EXECUTION_HEADS"',s)
 def test_u_statistic(self):
  one=dict(image_id="test",labels_by_run="G;O_g;I;I",
           rho_signed_by_run=".1;.2;.3;.4")
  clip=[dict(image_id="test",run=i,g1_chosen_good=int(i<2))
        for i in range(4)]
  for k in (1,2,3,4):
   v=cp2.exact_observed([one],clip,4,k)
   self.assertLessEqual(v["E0"],v["O_raw"]+1e-12)
   self.assertLessEqual(v["E1_CLIP"],v["O_orbit"]+1e-12)
  self.assertAlmostEqual(cp2.exact_observed([one],clip,4,4)["O_orbit"],1.)
 def test_cp3_cohorts(self):
  self.assertEqual(cp3.ff_key("S1_b22","60044"),"S1_b22_60044")
  self.assertEqual(cp3.ff_key(None,"S2_b24_dev80_60044"),
                   "S2_b24_dev80_60044")
  self.assertIn("shared_tau=float(np.quantile",inspect.getsource(cp3.score))
 def test_cp3_release_block(self):
  s=inspect.getsource(protocol.main)
  for x in ("CP3_RETURN_AUTHORIZED=False","NFE_matched_comparison_complete=False",
            "LPIPS_SSIM_complete=False","E0_only_primary_no_CLIP"):
   self.assertIn(x,s)
 def test_pipeline_stage_order(self):
  p=(ROOT/"scripts/track_r/xp_a15_cp23_cpu_pipeline.sh").read_text()
  self.assertLess(p.index("xp_a15_cp2_audit_cpu.py"),
                  p.index("xp_a15_cp2_publish.sh"))
  self.assertLess(p.index("xp_a15_cp2_publish.sh"),
                  p.index("xp_a15_cp3_score_cpu.py"))
if __name__=="__main__": unittest.main()
