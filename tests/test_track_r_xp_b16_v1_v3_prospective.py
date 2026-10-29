"""Pure regression tests for B16 CPU scoring, face CLIP and v1/v3."""
import ast,inspect,math,unittest
from pathlib import Path
from scripts.track_r import (
 xp_b16_v1_v3_score_cpu as s,
 xp_b16_v1_v3_faceclip_cpu as f,
 xp_b16_v1_v3_compare_cpu as c,
 xp_d1p2_a11_export_cpu as d,
 xp_d1p2_a11_protocol_cpu as p)
class TestB16Prospective(unittest.TestCase):
    def test_frozen_b16_tau_and_label(self):
        self.assertAlmostEqual(s.TAU["S1_b22"],.0448503)
        self.assertAlmostEqual(s.TAU["S2_b24_dev80"],.0449706)
        self.assertEqual(s.label(25.,26.,"000",.2,.04),"G")
        self.assertEqual(s.label(24.,25.1,"111",.2,.04),"O_g")
        self.assertEqual(s.label(24.,25.1,"101",.2,.04),"O_p")
        self.assertEqual(s.label(24.,24.4,"000",.04,.04),"C")
        self.assertEqual(s.label(24.,24.4,"000",.0401,.04),"I")
    def test_e1_exact_and_pl(self):
        r=[dict(run=i,rho=float(i),chosen_good=int(i==3),
                label="G" if i==3 else "I") for i in range(16)]
        self.assertAlmostEqual(c.e1_exact(r,16),1.0)
        self.assertAlmostEqual(c.e1_exact(r,1),1/16)
        self.assertAlmostEqual(c.conditional_pl(r,16,.95,.045),1.)
        self.assertAlmostEqual(c.conditional_pl(
          [dict(label="I") for _ in range(16)],16,1,1),0.)
    def test_full_grids_and_protocol_inputs(self):
        self.assertEqual(c.KS,(8,12,16))
        self.assertEqual(c.POLICIES,("E0","E1_CLIP","O_raw","O_orbit"))
        self.assertEqual(len(d.CLIP_FIELDS),9)
        self.assertEqual(len(d.STR_FIELDS),10)
        self.assertEqual(len(p.RESULTS),6)
        self.assertTrue(all(x.startswith("D1P2_") for x in p.RESULTS))
    def test_face_clip_is_cpu_only_and_frozen_n2(self):
        body=inspect.getsource(f.main)
        self.assertIn('model_source()',body)
        self.assertIn('PROMPTS_UP+PROMPTS_INV',body)
        self.assertIn('candidate_tensor(',body)
        self.assertIn('CUDA_VISIBLE_DEVICES',body)
        self.assertNotIn('.cuda(',body)
    def test_syntax_all(self):
        for m in (s,f,c,d,p):
            ast.parse(Path(m.__file__).read_text())
if __name__=="__main__":
    unittest.main()
