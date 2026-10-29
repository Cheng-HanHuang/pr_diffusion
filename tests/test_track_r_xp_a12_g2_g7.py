"""A12 G2-G7 prereg, native adapter and safety regression checks (CPU)."""
import ast,inspect,hashlib,math,unittest
from pathlib import Path
from scripts.track_r import xp_a12_g2_prepare_cpu as g2
from scripts.track_r import xp_a12_g1b_prepare_cpu as g1b
from scripts.track_r import xp_a12_g45_prepare_cpu as g45
from scripts.track_r import xp_a12_g3_prepare_cpu as g3
from scripts.track_r import xp_a12_campaign_autopilot as driver
from scripts.track_r import xp_a12_g367_readiness_cpu as gates

ROOT=Path(g2.__file__).resolve().parents[2]
STAGE_SCRIPTS=(
 "xp_a12_g2_prepare_cpu.py",
 "xp_a12_g1b_prepare_cpu.py",
 "xp_a12_g45_prepare_cpu.py",
 "xp_a12_g45_daps_gpu.py",
 "xp_a12_campaign_fast_gpu.py",
 "xp_a12_campaign_autopilot.py",
 "xp_a12_g3_prepare_cpu.py",
 "xp_a12_g3_pcdaps_gpu.py",
 "xp_a12_g367_readiness_cpu.py",
 "xp_a12_g7_c_clip_cpu.py",
 "xp_a12_g7_c2_gpu.py")
class A12Stages(unittest.TestCase):
 def test_all_sources_parse(self):
  for name in STAGE_SCRIPTS:
   path=ROOT/"scripts/track_r"/name
   ast.parse(path.read_text(),filename=str(path))
 def test_G2_full_paired_sigma_scopes(self):
  self.assertEqual(g2.SIGMAS,(("0.02",0.02),("0.10",0.10)))
  self.assertEqual(g2.seed("0.02","B22_60044",0),
                   g2.seed("0.02","B22_60044",0))
  self.assertNotEqual(g2.seed("0.02","B22_60044",0),
                      g2.seed("0.10","B22_60044",0))
  code=inspect.getsource(g2.main)
  self.assertIn("paired=clean+val*n",code)
  self.assertIn("n=(old-clean)/0.05",code)
  self.assertIn("n_roots=1440",code)
  self.assertIn("gt_paths_supplied=False",code)
 def test_G45_extension_indices_and_native(self):
  self.assertEqual(g45.seed("G4","S1_b22_60044",4),
                   g45.a6_seed("S1_b22_60044",4))
  self.assertNotEqual(g45.seed("G4","S1_b22_60044",4),
                      g45.seed("G4","S1_b22_60044",0))
  self.assertNotEqual(g45.seed("G5","ILSVRC2012_val_00049000.JPEG",4),
                      g45.seed("G5","ILSVRC2012_val_00049000.JPEG",0))
  native=(ROOT/"scripts/track_r/xp_a12_g45_daps_gpu.py").read_text()
  self.assertIn("native.generate(a,mf,row)",native)
  self.assertIn("verify_clean_sparse",native)
 def test_G3_four_native_particles(self):
  native=(ROOT/"scripts/track_r/xp_a12_g3_pcdaps_gpu.py").read_text()
  for word in ("sampler.get_start(1,model)",
      "sampler.sample(model,xstart,operator,y",
      "tuple(result.shape)==(1,4,3,256,256)",
      "load_model_native", "particles.pt",
      "MEAN_OF_FOUR_PARTICLE_METRICS_NOT_PIXEL_MEAN"):
   self.assertIn(word,native)
  self.assertEqual(gates.PC_HEAD,"59b2f046")
  self.assertEqual(g3.PC_HEAD,"59b2f046")
 def test_budget_global(self):
  self.assertEqual(driver.CAP["G2"],10.)
  self.assertEqual(driver.CAP["G3"],45.)
  self.assertEqual(driver.CAP["G4"],30.)
  self.assertEqual(driver.CAP["G5"],40.)
  self.assertEqual(driver.CAMPAIGN_CAP,150.)
  self.assertEqual(driver.PROJECT_CAP,400.)
  self.assertEqual(driver.N["G2"],2880)
  self.assertEqual(driver.N["G3"],180)
  self.assertEqual(driver.N["G4"],720)
  self.assertEqual(driver.N["G5"],400)
  s=inspect.getsource(driver.run_queue)
  for x in ("choose_gpu","STAGE_CEILING","CAMPAIGN_CEILING",
            "PROJECT_CEILING","STOP_A12","os.killpg"):
   self.assertIn(x,s)
 def test_g7_mapping_to_native_1000(self):
  script=(ROOT/"scripts/track_r/xp_a12_g7_c2_gpu.py").read_text()
  self.assertIn("native.M==1000",script)
  self.assertIn("index/999.",script)
  self.assertIn("model.tweedie",script)
  self.assertIn("gt_only",script.lower())
 def test_seal_and_conditional_stage(self):
  controller=(ROOT/"scripts/track_r/xp_a12_campaign_autopilot.py").read_text()
  self.assertIn("D4_outcomes_not_opened=True",controller)
  self.assertIn("G3_CONDITIONAL_CODE_ABSENT",controller) if False else None
  self.assertNotIn("psnr_rgb8",controller)
  self.assertNotIn("candidate_source",controller)
  self.assertIn("G3_CODE_ABSENT_SKIP_PER_A12",controller)
  g1bsrc=inspect.getsource(g1b.main)
  self.assertIn("G2_G3_G4_G5_must_have_launched=True",g1bsrc)
  self.assertIn("A12_campaign_projection_must_fit=True",g1bsrc)
if __name__=="__main__":
 unittest.main()
