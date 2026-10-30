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
 "xp_a12_g7_c2_gpu.py",
 "xp_a13_g1b_autopilot.py",
 "xp_a13_b16_protocol_cpu.py",
 "xp_a12_g6_prepare_cpu.py",
 "xp_a12_g6_daps_gpu.py",
 "xp_a12_g6_clean_autopilot.py",
 "xp_a12_g7_c2_autopilot.py",
 "xp_a13_f8_score_cpu.py",
 "xp_a13_f8_faceclip_cpu.py",
 "xp_a13_f8_v1_v3_compare_cpu.py",
 "xp_a13_f8_protocol_cpu.py")
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
 def test_G2_R07_catalog_uses_run_index(self):
  # R0.7's offline catalog is not a GPU manifest: it has run_index, not rep.
  r07=(ROOT/"scripts/track_r/r0_7_prepare_cpu.py").read_text()
  self.assertIn('"run_index":int(r["run_index"])',r07)
  code=inspect.getsource(g2.original)
  self.assertIn('rr.sort(key=lambda x:int(x["run_index"]))',code)
  self.assertIn('[int(r["run_index"]) for r in rr]==[0,1,2,3]',code)
  self.assertNotIn('x["rep"]',code)
  self.assertNotIn('r["rep"]',code)
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
 def test_A13_G2G5_cli_and_cp1_ledger(self):
  code=(ROOT/"scripts/track_r/xp_a12_campaign_autopilot.py").read_text()
  self.assertIn('p.add_argument("--g1b-prep",default=None)',code)
  self.assertIn('if a.g1b_prep else ()',code)
  self.assertIn('p.add_argument("--preflight-only",action="store_true")',code)
  publish=(ROOT/"scripts/track_r/xp_a13_cp1_publish.sh").read_text()
  self.assertIn('XP_A13_CP1_PROJECT_GPU_LEDGER.json',publish)
  self.assertIn('A10_included_in_project_prior_but_excluded_from_A12',publish)
 def test_g7_mapping_to_native_1000(self):
  script=(ROOT/"scripts/track_r/xp_a12_g7_c2_gpu.py").read_text()
  self.assertIn("native.M==1000",script)
  self.assertIn("index/999.",script)
  self.assertIn("model.tweedie",script)
  self.assertIn("gt_only",script.lower())
 def test_seal_and_conditional_stage(self):
  controller=(ROOT/"scripts/track_r/xp_a12_campaign_autopilot.py").read_text()
  self.assertIn("D4_outcomes_not_opened=True",controller)
  self.assertIn("g1b_final",controller)
  self.assertNotIn("campaign=A10+",controller)
  self.assertIn("G3_CONDITIONAL_CODE_ABSENT",controller) if False else None
  self.assertNotIn("psnr_rgb8",controller)
  self.assertNotIn("candidate_source",controller)
  self.assertIn("G3_CODE_ABSENT_SKIP_PER_A12",controller)
  g1bsrc=inspect.getsource(g1b.main)
  self.assertIn("A13_prereg_priority=\"G1a_then_G1b_immediately\"",g1bsrc)
  self.assertIn("v4_registered_sha256=",g1bsrc)
  g1b_ctl=(ROOT/"scripts/track_r/xp_a13_g1b_autopilot.py").read_text()
  self.assertIn("a12_before=float(g1a[\"stage_gpu_h\"])",g1b_ctl)
  self.assertNotIn("A10_FINAL_GPU_H+",g1b_ctl)
  self.assertIn("STAGE_CAP=40.",g1b_ctl)
  self.assertIn("PROJECT_CAP=400.",g1b_ctl)
  b16=(ROOT/"scripts/track_r/xp_a13_b16_protocol_cpu.py").read_text()
  self.assertIn("0.8186",b16)
  self.assertIn("0.0448503",b16)
  self.assertIn("0.0449706",b16)
  self.assertIn("abs(weighted[k]-ref)<0.002",b16)
 def test_A13_CP1_F8_contracts(self):
  scorer=(ROOT/"scripts/track_r/xp_a13_f8_score_cpu.py").read_text()
  self.assertIn("n_historical=520,n_new=920",scorer)
  self.assertIn("load_png_rgb8",scorer)
  self.assertIn("rho_signed(from_rgb8(x8),y)",scorer)
  self.assertIn("XP_F8_NEW_ROOT_COMPLETE.json",scorer)
  clip=(ROOT/"scripts/track_r/xp_a13_f8_faceclip_cpu.py").read_text()
  self.assertIn("PROMPTS_UP,PROMPTS_INV",clip)
  self.assertIn("CLIP_SHA",clip)
  forecast=(ROOT/"scripts/track_r/xp_a13_f8_v1_v3_compare_cpu.py").read_text()
  self.assertIn('len(prediction)==112',forecast)
  self.assertIn('FREEZES',forecast)
  protocol=(ROOT/"scripts/track_r/xp_a13_f8_protocol_cpu.py").read_text()
  self.assertIn('c598176:docs/track_r/protocol/gtfree_protocol.py',protocol)
  self.assertIn('D1_GAIN=0.0033265974759615347',protocol)
  self.assertIn('H2=bool(h2)',protocol)
if __name__=="__main__":
 unittest.main()
