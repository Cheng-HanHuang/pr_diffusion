"""A19 read-only orchestrator readiness: no unauthorized candidate scoring."""
import ast
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]

class A24Tests(unittest.TestCase):
 def test_inventory_is_cpu_only_and_no_candidate_reads(self):
  path=ROOT/"scripts/track_r/xp_a24_a19_readiness_cpu.py"
  script=path.read_text()
  ast.parse(script,filename=str(path))
  self.assertIn("XP_D4",script)
  self.assertIn("G5",script)
  self.assertIn("XP_N",script)
  self.assertIn("6400",script)
  self.assertIn("800",script)
  self.assertIn("DECISION_COLS",script)
  self.assertIn("before opening candidate outcomes",script)
  self.assertIn('GPU_RULES',script)
  self.assertIn('os.environ.get("CUDA_VISIBLE_DEVICES")!=""',script)
  self.assertNotIn("torch.load",script)
  self.assertNotIn("Image.open(",script)
  self.assertNotIn("subprocess.Popen",script)
  self.assertNotIn("os.kill",script)

 def test_stages_refuse_missing_native_code(self):
  from scripts.track_r.xp_a24_a19_readiness_cpu import SIZES,GPU_RULES,stage
  self.assertEqual(sum(SIZES.values()),13600)
  self.assertEqual(GPU_RULES["NOV4_CPU"],"CPU_ONLY")
  self.assertIn("DIRTY_OK",GPU_RULES["XPC3_PRIOR"])
  self.assertIn("DIRTY_OK",GPU_RULES["XPI2_PCDAPS"])
  self.assertEqual(stage("XP_C3",False,reason="missing")["status"],
                   "WAIT_IMPLEMENTATION_OR_INPUT")
  self.assertIsNone(stage("NOV4",True)["blocker"])

 def test_no_implicit_clean_gpu_for_quality(self):
  from scripts.track_r.xp_a24_a19_readiness_cpu import GPU_RULES
  self.assertIn("CLEAN_REQUIRED_IF_TIMING",GPU_RULES["XPK_G7_CP4"])
  self.assertIn("NOT_TIMING",GPU_RULES["TAB3_E1"])
  self.assertNotIn("CLEAN_ONLY",GPU_RULES["XPC3_PRIOR"])
if __name__=="__main__":unittest.main()
