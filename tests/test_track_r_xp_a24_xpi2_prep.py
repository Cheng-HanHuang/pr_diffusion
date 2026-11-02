"""A19 XP-I2 D1 ImageNet source freeze prevents using XP-N."""
from pathlib import Path
import ast,unittest
ROOT=Path(__file__).resolve().parents[1]
class A24I2PrepTests(unittest.TestCase):
 def test_prep_no_gt_and_no_native_sampling(self):
  p=ROOT/"scripts/track_r/xp_a24_xpi2_prepare_cpu.py"
  s=p.read_text()
  ast.parse(s,filename=str(p))
  self.assertIn("XP_IMAGENET100_Y_FROZEN",s)
  self.assertIn("imagenet256.pt",s)
  self.assertIn("n_bags=100",s)
  self.assertIn("generation_authorized=False",s)
  self.assertIn("two_bag_exact_replay_required=True",s)
  self.assertIn("one_bag_gpu_worker_required=True",s)
  self.assertNotIn("xp_a18_xpn_measurements",s)
  self.assertNotIn("torch.load",s)
  self.assertNotIn("sampler.sample(",s)
  self.assertNotIn("gt_path",s)
 def test_prep_seed_rule_frozen(self):
  from scripts.track_r.xp_a24_xpi2_prepare_cpu import seed
  n=[f"ILSVRC2012_val_{i:08d}.JPEG" for i in range(49000,49100)]
  vals=[seed(x) for x in n]
  self.assertEqual(len(set(vals)),100)
  self.assertTrue(all(0<=v<2**31 for v in vals))
  self.assertEqual(vals,[seed(x) for x in n])
if __name__=="__main__":unittest.main()
