"""A19 XP-I2 native adapter gating: D1 signed-y only, two-bag replay."""
from pathlib import Path
import ast,unittest
ROOT=Path(__file__).resolve().parents[1]
class A24I2NativeTests(unittest.TestCase):
 def test_native_script_frozen_source_and_no_gt(self):
  s=(ROOT/"scripts/track_r/xp_a24_xpi2_pcdaps_gpu.py").read_text()
  ast.parse(s)
  for v in ('"imagenet256.pt"','"num_channels=256"',
            '"num_res_blocks=2"','"32,16,8"',
            '"XP_IMAGENET100"','"A24_I2_STRICT_IMAGENET_WEIGHTS"',
            '"A24_I2_NATIVE_REPLAY_NOT_ACCEPTED"','"A24_I2_ONLY_TWO_REPLAY_BAGS"',
            '"A24_XPI2_NATIVE_FOUR_PARTICLE_BAG_PASS"',
            'sampler.sample(model,x0,operator,y',
            '"FOUR_PARTICLE_BAG"',
            '"MEAN_OF_FOUR_PARTICLE_METRICS_NOT_PIXEL_MEAN"'):
   self.assertIn(v,s)
  self.assertNotIn("torch.no_grad():",s.split("def native(",1)[1].split("def main(",1)[0])
  self.assertNotIn("gt_path",s)
  self.assertNotIn("test-ffhq",s)
 def test_no_production_without_accepted_two_bag_replay(self):
  s=(ROOT/"scripts/track_r/xp_a24_xpi2_pcdaps_gpu.py").read_text()
  self.assertIn('a.index in (0,1)',s)
  self.assertIn('proof.get("n_bags")==2',s)
  self.assertIn('proof.get("model_sha256")==IMAGE_MODEL_SHA',s)
  self.assertIn('--replay-receipt',s)
if __name__=="__main__":unittest.main()
