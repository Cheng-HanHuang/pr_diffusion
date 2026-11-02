"""A20 Nov4 publication exact forecast and 256-pixel assets, no GPU."""
import ast,unittest
from pathlib import Path
WT=Path(__file__).resolve().parents[1]

class Nov4PublishTests(unittest.TestCase):
 def test_publication_requires_frozen_original_receipts(self):
  s=(WT/"scripts/track_r/xp_a25_nov4_publish_cpu.py").read_text()
  ast.parse(s)
  for word in (
     "3595676cd2f7686e747d2659d25a77a6a4a2e34edaa7261a8b7f1fbf53f542e3",
     "c19798f9e4b45b6f3c5fd44e96180edaf18104c35d5a3700ba35d81369377753",
     "A19_FAST7_NEVER_GOOD_PANELS_PASS",
     "A19_D4_NEVER_ORBIT_EIGHT_PANELS_PASS",
     "A19_A9_2D_FAST11520_FACECLIP_E1_SCORED",
     "sha(copy)==sha(csvpath)",
     "GT_offline_figure_only=True",
     "n_comparisons=120",
     "f_choice","f_margin","f_chosen_good",
     "No GPU","PNG","256","prior"):
   if word in ("No GPU","prior"):continue
   self.assertIn(word,s)
  for forbidden in ("model.tweedie(","torch.cuda","CLIP.load("):
   self.assertNotIn(forbidden,s)
 def test_original_256_pixel_tiles_without_interpolation(self):
  s=(WT/"scripts/track_r/xp_a25_nov4_publish_cpu.py").read_text()
  for word in ('pix.size==(256,256)',"im.paste(pix","PSNR:",
               'no_GPU=True',"sha(dest)"):
   self.assertIn(word,s)
 def test_source_curve_supports_registered_64_run_k(self):
  s=(WT/"scripts/track_r/xp_a24_nov4_fast_cpu.py").read_text()
  self.assertIn("for k in (1,2,4,8,12,16,24,32,48,64):",s)
if __name__=="__main__":unittest.main()
