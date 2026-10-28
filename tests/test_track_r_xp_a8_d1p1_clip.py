"""A8 CLIP D1-P1 contract tests; no model files, GT or GPU."""
import unittest
from scripts.track_r import xp_a8_d1p1_clip_cpu as c

class A8CLIPContracts(unittest.TestCase):
    def test_g1_and_g2_frozen_prompts(self):
        self.assertEqual(c.G1_UP,("a photo.",))
        self.assertEqual(c.G1_INV,("an upside-down photo.",))
        self.assertEqual(len(c.G2_UP),6)
        self.assertEqual(len(c.G2_INV),6)
        self.assertNotEqual(c.G2_UP,c.G2_INV)
    def test_fixed_publication_schema(self):
        self.assertEqual(c.FIELDS,("image_id","run","label",
            "g1_choice","g1_margin","g1_chosen_good",
            "g2_choice","g2_margin","g2_chosen_good"))
    def test_pinned_d1_only(self):
        self.assertTrue(c.D1_LABEL_FILE.name.endswith("A8_STYLE_PER_IMAGE.csv"))
        self.assertEqual(len(c.D1_LABEL_SHA),64)
        self.assertEqual(len(c.CLIP_SHA),64)
        self.assertIn("b7af2f7135beb6",c.ORIGINAL_A6_HEAD)
if __name__=="__main__":unittest.main()
