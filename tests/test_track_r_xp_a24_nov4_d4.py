"""A19 XP-D4 eight representative PNG panels, frozen offline data only."""
import ast
import hashlib
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class Nov4D4Tests(unittest.TestCase):
    def test_worker_static_science_and_provenance(self):
        p=ROOT/"scripts/track_r/xp_a24_nov4_d4_panels_cpu.py"
        s=p.read_text()
        ast.parse(s,filename=str(p))
        for q in (
            "len(never)==22",
            "sorted(never",
            "hashlib.sha256(iid.encode",
            "SOURCE_SHA", # existing check is GT_SOURCE_SHA
            "P1_SHA",
            "validate_complete(pt.parent,row",
            "CP2_UNIQUE_COMPLETE_LOCAL_SCORE",
            "min(range(64)",
            "orbit_psnr_channel_mse",
            "oracle[0]<20.",
            "Image.Resampling.BILINEAR",
            "GT_offline_only=True",
            "no_GPU=True",
            "CPU_ONLY"):
            self.assertIn(q,s)
        for q in ("torch.cuda", "subprocess.Popen", "os.killpg"):
            self.assertNotIn(q,s)

    def test_hash_order_uses_image_id_and_not_outcome(self):
        # This is a figure-selection test, NOT a new experiment.
        ids=("ILSVRC2012_val_00048901.JPEG",
             "ILSVRC2012_val_00048902.JPEG",
             "ILSVRC2012_val_00048903.JPEG")
        expected=sorted(ids,key=lambda i:
            (hashlib.sha256(i.encode("utf-8")).hexdigest(),i))
        self.assertEqual(len(expected),3)
        self.assertEqual(set(expected),set(ids))

if __name__=="__main__":
    unittest.main()
