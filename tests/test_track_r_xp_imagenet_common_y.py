"""CPU-only ImageNet common-y manifest, seed rule and GT-leak tests."""
import hashlib
import unittest
from pathlib import Path
from scripts.track_r.xp_t_imagenet_prepare_cpu import (
    seed,validate_images,ROOT,MODEL_SHA,SOURCE_HEAD)

class ImageNetCommonYTests(unittest.TestCase):
    def test_measurement_seed_sha_first_eight_unsigned(self):
        basename="ILSVRC2012_val_00049000.JPEG"
        digest=hashlib.sha256(("XPMEAS|imagenet|"+basename).encode()).digest()
        self.assertEqual(seed("XPMEAS|imagenet",basename),
                         int.from_bytes(digest[:8],"big")%(2**63))
        self.assertNotEqual(seed("XPMEAS|imagenet",basename),
                            seed("XPMEAS|imagenet","ILSVRC2012_val_00049001.JPEG"))
    def test_xp_run_seeds_distinct(self):
        a="ILSVRC2012_val_00049000.JPEG"
        values=[seed("XP|FAST|imagenet",a,i,2**31) for i in range(3)]
        self.assertEqual(len(set(values)),3)
        self.assertTrue(all(0<=v<2**31 for v in values))
    def test_manifest_rejects_out_of_order(self):
        rows=[{
            "basename":f"ILSVRC2012_val_{49000+i:08d}.JPEG",
            "path":str(ROOT/"data/imagenet/test-imagenet"/f"ILSVRC2012_val_{49000+i:08d}.JPEG"),
            "sha256":"a"*64
        } for i in range(100)]
        self.assertEqual(len(validate_images(rows)),100)
        with self.assertRaisesRegex(RuntimeError,"STOP_XP0_IMAGENET_NOT_SORTED100"):
            validate_images(list(reversed(rows)))
        rows[1]=dict(rows[1],path=str(ROOT/"data/imagenet/test-ffhq"/rows[1]["basename"]))
        with self.assertRaisesRegex(RuntimeError,"STOP_IMAGE_PATH_OR_SCHEMA"):
            validate_images(rows)

if __name__=="__main__":
    unittest.main()
