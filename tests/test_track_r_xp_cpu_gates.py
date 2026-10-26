"""CPU-only regressions for metadata normalization and safe PAC symlinks."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.track_r import xp_r0n2_b2_a8_cpu as a8
from scripts.track_r import xp_0_inventory_cpu as xp0

class CpuGates(unittest.TestCase):
    def test_nvidia_ffhq_metadata_unpadded_id(self):
        metadata = {"2763": {"category": "validation"},
                    "62000": {"category": "training"}}
        self.assertEqual(a8.category_lookup(metadata,"02763"),"validation")
        self.assertEqual(a8.category_lookup(metadata,"62000"),"training")

    def test_nested_metadata_category(self):
        metadata={"2763":{"metadata":{"category":"training"}}}
        self.assertEqual(a8.category_lookup(metadata,"02763"),"training")

    def test_reject_missing_category(self):
        with self.assertRaisesRegex(RuntimeError,"STOP_B2_CATEGORY_UNRESOLVED"):
            a8.category_lookup({"1":{"category":"training"}},"02763")

    def test_fast_cohort_provenance_from_original_180(self):
        rows=[{"source":"S1_b22","image_id":f"{i:05d}"} for i in range(100)]
        rows += [{"source":"S2_b24_dev80","image_id":f"{i:05d}"} for i in range(100,180)]
        rows += [{"source":"S5_r0_fast","image_id":"00001"}]
        mapping=a8.frozen_cohort_map(rows)
        self.assertEqual(mapping["1"],"B22_random")
        self.assertEqual(mapping["101"],"DEV80_enriched")
        self.assertEqual(len(mapping),180)

    def test_fast_cohort_rejects_external_image(self):
        rows=[{"source":"S1_b22","image_id":str(i)} for i in range(100)]
        rows += [{"source":"S2_b24_dev80","image_id":str(i)} for i in range(100,180)]
        rows += [{"source":"S5_r0_fast","image_id":"999"}]
        with self.assertRaisesRegex(RuntimeError,"STOP_FAST_ID_NOT_IN_FROZEN_180"):
            a8.frozen_cohort_map(rows)

    def test_symlink_read_only_inside_pac_sandbox(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            x=root/"model.bin"
            x.write_bytes(b"synthetic")
            alias=root/"link"
            alias.symlink_to(x)
            with patch.object(xp0,"ROOT",root):
                self.assertEqual(xp0.safe(alias,True),alias)
                self.assertEqual(xp0.sha(alias),xp0.sha(x))

    def test_symlink_outside_sandbox_fails_closed(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            root=Path(d1)
            other=Path(d2)/"other.bin"
            other.write_bytes(b"synthetic")
            alias=root/"external"
            alias.symlink_to(other)
            with patch.object(xp0,"ROOT",root):
                with self.assertRaisesRegex(RuntimeError,"STOP_PATH_OUTSIDE_PAC_ROOT"):
                    xp0.safe(alias,True)

if __name__=="__main__":
    unittest.main()
