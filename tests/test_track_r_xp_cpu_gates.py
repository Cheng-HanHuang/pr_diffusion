"""CPU-only regressions for metadata normalization and safe PAC symlinks."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.track_r import xp_r0n2_b2_a8_cpu as a8
from scripts.track_r import xp_0_inventory_cpu as xp0
from collections import defaultdict,Counter

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

    def test_b2_composite_key_and_repair_counts(self):
        categories=defaultdict(Counter)
        prior={"S":{"000":3.0,"111":2.0},
               "S_mid":{"000":3.0,"111":2.0}}
        orbit={"raw_label":"ORBIT_ERROR","s_gt":"111","psnr_raw":19.0,
               "psnr_by_orbit":{"000":19.0,"111":28.0}}
        good={"raw_label":"GOOD","s_gt":"000","psnr_raw":27.0,
              "psnr_by_orbit":{"000":27.0,"111":20.0}}
        key=("S1_b22","B22_random","DAPS","validation")
        a8.update_b2_counts(categories,key,orbit,"111",prior)
        a8.update_b2_counts(categories,key,good,"000",prior)
        self.assertEqual(set(categories),{key})
        self.assertEqual(categories[key]["n"],2)
        self.assertEqual(categories[key]["good"],1)
        self.assertEqual(categories[key]["CLIP_fix"],1)
        self.assertEqual(categories[key]["S_fix"],1)
        self.assertEqual(categories[key]["CLIP_harm"],0)
        self.assertEqual(categories[key]["S_harm"],1)
        with self.assertRaisesRegex(RuntimeError,"STOP_B2_STRATUM_KEY"):
            a8.update_b2_counts(categories,("S1_b22","DAPS","validation"),
                                orbit,"111",prior)

    def test_export_row_integrity_bad_rank(self):
        row={"source":"S1_b22","image_id":"00001","n_runs":2,
             "run_labels":"GI","e0_rank_labels":"GI","e0_rank_order":"1,0",
             "orbit_best_member":"..","clip_fix_ok":"--",
             "dec_clip_face":"00","dec_2f":"00","dec_s_all8":"00",
             "n_GOOD":1,"n_ORBIT_global":0,"n_ORBIT_partial":0,
             "n_CONSISTENT":0,"n_INCONSISTENT":1}
        with patch.object(a8,"EXPECTED",Counter({"GOOD":1,"NONORBIT_INCONSISTENT":1})):
            with self.assertRaisesRegex(RuntimeError,"STOP_A8_ROW_INTEGRITY"):
                a8.validate_counts_export([row]*760,Counter({"GOOD":1,"NONORBIT_INCONSISTENT":1}))

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
