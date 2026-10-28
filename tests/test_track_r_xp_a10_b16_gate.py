"""A9 B16 400-root replay manifest partition and fail-closed contract tests.

Pure synthetic metadata: no GT, PAC data, GPU or native solver.
"""
from __future__ import annotations
import unittest
from unittest.mock import patch
from scripts.track_r import xp_a10_b16_gate_prepare_cpu as prep

def fake_catalog():
    roots={};scores={}
    for image in range(100):
        iid=f"{image:05d}"
        for rep in range(4):
            uid=f"S1_b22|SITCOM|{iid}|run{rep}"
            r={"uid":uid,"source":"S1_b22","family":"SITCOM",
               "image_id":iid,"run_index":rep,
               "measurement_path":f"/fake/y_{iid}.pt",
               "measurement_sha256":"yhash",
               "candidate_path":f"/fake/c_{iid}_{rep}.pt",
               "candidate_sha256":"chash",
               "gt_path":f"/fake/{iid}.png","gt_format":"png"}
            roots[uid]=r
            scores[uid]={"source":"S1_b22","family":"SITCOM",
                         "image_id":iid,"run_index":rep,
                         "psnr_raw":20.,"raw_label":"NONORBIT_INCONSISTENT",
                         "tau_signed_good_q95":0.045}
    return list(roots.values()),scores

class TestB16FullGate(unittest.TestCase):
    def test_partition_all_100_images_400_roots(self):
        catalog,scores=fake_catalog()
        with patch.object(prep,"source_sha",side_effect=lambda x:
               "yhash" if "/y_" in str(x) else "chash"),patch.object(
               prep,"native_seed_metadata",return_value=(43,{"native_tensor_hash":"x"*64})):
            roots,offline=prep.construct(catalog,scores)
        self.assertEqual(len(roots),len(offline))
        self.assertEqual(len(roots),400)
        self.assertEqual([r["rep"] for r in roots[:8]],[0,1,2,3]*2)
        self.assertEqual(len(set(r["image_id"] for r in roots)),100)
        self.assertNotIn("gt_path",roots[0])
        self.assertNotIn("historical_candidate_path",roots[0])
        self.assertEqual(offline[0]["historical_raw_label"],
                         "NONORBIT_INCONSISTENT")
    def test_missing_run_fails_closed(self):
        catalog,scores=fake_catalog()
        catalog.pop()
        with self.assertRaisesRegex(RuntimeError,"HIST_COUNT"):
            prep.construct(catalog,scores)
    def test_duplicate_run_fails_closed(self):
        catalog,scores=fake_catalog()
        catalog[1]["run_index"]=0
        with patch.object(prep,"source_sha",return_value="yhash"),patch.object(
                prep,"native_seed_metadata",return_value=(43,{})):
            with self.assertRaisesRegex(RuntimeError,"RUNS_0_3"):
                prep.construct(catalog,scores)

if __name__=="__main__":
    unittest.main()
