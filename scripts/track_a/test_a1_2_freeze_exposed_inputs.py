"""CPU-only A1.2 frozen source-provenance tests (no image/torch import)."""
import hashlib
import unittest
from copy import deepcopy
from scripts.track_a.a1_2_freeze_exposed_inputs import inspect_manifest,SELECTED,SELECTION_SHA,B22_MANIFEST_SHA,GT_ROOT,MEAS_ROOT

def artificial_doc():
    ids=list(SELECTED)
    rows=[]
    for i,ident in enumerate(ids):
        rows.append({
            "image_id":ident,
            "ground_truth_source_path":str(GT_ROOT/f"{(int(ident)//1000)*1000:05d}"/f"{ident}.png"),
            "measurement_path":str(MEAS_ROOT/"dummy"/f"ffhq{ident}_phase_noise005_meas5401.pt"),
            "ground_truth_source_sha256":"a"*64,
            "measurement_file_sha256":"b"*64,
            "ground_truth_tensor_sha256_prior_b22":"c"*64,
            "measurement_tensor_sha256":"d"*64,
        })
    return {
      "schema":"a1.2.exposed-b22-development-inventory.v1",
      "status":"PASS_EXPOSED_DEVELOPMENT_INVENTORY_NOT_PROSPECTIVE",
      "source_exposed_b22_sha256":B22_MANIFEST_SHA,
      "selected_image_ids":ids,
      "selected_id_list_sha256":SELECTION_SHA,
      "gpu_work":False,"protected_b24_confirmation_access":False,
      "rows":rows,
    }

class InputFreezeUnitTest(unittest.TestCase):
    def test_select8_metadata_pass_and_digest(self):
        o=artificial_doc()
        self.assertEqual(len(inspect_manifest(o)),8)
        self.assertEqual(hashlib.sha256("\n".join(SELECTED).encode()).hexdigest(),SELECTION_SHA)
        self.assertEqual([r["image_id"] for r in inspect_manifest(o)],list(SELECTED))

    def test_wrong_cohort_or_wrong_hash_fails(self):
        for field,value in (("schema","not B22"),("status","PROSPECTIVE"),
                            ("source_exposed_b22_sha256","0"*64),
                            ("selected_image_ids",list(reversed(SELECTED))),
                            ("gpu_work",True),("protected_b24_confirmation_access",True)):
            o=artificial_doc()
            o[field]=value
            with self.assertRaises(ValueError):inspect_manifest(o)

    def test_path_escape_and_hash_drift_fail(self):
        for field,value in (("ground_truth_source_path","/tmp/ffhq69073.png"),
                            ("measurement_path","/tmp/ffhq69073.pt"),
                            ("ground_truth_source_sha256","notasha"),
                            ("measurement_tensor_sha256","0"*63)):
            o=artificial_doc()
            o["rows"][0][field]=value
            with self.assertRaises(ValueError):inspect_manifest(o)

if __name__=="__main__":unittest.main()
