"""A20 R2 regression: native DAPS export boundary and chained failed ledgers."""
from __future__ import annotations
import ast,hashlib,importlib,subprocess,types,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
SCIENCE_SHA="298b88cdd8f4880fe35fd727c6d7c874dabcd9d09da55cd40a469788d5d19b27"
ORIGINAL_SHA="a6d37d9c4bd319a91cbe4704d8e3f4ac468517f0a56070c83bd6c5c565c980ce"
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class XPCA20R2(unittest.TestCase):
    def test_native_export_is_real_and_imported_correctly(self):
        from scripts.track_r.xp_t_daps4k_prepare_cpu import IMAGENET_MODEL_SHA
        from scripts.track_r.xp_t_daps4k_native_cpu import NATIVE,verify_clean_sparse
        self.assertTrue(callable(verify_clean_sparse))
        self.assertEqual(len(IMAGENET_MODEL_SHA),64)
        self.assertEqual(NATIVE.name,"DAPS_XP_NATIVE_4K")
        gpu_source=(ROOT/"scripts/track_r/xp_a25_xpc3_mid_gpu.py").read_text()
        self.assertIn("from scripts.track_r.xp_t_daps4k_prepare_cpu import IMAGENET_MODEL_SHA",gpu_source)
        self.assertIn("from scripts.track_r.xp_t_daps4k_native_cpu import NATIVE,verify_clean_sparse",gpu_source)
        self.assertNotIn("NATIVE,IMAGENET_MODEL_SHA,verify_clean_sparse)",gpu_source)
        ast.parse(gpu_source)

    def test_chain_is_immutable(self):
        import json
        science=ROOT/"docs/track_r/executors/xp/theory_inputs/XP_C3_A20_SOURCE_MANIFEST.json"
        frozen=ROOT/"docs/track_r/executors/xp/XP_C3_A20_CODE_FREEZE.json"
        r1=ROOT/"docs/track_r/executors/xp/XP_C3_A20_RECOVERY_R1.json"
        self.assertEqual(h(science),SCIENCE_SHA)
        spec=json.loads(frozen.read_text())
        self.assertEqual(spec["source_manifest_sha256"],SCIENCE_SHA)
        self.assertEqual(spec["scorer_script_sha256"],ORIGINAL_SHA)
        d=json.loads(r1.read_text())
        for k in ("corrected_scorer_sha256","corrected_controller_sha256","failed_charged_gpu_h"):
            self.assertIn(k,d)
        self.assertEqual(d["allowed_physical_gpus"],[1,2,3])
        self.assertTrue(d["scientific_definition_unchanged"])

    def test_r2_accounting_and_physical_gpu_mask(self):
        control=importlib.import_module("scripts.track_r.xp_a25_xpc3_autopilot")
        worker=importlib.import_module("scripts.track_r.xp_a25_xpc3_mid_gpu")
        self.assertEqual(control.ALLOWED_PHYSICAL_GPUS,(1,2,3))
        self.assertEqual(worker.ALLOWED_PHYSICAL_GPUS,(1,2,3))
        c=(ROOT/"scripts/track_r/xp_a25_xpc3_autopilot.py").read_text()
        for marker in (
          'R1_FAILED_RUN=BASE/"xp_a25_xpc3_prod_r1_20261103T163302Z"',
          'sha(r1_final)==r2["failed_r1_final_sha256"]',
          'sha(r1_log)==r2["failed_r1_worker_log_sha256"]',
          'total_prior=float(r1["charged_gpu_h"])',
          'abs(total_prior-prior-r1attempts[0]["wall_seconds"]/3600.)<1e-7',
          'charged_gpu_h=previous_charge',
          'RECOVERY_R2_REL',
          'r2["corrected_controller_sha256"]==sha(',
        ):
            self.assertIn(marker,c)
        ast.parse(c)

    def test_source_regression_and_r2_sha(self):
        m=importlib.import_module("scripts.track_r.xp_a25_xpc3_mid_gpu")
        class ReachedSource(Exception):pass
        with patch.object(m,"source",side_effect=ReachedSource):
            with self.assertRaises(ReachedSource):
                m.run(types.SimpleNamespace())
        s=(ROOT/"scripts/track_r/xp_a25_xpc3_mid_gpu.py").read_text()
        for marker in ("R2_CODE_SHA_AND_SCIENCE_UNCHANGED",
                       "R1_CODE_SHA_AND_SCIENCE_UNCHANGED",
                       "original_prior_score_sha256",
                       "ALLOWED_PHYSICAL_GPUS=(1,2,3)",
                       "saved_reconstruction=torch.load("):
            if marker=="original_prior_score_sha256":
                self.assertIn('spec["original_prior_score_sha256"]',s)
            else:self.assertIn(marker,s)

if __name__=="__main__":unittest.main()
