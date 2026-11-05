"""CPU-only fail-closed A23.5b GPU worker interlock tests."""
import ast
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.track_r import xp_a41_xpk_native_gpu as N

class A42Interlock(unittest.TestCase):
    def test_ast(self):
        ast.parse(inspect.getsource(N))
    def test_no_replay_receipt_means_no_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)
            run=base/"xp_a41_xpk_contract"
            run.mkdir()
            manifest={"status":"XP_K_A24_FROZEN_GT_FREE_100x4",
                      "original_GT_access":False,
                      "reference_paths_in_GPU_manifest":False}
            (run/"XPK_GPU_GT_FREE_MANIFEST.json").write_text(
                __import__("json").dumps(manifest))
            digest=N.sha(run/"XPK_GPU_GT_FREE_MANIFEST.json")
            (run/"XPK_SOURCE_FREEZE_RECEIPT.json").write_text(
                __import__("json").dumps({"sha_manifest":digest,"code_head":"head"}))
            args=SimpleNamespace(expected_head="head",family="FAST",
                phase="calibrated",run=str(run),manifest_sha=digest,
                image_id="60044",preflight_only=True,gpu=0)
            with patch.object(N,"BASE",base),patch.object(N,"git",side_effect=lambda wt,*args: "head" if "rev-parse" in args else ""):
                with self.assertRaisesRegex(RuntimeError,
                      "CALIBRATION_REPLAY_GATE_MISSING"):
                    N.valid_input(args)
    def test_direct_worker_requires_science_gates(self):
        s=inspect.getsource(N.valid_input)
        for fragment in (
            "CALIBRATION_REPLAY_GATE_MISSING",
            "CALIBRATION_REPLAY_GATE_NOT_ACCEPTED",
            "CALIBRATION_REPLAY_WRONG_TWO_ROOTS",
            "DAPS_F8_INDEPENDENT_GATE_MISSING",
            "DAPS_F8_INDEPENDENT_GATE_NOT_VERIFIED",
            "original_A4_PSNR_tolerance_db",
            "original_A4_label_must_match",
            "strict_thresholds_unchanged",
            "historical_native_source_head",
        ):
            self.assertIn(fragment,s)
        self.assertLess(s.index('if a.phase=="calibrated":'),
                        s.index('rows=m["historical_replay_rows"]'))
if __name__=="__main__":unittest.main()
