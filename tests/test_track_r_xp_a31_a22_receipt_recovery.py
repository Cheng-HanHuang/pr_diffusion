"""A22 stage-scoped Table3 receipt-only recovery contract (CPU, no PAC data)."""
from __future__ import annotations
import ast
import json
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCORE=ROOT/"scripts/track_r/xp_a30_a22_tab3_e1_cpu.py"
REPAIR=ROOT/"scripts/track_r/xp_a31_a22_tab3_receipt_recovery_cpu.py"
A7=ROOT/"docs/track_r/executors/xp/returns"

class A22ReceiptRecoveryTests(unittest.TestCase):
    def test_scoring_does_not_shadow_argparse_namespace(self):
        src=SCORE.read_text()
        ast.parse(src,str(SCORE))
        self.assertIn("image_rows,aggregate_row=summarize(",src)
        self.assertNotIn("a,b=summarize(",src)
        self.assertIn("implementation_head=a.expected_head",src)

    def test_repair_preserves_native_scoring_and_only_writes_missing_receipt(self):
        src=REPAIR.read_text()
        ast.parse(src,str(REPAIR))
        for marker in (
            "ORIGINAL_HEAD",
            "999d7eb9ee2a254a41c48354e3721155fa046e2a",
            "ORIGINAL_TABLE",
            "XP_A22_TAB3_IMAGENET_E1_G1.csv",
            "XP_A22_TAB3_IMAGE_E1_G1.csv",
            "not receipt_path.exists()",
            "receipt_path.open(\"x\")",
            "receipt_only_repair=True",
            "original_scoring_head=ORIGINAL_HEAD",
            "DAPS_G5_metrics_independently_revalidated=True",
            "SITCOM_G1_choice_copied_from_original_scoring=True",
            "A22_TABLE3_RECEIPT_REPAIR_AND_PUBLICATION_PASS",
        ):
            self.assertIn(marker,src)
        self.assertNotIn("torch.cuda",src)
        self.assertNotIn("torch.compile(",src)
        self.assertNotIn("clip_new(",src)
        self.assertNotIn("metrics_pretrained(",src)

    def test_existing_a7_sources_are_exact_registered_receipts(self):
        from scripts.track_r.xp_a30_a22_tab3_e1_cpu import A7_AGG
        for stage in ("D2","D3"):
            rec=json.loads((A7/f"XP_A7_IMAGE_K4_{stage}_20261028.json").read_text())
            self.assertEqual(A7_AGG[stage][1],
                             rec["local_per_run_detail_sha256"])

if __name__=="__main__":
    unittest.main()
