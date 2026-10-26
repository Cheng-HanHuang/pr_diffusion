"""Regression tests for CPU-only reuse of the frozen successful FAST GPU run."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.track_r import xp_fast_replay_recover_cpu as recover

class ReplayRecovery(unittest.TestCase):
    def test_recovery_scored_receipt_old_head_and_clean_science(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/"xp_fast_replay"
            run.mkdir()
            rows=[{"uid":"r"+str(i),"pass":True,"abs_delta_db":0.0,
                   "label":"GOOD","historical_label":"GOOD"}
                  for i in range(8)]
            payload={"status":"XP_FAST_REPLAY_PASS","head":"c"*40,
                     "generation_head":recover.SOURCE_HEAD,"n_runs":8,
                     "results":rows,"max_abs_delta_db":0.0,"n_png_exact":8}
            (run/"XP_FAST_REPLAY_OFFLINE_RECEIPT.json").write_text(json.dumps(payload))
            with patch.object(recover,"RUN",run),patch.object(recover,"ROOT",Path(tmp)):
                self.assertEqual(recover.validate_scored(recover.SOURCE_HEAD,"c"*40),payload)
                bad=dict(payload, generation_head="d"*40)
                (run/"XP_FAST_REPLAY_OFFLINE_RECEIPT.json").write_text(json.dumps(bad))
                with self.assertRaisesRegex(RuntimeError,"STOP_RECOVERED_REPLAY_SCHEMA"):
                    recover.validate_scored(recover.SOURCE_HEAD,"c"*40)

    def test_never_accept_numerical_or_label_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/"xp_fast_replay"
            run.mkdir()
            rows=[{"uid":str(i),"pass":True,"abs_delta_db":0.0,
                   "label":"GOOD","historical_label":"GOOD"} for i in range(8)]
            rows[4]["abs_delta_db"]=0.25
            payload={"status":"XP_FAST_REPLAY_PASS","head":"b"*40,
                     "generation_head":recover.SOURCE_HEAD,
                     "n_runs":8,"results":rows}
            (run/"XP_FAST_REPLAY_OFFLINE_RECEIPT.json").write_text(json.dumps(payload))
            with patch.object(recover,"RUN",run),patch.object(recover,"ROOT",Path(tmp)):
                with self.assertRaisesRegex(RuntimeError,"STOP_RECOVERED_REPLAY_SCHEMA"):
                    recover.validate_scored(recover.SOURCE_HEAD,"b"*40)
                rows[4]["abs_delta_db"]=0.0
                rows[4]["historical_label"]="ORBIT_ERROR"
                (run/"XP_FAST_REPLAY_OFFLINE_RECEIPT.json").write_text(json.dumps(payload))
                with self.assertRaisesRegex(RuntimeError,"STOP_RECOVERED_REPLAY_SCHEMA"):
                    recover.validate_scored(recover.SOURCE_HEAD,"b"*40)

if __name__=="__main__":
    unittest.main()
