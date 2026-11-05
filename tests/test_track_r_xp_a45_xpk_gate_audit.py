"""A45 independent XP-K gate auditor unit tests: no PAC/GPU required."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.track_r import xp_a45_xpk_gate_audit_cpu as M

class GateTests(unittest.TestCase):
    def test_fail_closed_A4(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            m={"historical_replay_rows":{"SITCOM":[
                {"uid":"a","image_id":"1","run":0},
                {"uid":"b","image_id":"1","run":1}]}}
            score={"status":"XP_K_A23_5B_ORIGINAL_TWO_ROOT_IDENTITY_REPLAY_PASS",
                   "family":"SITCOM","source_head":"head","passed":True,
                   "n_root":2,"original_A4_PSNR_tolerance_db":.1,
                   "original_A4_label_must_match":True,
                   "strict_thresholds_unchanged":True,
                   "rows":[{"uid":u,"pass_A4":True,
                            "computed_native_class":"G",
                            "frozen_historical_class":"G",
                            "abs_raw_psnr_delta_db":delta}
                           for u,delta in [("a",.1),("b",.0)]]}
            p=root/"score"/"SITCOM_REPLAY_A4_RECEIPT.json"
            p.parent.mkdir()
            p.write_text(json.dumps(score))
            with patch.object(M,"verified_group",return_value=["a","b"]):
                self.assertEqual(M.replay_gate(root,m,"head","SITCOM")["status"],"PASS")
                score["rows"][0]["abs_raw_psnr_delta_db"]=.100000001
                p.write_text(json.dumps(score))
                with self.assertRaisesRegex(RuntimeError,"REPLAY_A4_NUMERICAL"):
                    M.replay_gate(root,m,"head","SITCOM")
                score["rows"][0]["abs_raw_psnr_delta_db"]=.01
                score["rows"][0]["computed_native_class"]="C"
                p.write_text(json.dumps(score))
                with self.assertRaisesRegex(RuntimeError,"REPLAY_A4_NUMERICAL"):
                    M.replay_gate(root,m,"head","SITCOM")
                score["rows"][0]["computed_native_class"]="G"
                score["rows"][0]["abs_raw_psnr_delta_db"]=float("nan")
                p.write_text(json.dumps(score))
                with self.assertRaisesRegex(RuntimeError,"REPLAY_A4_NUMERICAL"):
                    M.replay_gate(root,m,"head","SITCOM")
    def test_no_early_calibration_without_replay_or_f8(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            m={"historical_replay_rows":{"DAPS":[{"uid":"a","image_id":"1","run":0},
                                              {"uid":"b","image_id":"1","run":1}]},
               "calibrated_rows":{"DAPS":[{"image_id":str(i),"run":j}
                                     for i in range(100) for j in range(4)]}}
            out=M.family_status(root,m,"head","DAPS")
            self.assertFalse(out["calibration_authorized"])
            self.assertEqual(out["calibrated_roots_complete"],0)
            self.assertEqual(out["f8"],"PENDING")
    def test_auditor_does_not_execute_native_gpu(self):
        import ast,inspect
        s=inspect.getsource(M)
        ast.parse(s)
        self.assertNotIn("torch.cuda",s)
        self.assertNotIn("subprocess.Popen",s)
        self.assertNotIn("nvidia-smi",s)
        self.assertIn("does_not_authorize_GPU_generation=True",s)
if __name__=="__main__":unittest.main()
