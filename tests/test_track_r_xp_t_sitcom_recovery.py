"""CPU-only regression: supervisor JSON serialization and pre-GPU recovery.

Synthetic fixtures only. Never touches frozen PAC data or GPU.
"""
from __future__ import annotations
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.track_r import xp_t_sitcom_supervisor as sup
from scripts.track_r import xp_t_sitcom_recovery_cpu as recovery

class SupervisorSerialization(unittest.TestCase):
    def test_exclusive_json_write_and_atomic_status_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"out.json"
            sup.dump_x(p,{"x":1,"status":"RECORDED"})
            self.assertEqual(json.loads(p.read_text()),
                             {"x":1,"status":"RECORDED"})
            with self.assertRaises(FileExistsError):
                sup.dump_x(p,{"x":2})
            sup.update(p,{"x":3,"status":"UPDATED"})
            self.assertEqual(json.loads(p.read_text()),
                             {"x":3,"status":"UPDATED"})

class RecoveryVerification(unittest.TestCase):
    def setup_fixture(self,root):
        run=Path(root)/"xp_t_sitcom_20261026T233451Z"
        run.mkdir()
        for name in ("gpu","pycache"):(run/name).mkdir()
        manifest=run/"XP_T_SITCOM_TASKS.json"
        manifest.write_text('{"frozen":"synthetic"}\n')
        digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
        (run/"XP_T_SITCOM_PREP.json").write_text(json.dumps({
            "status":"XP_T_SITCOM_PREP_PASS",
            "manifest_sha256":digest,"n_roots":18,"no_gpu_work":True}))
        (run/recovery.ORIGINAL_PID).write_bytes(b"")
        (run/recovery.ORIGINAL_LOG).write_text(
            "Traceback (most recent call last):\n"+
            recovery.EXPECTED_ERROR+"\n"+
            "ACTION NEEDED: XP_T_SITCOM_SUPERVISOR "+
            "dump() missing 1 required positional argument: 'fp'\n")
        return run,digest
    def test_exact_pre_gpu_abort_verifies_and_preserves(self):
        with tempfile.TemporaryDirectory() as root:
            run,sha=self.setup_fixture(root)
            before=(run/recovery.ORIGINAL_LOG).read_bytes()
            with (patch.object(recovery,"RUN",run),
                  patch.object(recovery,"MANIFEST_SHA",sha),
                  patch.object(recovery,"_old_supervisor_is_alive",
                               return_value=False)):
                evidence=recovery.audit_aborted_first_launch(run)
                self.assertEqual(evidence["old_pid"],3893497)
                self.assertTrue(evidence["no_gpu_workers_started"])
                self.assertEqual((run/recovery.ORIGINAL_LOG).read_bytes(),before)
                self.assertEqual((run/recovery.ORIGINAL_PID).stat().st_size,0)
                self.assertEqual(list((run/"gpu").iterdir()),[])
                self.assertEqual(list((run/"pycache").iterdir()),[])
    def test_active_original_or_any_gpu_artifact_blocks_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            run,sha=self.setup_fixture(root)
            with (patch.object(recovery,"RUN",run),
                  patch.object(recovery,"MANIFEST_SHA",sha),
                  patch.object(recovery,"_old_supervisor_is_alive",
                               return_value=True)):
                with self.assertRaisesRegex(RuntimeError,"OLD_SUPERVISOR_ACTIVE"):
                    recovery.audit_aborted_first_launch(run)
            (run/"gpu"/"candidate.pt").write_bytes(b"artifact")
            with (patch.object(recovery,"RUN",run),
                  patch.object(recovery,"MANIFEST_SHA",sha),
                  patch.object(recovery,"_old_supervisor_is_alive",
                               return_value=False)):
                with self.assertRaisesRegex(RuntimeError,"NONEMPTY_gpu"):
                    recovery.audit_aborted_first_launch(run)
    def test_authorization_bound_to_exact_old_receipts_and_head(self):
        with tempfile.TemporaryDirectory() as root:
            run,sha=self.setup_fixture(root)
            with (patch.object(recovery,"RUN",run),
                  patch.object(recovery,"MANIFEST_SHA",sha),
                  patch.object(recovery,"_old_supervisor_is_alive",
                               return_value=False)):
                evidence=recovery.audit_aborted_first_launch(run)
                (run/recovery.RECEIPT).write_text(json.dumps({
                    "status":"XP_T_SITCOM_PRE_GPU_RECOVERY_AUTHORIZED",
                    "expected_head":"synthetic-git-head",
                    "evidence":evidence}))
                (run/"XP_T_SITCOM_RECOVERY_SUPERVISOR.log").write_text("")
                with patch.object(recovery.subprocess,"check_output",
                                  return_value="synthetic-git-head\n"):
                    self.assertEqual(
                        recovery.audit_aborted_first_launch(
                            run,require_recovery_receipt=True),evidence)
                    (run/recovery.ORIGINAL_LOG).write_text(
                        "changed failed launch log")
                    with self.assertRaisesRegex(RuntimeError,
                        "UNEXPECTED_ORIGINAL_CRASH"):
                        recovery.audit_aborted_first_launch(
                            run,require_recovery_receipt=True)

if __name__=="__main__":
    unittest.main()
