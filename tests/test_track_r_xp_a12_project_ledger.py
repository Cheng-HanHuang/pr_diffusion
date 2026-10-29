"""Pure tests: A12 project GPU-h reconciliation stage boundaries."""
import ast,inspect,unittest
from pathlib import Path
from scripts.track_r import xp_a12_project_ledger_reconcile_cpu as m

class TestA12Ledger(unittest.TestCase):
    def test_disjoint_sources(self):
        names=[x[0] for x in m.MAIN+m.OPTIONAL]
        self.assertEqual(len(names),len(set(names)))
        self.assertEqual(names.count("A10_B16_REPLAY_F8"),1)
        self.assertIn("XP_NIGHT",names)
        self.assertIn("XP_A6",names)
        self.assertIn("XP_A16",names)
        self.assertIn("XP_FAST_REPLAY",names)
        self.assertIn("A10_FAILED_B16_GATE_1",names)
        self.assertIn("A10_FAILED_B16_GATE_2",names)
    def test_a10_not_double_charged(self):
        self.assertAlmostEqual(m.A10_EXPECTED,55.54578541457511)
        self.assertEqual(next(x for x in m.MAIN
            if x[0]=="A10_B16_REPLAY_F8")[2],"A10_total_gpu_h")
        check=inspect.getsource(m.check_cross)
        self.assertIn("xp_f8_legacy_replay_",check)
        self.assertIn("B16_PRODUCTION_STATE.json",check)
        self.assertIn("XP_F8_CONTROLLER_STATE.json",check)
        self.assertIn("A10_HISTORICAL_DAPS_WAVE_RECONCILIATION",check)
    def test_nonterminal_legacy_gate_upper_billing(self):
        code=inspect.getsource(m.one)
        self.assertIn("NONTERMINAL_SOURCE_",code)
        self.assertIn("CONSERVATIVE_HISTORICAL_GATE_ACCOUNTING",code)
        self.assertIn("billing_is_upper_bound",code)
        self.assertIn("exact_gpu_time_known",code)
        task=next(x for x in m.MAIN if x[0]=="A10_FAILED_B16_GATE_2")
        self.assertEqual(task[1],"XP_A12_LEGACY_B16_GATE_104836_CONSERVATIVE_CHARGE.json")
        self.assertEqual(task[2],"charged_gpu_h_upper_bound")
    def test_upper_bound_source_is_nonexact(self):
        from scripts.track_r import xp_a12_legacy_gate_upper_charge_cpu as u
        source=inspect.getsource(u.main)
        self.assertIn('uncertain_worker_confirmed_absent_utc',source)
        self.assertIn('exact_gpu_time_known=False',source)
        self.assertIn('charged_gpu_h_upper_bound',source)
        self.assertIn('PIN_STATE',source)
        self.assertIn('PIN_EVENTS',source)
        # LOCK_EX on Linux needs a write-capable descriptor; opening
        # in 'r' caused EBADF before any GPU admission.
        self.assertIn('lockp.open("a+")',source)
        self.assertNotIn('lockp.open("r")',source)
    def test_fail_closed(self):
        code=inspect.getsource(m.main)
        self.assertLess(code.index("time_sources()"),
                        code.index("with output.open"))
        self.assertIn("audit_folders(stages)",code)
        self.assertIn("project_total+A12_CAP<=PROJECT_CAP",code)
        self.assertIn("A12_PROJECT_LEDGER_STRICT_VERIFIED_PASS",code)
    def test_syntax(self):
        ast.parse(Path(m.__file__).read_text())
if __name__=="__main__":
    unittest.main()
