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
    def test_nonterminal_guard(self):
        code=inspect.getsource(m.one)
        self.assertIn('doc.get("active")==[]',code)
        self.assertIn("LOCK_EX|fcntl.LOCK_NB",code)
        self.assertIn("NONTERMINAL_SOURCE_",code)
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
