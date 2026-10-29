"""G1a autonomous admission, safety and replay gating regression tests."""
import ast,inspect,unittest
from pathlib import Path
from scripts.track_r import xp_a12_g1a_autopilot as m
class TestG1aAutopilot(unittest.TestCase):
    def test_stage_and_project_caps(self):
        self.assertEqual(m.STAGE_CAP_GPU_H,25.)
        self.assertEqual(m.A12_CAMPAIGN_CAP_GPU_H,150.)
        self.assertEqual(m.PROJECT_CAP_GPU_H,400.)
        self.assertAlmostEqual(m.A10_FINAL_GPU_H,55.54578541457511)
        self.assertEqual(m.MAX_ATTEMPTS,2)
    def test_exact_historical_replay_gate_before_production(self):
        code=inspect.getsource(m.main)
        self.assertLess(code.index('stage_loop(a,run,prep,receipt,refs,"replay",state)'),
                        code.index('stage_loop(a,run,prep,receipt,refs,"production",state)'))
        self.assertIn('XP_A12_G1A_REPLAY2_PIXEL_EXACT_PASS',code)
        check=inspect.getsource(m.verify)
        self.assertIn('ImageChops.difference',check)
        self.assertIn('HISTORICAL_REPLAY_PIXEL_MISMATCH_',check)
        self.assertIn('validate_complete',check)
    def test_frozen_1600_and_100_chunks(self):
        j=m.production()
        self.assertEqual(len(j),100)
        self.assertEqual([v["start"] for v in j],list(range(0,1600,16)))
        self.assertTrue(all(v["count"]==16 for v in j))
        self.assertEqual(len(m.workers()),2)
    def test_no_gpu_without_cumulative_project_ledger(self):
        s=inspect.getsource(m.preflight)
        self.assertIn('XP_A12_CUMULATIVE_PROJECT_LEDGER_VERIFIED',s)
        self.assertIn('has_every_stage_since_XP0',s)
        self.assertIn('project_total_before_A12_gpu_h',s)
        self.assertIn('PROJECT_HARD_CAP_HEADROOM',s)
        self.assertIn('source_receipt_sha256',s)
    def test_runtime_co_tenancy_and_attempt_charges(self):
        s=inspect.getsource(m.stage_loop)
        self.assertIn('choose_gpu',s)
        self.assertIn('fraction=.80',s)
        self.assertIn('foreign',s)
        self.assertIn('charge(run,job,v,ret)',s)
        self.assertIn('os.killpg',s)
        self.assertIn('STAGE_GPUH_CEILING',s)
        self.assertIn('STOP_G1A',s)
    def test_outcome_seal(self):
        s=inspect.getsource(m.main)
        self.assertIn('D4_outcomes_opened=False',s)
        self.assertNotIn('psnr(',s)
        self.assertNotIn('gt_source(',s)
    def test_syntax(self):
        path=Path(m.__file__).resolve().parents[2]
        ast.parse((path/"scripts/track_r/xp_a12_g1a_autopilot.py").read_text())
if __name__=="__main__":unittest.main()
