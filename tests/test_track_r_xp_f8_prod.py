"""Static and pure protocol regression tests for embargo-safe F8 920."""
from __future__ import annotations
import ast,hashlib,inspect,unittest
from pathlib import Path
from scripts.track_r import xp_f8_prod_prepare_cpu as prep
from scripts.track_r import xp_f8_prod_worker_gpu as gpu
from scripts.track_r import xp_f8_prod_controller as ctl

class TestF8NewProduction(unittest.TestCase):
    def test_exact_920_and_seed_protocol(self):
        self.assertEqual(prep.A10_LIMIT,70.)
        self.assertEqual(prep.DEV80_SEED_DOMAIN,"B24_DEV80_DAPS_SOLVER_V1")
        self.assertEqual(prep.SCREEN_SHA,
          "b516c8154cbbb790d8a3592b86736bb0d4bd47d0833d85ecf3d6a9d710e950ba")
        for rep in range(2,8):
            self.assertNotIn(22020+rep*1000,(22020,23020))
        for label,iid in (("A","01113"),("C","02763")):
            seeds=[]
            for rep in range(8):
                parts=(prep.DEV80_SEED_DOMAIN,prep.SCREEN_SHA,
                       label,iid,str(rep))
                seed=int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:16],16)&((1<<63)-1)
                seeds.append(seed%(2**32))
            self.assertEqual(len(set(seeds)),8)
    def test_frozen_8_of_8_offline_gate_receipts(self):
        self.assertEqual(prep.GATE_SHA[
          "XP_F8_LEGACY_REPLAY_WAVE1_OFFLINE.json"],
          "45e69880b04adb3a8762a940b4cb3630a4f055ec7784368ef9d891476320b1da")
        self.assertEqual(prep.GATE_SHA[
          "XP_F8_LEGACY_REPLAY_WAVE2_OFFLINE.json"],
          "ab34db50115ee7daef5b55ac81b321b47a318b67028ab21cda8a92b9e52d0385")
    def test_no_gt_in_new_worker_contract(self):
        w=inspect.getsource(gpu.do)
        self.assertIn("check_cfg",w)
        self.assertIn("native_command",w)
        self.assertIn("no_GT_scoring=True",w)
        self.assertIn("no_labels_calculated=True",w)
        self.assertIn("no_scientific_outcome_opened=True",w)
        self.assertNotIn("gt_source(",w)
        self.assertNotIn("rho_signed(",w)
        self.assertNotIn("psnr(",w)
    def test_hard_budget_and_staged_run(self):
        c=inspect.getsource(ctl.run)
        self.assertIn("RESERVE_PER_JOB_SECONDS",c)
        self.assertIn("STOP_F8",c)
        self.assertIn("write_cost",c)
        self.assertIn("RESERVE_PER_JOB_SECONDS=1800.",inspect.getsource(ctl))
        self.assertIn("max_completed_this_session",c)
        self.assertIn("clean_only=False",c)
        self.assertIn("start_new_session=True",c)
        self.assertIn("XP_F8_920_GENERATION_COMPLETE_EMBARGOED",c)
    def test_image_schedule_outcome_blind(self):
        p=inspect.getsource(prep.manifest_rows)
        self.assertIn("XP_F8_SCHEDULE_IMAGE_BLOCK_V1",p)
        self.assertIn("for key in order",p)
        self.assertIn('range(2,8)',p)
        self.assertIn('range(4,8)',p)
    def test_python_syntax(self):
        for module in (prep,gpu,ctl):
            ast.parse(Path(module.__file__).read_text())
if __name__=="__main__":
    unittest.main()
