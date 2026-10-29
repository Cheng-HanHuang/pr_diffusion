"""A12 G1a GT-free 1600-root fixed domain and two-root replay contracts."""
import ast,hashlib,inspect,unittest
from pathlib import Path
from scripts.track_r import xp_a12_g1a_prepare_cpu as m
class TestA12G1a(unittest.TestCase):
    def test_historical_seed_domain(self):
        self.assertEqual(m.HIST_SHA,"3107d40c495886c52ea2a407f4d8f74079278fffffaa93f4f55eff1edb5272f7")
        self.assertEqual(m.COMMON_Y_SHA,m.COMMON_Y_SHA)
        self.assertEqual(len({m.native_seed("XP|FAST|imagenet","sample.JPEG",r,2**31)
                              for r in range(32)}),32)
        self.assertTrue(all(m.native_seed("XP|FAST|imagenet","sample.JPEG",r,2**31)
                            not in {m.native_seed("XP|FAST|imagenet","sample.JPEG",x,2**31)
                                    for x in range(4,16)}
                            for r in range(16,32)))
    def test_generation_has_no_gt_and_no_score(self):
        code=inspect.getsource(m.main)
        self.assertIn("range(16,32)",code)
        self.assertIn("len(new)==1600",code)
        self.assertIn("len(replay)==2",code)
        self.assertIn("GT_OR_OUTCOME_LEAK",code)
        self.assertIn("historical_two_root_replay_required_before_GPU_production=True",code)
        self.assertIn("cumulative_project_ledger_required_before_GPU_admission=True",code)
    def test_native_sampling_is_unchanged_across_producer_check(self):
        # A12 has one added preflight contract for the earlier frozen
        # input HEAD, but the execution path remains verbatim FAST.
        root=Path(m.__file__).resolve().parents[2]
        old=(root/"scripts/track_r/xp_fast_gpu.py").read_text()
        new=(root/"scripts/track_r/xp_a12_g1a_fast_gpu.py").read_text()
        self.assertEqual(old[old.index("def execute(a):"):],
                         new[new.index("def execute(a):"):])
        self.assertIn("STOP_A12_G1A_PRECONTROLLER_SOURCE_DRIFT",new)
        self.assertIn("A12_G1A_FROZEN_PRODUCER",new)
    def test_syntax(self):
        root=Path(m.__file__).resolve().parents[2]
        for file in ("xp_a12_g1a_prepare_cpu.py","xp_a12_g1a_fast_gpu.py"):
            ast.parse((root/"scripts/track_r"/file).read_text())
if __name__=="__main__":
    unittest.main()
