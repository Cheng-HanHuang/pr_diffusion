"""A24 XP-I2 supervisor structural and replay-policy regression (CPU only)."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class A24I2AutopilotTests(unittest.TestCase):
    def test_native_controller_compiles_and_has_hard_gates(self):
        p = ROOT / "scripts/track_r/xp_a24_xpi2_autopilot.py"
        s = p.read_text()
        ast.parse(s, filename=str(p))
        for token in (
            "STAGE_CAP_H = 40.0",
            "TAIL_CAP_H = 60.0",
            "PROJECT_CAP_H = 400.0",
            "REQUEST_MIB = 16384",
            "fraction=.80, clean_only=False",
            "a.roots_source_head",
            "XP_N_CONTINUATION_FINAL_STATUS.json",
            "A24_XPI2_TWO_BAG_EXACT_REPLAY_PASS",
            "I2_REPLAY_NOT_BITWISE_IDENTICAL",
            "I2_UNRECEIPTED_PARTIAL_BAG",
            "live[(str(run), i)]",
            "kill_ours(live)",
            "project_prior_ledger_sha256",
            "scripts/track_r/xp_a24_xpi2_pcdaps_gpu.py"):
            self.assertIn(token, s)
        for forbidden in ("torch.load", "Image.open(", "gt_path", "test-ffhq"):
            self.assertNotIn(forbidden, s)

    def test_replay_comparison(self):
        from scripts.track_r.xp_a24_xpi2_autopilot import replay_equal
        a = {"particles_pt_sha256": "a", "particle_pngs": [
            {"sha256": "0"}, {"sha256": "1"}, {"sha256": "2"},
            {"sha256": "3"}]}
        b = {"particles_pt_sha256": "a", "particle_pngs": [
            dict(q) for q in a["particle_pngs"]]}
        self.assertTrue(replay_equal(a, b))
        b["particle_pngs"][0]["sha256"] = "changed"
        self.assertFalse(replay_equal(a, b))
        b["particle_pngs"][0]["sha256"] = "0"
        b["particles_pt_sha256"] = "changed"
        self.assertFalse(replay_equal(a, b))

if __name__ == "__main__":
    unittest.main()
