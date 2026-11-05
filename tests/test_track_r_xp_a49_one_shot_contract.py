"""A49 one-shot XP-K controller contract tests. CPU-only."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CTRL = ROOT / "scripts/track_r/xp_a49_xpk_one_shot_autopilot.py"
LAUNCH = ROOT / "scripts/track_r/xp_a49_xpk_one_shot_launch.sh"

class A49Contract(unittest.TestCase):
    def test_ast_and_single_controller_entry(self):
        src = CTRL.read_text()
        tree = ast.parse(src, filename=str(CTRL))
        top = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in (
            "ensure_preflight", "run_task_queue", "score_replay",
            "pilot_projection", "score_calibration", "make_cp4_report",
            "publish", "main",
        ):
            self.assertIn(name, top)

    def test_registered_order_and_hard_science_gates(self):
        s = CTRL.read_text()
        main = s[s.index("def main(a):"):s.index('if __name__ == "__main__":')]
        order = [
            "ensure_preflight(run, a)",
            '"IDENTITY_REPLAY"',
            "score_replay(run, a, family)",
            "ensure_f8_gate(run, a.expected_head)",
            '"CALIBRATION_PILOT"',
            "pilot_projection(run, a)",
            '"CALIBRATION_FULL"',
            "score_calibration(run, a, family)",
            "make_cp4_report(run, a)",
            "publish(run, a, report)",
        ]
        positions = [main.index(x) for x in order]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('0 <= r["abs_raw_psnr_delta_db"] <= 0.1', s)
        self.assertIn("STOP_ASK_A23_5B_GATE", s)
        self.assertIn("F8_OFFLINE_EXACT_8", s)
        self.assertIn("no_panel_reduction", s)

    def test_resource_wait_is_not_science_failure(self):
        s = CTRL.read_text()
        self.assertIn('status="WAIT_SAFE_GPU"', s)
        self.assertIn("time.sleep(a.wait_seconds)", s)
        self.assertIn('"foreign_pids"', s)
        self.assertIn("p.terminate()", s)
        self.assertIn("projected <= 0.80 * c", s)
        self.assertIn(">= 4 * 1024", s)

    def test_preferred_replay_gpus_and_two_gpu_default(self):
        s = CTRL.read_text()
        self.assertIn("PREFERRED_REPLAY_GPUS = (1, 0, 2, 3)", s)
        self.assertIn('p.add_argument("--max-gpus", type=int, default=2)', s)

    def test_crash_accounting_is_durable(self):
        s = CTRL.read_text()
        self.assertIn('j.mutate(jp, "START"', s)
        self.assertIn('j.mutate(run / "XPK_GPU_EVENTS.jsonl", "END"', s)
        self.assertIn("recover_unfinished(run, a)", s)
        self.assertIn("unfinished_workers", s)

    def test_publication_is_one_commit_then_same_commit_retried(self):
        s = CTRL.read_text()
        self.assertIn("results(track-r): publish XP-K CP4 A49 final return", s)
        self.assertIn('f"{pub}:refs/heads/{a.branch}"', s)
        self.assertIn("REMOTE_BRANCH_MOVED_DURING_CAMPAIGN", s)
        self.assertIn("XP_K_CP4_ALL_THREE_FAMILIES_PUBLISHED", s)

    def test_launcher_is_one_background_command_contract(self):
        s = LAUNCH.read_text()
        self.assertIn("nohup", s)
        self.assertIn("xp_a49_xpk_one_shot_autopilot", s)
        self.assertIn("MAX_GPUS=2", s)
        self.assertIn("LAST_XPK_A49_RUN.txt", s)
        self.assertIn("checkout --detach", s)

if __name__ == "__main__":
    unittest.main()
