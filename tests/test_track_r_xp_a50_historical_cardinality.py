"""A50 regression: historical B22 cardinality must not be conflated with XP-K K=4."""
import ast
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MAN=ROOT/"scripts/track_r/xp_a41_xpk_manifest_cpu.py"
SCORE=ROOT/"scripts/track_r/xp_a41_xpk_score_cpu.py"

class HistoricalCardinality(unittest.TestCase):
    def test_manifest_keeps_new_k4_but_daps_history_k2(self):
        s=MAN.read_text()
        ast.parse(s)
        self.assertIn('K=4',s)
        self.assertIn('HIST_K={"SITCOM":4,"FAST":4,"DAPS":2}',s)
        self.assertIn('len(panel)==100*hist_k',s)
        self.assertIn('refs=[group[image_id][i] for i in range(hist_k)]',s)

    def test_scorer_does_not_invent_daps_historical_runs_2_3(self):
        s=SCORE.read_text()
        ast.parse(s)
        self.assertIn('expected_hist=200 if family=="DAPS" else 400',s)
        self.assertIn('family_tau=next(iter(tau_values))',s)
        self.assertIn('native_tau=family_tau',s)
        self.assertNotIn('rr[row["run"]]',s)
        self.assertIn('historical_roots_per_image=(2 if family=="DAPS" else 4)',s)

if __name__=="__main__":
    unittest.main()
