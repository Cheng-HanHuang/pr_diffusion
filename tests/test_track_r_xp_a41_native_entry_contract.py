"""CPU-only structural tests: never claim a native GPU result from this file."""
import ast
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
GPU=ROOT/"scripts/track_r/xp_a41_xpk_native_gpu.py"
MANIFEST=ROOT/"scripts/track_r/xp_a41_xpk_manifest_cpu.py"
SCORE=ROOT/"scripts/track_r/xp_a41_xpk_score_cpu.py"

class NativeEntryContract(unittest.TestCase):
    def test_native_worker_cli_is_reachable_at_module_level(self):
        tree=ast.parse(GPU.read_text(),filename=str(GPU))
        top={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        for name in ("valid_input","sit_run","fast_run","daps_run","main"):
            self.assertIn(name,top,f"{name} must not be nested inside a family worker")
        self.assertEqual(sum(isinstance(n,ast.FunctionDef) and n.name=="main" for n in tree.body),1)
        entry=[n for n in tree.body if isinstance(n,ast.If)
               and isinstance(n.test,ast.Compare)
               and isinstance(n.test.left,ast.Name)
               and n.test.left.id=="__name__"]
        self.assertEqual(len(entry),1,"missing top-level __main__ CLI")
        self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                            and n.func.id=="main" for n in ast.walk(entry[0])))
    def test_each_family_has_real_two_mode_native_entry(self):
        s=GPU.read_text()
        for family in ("SITCOM","FAST","DAPS"):
            self.assertIn(f'if a.family=="{family}"',s)
        self.assertIn('choices=("historical_replay","calibrated")',s)
        self.assertIn('if a.phase=="calibrated"',s)
        self.assertIn("def complete_valid(",s)
        self.assertIn("no_GT_on_GPU=True",s)
        self.assertIn("XPK_GROUP_COMPLETE.json",s)
    def test_source_and_scoring_are_separate_cpu_stages(self):
        for p in (MANIFEST,SCORE):
            ast.parse(p.read_text(),filename=str(p))
        self.assertIn("XPK_GPU_GT_FREE_MANIFEST.json",MANIFEST.read_text())
        self.assertIn("XPK_CPU_ONLY_HIST_REFERENCES.json",MANIFEST.read_text())
        self.assertIn("0.1",SCORE.read_text())
        self.assertIn("historical_replay",GPU.read_text())
if __name__=="__main__":
    unittest.main()
