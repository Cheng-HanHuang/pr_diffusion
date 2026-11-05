"""A40 CPU-only actual native source binding/one-param gate contracts."""
import ast
import unittest
from pathlib import Path
from scripts.track_r import xp_a40_xpk_native_binding_cpu as M
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts/track_r/xp_a40_xpk_native_binding_cpu.py"
A39=ROOT/"scripts/track_r/xp_a39_a24_xpk_cpu_preflight.py"
class NativeBindingCPUContract(unittest.TestCase):
    def test_real_native_files_and_original_daps_lineage(self):
        self.assertEqual(M.PIN_SITCOM,
            "275ab67efbd8146bffca20155171ba6be1169c09")
        self.assertEqual(M.PIN_FAST,
            "a9758b38906b7c7950fa04491f37f6634948db26")
        self.assertEqual(M.PIN_DAPS,
            "e7a77d094167084faed19b599b96673b7bb11447")
        self.assertEqual(M.PATCH_DAPS_SAMPLER_BLOB,
            "ae1fc6908bb1e7f0c0349d4f82cdf20364389db8")
        self.assertIn("pr_diffusion_b19_solver/external/daps",str(M.DAPS))
        self.assertNotIn("DAPS_XP_NATIVE_4K",str(M.DAPS))
        self.assertAlmostEqual(M.TARGET_DAPS,0.07071067811865475,places=14)
        self.assertEqual(M.F8_MANIFEST_SHA,
            "0929a94537c30faa893346a000c6cca5b51d5f45b48c6976cccb1c64899f5c215f97b4f1"[:64] if False else
            "0929a94537c30faa893346a000c6cca5b51d5f45b48fbc2dfb0f6e64fa2537b9")
    def test_static_ast_and_registered_source_bindings_in_script(self):
        s=SCRIPT.read_text()
        ast.parse(s,str(SCRIPT))
        for pattern in (
           'checked_native_source(SITCOM,PIN_SITCOM,"sampler.py")',
           'self.tau ** 2',
           'LangevinDynamics(**lgvd_config)',
           '"self.tau" not in euler_src',
           'checked_native_source(FAST,PIN_FAST,"main.py")',
           'self.epsilon = float(epsilon)',
           '(self.epsilon / nrm).clamp_max(1.0)',
           'epsilon=conf["epsilon"]',
           'native_command(row,dest)',
           'task.pixel.mcmc_sampler_config.tau=',
           'set(left)==set(right) and changes==[key]',
           'A23_5b_gate_ii_two_B22_root_replay_passed=False',
           'calibrated_XP_K_generation_authorized=False',
           'no_gpu_jobs_launched=True',
        ):
            self.assertIn(pattern,s)
    def test_science_gate_cannot_be_promoted_from_config_alone(self):
        s=SCRIPT.read_text()
        self.assertIn("A23_5b_gate_i_committed_actual_adapter_diff=False",s)
        self.assertIn("A40_NATIVE_BINDINGS_PARTIAL_STOP",s)
        self.assertIn("SCIENCE_GATE_FAILURE_STOP_ASK",s)
        self.assertNotIn("torch.cuda",s)
        self.assertNotIn("subprocess.Popen",s)
        self.assertNotIn("sampler.sample(",s)
    def test_A39_preserves_unrun_replay(self):
        s=A39.read_text()
        self.assertIn('gate_ii_original_two_B22_root_identity_replay',s)
        self.assertIn("NOT_RUN_ALL_FAMILIES",s)
        self.assertIn("GPU_generation_authorized",s)
if __name__=="__main__":
    unittest.main()
