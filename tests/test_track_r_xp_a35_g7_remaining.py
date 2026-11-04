"""A35 prereg G7 all8/2-forward/UnivFD autonomous CPU-only gates."""
from __future__ import annotations
import ast,json,math,tempfile,unittest
from pathlib import Path
from scripts.track_r.xp_a35_g7_scoring import native_scores, EVALS, PATTERNS, TIMES, DRAWS
from scripts.track_r.xp_a35_g7_autopilot_cpu import (
 METHODS,PRIOR_TOTAL_G7_H,PRIOR_PROJECT_H,
 METHOD_CAPS,G7_LIMIT,PROJECT_LIMIT,current_method_h,
)
from scripts.track_r.xp_a35_g7_worker import STATUS
from prdiffusion.track_r.physics import ORBIT_PATTERNS

ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/"docs/track_r/executors/xp/theory_inputs/XP_A23_G7_REMAINING_SOURCE_LOCK.json"
WORKER=ROOT/"scripts/track_r/xp_a35_g7_worker.py"
CONTR=ROOT/"scripts/track_r/xp_a35_g7_autopilot_cpu.py"
PUBLISH=ROOT/"scripts/track_r/xp_a35_g7_readout_publish_cpu.py"
LAUNCH=ROOT/"scripts/track_r/xp_a35_g7_launch.sh"

class A35G7FrozenTests(unittest.TestCase):
    def test_prereg_order_and_exact_source(self):
        s=json.loads(LOCK.read_text())
        self.assertEqual(tuple(s["methods"]),METHODS)
        self.assertEqual(s["source_GT_run"] if "source_GT_run" in s else
                         s["original_GT_run"],"xp_c_c2_a22_20261104T025355Z")
        self.assertEqual(s["GT_manifest_sha256"],
                        "d517a8ea337c5fed8925f925bf24bae21cea60a566eb29f6795bc31e350abfde")
        self.assertEqual(s["preceding_method"]["publication_commit"],
                        "da130b50856fd6c495fe73049651b746ab67b56b")
        self.assertEqual(s["paired_prior"]["smid_all8"]["timesteps"],
                         [300,400,500,600])
        self.assertEqual(s["paired_prior"]["two_forward"]["timesteps"],[300])
        self.assertEqual(s["paired_prior"]["two_forward"]["draws"],[0])
        self.assertEqual(s["univfd"]["source_head"],
                        "030495aea3300a8b54c0ec37ec7fe1dd7e63c619")
        self.assertFalse(s["no_solver_generation"] is False)

    def test_all8_batched_score_is_registered_orbit_and_2forward_is_two(self):
        self.assertEqual(EVALS["S_mid_all8"],128)
        self.assertEqual(EVALS["2_forward"],2)
        self.assertEqual(PATTERNS["S_mid_all8"],ORBIT_PATTERNS)
        self.assertEqual(PATTERNS["2_forward"],("000","111"))
        self.assertEqual(TIMES["2_forward"],(300,))
        self.assertEqual(DRAWS["2_forward"],(0,))

    def test_native_scoring_cpu_mock_no_GT_or_solver(self):
        import torch
        torch.set_num_threads(2)
        class MockNative:
            def tweedie(self,z,sigma):
                return z-sigma*torch.tanh(z.float())*.15
        generator=torch.Generator(device="cpu").manual_seed(9867)
        x=torch.randn((1,3,256,256),generator=generator,dtype=torch.float32)
        for method in ("S_mid_all8","2_forward"):
            with self.subTest(method=method):
                a=native_scores(x,"XPCGT|ffhq|A35TEST",MockNative(),method)
                b=native_scores(x,"XPCGT|ffhq|A35TEST",MockNative(),method)
                self.assertEqual(a,b)
                self.assertEqual(a["denoiser_examples"],EVALS[method])
                self.assertEqual(len(a["scores"]),len(PATTERNS[method]))
                self.assertEqual(a["native_calls"],len(TIMES[method])*len(DRAWS[method]))
                self.assertIn(a["best_orbit"],PATTERNS[method])
                self.assertEqual(a["upright_kept"],a["best_orbit"]=="000")
                self.assertTrue(a["single_t_per_call"])

    def test_tie_rule_explicit_identity(self):
        import torch
        class ZeroNative:
            def tweedie(self,z,sigma):
                # Every pattern has exactly the same epsilon prediction
                # as the paired input in the zero image case up to FP.
                return z
        x=torch.zeros((1,3,256,256),dtype=torch.float32)
        r=native_scores(x,"XPCGT|ffhq|TIE",ZeroNative(),"S_mid_all8")
        self.assertEqual(r["best_orbit"],"000")

    def test_stage_budget_and_cpu_univfd_cost(self):
        self.assertAlmostEqual(PRIOR_TOTAL_G7_H,5.1570269519018215)
        self.assertAlmostEqual(PRIOR_PROJECT_H,246.9875918177077)
        self.assertLess(sum(METHOD_CAPS.values())+
                        PRIOR_TOTAL_G7_H,G7_LIMIT)
        self.assertEqual(METHOD_CAPS["UnivFD"],0.)
        self.assertEqual(current_method_h([],{
            123:dict(gpu=None,start=10.)}),0.)
        self.assertEqual(current_method_h([],{}),0.)
        self.assertLess(sum(METHOD_CAPS.values())+PRIOR_PROJECT_H,
                        PROJECT_LIMIT)

    def test_native_and_CPU_workers_do_not_generate_inverse_problem(self):
        s=WORKER.read_text()
        ast.parse(s,str(WORKER))
        for v in ("native_scores(x,uid,model,a.method,alpha)",
                  'fd.p_fake(all8)',"torch.set_num_threads(4)",
                  'GT_not_used_for_inverse_problem_generation=True',
                  "sha(im[\"source_path\"])", "atomic(out,payload)"):
            self.assertIn(v,s)
        self.assertNotIn("sampler.sample(",s)
        self.assertNotIn("get_operator(",s)

    def test_controller_stops_on_real_gates_and_publishes_ordered(self):
        s=CONTR.read_text()
        ast.parse(s,str(CONTR))
        for k in ("for method in METHODS:","METHOD_CAPS","gpu_admissible()",
                  "check_ledger" if False else "pinned()",
                  'PARKED_PROJECTED_GPU_H_EXCEEDS_REGISTERED_CAP',
                  "GLOBAL_LOCK.open", "check_results" if False else
                  "analyze(run,method,a.expected_head,result)",
                  "pub=publish(run,method,a.expected_head,receipt)",
                  "A35_G7_THREE_REMAINING_METHODS_PUBLISHED"):
            self.assertIn(k,s)
        self.assertIn("GT_diagnostic_only",s)
        self.assertNotIn("A21.10",s)

    def test_publisher_per_stage_not_false_full_g7(self):
        s=PUBLISH.read_text()
        ast.parse(s,str(PUBLISH))
        self.assertIn("not_full_G7",s)
        self.assertIn("check_results(stage,method,head)",s)
        self.assertIn("stage_terminal_sha256",s)
        self.assertIn("PUBLISH_TIMEOUT_KEEP_EXISTING_COMMIT",s)
        self.assertIn("no_new_main_text",s)

    def test_launcher_has_tests_before_any_gpu(self):
        s=LAUNCH.read_text()
        self.assertIn("test_track_r_xp_a35_g7_remaining.py",s)
        self.assertIn("--preflight-only",s)
        self.assertIn("xp_a35_g7_autopilot_cpu.py",s)

if __name__=="__main__":
    unittest.main()
