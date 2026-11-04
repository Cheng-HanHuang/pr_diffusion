"""A23 G7 native S_mid source-lock and CPU-only A20/A21 numerical pilot tests."""
from __future__ import annotations
import ast
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/"docs/track_r/executors/xp/theory_inputs/XP_A23_G7_GT_SOURCE_LOCK.json"
WORKER=ROOT/"scripts/track_r/xp_a33_a23_g7_smid_pilot_gpu.py"

class A23G7SourceLockTests(unittest.TestCase):
    def test_source_lock_is_prospective_and_points_to_accepted_280_GT(self):
        d=json.loads(LOCK.read_text())
        self.assertIn("FROZEN_BEFORE_NEW_RESULTS",d["status"])
        self.assertEqual(d["source_image_manifest_sha256"],
                         "d517a8ea337c5fed8925f925bf24bae21cea60a566eb29f6795bc31e350abfde")
        self.assertEqual(d["source_c2_task_manifest_sha256"],
                         "75e8fe296932044eb4ff1cd30300ef58a8f48ece02acd9b22ba3aa6f3bd7c7ac")
        self.assertEqual(d["n_images"],280)
        self.assertEqual((d["FFHQ"],d["ImageNet"]),(180,100))
        self.assertFalse(d["GPU_launch_authorized_by_this_freeze"])
        self.assertEqual(d["g7_order"],["S_mid_global2","S_mid_all8","2_forward","UnivFD"])
        self.assertEqual(d["S_mid_global2"]["timesteps_DDPM1000"],[300,400,500,600])
        self.assertEqual(d["S_mid_global2"]["draws_per_time"],[0,1,2,3])
        self.assertEqual(d["S_mid_global2"]["rotations"],["000","111"])
        self.assertEqual(d["S_mid_global2"]["seed_uid_rule"],
                         "XPCGT|<dataset>|<basename>")
        self.assertEqual(d["S_mid_global2"]["n_orbit_noise_examples_per_image"],32)
        self.assertEqual(d["S_mid_global2"]["batch_rule"],
                         "single timestep per native forward, A21 B8 versus B2 frozen numerical equivalence")
        self.assertAlmostEqual(d["GPU_budget"]["unspent_G7_h"],10-2.6265298946173345)

    def test_native_prospective_script_reuses_source_math(self):
        src=WORKER.read_text()
        ast.parse(src,str(WORKER))
        for token in [
            "from scripts.track_r.xp_a25_xpc3_mid_gpu import (",
            "fix_exact_native_timestep,orient_sm",
            "from scripts.track_r.xp_a29_a21_gpu_worker import score_group",
            "source(a.index,a.expected_head)",
            "score_group([(dict(uid=uid),x)],model,torch,alpha,",
            "orient_sm(x,uid,model,torch,alpha,prior_seed)",
            "A20_VERSUS_A21_B8_NUMERICAL_EQUIVALENCE",
            "A23_G7_S_MID_CPU_SOURCE_PREFLIGHT_PASS",
            "A23_G7_S_MID_PILOT_IMAGE_PASS",
            "no_inverse_problem_generation=True"
        ]:self.assertIn(token,src)
        self.assertNotIn("sampler.sample(",src)
        self.assertNotIn("original_A21_optional_gpu_replication",src)

    def test_ffhq_and_imagenet_native_checkpoint_config(self):
        d=json.loads(LOCK.read_text())
        f=d["models"]["ffhq"];im=d["models"]["imagenet"]
        self.assertEqual(f["num_channels"],128)
        self.assertEqual(f["num_res_blocks"],1)
        self.assertEqual(f["attention_resolutions"],16)
        self.assertEqual(im["num_channels"],256)
        self.assertEqual(im["num_res_blocks"],2)
        self.assertEqual(im["attention_resolutions"],"32,16,8")

    def test_cpu_b2_b8_math_on_mock_native_tweedie(self):
        try:
            import torch
            from scripts.track_r.xp_a25_xpc3_mid_gpu import orient_sm
            from scripts.track_r.xp_a29_a21_gpu_worker import score_group
            from prdiffusion.track_r.prior_score import frozen_alpha_schedule,prior_seed
        except (ImportError,ModuleNotFoundError):
            self.skipTest("PyTorch not available")
        torch.set_num_threads(2)
        class MockNative:
            def tweedie(self,z,sigma):
                return z-sigma*(z.sigmoid()-.5)*2
        g=torch.Generator(device="cpu").manual_seed(1234)
        x=torch.randn((1,3,256,256),generator=g,dtype=torch.float32)
        alpha=frozen_alpha_schedule()
        uid="XPCGT|ffhq|UNITTEST"
        a=orient_sm(x,uid,MockNative(),torch,alpha,prior_seed)
        b=score_group([(dict(uid=uid),x)],MockNative(),torch,alpha,prior_seed,8)[0]
        self.assertEqual(a["prior_choice"],b["prior_choice"])
        for k in ("L000","L111","prior_margin"):
            self.assertLessEqual(abs(float(a[k])-float(b[k])),1e-5)

if __name__=="__main__":
    unittest.main()
